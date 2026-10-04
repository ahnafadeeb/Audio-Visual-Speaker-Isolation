"""Speaker / room adaptation for AV-TSE: ``python -m app.adapt``.

The released AV-MossFormer2 was trained on VoxCeleb2 at clean-ish SNRs. On the
team's own phone recordings it separates the two readers well in a quiet
room, and it degrades quickly once a room gets loud. Measured by Whisper WER
on the 37 s take with real noise mixed in (face 0 / face 1):

    clean 3.5 / 5.4 %    cafe 15 dB 10.5 / 21.7 %    babble 15 dB 37.2 / 25.0 %

Nothing downstream of the model recovers that: post-hoc enhancement and the
strict canceller both raised WER in noise, and enhancing BEFORE the model
destroyed the separation. What does help is showing the model these voices
under noise. This module fine-tunes the last few transformer layers and the
output heads on a recording of the same people, with noise mixed in, and
saves the changed tensors as an *adapter* that loads over the released
weights (avtse.load_model(adapter=...)). Trained on the 20 s take, tested on
the 37 s take it never saw, same WER path:

    clean 3.5 / 7.6 %    cafe 15 dB 3.5 / 5.4 %      babble 15 dB 4.7 / 13.0 %

and on a ground-truth remix of that take it leaks less of the other reader
(strict mode 16.3 / 13.4 -> 16.9 / 16.0 dB; in pauses -25.6 -> -30.3 dB).

An adapter is SPECIFIC to the people it was trained on. On a clip of three
strangers (ground-truth remix, strict mode) it lowered isolation from
24.0 / 25.8 dB to 20.0 / 17.1 dB. So it stores the SFace embedding of every
face it was trained on, and a job applies it per face, only to faces that
match one (cosine >= AVTSEConfig.adapter_face_match; the same person across
two takes measured 0.86-0.96, different people at most 0.34 over seven
clips). Everyone else is extracted by the released model.

How a training example is built, from one calibration recording that the
pipeline has already processed:

* target: face i's own separated channel over a random 2 s window, with face
  i's mouth crops from the same 2 s. These are the model's own outputs
  (pseudo-targets), so they are only as clean as a quiet-room run is -- which
  is why the calibration recording should be made somewhere quiet. Solo takes
  (one person on screen) are the cleanest possible targets.
* interferer: another face's channel from a DIFFERENT moment, at +-6 dB, so
  the model has to use the lips rather than timing to tell them apart.
* noise: babble (other people talking, from adapters/noise/), synthetic room
  noise (fan rumble, hiss, mains hum), recorded room tone of the venue when
  given, or none, at 0-25 dB SNR.

Usage::

    python -m app.adapt train team runs/<job-id> [more runs or videos ...]
    python -m app.adapt train team calib.mp4 --room venue_tone.m4a
    python -m app.adapt use team        # new jobs use it (adapters/active.txt)
    python -m app.adapt use none        # back to the released weights
    python -m app.adapt list

Training needs the GPU to itself for ~1 s per step at full power (4.6 GiB
peak at batch 4; ~2.3 s/step on a laptop GPU capped at 24 W); don't process
videos in the web app while it runs.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ADAPTERS_DIR, CONFIG, RUNS_DIR

log = logging.getLogger(__name__)

SR = 16000
SPF = SR // 25                 # audio samples per video frame
WIN_FRAMES = 50                # 2 s training windows
WIN = WIN_FRAMES * SPF
NOISE_DIR = ADAPTERS_DIR / "noise"
ACTIVE = ADAPTERS_DIR / "active.txt"

#: Output heads trained alongside the last N transformer layers.
HEADS = ("masknet.conv1d_out", "masknet.conv1_decoder", "masknet.output",
         "masknet.output_gate", "decoder")
_LAYER = re.compile(r"flashT\.(?:layers|fsmn)\.(\d+)\.")
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
             ".mp4", ".mov", ".mkv", ".webm", ".avi"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Face:
    """One face of one calibration run: its channel and its mouth crops."""
    run: str
    index: int
    audio: np.ndarray          # (T,) separated channel, 16 kHz
    roi: np.ndarray            # (F, 112, 112)
    starts: np.ndarray         # valid window start frames (on screen, speaking)


def _load_run(run_dir: Path) -> list[Face]:
    from .roi import boxes_to_array, extract_roi, n_video_frames_for
    import soundfile as sf

    st, sr = sf.read(run_dir / "stems_raw.wav", dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{run_dir}: stems_raw.wav is {sr} Hz, expected {SR}")
    st = st.T
    t = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    if len(t["tracks"]) != st.shape[0]:
        raise ValueError(f"{run_dir}: {len(t['tracks'])} tracks but {st.shape[0]} "
                         f"channels -- not an AV-TSE run?")
    n_vf = n_video_frames_for(st.shape[1])
    faces = []
    for i, tr in enumerate(t["tracks"]):
        r = extract_roi(str(run_dir / "video.mp4"), boxes_to_array(tr["boxes"]),
                        n_vf, (t["width"], t["height"]))
        n = min(len(r.roi), st.shape[1] // SPF)
        # 40 ms frame levels; a window is usable when the face is on screen
        # for all of it and it holds speech (not a 2 s pause).
        lvl = 10 * np.log10(np.mean(st[i, : n * SPF].reshape(n, SPF) ** 2, 1) + 1e-12)
        speech = lvl > np.percentile(lvl, 95) - 30
        on = ~r.absent[:n] if r.absent is not None else np.ones(n, bool)
        ok = [s for s in range(0, n - WIN_FRAMES)
              if on[s:s + WIN_FRAMES].all() and speech[s:s + WIN_FRAMES].mean() > 0.2]
        faces.append(Face(run_dir.name, i, st[i], r.roi, np.asarray(ok, int)))
        log.info("%s face %d: %.1f s, %d usable 2 s windows, face %.0f px",
                 run_dir.name, i, st.shape[1] / SR, len(ok), r.median_face_px)
    return faces


def face_embeddings(run_dir: Path, n: int = 16) -> list[np.ndarray | None]:
    """One SFace embedding per track of a finished run (mean of ``n`` frames).

    Stored in the adapter so a job can tell whether the people on screen are
    the people it was trained on -- see :func:`match_faces`.
    """
    import cv2
    from .reid import FaceEmbedder

    t = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    emb = FaceEmbedder()
    if not emb.available:
        return [None] * len(t["tracks"])
    want: dict[int, list[tuple[int, tuple]]] = {}
    for i, tr in enumerate(t["tracks"]):
        seen = [(f, tuple(b)) for f, b in enumerate(tr["boxes"]) if b is not None]
        for f, b in (seen[k] for k in np.linspace(0, len(seen) - 1, min(n, len(seen))).astype(int)) if seen else ():
            want.setdefault(f, []).append((i, b))
    sums = [np.zeros(128, np.float32) for _ in t["tracks"]]
    cap = cv2.VideoCapture(str(run_dir / "video.mp4"))
    try:
        for f in sorted(want):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            for i, b in want[f]:
                e = emb.embed(frame, b)
                if e is not None:
                    sums[i] += e
    finally:
        cap.release()
    return [s / np.linalg.norm(s) if np.linalg.norm(s) > 0 else None for s in sums]


_FACES_CACHE: dict[tuple[str, int], np.ndarray | None] = {}


def adapter_faces(path: Path) -> np.ndarray | None:
    """(K, 128) face embeddings an adapter was trained on; None if it has none."""
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _FACES_CACHE:
        from .avtse import read_adapter
        faces = read_adapter(path).get("faces") or []
        _FACES_CACHE.clear()
        _FACES_CACHE[key] = np.asarray(faces, np.float32) if faces else None
    return _FACES_CACHE[key]


def match_faces(path: Path, embs: list[np.ndarray | None]) -> list[float] | None:
    """Best cosine of each job face against the adapter's faces.

    None when the adapter stores no faces (it then applies to every face).
    A face with no embedding scores -1, i.e. never matches.
    """
    known = adapter_faces(path)
    if known is None:
        return None
    return [float((known @ e).max()) if e is not None else -1.0 for e in embs]


def _decode(path: Path) -> np.ndarray:
    from . import media
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        media.extract_audio(path, wav, SR)
        return media.read_audio(wav, SR)


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.sqrt(np.mean(v ** 2)) + 1e-9)


class NoiseBank:
    """Babble files, venue room tone, and a synthetic room-noise generator."""

    def __init__(self, files: list[Path], room: list[Path], rng) -> None:
        self.rng = rng
        self.babble = [x for x in (_decode(f) for f in files) if len(x) > WIN * 2]
        self.room = [x for x in (_decode(f) for f in room) if len(x) > WIN]
        log.info("noise bank: %d babble sources (%.0f s), %d room-tone clips (%.0f s)",
                 len(self.babble), sum(map(len, self.babble)) / SR,
                 len(self.room), sum(map(len, self.room)) / SR)

    def _segment(self, src: np.ndarray, n: int) -> np.ndarray:
        o = int(self.rng.integers(0, len(src) - n))
        return src[o:o + n]

    def _synthetic_room(self, n: int) -> np.ndarray:
        rng = self.rng
        f = np.fft.rfftfreq(n, 1 / SR)
        col = np.fft.irfft(np.fft.rfft(rng.standard_normal(n))
                           / np.maximum(f, 1.0) ** rng.uniform(0.3, 1.0), n)
        mains = rng.choice([50, 60])
        t = np.arange(n) / SR
        hum = sum(np.sin(2 * np.pi * mains * k * t + rng.uniform(0, 6)) / k for k in (1, 2, 3))
        return _norm(col) * rng.uniform(0.5, 1.5) + _norm(hum) * rng.uniform(0, 0.4)

    def draw(self, n: int) -> np.ndarray | None:
        kinds, p = ["room", "none"], [0.25, 0.15]
        if self.babble:
            kinds += ["babble", "cafe"]; p += [0.35, 0.25]
        if self.room:
            kinds.append("venue"); p.append(0.30)
        kind = self.rng.choice(kinds, p=np.asarray(p) / sum(p))
        if kind == "none":
            return None
        out = np.zeros(n)
        if kind in ("babble", "cafe"):
            for s in self.rng.choice(len(self.babble), size=int(self.rng.integers(3, 8))):
                out += _norm(self._segment(self.babble[s], n))
            out = _norm(out)
        if kind in ("room", "cafe"):
            out = out + self._synthetic_room(n)
        if kind == "venue":
            out = _norm(self._segment(self.room[int(self.rng.integers(len(self.room)))], n))
            if self.babble and self.rng.random() < 0.5:     # a room with people in it
                out = out + _norm(self._segment(
                    self.babble[int(self.rng.integers(len(self.babble)))], n)) * self.rng.uniform(0.3, 1.0)
        return _norm(out).astype(np.float32)


class Sampler:
    def __init__(self, faces: list[Face], noise: NoiseBank, rng) -> None:
        self.rng, self.noise = rng, noise
        # Every window trains. Holding the last 15% of a 20 s take out for
        # validation cost measurable quality (see train()); what validates an
        # adapter is a different take, not a slice of this one.
        self.faces = [(f, f.starts) for f in faces if len(f.starts)]
        if not self.faces:
            raise ValueError("no usable 2 s speech windows in the calibration runs")
        w = np.array([len(k) for _, k in self.faces], float)
        self.weights = w / w.sum()

    def _interferer_for(self, face: Face) -> tuple[Face, np.ndarray]:
        # Another face of the SAME run when there is one: two people in one
        # recording are certainly different people. A solo take is paired with
        # other runs' faces instead -- record one solo take per person.
        same = [(f, k) for f, k in self.faces if f.run == face.run and f.index != face.index]
        other = [(f, k) for f, k in self.faces if f.run != face.run]
        pool = same or other
        if not pool:
            raise ValueError("need two different faces: a two-person run, or solo takes of two people")
        return pool[int(self.rng.integers(len(pool)))]

    def draw(self):
        rng = self.rng
        f, starts = self.faces[int(rng.choice(len(self.faces), p=self.weights))]
        fa = int(starts[int(rng.integers(len(starts)))])
        g, gstarts = self._interferer_for(f)
        fb = int(gstarts[int(rng.integers(len(gstarts)))])
        if g.run == f.run:
            for _ in range(20):                   # a different moment of the other face
                if abs(fb - fa) >= WIN_FRAMES:
                    break
                fb = int(gstarts[int(rng.integers(len(gstarts)))])
        tgt = f.audio[fa * SPF: fa * SPF + WIN]
        itf = g.audio[fb * SPF: fb * SPF + WIN]
        itf = itf * np.sqrt(np.mean(tgt ** 2) / (np.mean(itf ** 2) + 1e-12)) * 10 ** (rng.uniform(-6, 6) / 20)
        # Always an interferer. A version that left it out 15% of the time
        # (to mimic turn-taking) taught the model to pass whatever voice it
        # hears: on the held-out take, pause leak went from -21.9 to -17.3 dB,
        # worse than the released weights.
        mix = tgt + itf
        n = self.noise.draw(WIN)
        if n is not None:
            mix = mix + n * np.sqrt(np.mean(mix ** 2)) * 10 ** (-rng.uniform(0, 25) / 20)
        gain = 10 ** (rng.uniform(-10, 6) / 20) * 0.3 / (np.abs(mix).max() + 1e-6)
        return ((mix * gain).astype(np.float32), f.roi[fa:fa + WIN_FRAMES],
                (tgt * gain).astype(np.float32))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _sisnr(e, t):
    """Scale-invariant SNR per row, dB -- the loss the model was trained with."""
    import torch
    t = t - t.mean(-1, keepdim=True)
    e = e - e.mean(-1, keepdim=True)
    s = (e * t).sum(-1, keepdim=True) / (t.pow(2).sum(-1, keepdim=True) + 1e-8) * t
    return 10 * torch.log10(s.pow(2).sum(-1) / ((e - s).pow(2).sum(-1) + 1e-8) + 1e-8)


def _batch(samples, device):
    import torch
    x = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
    v = torch.from_numpy(np.stack([s[1] for s in samples])).float().to(device)
    y = torch.from_numpy(np.stack([s[2] for s in samples])).to(device)
    return x, v, y


def _evaluate(model, val, device, bsz) -> float:
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(val), bsz):
            x, v, y = _batch(val[i:i + bsz], device)
            est = model(x, v).reshape(len(x), -1)[:, :WIN].float()
            out.append(_sisnr(est, y))
    return float(torch.cat(out).mean())


def train(name: str, runs: list[Path], *, noise_files: list[Path], room_files: list[Path],
          steps: int = 800, last_layers: int = 4, batch: int = 4, lr: float = 3e-5,
          seed: int = 0, progress=print) -> Path:
    import torch
    from .avtse import REPO_ID, load_model

    if not torch.cuda.is_available():
        raise RuntimeError("adapter training needs a CUDA GPU")
    device = "cuda"
    rng = np.random.default_rng(seed)
    faces = [f for r in runs for f in _load_run(r)]
    known = [e.tolist() for r in runs for e in face_embeddings(r) if e is not None]
    if not known:
        log.warning("no face embeddings (re-ID weights missing?); the adapter "
                    "will apply to EVERY face, including strangers")
    noise = NoiseBank(noise_files, room_files, rng)
    if not noise.babble:
        log.warning("no babble noise found (put speech recordings of OTHER people in %s); "
                    "training against room noise only", NOISE_DIR)
    tr = Sampler(faces, noise, rng)
    # In-sample, fixed draws: a check that training moved the right way, not
    # a generalisation estimate.
    va = Sampler(faces, noise, np.random.default_rng(seed + 1))
    val = [va.draw() for _ in range(48)]

    # A private copy: load_model's cached instance is the one the web app runs.
    model = copy.deepcopy(load_model(device))
    idx = sorted({int(m.group(1)) for n, _ in model.named_parameters()
                  if (m := _LAYER.search(n))})
    first = idx[-1] + 1 - last_layers
    names = []
    for n, p in model.named_parameters():
        m = _LAYER.search(n)
        on = bool((m and int(m.group(1)) >= first) or any(h in n for h in HEADS))
        p.requires_grad_(on)
        if on:
            names.append(n)
    params = [p for p in model.parameters() if p.requires_grad]
    base_val = _evaluate(model, val, device, batch)
    progress(f"training last {last_layers} layers + heads "
             f"({sum(p.numel() for p in params) / 1e6:.1f}M params), {steps} steps; "
             f"validation SI-SNR before: {base_val:.2f} dB")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    t0, hist = time.time(), []
    for step in range(steps):
        x, v, y = _batch([tr.draw() for _ in range(batch)], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            est = model(x, v).reshape(batch, -1)[:, :WIN].float()
        loss = -_sisnr(est, y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        hist.append(-loss.item())
        if (step + 1) % 50 == 0 or step + 1 == steps:
            el = time.time() - t0
            progress(f"step {step + 1}/{steps}  train SI-SNR {np.mean(hist[-50:]):5.2f} dB  "
                     f"{el:4.0f} s, ~{el / (step + 1) * (steps - step - 1):4.0f} s left")
    new_val = _evaluate(model, val, device, batch)
    progress(f"validation SI-SNR: {base_val:.2f} -> {new_val:.2f} dB")

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    out = ADAPTERS_DIR / f"{name}.pt"
    keep = set(names)
    torch.save({
        "format": 1,
        "base": REPO_ID,
        "state": {n: p.detach().cpu() for n, p in model.named_parameters() if n in keep},
        "last_layers": last_layers,
        "steps": steps,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [str(r) for r in runs],
        "noise": [str(f) for f in noise_files] + [str(f) for f in room_files],
        "val_sisnr": [round(base_val, 3), round(new_val, 3)],
        "faces": known,
    }, out)
    del model, opt
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _source_to_run(src: str) -> Path:
    """A run dir, a job id under runs/, or a video to process first."""
    p = Path(src)
    for cand in (p, RUNS_DIR / src):
        if (cand / "stems_raw.wav").is_file() and (cand / "tracks.json").is_file():
            return cand
    if p.is_file():
        import dataclasses
        from .pipeline import Pipeline
        out = RUNS_DIR / f"adapt_{p.stem}"
        if (out / "stems_raw.wav").is_file():
            print(f"{p.name}: reusing {out}")
            return out
        # Pseudo-targets come from the RELEASED model: training an adapter on
        # another adapter's output would compound whatever that one got wrong.
        cfg = dataclasses.replace(CONFIG, avtse=dataclasses.replace(CONFIG.avtse, adapter="none"))
        print(f"{p.name}: processing with the released model -> {out}")
        Pipeline(cfg).run(p, out)
        return out
    raise FileNotFoundError(f"{src}: not a run directory, a job id in {RUNS_DIR}, or a video file")


def _files(paths: list[str]) -> list[Path]:
    out = []
    for s in paths:
        p = Path(s)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(s)
    return out


def _set_active(name: str | None) -> None:
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    if name is None:
        ACTIVE.unlink(missing_ok=True)
    else:
        ACTIVE.write_text(name + "\n", encoding="utf-8")


def _active() -> str | None:
    try:
        return ACTIVE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.adapt", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fine-tune an adapter on calibration recordings")
    t.add_argument("name", help="adapter name (letters, digits, - and _)")
    t.add_argument("sources", nargs="+", help="run dirs, job ids, or videos of the same people")
    t.add_argument("--noise", nargs="*", default=None,
                   help=f"babble files/dirs: speech of OTHER people (default {NOISE_DIR})")
    t.add_argument("--room", nargs="*", default=[], help="room-tone recordings of the venue")
    t.add_argument("--steps", type=int, default=800)
    t.add_argument("--layers", type=int, default=4, help="last N transformer layers to train")
    t.add_argument("--no-activate", action="store_true", help="don't make it the active adapter")
    u = sub.add_parser("use", help="pick the adapter new jobs use ('none' = released weights)")
    u.add_argument("name")
    sub.add_parser("list", help="list adapters")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    if a.cmd == "list":
        cur = _active()
        found = sorted(ADAPTERS_DIR.glob("*.pt")) if ADAPTERS_DIR.is_dir() else []
        if not found:
            print("no adapters yet -- see `python -m app.adapt train -h`")
        import torch
        for f in found:
            try:
                m = torch.load(f, map_location="cpu", weights_only=True)
                info = (f"last {m.get('last_layers')} layers, {m.get('steps')} steps, "
                        f"{m.get('created')}, val SI-SNR {m.get('val_sisnr')}, from {m.get('sources')}")
            except Exception as exc:                          # noqa: BLE001
                info = f"unreadable: {exc}"
            print(f"{'*' if f.stem == cur else ' '} {f.stem:20s} {info}")
        print(f"\nactive: {cur or 'none (released weights)'}")
        return 0

    if a.cmd == "use":
        if a.name.lower() == "none":
            _set_active(None)
            print("new jobs use the released weights")
            return 0
        if not (ADAPTERS_DIR / f"{a.name}.pt").is_file():
            print(f"no adapter {a.name!r} in {ADAPTERS_DIR}", file=sys.stderr)
            return 1
        _set_active(a.name)
        print(f"new jobs use adapter {a.name!r}")
        return 0

    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.name) or a.name.lower() in ("none", "auto"):
        print(f"bad adapter name {a.name!r}", file=sys.stderr)
        return 1
    runs = [_source_to_run(s) for s in a.sources]
    noise = _files(a.noise) if a.noise is not None else (_files([str(NOISE_DIR)]) if NOISE_DIR.is_dir() else [])
    out = train(a.name, runs, noise_files=noise, room_files=_files(a.room),
                steps=a.steps, last_layers=a.layers)
    print(f"saved {out}")
    if not a.no_activate:
        _set_active(a.name)
        print(f"active: new jobs use adapter {a.name!r} (`python -m app.adapt use none` to undo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
