"""Where does the user's mid-clip "speaker swap" actually come from?

``diag_perm.py`` exonerated the chunk stitcher on this clip: two independent
ground truths (pitch, and a decisive 5 s-overlap reference) both say **no
permutation flip**, and the shipped overlap-correlation rule was 3/3 correct with
margins 0.95-1.80.  So the swap the user heard is downstream of separation.

Three suspects remain, and this script measures all three on the *shipped*
artefacts -- ``stems_raw.wav``, ``stems_demo.wav``, ``tracks.json`` -- so it is
measuring what the user actually heard, not a re-derivation of it.

  1. **the matcher** (`app/matching.py`).  It solves track<->stem ONCE over the
     whole clip.  If the correct pairing is not constant in the evidence -- or if
     the evidence is just weak -- the single global answer is right for part of
     the clip and wrong for the rest, which *sounds exactly like a mid-clip
     swap* with no permutation flip anywhere.  Its confidence on this clip was
     0.0376, below `min_confidence = 0.05`, for BOTH tracks.
  2. **vision track identity** (`app/vision.py::_assign`).  Greedy IoU against
     each track's last seen box, no motion prediction.  If two faces are ever
     confused, the lip signals swap between tracks -- and so does the click
     target.
  3. **the gate** (`app/dsp.py::apply_gate`).  A differential veto charged to the
     wrong stem silences the wrong channel for a stretch.

Run::

    PYTHONPATH=. python scripts/diag_swap.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dsp                                     # noqa: E402
from app.config import CONFIG                           # noqa: E402
from app.matching import energy_envelope                # noqa: E402

SR = CONFIG.audio.sample_rate
WIN_S = 4.0                     # sliding window for the windowed matcher


def _z(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def f0_median(x: np.ndarray, *, fmin: float = 70.0, fmax: float = 330.0,
              frame_ms: float = 40.0, hop_ms: float = 20.0,
              clarity: float = 0.45) -> tuple[float | None, float]:
    """Median f0 over voiced frames, and the voiced fraction.  ``None`` = no
    voiced frame in the window (the project's contract for "no measurement")."""
    frame = int(SR * frame_ms / 1000.0)
    hop = int(SR * hop_ms / 1000.0)
    if x.size < frame:
        return None, 0.0
    n_fr = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_fr)[:, None]
    seg = x[idx]
    seg = seg - seg.mean(axis=1, keepdims=True)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    ref = np.percentile(rms, 95)
    loud = rms > max(ref * 10 ** (-25.0 / 20.0), 1e-6)
    lo, hi = int(SR / fmax), int(SR / fmin)
    f0s = []
    for i in np.flatnonzero(loud):
        s = seg[i]
        best_r, best_l = 0.0, 0
        for lag in range(lo, min(hi + 1, frame - 1)):
            a, b = s[:-lag], s[lag:]
            d = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b))) + 1e-20
            r = float(np.dot(a, b)) / d
            if r > best_r:
                best_r, best_l = r, lag
        if best_r >= clarity and best_l > 0:
            f0s.append(SR / best_l)
    if not f0s:
        return None, 0.0
    return float(np.median(f0s)), len(f0s) / n_fr


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    meta = json.loads((job / "meta.json").read_text())
    tj = json.loads((job / "tracks.json").read_text())
    fps = float(tj["fps"])
    tracks = tj["tracks"]
    print(f"{job.name}: {len(tracks)} tracks @ {fps} fps, "
          f"{tj['n_frames']} frames")
    print(f"  track keys: {sorted(tracks[0].keys())}")
    for t in tracks:
        print(f"  track {t['id']} '{t['label']}' -> channel {t['channel']} "
              f"conf {t['confidence']} reliable {t['reliable']}")
    print(f"  meta assignment {meta.get('assignment')}  "
          f"confidence {meta.get('confidence')}")
    print(f"  meta alignment {meta.get('alignment')}")

    raw, sr = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    demo, _ = sf.read(job / "stems_demo.wav", dtype="float64", always_2d=True)
    raw, demo = raw.T, demo.T                            # (n_ch, T)
    n_ch = raw.shape[0]
    dur = raw.shape[1] / sr
    print(f"  stems: {n_ch} channels, {dur:.2f} s")

    # ---- 1. vision: does either track's box jump or cross the other? ------- #
    print("\n=== suspect 2: vision track identity ===")
    cx = []
    for t in tracks:
        c = []
        for b in t["boxes"]:
            c.append(None if b is None else (b[0] + b[2] / 2.0,
                                             b[1] + b[3] / 2.0))
        cx.append(c)
    n_fr = min(len(c) for c in cx)
    # A greedy-IoU identity swap shows up as a large single-frame centroid jump,
    # and (for two tracks) as the x-ordering of the two centroids reversing.
    for i, c in enumerate(cx):
        pres = [k for k in range(n_fr) if c[k] is not None]
        jumps = [(k, np.hypot(c[k][0] - c[p][0], c[k][1] - c[p][1]))
                 for p, k in zip(pres, pres[1:])]
        big = [(k / fps, d) for k, d in jumps if d > 0.05]
        gaps = [(p / fps, (k - p) / fps) for p, k in zip(pres, pres[1:])
                if k - p > 1]
        print(f"  track {i}: present {len(pres)}/{n_fr}  "
              f"max centroid jump {max((d for _, d in jumps), default=0):.4f}  "
              f"jumps>0.05: {len(big)}")
        if big[:6]:
            print(f"      first big jumps (t, dist): "
                  f"{[(round(a, 2), round(b, 3)) for a, b in big[:6]]}")
        if gaps:
            print(f"      dropout gaps (start_s, len_s): "
                  f"{[(round(a, 2), round(b, 2)) for a, b in gaps[:8]]}"
                  f"{' ...' if len(gaps) > 8 else ''}  n={len(gaps)}")
    if len(cx) == 2:
        order = []
        for k in range(n_fr):
            a, b = cx[0][k], cx[1][k]
            order.append(None if a is None or b is None else int(a[0] < b[0]))
        seen = [o for o in order if o is not None]
        flips = sum(1 for p, q in zip(seen, seen[1:]) if p != q)
        print(f"  x-ordering of the two centroids reverses {flips} time(s) "
              f"over {len(seen)} co-present frames"
              + ("  <-- clean: the two faces never cross" if flips == 0
                 else "  <-- INVESTIGATE"))

    # ---- 2. windowed matcher: is the correct pairing constant? ------------- #
    print("\n=== suspect 1: the matcher (solved once, globally) ===")
    mcfg = CONFIG.match
    envs = np.stack([energy_envelope(raw[j], sr, mcfg.env_frame_ms,
                                     smooth=mcfg.smooth_kernel)
                     for j in range(n_ch)])
    n_env = envs.shape[1]
    lipm = []
    for t in tracks:
        lip = np.array([np.nan if v is None else float(v)
                        for v in t.get("lip", [])], dtype=np.float64) \
            if t.get("lip") is not None else None
        if lip is None or lip.size == 0:
            lipm.append(None)
            continue
        r, ok = dsp.resample_hold(lip, fps, n_env, 1000.0 / mcfg.env_frame_ms)
        d = np.abs(np.diff(r, prepend=r[:1]))
        d[~ok] = 0.0
        lipm.append(d)

    if any(l is None for l in lipm):
        print("  tracks.json carries no lip signal; recomputing from video")
        from app.vision import FaceAnalyzer
        fa = FaceAnalyzer(CONFIG.vision)
        vt, _ = fa.analyze(str(job / "video.mp4"))
        lipm = []
        for t in vt:
            r, ok = dsp.resample_hold(np.asarray(t.lip, dtype=np.float64), fps,
                                      n_env, 1000.0 / mcfg.env_frame_ms)
            d = np.abs(np.diff(r, prepend=r[:1]))
            d[~ok] = 0.0
            lipm.append(d)

    n_tr = len(lipm)
    glob = np.array([[float(np.dot(_z(lipm[i]), _z(envs[j]))) / n_env
                      for j in range(n_ch)] for i in range(n_tr)])
    print(f"  GLOBAL score[track][channel] = {np.round(glob, 5).tolist()}")
    print(f"    global argmax pairing: "
          f"{[int(np.argmax(glob[i])) for i in range(n_tr)]}")

    w = max(1, int(WIN_S * 1000.0 / mcfg.env_frame_ms))
    print(f"  windowed ({WIN_S:g} s) score, and the pairing each window prefers:")
    hdr = "   t0     " + "  ".join(f"t{i}c{j}" for i in range(n_tr)
                                   for j in range(n_ch))
    print(hdr + "   prefers  margin")
    prefer = []
    for st in range(0, max(1, n_env - w + 1), w):
        sl = slice(st, st + w)
        sc = np.array([[float(np.dot(_z(lipm[i][sl]), _z(envs[j][sl]))) / w
                        for j in range(n_ch)] for i in range(n_tr)])
        if n_tr == 2 and n_ch == 2:
            keep = sc[0, 0] + sc[1, 1]
            flip = sc[0, 1] + sc[1, 0]
            p = [0, 1] if keep >= flip else [1, 0]
            m = abs(keep - flip)
        else:
            p, m = [int(np.argmax(sc[i])) for i in range(n_tr)], 0.0
        prefer.append((st * mcfg.env_frame_ms / 1000.0, p, m))
        vals = "  ".join(f"{sc[i, j]:+.3f}" for i in range(n_tr)
                         for j in range(n_ch))
        print(f"  {st * mcfg.env_frame_ms / 1000.0:5.1f}  {vals}   {p}  {m:+.4f}")
    changes = sum(1 for a, b in zip(prefer, prefer[1:]) if a[1] != b[1])
    print(f"  the preferred pairing changes {changes} time(s) across "
          f"{len(prefer)} windows")

    # ---- 3. who is actually who?  pitch per channel, over time ------------- #
    print("\n=== identity check: pitch per channel, per window ===")
    step = int(WIN_S * sr)
    print("   t0      " + "  ".join(f"ch{j} f0/voiced" for j in range(n_ch)))
    for st in range(0, raw.shape[1], step):
        cells = []
        for j in range(n_ch):
            f, v = f0_median(raw[j, st:st + step])
            cells.append("  n/a      " if f is None
                         else f"{f:6.1f}Hz/{v * 100:4.1f}%")
        print(f"  {st / sr:5.1f}  " + "  ".join(cells))

    # ---- 4. what the user hears on each button ---------------------------- #
    print("\n=== what each button actually plays (gated / demo stems) ===")
    print("   t0     " + "  ".join(f"ch{j} raw/demo dB" for j in range(n_ch))
          + "     lip activity per track")
    vis = []
    for i in range(n_tr):
        v, ok = dsp.visual_activity(
            _lip_of(tracks, i, job), src_fps=fps,
            n_frames=int(np.ceil(raw.shape[1] / (sr * CONFIG.gate.frame_ms / 1000))),
            frame_ms=CONFIG.gate.frame_ms,
            smooth_ms=CONFIG.gate.visual_smooth_ms,
            hold_ms=CONFIG.gate.visual_hold_ms,
            min_dynamic_range=CONFIG.gate.visual_min_dynamic_range)
        vis.append((v, ok))
    gfr = int(sr * CONFIG.gate.frame_ms / 1000)
    for st in range(0, raw.shape[1], step):
        cells = []
        for j in range(n_ch):
            r = raw[j, st:st + step]
            d = demo[j, st:st + step]
            rd = 10 * np.log10((r ** 2).mean() + 1e-12)
            dd = 10 * np.log10((d ** 2).mean() + 1e-12)
            zero = float(np.mean(d == 0.0)) * 100
            cells.append(f"{rd:+6.1f}/{dd:+6.1f} z{zero:4.0f}%")
        f0, f1 = st // gfr, (st + step) // gfr
        lv = "  ".join(
            f"t{i}={np.mean(vis[i][0][f0:f1][vis[i][1][f0:f1]]):.2f}"
            if np.any(vis[i][1][f0:f1]) else f"t{i}=n/a"
            for i in range(n_tr))
        print(f"  {st / sr:5.1f}  " + "  ".join(cells) + "   " + lv)


def _lip_of(tracks, i, job) -> np.ndarray:
    t = tracks[i]
    if t.get("lip"):
        return np.array([np.nan if v is None else float(v) for v in t["lip"]],
                        dtype=np.float64)
    from app.vision import FaceAnalyzer
    global _VT
    if "_VT" not in globals():
        _VT = FaceAnalyzer(CONFIG.vision).analyze(str(job / "video.mp4"))[0]
    return np.asarray(_VT[i].lip, dtype=np.float64)


if __name__ == "__main__":
    main()
