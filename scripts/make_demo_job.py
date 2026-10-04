"""Build a synthetic job with ZERO machine learning.

The walking skeleton.  It exercises every contract the browser depends on --
multichannel WAV channel order, ``tracks.json`` box geometry, the null-box
convention, the drift corrector, the gain switch -- without loading torch,
speechbrain, or mediapipe.  If the UI misbehaves, this tells you in seconds
whether the fault is in the frontend or in the models.

    python scripts/make_demo_job.py                 # -> runs/demo, open /?job=demo
    python scripts/make_demo_job.py --video-only    # -> a clip to upload for real

The two "voices" are synthetic glottal buzzes at different pitches, and each
face's mouth is driven by *the same envelope* that drives its voice.  So the
lips genuinely match the audio: clicking the moving mouth really should select
the voice you hear.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dsp, media                                    # noqa: E402
from app.config import CONFIG, RUNS_DIR, ensure_dirs          # noqa: E402
from app.serialization import dumps as json_dumps             # noqa: E402

W, H = 640, 360
FPS = 25.0
SR = CONFIG.audio.sample_rate

# (start, end) speech turns in seconds.  The 5.5-7.0 window is deliberate
# overlap -- that is where naive gating fails and where the demo has to hold up.
TURNS_A = [(0.4, 2.6), (5.5, 7.8), (10.4, 12.0)]
TURNS_B = [(2.9, 5.2), (6.6, 9.0), (12.4, 14.6)]
DURATION = 15.0


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

def _envelope(turns: list[tuple[float, float]], n: int, sr: float) -> np.ndarray:
    """Speech-like amplitude envelope: syllabic ripple inside each turn."""
    t = np.arange(n) / sr
    env = np.zeros(n, dtype=np.float64)
    for start, end in turns:
        m = (t >= start) & (t < end)
        if not m.any():
            continue
        local = t[m] - start
        # ~4 Hz syllable rate, never quite reaching zero mid-turn
        syl = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 3.8 * local + 0.7))
        # 60 ms raised-cosine edges so the turn itself has no click
        edge = np.minimum(local, (end - start) - local)
        fade = np.clip(edge / 0.06, 0, 1)
        env[m] = syl * (0.5 - 0.5 * np.cos(np.pi * fade))
    return env


def _voice(f0: float, formants: list[tuple[float, float]],
           env: np.ndarray, sr: float, seed: int) -> np.ndarray:
    """A cheap two-formant buzz.  Not speech, but it has the right structure:
    a harmonic stack under a moving envelope, which is all the energy-envelope
    matcher and the gate ever look at."""
    rng = np.random.default_rng(seed)
    n = len(env)
    t = np.arange(n) / sr

    # Slight pitch wobble so the two voices never phase-lock into one tone.
    jitter = 1.0 + 0.02 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0 * jitter) / sr

    sig = np.zeros(n)
    for k in range(1, 25):
        if f0 * k > sr / 2 * 0.9:
            break
        gain = 1.0 / k
        for fc, bw in formants:                      # crude formant emphasis
            gain += 0.9 / (1 + ((f0 * k - fc) / bw) ** 2)
        sig += gain * np.sin(k * phase + rng.uniform(0, 6.28))

    sig /= np.abs(sig).max() + 1e-12
    sig += 0.004 * rng.standard_normal(n)            # a little breath
    return (sig * env).astype(np.float32)


def build_audio() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(DURATION * SR)
    env_a = _envelope(TURNS_A, n, SR)
    env_b = _envelope(TURNS_B, n, SR)
    a = _voice(118.0, [(650, 90), (1180, 140)], env_a, SR, seed=1) * 0.80
    b = _voice(207.0, [(430, 80), (2100, 200)], env_b, SR, seed=2) * 0.72
    return np.stack([a, b]), env_a, env_b


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

FACES = [
    # (centre x, centre y, radius x, radius y, skin, label)
    (0.28, 0.50, 0.115, 0.185, (214, 176, 148), "Speaker A"),
    (0.72, 0.50, 0.115, 0.185, (196, 158, 132), "Speaker B"),
]


def _draw_frame(frame: np.ndarray, openness: list[float], bob: list[float]) -> None:
    yy, xx = np.mgrid[0:H, 0:W]
    frame[:, :] = np.linspace(28, 46, H, dtype=np.uint8)[:, None, None]

    for (cx, cy, rx, ry, skin, _), open_amt, dy in zip(FACES, openness, bob):
        px, py = cx * W, (cy + dy) * H
        ax, ay = rx * W, ry * H

        head = ((xx - px) / ax) ** 2 + ((yy - py) / ay) ** 2 <= 1.0
        frame[head] = skin

        for ex in (-0.38, 0.38):                                     # eyes
            exx, eyy = px + ex * ax, py - 0.30 * ay
            eye = ((xx - exx) / (0.13 * ax)) ** 2 + ((yy - eyy) / (0.09 * ay)) ** 2 <= 1.0
            frame[eye] = (30, 30, 38)

        # Mouth height tracks the voice envelope: this is the visual signal the
        # real pipeline would extract from the lips.
        mh = (0.03 + 0.20 * open_amt) * ay
        myy = py + 0.42 * ay
        mouth = ((xx - px) / (0.34 * ax)) ** 2 + ((yy - myy) / mh) ** 2 <= 1.0
        frame[mouth] = (86, 34, 42)


def build_video(path: Path, env_a: np.ndarray, env_b: np.ndarray) -> list[list]:
    n_frames = int(DURATION * FPS)
    per = SR / FPS

    def sample(env: np.ndarray, i: int) -> float:
        lo, hi = int(i * per), int((i + 1) * per)
        return float(np.clip(env[lo:hi].max() if hi > lo else 0.0, 0, 1))

    cmd = [
        media.ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", f"{FPS:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    boxes: list[list] = [[], []]
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    try:
        for i in range(n_frames):
            oa, ob = sample(env_a, i), sample(env_b, i)
            bob = [0.006 * np.sin(i / 9.0), 0.006 * np.sin(i / 11.0 + 1.7)]
            _draw_frame(frame, [oa, ob], bob)
            proc.stdin.write(frame.tobytes())

            for k, (cx, cy, rx, ry, _, _) in enumerate(FACES):
                # Frames 300-325 drop Speaker B entirely, to prove the null-box
                # path: the overlay must vanish, not freeze on a stale box.
                if k == 1 and 300 <= i < 325:
                    boxes[k].append(None)
                    continue
                y = cy + bob[k]
                boxes[k].append([round(cx - rx, 4), round(y - ry, 4),
                                 round(2 * rx, 4), round(2 * ry, 4)])
        proc.stdin.close()
    finally:
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed writing the demo video:\n"
                               + "\n".join(err.splitlines()[-12:]))
    return boxes


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-id", default="demo")
    ap.add_argument("--video-only", action="store_true",
                    help="write only a mixed-audio clip, for uploading to the real pipeline")
    args = ap.parse_args()

    ensure_dirs()
    t0 = time.time()

    print("synthesising audio ...")
    stems_raw, env_a, env_b = build_audio()

    if args.video_only:
        out = RUNS_DIR / "synthetic_input.mp4"
        silent = out.with_name("_silent.mp4")
        mix_wav = out.with_name("_mix.wav")
        print("rendering video ...")
        build_video(silent, env_a, env_b)
        media.write_multichannel_wav(stems_raw.sum(axis=0), mix_wav, SR)
        media._run([media.ffmpeg_exe(), "-y", "-i", str(silent), "-i", str(mix_wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", str(out)])
        silent.unlink(missing_ok=True)
        mix_wav.unlink(missing_ok=True)
        print(f"\nupload this through the UI -> {out}")
        return

    job_dir = RUNS_DIR / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print("rendering video ...")
    boxes = build_video(job_dir / "video.mp4", env_a, env_b)

    print("gating ...")
    stems_demo = dsp.apply_gate(stems_raw, sample_rate=SR, cfg=CONFIG.gate)

    # One gain across both files, exactly as pipeline.py does it -- the demo job
    # exists to exercise the real contracts, so it must not take a shortcut the
    # real exporter does not take.
    gain = media.peak_gain(stems_demo, stems_raw)
    if gain != 1.0:
        stems_demo = stems_demo * gain
        stems_raw = stems_raw * gain
    media.write_multichannel_wav(stems_demo, job_dir / "stems_demo.wav", SR, gain=1.0)
    media.write_multichannel_wav(stems_raw, job_dir / "stems_raw.wav", SR, gain=1.0)

    n_frames = len(boxes[0])
    (job_dir / "tracks.json").write_text(json.dumps({
        "fps": FPS, "width": W, "height": H, "n_frames": n_frames,
        "tracks": [
            {"id": k, "label": FACES[k][5], "channel": k,
             "confidence": 1.0, "reliable": True, "boxes": boxes[k]}
            for k in range(2)
        ],
    }), encoding="utf-8")

    stats = {f"channel_{i}": dsp.silence_stats(stems_demo[i]) for i in range(2)}
    stats["whisper_db"] = {
        f"channel_{i}": dsp.whisper_db(
            stems_demo[i], stems_raw[i],
            np.abs(np.delete(stems_raw, i, axis=0)).max(axis=0),
            sample_rate=SR)
        for i in range(2)
    }
    stats["export_gain"] = round(gain, 6)
    # Every key pipeline.py emits, including `visual`, `alignment` and `config`.
    # The demo job's whole purpose is to exercise the browser's contracts, so a
    # meta.json missing keys the real pipeline writes would let a consumer that
    # assumes they exist pass here and fail on the first real clip.
    meta = {
        "sample_rate": SR, "duration": DURATION, "fps": FPS,
        "width": W, "height": H, "n_speakers": 2, "n_tracks": 2,
        "vision_backend": "synthetic", "separator": "synthetic", "device": "none",
        "assignment": [0, 1], "confidence": [1.0, 1.0],
        "silence": stats,
        "alignment": {"boundaries": 0, "flipped": 0,
                      "held_boundaries": [], "min_margin": 0.0},
        "visual": {"enabled": bool(CONFIG.gate.visual_fusion),
                   "fused_stems": 0, "total_stems": 2, "veto_cost": [0.0, 0.0]},
        "elapsed_s": round(time.time() - t0, 2),
        "config": CONFIG.to_dict(),
        "synthetic": True,
    }
    # json_dumps, not json.dumps: a perfectly-muted synthetic channel reports a
    # non-finite floor, and this file is served straight to the browser.
    (job_dir / "meta.json").write_text(json_dumps(meta, indent=2, default=str),
                                       encoding="utf-8")

    print(json_dumps(stats, indent=2))
    print(f"\nartefacts -> {job_dir}")
    print(f"start the server, then open:  http://127.0.0.1:8000/?job={args.job_id}")


if __name__ == "__main__":
    main()
