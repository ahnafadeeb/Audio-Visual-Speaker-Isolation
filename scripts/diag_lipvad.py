"""Does the mouth say the same thing the voice does?

The finding that motivates this: AV-TSE always emits a voice. If the face it
is conditioned on stops talking, it emits whoever else is audible -- including
someone off-screen, who is not any tracked face. ``verify_identity.py`` shows
that happening on the reference clip: face 0's stem changes speaker at 4 s.

No audio-side statistic can catch that, because the intruder is a real, loud,
perfectly good voice. The only evidence that the on-screen person stopped
talking is that their mouth stopped moving. This measures exactly that, from
the same ``lip`` signal ``app.vision`` already computes and currently discards.

The feature is inner-lip area / interocular^2 -- dimensionless, so an absolute
threshold is meaningful across speakers and camera distances. It is
deliberately NOT normalised per track: dividing each face by its own maximum is
what makes a silent mouth look as busy as a talking one, and would erase the
distinction this whole script exists to find.

    python scripts/diag_lipvad.py runs/862bd92a01ac
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Speech moves the jaw at roughly the syllable rate. Below this band lies
#: head-nodding and slow expression; above it lies detector jitter. Isolating
#: the band is what separates "talking" from "has a face".
SPEECH_LO_HZ = 1.5
SPEECH_HI_HZ = 8.0


def lip_activity(lip: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame speech-band modulation depth of the lip-aperture signal."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    x = np.asarray(lip, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 8:
        return np.zeros_like(x)
    # Hold across dropouts rather than interpolating: a mouth hidden behind a
    # hand has not smoothly travelled to wherever it reappears, and a ramp
    # invented across the gap is itself low-frequency energy we would then
    # measure as motion.
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(x)), -1))
    idx[idx < 0] = np.flatnonzero(ok)[0]
    x = x[idx]

    sos = butter(3, [SPEECH_LO_HZ, SPEECH_HI_HZ], btype="band", fs=fps, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x - x.mean())))
    env[~ok] = 0.0          # absent face is not evidence of speech
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--video", default="video.mp4")
    args = ap.parse_args()

    from app.config import Config
    from app.vision import FaceAnalyzer

    cfg = Config()
    an = FaceAnalyzer(cfg.vision)
    print(f"backend: {an.backend}")
    tracks, meta = an.analyze(str(args.run_dir / args.video))
    fps = meta["fps"]
    print(f"{len(tracks)} tracks, {meta['n_frames']} frames @ {fps:g} fps")

    acts = []
    for t in tracks:
        lip = np.asarray(t.lip, dtype=np.float64)
        e = lip_activity(lip, fps)
        acts.append(e)
        print(f"  track {t.track_id}: lip valid {np.isfinite(lip).sum()}/{len(lip)}"
              f"  aperture med {np.nanmedian(lip):.4f}"
              f"  activity med {np.median(e):.4f}  p95 {np.percentile(e, 95):.4f}")

    # Per-second table in ABSOLUTE units -- the numbers must be comparable
    # between the two faces for a shared threshold to exist at all.
    n_s = int(meta["n_frames"] / fps)
    print("\n  t(s) | " + " | ".join(f"track {t.track_id} activity" for t in tracks))
    print("-" * (9 + 26 * len(tracks)))
    for s in range(n_s):
        lo, hi = int(s * fps), int((s + 1) * fps)
        row = f"{s:6d} "
        for e in acts:
            v = float(e[lo:hi].mean())
            row += f"| {v:.4f} {'#' * int(min(v, 0.05) / 0.0025):<20} "
        print(row)

    # np.save so a threshold can be fitted without re-running the detector,
    # which is the expensive part.
    out = args.run_dir / "lip_activity.npz"
    np.savez(out, fps=fps, **{f"track{t.track_id}": a for t, a in zip(tracks, acts)},
             **{f"lip{t.track_id}": np.asarray(t.lip, dtype=np.float64) for t in tracks})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
