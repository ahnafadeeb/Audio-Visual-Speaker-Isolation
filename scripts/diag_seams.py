"""Who is in this stem, second by second?

``verify_avtse.py`` reduces a stem to one median f0. That is the right gate
but the wrong diagnostic: a stem that is the correct speaker for 25 s and the
wrong one for 7 s has a median somewhere in between, which reads as "slightly
off" when the truth is "swapped partway through".

This prints a per-window f0 trace and marks the chunk boundaries, so a failure
that is caused by chunking looks different from one that is not:

    boundary-aligned flips  -> the seam logic is wrong
    flips anywhere else     -> the visual cue failed there (occlusion, profile,
                               undersized face), and chunking is innocent

    python scripts/diag_seams.py runs/862bd92a01ac/spike
    python scripts/diag_seams.py runs/862bd92a01ac/spike --chunk-s 9.312
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_f0 import YIN_THRESH, framify, yin  # noqa: E402
from verify_avtse import MALE_MAX  # noqa: E402


def trace(x: np.ndarray, sr: int, win_s: float) -> list[tuple[float, float, float, float]]:
    """(t, median f0, voiced fraction, rms dB) per window."""
    out = []
    n = int(win_s * sr)
    for s in range(0, len(x) - n // 2, n):
        seg_x = x[s : s + n]
        seg, db, _ = framify(np.asarray(seg_x, dtype=np.float64), sr)
        f0, ape = yin(seg, sr)
        m = (ape < YIN_THRESH) & np.isfinite(f0)
        rms = 20 * np.log10(np.sqrt((seg_x**2).mean()) + 1e-12)
        med = float(np.median(f0[m])) if m.sum() >= 5 else float("nan")
        out.append((s / sr, med, float(m.mean()), rms))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spike_dir", type=Path)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--chunk-s", type=float, default=None,
                    help="body length used at extraction; marks the seams")
    ap.add_argument("--fade-s", type=float, default=0.128)
    args = ap.parse_args()

    faces = sorted(args.spike_dir.glob("face*.wav"))
    if not faces:
        print(f"no face*.wav in {args.spike_dir}")
        return 2

    traces = []
    for p in faces:
        x, sr = sf.read(str(p), dtype="float64")
        traces.append(trace(x, sr, args.win))
    n_win = min(len(t) for t in traces)

    seams: set[int] = set()
    if args.chunk_s:
        step = args.chunk_s - args.fade_s
        t = step
        while t < n_win * args.win:
            seams.add(int(t / args.win))
            t += step

    hdr = "  t(s) " + "".join(f"| face {i}: f0   voiced  rms  " for i in range(len(faces)))
    print(hdr)
    print("-" * len(hdr))
    flips = [0] * len(faces)
    for w in range(n_win):
        row = f"{traces[0][w][0]:6.1f} "
        for i, tr in enumerate(traces):
            _, f0, v, rms = tr[w]
            lbl = "  ?  " if not np.isfinite(f0) else ("male " if f0 < MALE_MAX else "FEM  ")
            row += f"| {f0:6.1f} {lbl} {v*100:4.0f}% {rms:6.1f} "
        mark = "  <== chunk seam" if w in seams else ""
        print(row + mark)

    # Per-face verdict: which label dominates, and how often it is contradicted.
    print()
    for i, tr in enumerate(traces[: len(faces)]):
        lab = [("male" if f0 < MALE_MAX else "female")
               for _, f0, _, _ in tr[:n_win] if np.isfinite(f0)]
        if not lab:
            print(f"face {i}: no voiced windows")
            continue
        dom = max(set(lab), key=lab.count)
        agree = lab.count(dom)
        flips[i] = len(lab) - agree
        print(f"face {i}: dominant {dom:6}  {agree}/{len(lab)} windows agree"
              f"   ({flips[i]} contradict)")
    if seams:
        print(f"\nseam windows: {sorted(seams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
