"""Did the female stem collapse, and does that explain what the user heard?

Item 4 of the bug report says the female speaker's stem collapsed onto the male
channel.  I previously called that refuted.  That was wrong, and the reason it
was wrong is instructive: I compared channel 0's f0 against channel 1's, but
channel 1 is 73.9% exact zero, so it was a real voice against separator residual.
Measuring the *mixture* instead -- conditioned on which mouth is moving, no
separator and no matcher in the path -- gives clean references via YIN:

    the man's windows   234.7 Hz
    the woman's windows 393.5 Hz          (ratio 1.68, so unambiguous ordering)

That makes a separator-independent label available for every audio frame: run YIN
on the MIXTURE and ask which reference its f0 is closer to.  No vision, no
assignment, no correlation -- so it can arbitrate all of them.

What this script then establishes, in order:

  A  the label itself, and how much each speaker actually talks;
  B  cross-validation of the audio label against the visual one, which doubles as
     a measurement of how informative the lip signal is at all -- the matcher's
     evidence was a coin flip and this says whether that is the feature's fault;
  C  per-channel selectivity: dB in MAN frames vs WOMAN frames.  Correct
     separation is strongly selective in opposite directions;
  D  where each channel's p95 gate reference comes from.  This is the crux.  The
     Schmitt trigger's thresholds are relative to each stem's OWN p95 frame
     energy, so if channel 1's p95 is set by the *man's leakage* rather than by
     the woman's voice, her real speech sits below `open_db` and the gate closes
     on her -- "selecting Female gave complete silence" -- with every threshold
     behaving exactly as designed;
  E  the predicted gate outcome per channel, checked against the shipped
     `stems_demo.wav` so the prediction is falsifiable rather than a story.

Run::

    PYTHONPATH=. python scripts/diag_collapse.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media                                    # noqa: E402
from app.config import CACHE_DIR, CONFIG                 # noqa: E402
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,  # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin             # noqa: E402

SUBNASALE = 2


def aperture(lm: np.ndarray) -> np.ndarray:
    n = lm.shape[0]
    out = np.full(n, np.nan)
    for f in range(n):
        p = lm[f]
        if not np.isfinite(p[LEFT_EYE_OUTER, 0]):
            continue
        eL, eR = p[LEFT_EYE_OUTER, :2], p[RIGHT_EYE_OUTER, :2]
        inter = float(np.linalg.norm(eL - eR))
        vspan = float(np.linalg.norm((eL + eR) / 2.0 - p[SUBNASALE, :2]))
        ring = p[INNER_LIP_RING, :2]
        if inter < 1e-6 or vspan < 1e-6 or not np.isfinite(ring).all():
            continue
        out[f] = _shoelace(ring.astype(np.float64)) / (inter * vspan)
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum() > 1:
        out[ok] = np.argsort(np.argsort(x[ok])) / (ok.sum() - 1)
    return out


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps, n_tr = float(tj["fps"]), len(tj["tracks"])
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)
    raw, _ = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    demo, _ = sf.read(job / "stems_demo.wav", dtype="float64", always_2d=True)
    raw, demo = raw.T, demo.T
    n_ch = raw.shape[0]

    # ---- A: label every frame from the MIXTURE alone ----------------------- #
    seg, mdb, t = framify(mix, sr)
    f0, ape = yin(seg, sr)
    voiced = (ape < YIN_THRESH) & np.isfinite(f0)
    F_MAN, F_WOM = 234.7, 393.5
    split = float(np.sqrt(F_MAN * F_WOM))                 # geometric midpoint
    lab = np.zeros(len(f0), dtype=np.int8)                # 0 unvoiced, 1 man, 2 woman
    lab[voiced & (f0 < split)] = 1
    lab[voiced & (f0 >= split)] = 2
    print(f"{job.name}: {len(f0)} frames, split at {split:.1f} Hz "
          f"(man ref {F_MAN}, woman ref {F_WOM})")
    print(f"=== A: who speaks, from the mixture only ===")
    for k, nm in ((1, "MAN  "), (2, "WOMAN")):
        m = lab == k
        print(f"  {nm}: {int(m.sum()):4d} frames ({m.mean() * 100:5.1f}%)  "
              f"median f0 {np.median(f0[m]):6.1f} Hz  "
              f"mixture level {np.median(mdb[m]):+6.1f} dB")
    print(f"  unvoiced/ambiguous: {int((lab == 0).sum())} frames "
          f"({(lab == 0).mean() * 100:5.1f}%)")

    # ---- B: does the visual signal agree with the audio label? ------------- #
    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    aps = [aperture(lm[i]) for i in range(n_tr)]
    w_v = max(2, int(round(0.4 * fps)))
    ker = np.ones(w_v) / w_v
    wr = []
    for i in range(n_tr):
        d = np.abs(np.diff(aps[i], prepend=aps[i][:1]))
        num = np.convolve(np.nan_to_num(d), ker, mode="valid")
        cov = np.convolve(np.isfinite(d).astype(float), ker, mode="valid")
        wr.append(_rank(np.where(cov > 0.6, num / np.maximum(cov, 1e-9), np.nan)))
    n_w = min(len(r) for r in wr)
    wi = np.clip((t * fps - w_v / 2).astype(int), 0, n_w - 1)
    r0, r1 = wr[0][:n_w][wi], wr[1][:n_w][wi]
    okv = np.isfinite(r0) & np.isfinite(r1)

    print("\n=== B: does the lip signal know who is speaking? ===")
    print("    audio label vs 'whose mouth is moving more' -- 50% is a coin flip")
    for thr in (0.0, 0.2, 0.4):
        m = okv & (lab > 0) & (np.abs(r0 - r1) > thr)
        if m.sum() < 20:
            print(f"  |rank gap| > {thr}: {int(m.sum())} frames (too few)")
            continue
        pred = np.where(r0[m] > r1[m], 1, 2)               # track0=man, track1=woman
        acc = float(np.mean(pred == lab[m]))
        print(f"  |rank gap| > {thr}: n={int(m.sum()):4d}  agreement "
              f"{acc * 100:5.1f}%"
              + ("   <-- informative" if acc > 0.6 else
                 "   <-- no better than chance" if acc < 0.58 else ""))

    # ---- C: per-channel selectivity --------------------------------------- #
    print("\n=== C: per-channel level by speaker (stems_raw, ungated) ===")
    print("    channel i carries the stem the matcher gave track i;")
    print("    track 0 = MAN, track 1 = WOMAN")
    cdb = []
    for j in range(n_ch):
        s, d, _ = framify(raw[j], sr)
        cdb.append(d)
    cdb = np.stack(cdb)
    for j in range(n_ch):
        a = float(np.median(cdb[j][lab == 1]))
        b = float(np.median(cdb[j][lab == 2]))
        owner = "MAN" if j == 0 else "WOMAN"
        want = a - b if j == 0 else b - a
        print(f"  channel {j} (owner {owner:5}): MAN frames {a:+6.1f} dB   "
              f"WOMAN frames {b:+6.1f} dB   selectivity for its owner "
              f"{want:+5.1f} dB")
    # The cross term the user hears on the male button.
    print(f"  -> during WOMAN frames, channel 0 sits {float(np.median(cdb[0][lab == 2])):+.1f} dB "
          f"vs {float(np.median(cdb[0][lab == 1])):+.1f} dB during MAN frames")
    print(f"  -> the woman's own channel is "
          f"{float(np.median(cdb[0][lab == 1])) - float(np.median(cdb[1][lab == 2])):+.1f} dB "
          f"quieter on her speech than the man's is on his")

    # ---- D: where does each channel's gate reference come from? ------------ #
    print("\n=== D: what sets each channel's p95 gate reference? ===")
    print("    the Schmitt thresholds are relative to each stem's OWN p95 frame")
    print("    energy, so whoever owns the loudest 5% owns the threshold")
    for j in range(n_ch):
        p95 = float(np.percentile(cdb[j], 95))
        top = cdb[j] >= p95
        comp = {k: float(np.mean(lab[top] == k)) for k in (0, 1, 2)}
        owner = 1 if j == 0 else 2
        print(f"  channel {j}: p95 = {p95:+6.1f} dB   top-5% frames are "
              f"{comp[1] * 100:5.1f}% MAN, {comp[2] * 100:5.1f}% WOMAN, "
              f"{comp[0] * 100:5.1f}% unvoiced")
        if comp[owner] < comp[3 - owner]:
            print(f"    ^^ channel {j}'s reference is set by the WRONG speaker: "
                  f"its own gate\n       threshold is calibrated on the "
                  f"interferer's leakage.")

    # ---- E: predicted vs actual gate outcome ------------------------------ #
    print("\n=== E: does that predict the silence the user heard? ===")
    open_db = CONFIG.gate.open_db
    for j in range(n_ch):
        p95 = float(np.percentile(cdb[j], 95))
        owner = 1 if j == 0 else 2
        own = lab == owner
        above = float(np.mean(cdb[j][own] > p95 + open_db))
        print(f"  channel {j}: {above * 100:5.1f}% of its OWNER's speech frames "
              f"clear open_db ({open_db:+.0f} dB re p95)")
        ddb, _, _ = framify(demo[j], sr)[1], None, None
        z = float(np.mean(demo[j] == 0.0))
        # Measured on the shipped demo, so the prediction is falsifiable.
        kept = float(np.mean(ddb[own] > p95 + open_db - 3.0))
        print(f"             shipped stems_demo: {z * 100:5.1f}% exact zero "
              f"overall, {kept * 100:5.1f}% of owner frames survive")


if __name__ == "__main__":
    main()
