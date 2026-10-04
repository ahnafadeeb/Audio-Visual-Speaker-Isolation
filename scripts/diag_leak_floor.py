"""Is the leak separable from quiet target speech in a RIVAL-relative quantity?

The user's report is two symptoms of one cause: "if the sound of the selected
speaker is not loud but the other non selected person is loud then the loud
person's voice leaks", and "the selected speaker's sound is a bit suppressed in
the process of reducing leaks".

The gate's decision variable is ``rel = ldb - p95(own stem)`` (dsp.gate_mask),
a purely INTRA-stem quantity.  The separator's residual, however, scales with
the RIVAL's absolute level -- it is a fixed fraction of what leaked through the
TF mask.  So the leak-to-target ratio rises exactly when the target is quiet and
the rival is loud, and no threshold on ``rel`` can see that: the gate cannot
distinguish "my quiet speech" from "their loud residual" because both are simply
energy in this stem.

Meanwhile ``open_db = -20`` sits only 3.8 dB below the target's own quiet speech
(p10 = -16.2 dB rel p95, per config), and ``visual_veto_db = 4.0`` lifts
``effective_open`` to -16.0 -- THROUGH it.  Hence both symptoms at once.

Full-band dominance gating is already REFUTED (dsp.py:265-269): dominance on
target-active frames has a 10th percentile of -4.5 dB, so requiring the target
to be louder than the rival cuts 40-70% of genuine target speech.  This script
asks a strictly weaker question:

    Is there a margin M such that "target is more than M dB BELOW the rival"
    identifies leak frames while almost never touching real target speech?

That is a leak FLOOR, not a dominance test.  Dominance asks the target to win;
a floor only asks it not to lose by 25 dB.  If the two distributions separate at
some M, the fix is a rival-relative floor on the gate's open threshold, and the
p10 = -4.5 dB refutation does not apply to it.

Run::

    .venv/Scripts/python.exe scripts/diag_leak_floor.py runs/verify3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def frames(x: np.ndarray, n: int) -> np.ndarray:
    m = len(x) // n * n
    return x[:m].reshape(-1, n)


def rms_db(x: np.ndarray, n: int) -> np.ndarray:
    e = np.sqrt((frames(x, n) ** 2).mean(1)) + 1e-12
    return 20.0 * np.log10(e)


def pct(a: np.ndarray, qs) -> str:
    if a.size == 0:
        return "  (no frames)"
    return "  ".join(f"p{q}={np.percentile(a, q):+6.1f}" for q in qs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--frame-ms", type=float, default=20.0)
    a = ap.parse_args()

    d = Path(a.job_dir)
    raw, sr = sf.read(d / "stems_raw.wav", always_2d=True)
    raw = raw.T.astype(np.float32)
    n = max(1, int(sr * a.frame_ms / 1000.0))
    if raw.shape[0] < 2:
        print("need 2 stems")
        return 1

    print(f"{d}  sr={sr}  frame={a.frame_ms:g} ms  "
          f"{raw.shape[1] / sr:.2f} s  {raw.shape[0]} stems\n")

    L = [rms_db(s, n) for s in raw]
    k = min(len(L[0]), len(L[1]))
    L = [x[:k] for x in L]
    P95 = [float(np.percentile(x, 95)) for x in L]
    REL = [L[i] - P95[i] for i in (0, 1)]        # what the gate actually sees

    # Activity from the RAW stems, same definition fit_gate.score uses.
    ACT = [REL[i] > -20.0 for i in (0, 1)]

    print("decision variable the gate sees:  rel = frame_db - p95(own stem)")
    for i in (0, 1):
        print(f"  stem {i}  rel over ALL frames        {pct(REL[i], (5, 10, 25, 50, 90))}")
    print()

    # ---- the two populations, in a rival-relative quantity ----------------- #
    # dom = how far this stem is ABOVE the rival, in dB.  Negative = rival wins.
    rows = []
    for i in (0, 1):
        j = 1 - i
        dom = L[i] - L[j]
        target_only = ACT[i] & ~ACT[j]        # real speech we must NOT cut
        leak_only = ACT[j] & ~ACT[i]          # rival talks alone: any output is leak
        rows.append((i, dom, target_only, leak_only))
        print(f"stem {i}:  target-alone {target_only.sum():4d} frames   "
              f"rival-alone {leak_only.sum():4d} frames")
        print(f"  dom on TARGET-alone frames (must stay ABOVE the floor)")
        print(f"    {pct(dom[target_only], (1, 2, 5, 10, 25, 50))}")
        print(f"  dom on RIVAL-alone frames  (the leak; want it BELOW the floor)")
        print(f"    {pct(dom[leak_only], (50, 75, 90, 95, 99))}")
        print()

    # ---- sweep the floor -------------------------------------------------- #
    # The "must not cut" population is EVERY frame where the target is clearly
    # active -- including OVERLAP, where both speak at once.  Overlap is where
    # the -4.5 dB dominance refutation came from, so scoring the floor only on
    # target-alone frames would repeat that mistake in a new form: it would
    # exclude the very frames that killed the previous attempt.
    both = ACT[0] & ACT[1]
    print(f"frame census: target-alone {int((ACT[0] & ~ACT[1]).sum())}, "
          f"rival-alone {int((ACT[1] & ~ACT[0]).sum())}, "
          f"OVERLAP {int(both.sum())}, silent {int((~ACT[0] & ~ACT[1]).sum())}, "
          f"total {k}")
    print()
    for i, dom, tgt, leak in rows:
        ov = ACT[i] & ACT[1 - i]
        print(f"stem {i}  dom on OVERLAP frames (target genuinely speaking "
              f"under the rival)")
        print(f"    {pct(dom[ov], (1, 5, 10, 25, 50))}")
    print()

    print("floor sweep:  gate stays shut where  own_db < rival_db - M")
    print("  M     leak caught   target lost (ALL active, incl. overlap)   verdict")
    best = None
    for M in (6, 8, 10, 12, 15, 18, 20, 25, 30):
        caught = lost = 0.0
        cn = ln = 0
        for i, dom, tgt, leak in rows:
            active = ACT[i]                     # <- includes overlap
            if leak.any():
                caught += float((dom[leak] < -M).mean()); cn += 1
            if active.any():
                lost += float((dom[active] < -M).mean()); ln += 1
        caught = caught / cn if cn else 0.0
        lost = lost / ln if ln else 0.0
        ok = lost <= 0.01 and caught > 0.0
        if ok and (best is None or caught > best[1]):
            best = (M, caught, lost)
        print(f"  {M:2d}    {caught:7.1%}          {lost:7.2%}"
              f"                          {'usable' if ok else 'CUTS SPEECH'}")

    print()
    if best:
        M, c, l = best
        print(f"=> a floor at rival - {M} dB catches {c:.1%} of leak frames "
              f"while cutting {l:.2%} of genuine target speech.")
        print("   The p10 = -4.5 dB dominance refutation does not apply: this")
        print("   asks the target not to lose by a wide margin, not to win.")
    else:
        print("=> NO usable floor on this clip: the leak is not separable from")
        print("   target speech by level alone.  A spectral or visual term is")
        print("   required instead; do not ship a level floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
