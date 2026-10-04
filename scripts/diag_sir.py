"""How much separation did SepFormer actually achieve on this clip?

    .venv\\Scripts\\python.exe scripts\\diag_sir.py runs/<job_id>

The gate refit hit a floor: with ``open_db`` at the edge of the retain cliff,
only ~55% of the interferer's frames reach bit-exact zero and both channels stay
open through half the clip.  Two very different things could be true, and the
gate metrics cannot distinguish them:

  A. **The gate is leaving money on the table.**  The stems really are well
     separated, and a smarter decision rule would close the gate on frames the
     current one keeps open.
  B. **The separator did not separate.**  The residual of speaker B inside
     stem A is genuinely loud during A's turn, so stem A's own energy never
     drops far below its p95 and NO level-based rule can close the gate without
     also cutting A's quiet speech.

GateConfig records the measured relationship for case B: exact-zero fraction vs
input leakage was 100% at -12 dB, 70% at -9 dB, 23% at -6 dB.  Our ~25% sits
near the -6 dB row, which *predicts* a separator running at about 6 dB SIR --
but that is an inference from a synthetic curve, not a measurement of this clip.
So measure it directly.

The estimate is frame-local and reference-free, which it has to be: there is no
ground-truth isolated speech for a real recording.  During frames where exactly
one speaker is clearly active, the OTHER stem contains nothing but leakage plus
that speaker's own silence floor, so

    SIR_j ~= 10*log10( E[stem_j | j active] / E[stem_j | only-other active] )

is a lower bound on the separation actually delivered.  It is a lower bound
rather than an equality because the "only-other active" frames also contain
stem j's own room tone and breath, which inflate the denominator and make the
separator look slightly worse than it is.  Erring in that direction is the safe
one here: it cannot manufacture a case for blaming the separator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from app.config import CONFIG                                   # noqa: E402


def _frames(v: np.ndarray, n: int) -> np.ndarray:
    m = len(v) // n * n
    return v[:m].reshape(-1, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir")
    ap.add_argument("--stems", default="stems_raw")
    args = ap.parse_args()

    job = Path(args.job_dir)
    x, sr = sf.read(str(job / f"{args.stems}.wav"), dtype="float32", always_2d=True)
    stems = x.T
    if stems.shape[0] != 2:
        print("this diagnostic assumes 2 stems")
        return 1

    g = CONFIG.gate
    n = max(1, int(sr * g.frame_ms / 1000.0))
    E = [(_frames(s, n) ** 2).mean(1) + 1e-20 for s in stems]
    k = min(len(E[0]), len(E[1]))
    db = [10.0 * np.log10(e[:k]) for e in E]

    print("=" * 78)
    print(f"separator quality -- {job.name}  ({k} frames @ {g.frame_ms:.0f} ms, "
          f"{args.stems})")
    print("=" * 78)

    act = []
    for i in range(2):
        p95 = float(np.percentile(db[i], 95))
        act.append(db[i] > p95 - 20.0)
        print(f"  stem {i}: p95 {p95:7.1f} dBFS   active {act[i].mean():5.1%}")

    only = [act[0] & ~act[1], act[1] & ~act[0]]
    both = act[0] & act[1]
    print(f"  only-0 {only[0].mean():5.1%}   only-1 {only[1].mean():5.1%}   "
          f"both {both.mean():5.1%}   neither {(~act[0] & ~act[1]).mean():5.1%}")

    print(f"\n{'stem':<6} {'own turn':>10} {'other turn':>11} {'SIR':>8}   verdict")
    sirs = []
    for j in range(2):
        own = db[j][only[j]]
        oth = db[j][only[1 - j]]
        if own.size == 0 or oth.size == 0:
            print(f"{j:<6}  no unambiguous frames")
            continue
        sir = float(own.mean() - oth.mean())
        sirs.append(sir)
        if sir >= 20:
            verdict = "clean -- the gate is the bottleneck"
        elif sir >= 12:
            verdict = "adequate"
        elif sir >= 8:
            verdict = "marginal"
        else:
            verdict = "POOR -- no level rule can gate this"
        print(f"{j:<6} {own.mean():>9.1f}  {oth.mean():>10.1f}  {sir:>7.1f}   {verdict}")

    if not sirs:
        return 1
    m = float(np.mean(sirs))
    print(f"\nmean SIR {m:.1f} dB")

    # The number the gate actually sees.  gate thresholds are relative to the
    # stem's OWN p95, so what matters is not absolute SIR but how far the
    # interferer's residual sits below that reference -- and how that compares
    # to where the stem's own quiet speech lives.  If the two overlap, no
    # threshold separates them, which is the whole reason the fusion veto
    # exists.
    print(f"\nwhat the gate sees (dB relative to each stem's own p95):")
    print(f"  {'stem':<6} {'residual p50':>13} {'residual p90':>13} "
          f"{'own speech p10':>15} {'own p25':>9}   overlap")
    for j in range(2):
        p95 = float(np.percentile(db[j], 95))
        res = db[j][only[1 - j]] - p95
        own = db[j][only[j]] - p95
        if res.size == 0 or own.size == 0:
            continue
        r50, r90 = np.percentile(res, [50, 90])
        o10, o25 = np.percentile(own, [10, 25])
        # A threshold can separate them only if the interferer's loud tail sits
        # below the target's quiet tail.
        gap = o10 - r90
        tag = f"{gap:+.1f} dB headroom" if gap > 0 else f"{-gap:.1f} dB OVERLAP"
        print(f"  {j:<6} {r50:>12.1f} {r90:>13.1f} {o10:>15.1f} {o25:>9.1f}   {tag}")

    print()
    if m < 8:
        print("=> The separator, not the gate, is the binding constraint.  Tuning")
        print("   thresholds past this point trades intelligibility for silence at")
        print("   roughly 1:1, which is what the sweep already showed.")
    elif m < 14:
        print("=> Marginal separation.  The gate can help but cannot finish the job;")
        print("   expect a residual floor no threshold removes.")
    else:
        print("=> The stems are well separated -- the remaining leakage is the")
        print("   gate's decision rule, not the separator's output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
