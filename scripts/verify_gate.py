"""Does the identity gate remove the intruder without eating the target?

Runs :mod:`app.identity` over a spike directory and reports, per face:

  * who was found, and which identity each face claimed
  * how much of the stem was muted, and when
  * the margin between kept and muted windows -- the number that says whether
    the decision was easy or a coin flip
  * leakage in the muted regions, before and after

Writes ``face*_id.wav`` next to the inputs so the result can be listened to.

    python scripts/verify_gate.py runs/862bd92a01ac/spike
    python scripts/verify_gate.py runs/862bd92a01ac/spike --device cuda
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.identity import IdentityConfig, covered, enroll, gate_intruders  # noqa: E402

#: Exactly ``faceN.wav``. A looser glob re-ingests this script's own outputs
#: as extra faces on the second run -- which happened, and silently turned a
#: 2-face clip into a 4-face one.
FACE_RE = re.compile(r"^face\d+$")


def face_stems(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("face*.wav") if FACE_RE.match(p.stem))


def db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12))


def runs_of(mask: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Contiguous True spans of ``mask`` as (start_s, end_s)."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(s / sr, e / sr)
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spike_dir", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-mixture", action="store_true",
                    help="omit the mixture from enrollment, to show what it buys")
    args = ap.parse_args()

    faces = face_stems(args.spike_dir)
    if not faces:
        print(f"no faceN.wav in {args.spike_dir}")
        return 2
    stems, sr = [], None
    for p in faces:
        x, sr = sf.read(str(p), dtype="float32")
        stems.append(x)

    mix = None
    mp = args.spike_dir / "mixture_ref.wav"
    if mp.exists() and not args.no_mixture:
        mix, _ = sf.read(str(mp), dtype="float32")

    cfg = IdentityConfig()
    plan, per_face = enroll(stems, sr, cfg, mixture=mix, device=args.device)

    print(f"identities found: {plan.n_clusters}"
          f"   ({len(plan.refs)} claimed by faces, {len(plan.others)} unclaimed)")
    if len(plan.others):
        print("  unclaimed identities are people who are audible but never "
              "tracked -- off-screen speakers.")
    print()

    total_muted = 0.0
    for i, (t, E) in enumerate(per_face):
        s = plan.score(E, i)
        gated, env = gate_intruders(stems[i], t, s, sr, cfg)

        muted = env < 0.5
        frac = float(muted.mean())
        total_muted += frac
        # Split muted time by cause. "Unverified" is audio the loudness gate
        # skipped, so no identity was ever measured there; reporting it as a
        # verdict would overstate what the gate knows.
        seen = covered(t, len(env), sr, cfg.win_s)
        n_int = float((muted & seen).mean())
        n_unv = float((muted & ~seen).mean())

        keep_s, drop_s = s[s >= 0], s[s < 0]
        print(f"face {i}:  purity {plan.purity[i]:.2f}"
              f"   nearest rival cos {plan.nearest_rival[i]:+.3f}")
        print(f"   score: kept {keep_s.mean():+.3f} (n={len(keep_s)})"
              + (f"   muted {drop_s.mean():+.3f} (n={len(drop_s)})"
                 if len(drop_s) else "   muted -- none --"))
        if len(keep_s) and len(drop_s):
            margin = float(np.percentile(keep_s, 5) - np.percentile(drop_s, 95))
            auc = float((keep_s[:, None] > drop_s[None, :]).mean())
            print(f"   separability: margin {margin:+.3f}   AUC {auc:.3f}"
                  f"   (vs its own clustering -- see fit_identity.py)")
        print(f"   muted {frac * 100:.1f}% of the stem"
              f"   ({len(runs_of(muted, sr))} spans)"
              f"   = {n_int * 100:.1f}% wrong identity"
              f" + {n_unv * 100:.1f}% never verified")
        for a, b in runs_of(muted, sr)[:12]:
            print(f"      {a:6.2f} - {b:6.2f}s  ({b - a:.2f}s)")

        if muted.any():
            before, after = db(stems[i][muted]), db(gated[muted])
            print(f"   muted regions: {before:.1f} dB -> {after:.1f} dB"
                  f"   ({after - before:+.1f} dB)")
            exact = float((gated[muted] == 0.0).mean())
            print(f"   exact-zero samples in muted regions: {exact * 100:.2f}%")

        out = args.spike_dir / f"face{i}_id.wav"
        sf.write(str(out), gated, sr, subtype="FLOAT")
        print(f"   wrote {out.name}\n")

    print("=" * 60)
    print(f"  {total_muted / len(per_face) * 100:.1f}% of stem time muted on average")
    print("  LISTEN to face*_id.wav: the intruder should be gone, and the "
          "target's own speech untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
