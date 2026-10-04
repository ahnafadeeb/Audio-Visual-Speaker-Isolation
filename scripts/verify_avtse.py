"""Step-1 gate: did AV-TSE extract the RIGHT speaker?

Correlation cannot answer this -- that is the whole finding of
``docs/DIAG_MATCHER.md``. Pitch can. The reference clip pairs a man
(track 0, screen-left) with a woman (track 1, screen-right), so median f0
attributes each stem to a speaker on physiology alone, with no reference to
the statistic that was measured to be broken.

Scores v3 against v2 on the same clip, same ground truth, so the comparison
is like-for-like.

    python scripts/verify_avtse.py runs/862bd92a01ac
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

#: Conventional adult speaking-f0 bands. Deliberately wide and non-overlapping
#: in the middle -- we only need to separate "male" from "female", not to
#: estimate either precisely.
MALE_MAX = 165.0
FEMALE_MIN = 165.0

#: Ground truth for the reference clip, from docs/DIAG_MATCHER.md:
#: "man screen-left, woman screen-right".
GROUND_TRUTH = {"862bd92a01ac": ["male", "female"]}


def median_f0(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(median f0 over loud voiced frames, voiced fraction)."""
    seg, db, _ = framify(np.asarray(x, dtype=np.float64), sr)
    f0, ape = yin(seg, sr)
    m = (ape < YIN_THRESH) & np.isfinite(f0) & (db > np.percentile(db, 95) - 25)
    if m.sum() < 20:
        return float("nan"), float(m.mean())
    return float(np.median(f0[m])), float(m.mean())


def classify(f0: float) -> str:
    if not np.isfinite(f0):
        return "unknown"
    return "male" if f0 < MALE_MAX else "female"


def score(name: str, stems: list[np.ndarray], sr: int, truth: list[str]) -> int:
    print(f"\n=== {name} ===")
    correct = 0
    for i, s in enumerate(stems):
        f0, voiced = median_f0(s, sr)
        got = classify(f0)
        want = truth[i] if i < len(truth) else "?"
        ok = got == want
        correct += ok
        print(f"  face {i}: median f0 {f0:6.1f} Hz  voiced {voiced*100:4.1f}%"
              f"   -> {got:7} (want {want:7}) {'OK' if ok else 'WRONG'}")
    print(f"  score: {correct}/{len(stems)}")
    return correct


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=None,
                    help="truncate every stem, so a partial v3 run compares "
                         "like-for-like against the full v2 export")
    args = ap.parse_args()

    run = args.run_dir
    lim = int(args.seconds * args.sr) if args.seconds else None
    truth = GROUND_TRUTH.get(run.name)
    if truth is None:
        print(f"no ground truth recorded for {run.name}; add it to GROUND_TRUTH")
        return 2

    results = {}

    spike = sorted((run / "spike").glob("face*.wav"))
    if spike:
        stems = [sf.read(str(p), dtype="float64")[0][:lim] for p in spike]
        results["v3 AV-TSE"] = score("v3  AV-TSE (face-conditioned)", stems, args.sr, truth)

    raw = run / "stems_raw.wav"
    if raw.exists():
        import json
        data, _ = sf.read(str(raw), dtype="float64", always_2d=True)
        # NOT meta["assignment"]: that is indexed by track holding STEM, and
        # plan_channels has already baked it into the export order. The
        # authoritative track -> WAV-channel map is tracks.json's "channel"
        # field, which is what the browser uses when a face is clicked.
        tj = json.loads((run / "tracks.json").read_text())
        stems, labels = [], []
        for tr in tj["tracks"]:
            ch = tr.get("channel")
            if ch is None or ch >= data.shape[1]:
                continue
            stems.append(data[:, ch][:lim])
            labels.append(ch)
        if stems:
            print(f"\n(v2 track -> channel map: {labels})")
            results["v2 SepFormer"] = score(
                "v2  SepFormer + Hungarian matcher (as heard)", stems, args.sr, truth)

    print("\n" + "=" * 58)
    for k, v in results.items():
        print(f"  {k:<16} {v}/{len(truth)} faces correct")
    if "v3 AV-TSE" in results:
        print()
        print("  PASS" if results["v3 AV-TSE"] == len(truth) else "  FAIL",
              "- v3 must score", f"{len(truth)}/{len(truth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
