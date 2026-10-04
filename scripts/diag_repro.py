"""Is the vision -> matching path reproducible run-to-run on identical input?

    .venv\\Scripts\\python.exe scripts\\diag_repro.py runs/<job_id> [--runs 3]

This probe exists because of a contradiction that could not be explained by any
input difference.  ``meta.json`` for the test job shipped assignment ``[1, 0]``.
Re-running ``diag_match.py`` on the very same ``video.mp4`` -- the same file the
pipeline analysed, byte for byte, with the same stems -- returned ``[0, 1]``, the
opposite pairing, and reported a slightly different landmark count (776 vs 774
frames on track 0).

Two candidate causes, and they have very different consequences:

  A. **Something in the code changed** between the two runs.  Then the earlier
     numbers describe an older pipeline and the contradiction is bookkeeping.
  B. **The path is nondeterministic.**  MediaPipe runs its graph across a thread
     pool with an XNNPACK delegate; if frame results are not bit-reproducible,
     the lip signal differs slightly on every run.  A matcher whose margin is
     0.04 -- at the same order as the pure-noise margin of 0.068 measured in
     check_fusion -- can then flip on that jitter alone.

Cause B is much worse than a mis-assignment, because it means no single run of
any diagnostic can be quoted: every number is one draw from a distribution, and
"the pairing is [0, 1]" is not even a property of the clip.

So this measures it the only way that settles it: run the identical analysis N
times in one process and compare.  Reported per run:

    landmarks     frames where FaceMesh returned a face, per track
    lip checksum  bit-exact hash of the lip signal -- catches jitter a mean or
                  a NaN count would hide
    assignment    the pairing, which is what the user hears
    margin, p     the two numbers app/matching.py reports

If the checksums differ, cause B is confirmed and the honest reporting unit is a
distribution over runs, not a number.  If they are identical, the contradiction
is cause A and the earlier artifact predates a code change.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from app import matching                                        # noqa: E402
from app.config import CONFIG                                   # noqa: E402
from _job import pipeline_video                                 # noqa: E402


def lip_hash(lip) -> str:
    """Bit-exact digest of a lip signal, NaNs included.

    ``np.asarray(...).tobytes()`` and not a rounded string: the question is
    whether the floats are identical, and rounding to print precision is exactly
    how run-to-run jitter goes unnoticed.  NaN payload bits are stable for
    numpy's own NaN, so a NaN in the same slot hashes the same.
    """
    a = np.asarray(lip, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    job = Path(args.job_dir)
    video = pipeline_video(job)
    x, sr = sf.read(str(job / "stems_raw.wav"), dtype="float32", always_2d=True)
    stems = x.T

    print("=" * 78)
    print(f"reproducibility probe -- {job.name}")
    print(f"  video {video.name} (the file the pipeline analysed)")
    print(f"  stems stems_raw.wav {stems.shape[0]}ch -- FIXED across runs, so any")
    print(f"        variation below comes from vision alone")
    print("=" * 78)

    from app.vision import FaceAnalyzer

    records = []
    for run in range(args.runs):
        fa = FaceAnalyzer(CONFIG.vision)
        tracks, vmeta = fa.analyze(str(video))
        m = matching.match_stems_to_tracks(
            stems, tracks, sample_rate=sr, video_fps=vmeta["fps"],
            cfg=CONFIG.match)
        rec = {
            "n_tracks": len(tracks),
            "n_frames": vmeta["n_frames"],
            "landmarks": [int(np.isfinite(np.asarray(t.lip, dtype=np.float64)).sum())
                          for t in tracks],
            "hashes": [lip_hash(t.lip) for t in tracks],
            "assignment": list(m.assignment),
            "margin": max(m.confidence, default=0.0),
            "p": m.significance,
            "score": m.score.copy(),
        }
        records.append(rec)
        print(f"\nrun {run}: {rec['n_tracks']} tracks, {rec['n_frames']} frames")
        print(f"  landmarks   {rec['landmarks']}")
        print(f"  lip hashes  {rec['hashes']}")
        print(f"  assignment  {rec['assignment']}   margin {rec['margin']:.4f}   "
              f"p {rec['p']:.4f}")

    print("\n" + "=" * 78)
    ref = records[0]
    same_hash = all(r["hashes"] == ref["hashes"] for r in records)
    same_assign = all(r["assignment"] == ref["assignment"] for r in records)
    margins = [r["margin"] for r in records]

    if same_hash:
        print("VERDICT: vision is BIT-REPRODUCIBLE across runs.")
        print("  The lip signals hash identically, so a stored artifact that")
        print("  disagrees with a fresh run reflects a CODE CHANGE since it was")
        print("  written, not nondeterminism.  Re-run the pipeline to refresh it.")
    else:
        print("VERDICT: vision is NOT reproducible run-to-run.")
        print("  Identical input, identical stems, different lip signals.  Every")
        print("  single-run number in the diagnostics is one draw from a")
        print("  distribution and must be reported as such.")
        # Quantify the jitter where it matters: the score matrix the pairing
        # is read off.  A tiny lip difference is only a problem if it moves
        # the decision, and the decision is a comparison of these numbers.
        S = np.stack([r["score"] for r in records])
        print(f"\n  score-matrix spread across runs (max - min per cell):")
        spread = S.max(0) - S.min(0)
        for i in range(spread.shape[0]):
            print("    " + "".join(f"{v:>12.5f}" for v in spread[i]))
        print(f"  largest cell spread {spread.max():.5f}  vs  margin "
              f"{np.mean(margins):.5f}")
        if spread.max() >= np.mean(margins):
            print("  => the run-to-run jitter is as large as the margin itself.")
            print("     The pairing is decided by noise in the detector.")

    if not same_assign:
        print(f"\n  !! the ASSIGNMENT itself changed across runs: "
              f"{[r['assignment'] for r in records]}")
        print("     This is the reported speaker-swap symptom, reproduced without")
        print("     touching the clip.")
    elif not same_hash:
        print(f"\n  the assignment held at {ref['assignment']} across all "
              f"{args.runs} runs despite the jitter")
        print("     -- stable here, but margin "
              f"{np.mean(margins):.4f} is not a safe distance from a flip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
