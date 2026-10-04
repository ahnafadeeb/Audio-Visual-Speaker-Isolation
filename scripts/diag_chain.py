"""The whole stem -> channel -> track chain, in one place, with no hardcoded truth.

    PYTHONPATH=. .venv\\Scripts\\python.exe scripts\\diag_chain.py runs/pair_v2

Why this script exists, stated plainly: the three index spaces that
``app/channels.py`` was written to keep apart have now produced a wrong
conclusion **twice**, once in this project's own docs and once in my analysis of
the fresh run.  Both times the mistake was the same shape -- a number measured in
one space, compared against a number that lived in another, with nothing in the
output to say which was which.

    stem     what the separator emitted, 0..n_stems-1
    channel  a column of stems_demo.wav / stems_raw.wav, 0..n_ch-1
    track    a detected face, 0..n_tracks-1, the order the buttons are drawn in

Two hops connect them, and both are permutations, so both can silently invert a
conclusion:

    stem -> channel   plan_channels() reorders the export so channel i belongs to
                      track i.  A pairing of [1, 0] therefore ships as channels
                      [0, 1] -- the assignment and the shipped binding look like
                      DIFFERENT permutations while describing the same decision.
    channel -> track  tracks.json's `channel` field, which is what the browser
                      binds its buttons to.

``scripts/diag_pose.py`` hardcodes ``TRUTH = [1, 0]`` in *channel* space while
``meta.json`` stores ``assignment`` in *stem* space.  The two are numerically
identical and semantically opposite.  Nothing in either output says so.

So this reports every hop explicitly and derives each one by measurement:

  1. Run the separator (deterministic here -- see diag_sep_repro.py) to get the
     stems in STEM order, and read stems_raw.wav to get them in CHANNEL order.
  2. Recover the stem->channel permutation by cross-correlation, rather than
     deriving it from `assignment` -- so the report cannot inherit the very
     bookkeeping error it exists to catch.
  3. Attribute each stem to a voice by octave-safe YIN f0 (the estimator
     diag_f0.py established for this clip; plain autocorrelation and HPS both
     misled earlier analyses by an octave).
  4. Print the chain, and say whether each FACE hears its own voice.

The verdict needs no ground-truth constant: it needs only which face is male, and
that is read from ``--male-track`` (default 0, confirmed by eye on this clip:
track 0 is the man at centroid x 0.297, track 1 the woman at 0.757).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from app import media                                           # noqa: E402
from app.config import CACHE_DIR, CONFIG                        # noqa: E402
from app.separation import build_separator                      # noqa: E402
from _job import pipeline_audio                                 # noqa: E402
from diag_f0 import YIN_THRESH, framify, yin                    # noqa: E402

# Conventional adult ranges.  Used only to LABEL a measured median, never to
# threshold a decision -- the decision is the comparison between the two stems.
MALE_F0 = (85.0, 155.0)
FEMALE_F0 = (165.0, 265.0)


def voiced_median(x: np.ndarray, sr: int) -> tuple[float, float, int]:
    """(median f0 over voiced frames, voiced fraction, n voiced frames)."""
    seg, db, _ = framify(np.asarray(x, dtype=np.float64), sr)
    f0, ape = yin(seg, sr)
    m = (ape < YIN_THRESH) & np.isfinite(f0)
    if m.sum() < 6:
        return float("nan"), float(m.mean()), int(m.sum())
    return float(np.median(f0[m])), float(m.mean()), int(m.sum())


def label(f0: float) -> str:
    if not np.isfinite(f0):
        return "unattributable"
    if MALE_F0[0] <= f0 <= MALE_F0[1]:
        return "MALE"
    if FEMALE_F0[0] <= f0 <= FEMALE_F0[1]:
        return "FEMALE"
    return f"outside both ranges"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", nargs="?", default="runs/pair_v2")
    ap.add_argument("--male-track", type=int, default=0,
                    help="index of the track whose face is the male speaker "
                         "(confirmed visually; default 0 for the test clip)")
    args = ap.parse_args()

    job = Path(args.job_dir)
    sr = CONFIG.audio.sample_rate
    tj = json.loads((job / "tracks.json").read_text())
    tracks = tj["tracks"]
    meta = json.loads((job / "meta.json").read_text())

    print("=" * 78)
    print(f"stem -> channel -> track chain -- {job.name}")
    print("=" * 78)

    # -- what the pipeline recorded, with its space named ------------------- #
    assignment = meta.get("assignment")
    channel_of = [t.get("channel") for t in tracks]
    print(f"  meta.assignment  {assignment}   (track -> STEM, pre-export)")
    print(f"  tracks[].channel {channel_of}   (track -> CHANNEL, what the browser binds)")
    print("  These two are DIFFERENT permutations of the same decision whenever")
    print("  plan_channels() renumbers -- comparing them directly is the error.")

    # -- hop 1, measured: stem order vs channel order ----------------------- #
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mix.wav"
        media.extract_audio(pipeline_audio(job), wav, sr)
        mix = media.read_audio(wav, sr)

    device = CONFIG.runtime.resolve_device()
    sep = build_separator(CONFIG.runtime.separator, device=device,
                          cache_dir=str(CACHE_DIR / "models"),
                          chunk_s=CONFIG.audio.resolve_chunk_s(device),
                          overlap_s=CONFIG.audio.overlap_s)
    stems = np.asarray(sep.separate(mix, sr), dtype=np.float64)
    if stems.shape[0] > stems.shape[1]:
        stems = stems.T
    del sep

    ch, _ = sf.read(str(job / "stems_raw.wav"), dtype="float64", always_2d=True)
    ch = ch.T
    n_s, n_c = stems.shape[0], ch.shape[0]
    n = min(stems.shape[1], ch.shape[1])

    def corr(u: np.ndarray, v: np.ndarray) -> float:
        u = u[:n] - u[:n].mean()
        v = v[:n] - v[:n].mean()
        d = float(np.sqrt((u @ u) * (v @ v)))
        return float(u @ v / d) if d > 0 else 0.0

    C = np.array([[corr(stems[i], ch[j]) for j in range(n_c)] for i in range(n_s)])
    print(f"\n  hop 1 MEASURED: separator stems (rows) vs stems_raw.wav channels (cols)")
    print("         (raw is post-Wiener, so |r| < 1 even on the diagonal;")
    print("          the argmax is what identifies the permutation)")
    for i in range(n_s):
        print("    stem %d  " % i + "  ".join(f"ch{j} {C[i, j]:+.4f}" for j in range(n_c)))
    stem_to_ch = [int(np.argmax(np.abs(C[i]))) for i in range(n_s)]
    ok_perm = sorted(stem_to_ch) == list(range(n_c))
    print(f"    => stem -> channel  {stem_to_ch}"
          f"{'' if ok_perm else '   !! NOT a permutation, attribution unsafe'}")

    derived = None
    if assignment and ok_perm:
        # plan_channels: channel of track i is the position of its stem in `order`.
        # Recomputing it here is a cross-check on hop 1, not the source of it.
        derived = [stem_to_ch[s] if s is not None and 0 <= s < n_s else None
                   for s in assignment]
        agree = derived == channel_of
        print(f"    cross-check: assignment {assignment} through the measured hop "
              f"gives {derived}, tracks.json says {channel_of} -- "
              f"{'consistent' if agree else 'INCONSISTENT'}")

    # -- attribution: which voice is in each stem --------------------------- #
    print(f"\n  f0 attribution (octave-safe YIN, the estimator diag_f0.py settled on)")
    print(f"    male range {MALE_F0[0]:g}-{MALE_F0[1]:g} Hz, "
          f"female {FEMALE_F0[0]:g}-{FEMALE_F0[1]:g} Hz")
    f0_mix, vf_mix, nv_mix = voiced_median(mix, sr)
    print(f"    mixture      f0 {f0_mix:6.1f} Hz   voiced {100 * vf_mix:5.1f}% "
          f"(n={nv_mix})   -- both voices, so expect no clean label")
    stem_voice, ch_voice = [], []
    for i in range(n_s):
        f0, vf, nv = voiced_median(stems[i], sr)
        stem_voice.append(f0)
        print(f"    stem    {i}    f0 {f0:6.1f} Hz   voiced {100 * vf:5.1f}% "
              f"(n={nv:4d})   -> {label(f0)}")
    for j in range(n_c):
        f0, vf, nv = voiced_median(ch[j], sr)
        ch_voice.append(f0)
        print(f"    channel {j}    f0 {f0:6.1f} Hz   voiced {100 * vf:5.1f}% "
              f"(n={nv:4d})   -> {label(f0)}")

    # -- the verdict, per FACE ---------------------------------------------- #
    print(f"\n  verdict -- track {args.male_track} is the male face (given)")
    male_stems = [i for i in range(n_s) if label(stem_voice[i]) == "MALE"]
    female_stems = [i for i in range(n_s) if label(stem_voice[i]) == "FEMALE"]
    if len(male_stems) != 1 or len(female_stems) != 1:
        print(f"    f0 does not cleanly split the stems (male {male_stems}, "
              f"female {female_stems}); cannot arbitrate the pairing here.")
        return 0
    ms, fs = male_stems[0], female_stems[0]
    print(f"    stem {ms} is the man's voice, stem {fs} is the woman's")

    bad = 0
    for i, t in enumerate(tracks):
        c = t.get("channel")
        if c is None:
            print(f"    track {i} ({t.get('label')}): no channel -- unmatched")
            continue
        heard = label(ch_voice[c]) if c < n_c else "?"
        want = "MALE" if i == args.male_track else "FEMALE"
        good = heard == want
        bad += not good
        print(f"    track {i} ({t.get('label'):9s}) -> channel {c} -> "
              f"f0 {ch_voice[c]:6.1f} Hz = {heard:6s}   want {want:6s}   "
              f"{'OK' if good else 'WRONG'}")
    print()
    if bad == 0:
        print("  PAIRING CORRECT: every face hears its own voice.")
    else:
        print(f"  PAIRING INVERTED: {bad} of {len(tracks)} faces hear the wrong voice.")
        print("  One press of the UI's swap control fixes it, and because the")
        print("  separator is bit-deterministic here (diag_sep_repro.py) this is")
        print("  reproducible rather than a per-run coin flip -- so the press")
        print("  always lands the same way on this clip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
