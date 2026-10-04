"""Who is in each stem?  f0 histograms plus formants, which cannot octave-error.

``diag_scale.py`` refuted two hypotheses and exposed a third that invalidates a
chain of earlier work:

  * input scale is NOT the problem.  Purity was 0.590 at every gain across 60 dB
    and output RMS tracked input 1:1, so SepFormer is scale-equivariant here and
    the missing input normalisation in ``separation.py`` is not the defect.
  * the mixture reads a median f0 of 317 Hz with 49% of voiced frames above
    303.9 Hz -- yet the two separated stems read 149.3 Hz and 226.3 Hz, with
    nothing above 304 Hz at all.

The second is the important one.  A two-talker frame is not periodic at either
speaker's period, so YIN's difference function never dips below threshold, the
``argmin`` fallback fires, and the answer lands anywhere -- typically an octave
high.  **Pitch measured on a mixture is not a speaker label for overlapping
speech**, and every label in ``diag_collapse.py`` was built on exactly that,
with a male/female split of 303.9 Hz derived from a 393.5 Hz "female" reference
that is itself far above the 180-250 Hz the bug report quotes.  So the
selectivity numbers, the 15.8% gate prediction, and the pitch-based exoneration
of the matcher all have to be withdrawn and redone.

The stems, being close to single-talker, are where pitch *is* measurable.  This
script asks who they are, three independent ways:

  A  **f0 histograms** at fmax=300 Hz, per stem and for the mixture.  A clean
     single-speaker stem is unimodal; a satellite mode at exactly 2x the main one
     is the octave artefact made visible; two well-separated modes in one stem is
     contamination.
  B  **formants F1-F3** by LPC root-solving.  These measure vocal-tract length,
     not periodicity, so they cannot inherit an octave error.  Adult female F3
     runs ~2800-3300 Hz against a male ~2400-2700 Hz, which settles sex
     independently of f0.
  C  **the shipped artefact.**  The same two measurements on ``stems_raw.wav``
     channels 0 and 1 -- after the mask, after ``plan_channels``, after the
     export gain.  Track 0 is the man and track 1 the woman (annotated frames,
     ``diag_id``), and channel i carries the stem the matcher gave track i, so:

         channel 0 reads MALE   -> the shipped pairing is correct
         channel 0 reads FEMALE -> the matcher is inverted, which is precisely
                                   "selecting Male played Female audio"

Run::

    PYTHONPATH=. python scripts/diag_ident.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media                                      # noqa: E402
from app.config import CACHE_DIR, CONFIG                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin               # noqa: E402
from diag_id import formants                               # noqa: E402

FMAX_SPEECH = 300.0          # above any ordinary speaking f0, below its double
BINS = np.arange(60, 310, 10)


def voiced_f0(x: np.ndarray, sr: int, *, fmax: float = FMAX_SPEECH
              ) -> tuple[np.ndarray, float, float]:
    """(f0 of the voiced frames, voiced fraction, doubling-ambiguity fraction).

    The ambiguity fraction is the share of voiced frames where the difference
    function is *also* sub-threshold at half the chosen period -- i.e. frames
    where the estimator had a genuine octave choice to make.  A large value is
    the warning label this whole investigation needed earlier.
    """
    seg, db, _ = framify(x, sr)
    f0, ape = yin(seg, sr, fmax=fmax)
    ref = float(np.percentile(db, 95))
    v = (ape < YIN_THRESH) & np.isfinite(f0) & (db > ref - 25.0)
    if v.sum() < 10:
        return np.empty(0), float(v.mean()), float("nan")
    # Re-run one octave lower to see whether the low candidate also explains it.
    f0lo, apelo = yin(seg, sr, fmax=fmax / 2.0)
    amb = float(np.mean((apelo[v] < YIN_THRESH)
                        & (np.abs(f0[v] / np.maximum(f0lo[v], 1e-9) - 2.0) < 0.15)))
    return f0[v], float(v.mean()), amb


def hist_line(f0: np.ndarray) -> str:
    h, _ = np.histogram(f0, bins=BINS)
    if h.max() == 0:
        return " " * (len(BINS) - 1)
    return "".join("#" if v > h.max() * 0.5 else ("+" if v > h.max() * 0.2
                   else ("." if v > 0 else " ")) for v in h)


def modes(f0: np.ndarray) -> list[tuple[float, int]]:
    """Local maxima of the 10 Hz histogram holding >=15% of the peak."""
    h, e = np.histogram(f0, bins=BINS)
    out = []
    for i in range(len(h)):
        lo = h[i - 1] if i > 0 else 0
        hi = h[i + 1] if i + 1 < len(h) else 0
        if h[i] >= max(lo, hi) and h[i] > 0.15 * h.max():
            out.append((float((e[i] + e[i + 1]) / 2), int(h[i])))
    return sorted(out, key=lambda p: -p[1])


def describe(tag: str, x: np.ndarray, sr: int) -> dict:
    f0, vfrac, amb = voiced_f0(x, sr)
    F = formants(x, sr, n_form=3, order=16)
    print(f"\n  {tag}")
    if f0.size == 0:
        print("    no voiced frames")
        return {}
    q = np.percentile(f0, [10, 50, 90])
    print(f"    f0  p10/p50/p90 {q[0]:6.1f}/{q[1]:6.1f}/{q[2]:6.1f} Hz   "
          f"voiced {vfrac * 100:5.1f}%   n={f0.size}   "
          f"octave-ambiguous frames {amb * 100:5.1f}%")
    print(f"    60Hz [{hist_line(f0)}] 300Hz")
    md = modes(f0)
    print("    modes: " + ", ".join(f"{m:.0f} Hz (n={c})" for m, c in md[:3]))
    fs = "  ".join("n/a" if not np.isfinite(v) else f"F{k + 1} {v:6.0f} Hz"
                   for k, v in enumerate(F))
    print(f"    formants: {fs}")
    # Sex call: two independent votes, reported separately so a disagreement is
    # visible rather than averaged away.
    v_f0 = "female" if q[1] >= 185 else ("male" if q[1] <= 165 else "ambiguous")
    f3 = F[2] if len(F) > 2 else float("nan")
    v_fm = ("female" if f3 >= 2800 else ("male" if f3 <= 2650 else "ambiguous")
            ) if np.isfinite(f3) else "n/a"
    print(f"    -> f0 says {v_f0}, F3 says {v_fm}"
          + ("   (agree)" if v_f0 == v_fm else "   (DISAGREE)"))
    return {"f0": float(q[1]), "vote_f0": v_f0, "vote_f3": v_fm,
            "voiced": vfrac, "F": F}


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    meta = json.loads((job / "meta.json").read_text())
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)
    pre = np.asarray(np.load(CACHE_DIR / "diag_pose"
                             / f"{job.name}_stems_premask.npy"), dtype=np.float64)
    raw, _ = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T

    print(f"{job.name}: f0 ceiling {FMAX_SPEECH:g} Hz (was 400, which admitted "
          f"the doubled band)")

    print("\n=== A/B: the mixture, for reference ===")
    describe("mixture", mix, sr)

    print("\n=== A/B: raw separator output, in STEM order ===")
    st = [describe(f"pre-mask stem {j}", pre[j], sr) for j in range(pre.shape[0])]

    print("\n=== C: the shipped artefact, in CHANNEL order ===")
    print("    channel i carries the stem the matcher gave track i;")
    print("    track 0 = the MAN, track 1 = the WOMAN (annotated frames)")
    ch = [describe(f"stems_raw.wav channel {j}", raw[j], sr)
          for j in range(raw.shape[0])]

    print(f"\n=== verdict ===")
    print(f"  meta assignment {meta['assignment']} (stem index per track)")
    want = ["male", "female"]                       # track 0 man, track 1 woman
    for j, c in enumerate(ch):
        if not c:
            print(f"  channel {j}: not measurable")
            continue
        votes = [v for v in (c["vote_f0"], c["vote_f3"])
                 if v in ("male", "female")]
        call = votes[0] if len(set(votes)) == 1 and votes else "ambiguous"
        ok = call == want[j]
        print(f"  channel {j} (owner {'MAN' if j == 0 else 'WOMAN'}): reads "
              f"{call.upper()}  f0 {c['f0']:.0f} Hz  "
              f"F3 {c['F'][2] if len(c['F']) > 2 else float('nan'):.0f} Hz"
              + ("   OK" if ok else "   <-- MISMATCH"))
    if all(c for c in ch) and len(ch) == 2:
        calls = []
        for c in ch:
            votes = [v for v in (c["vote_f0"], c["vote_f3"])
                     if v in ("male", "female")]
            calls.append(votes[0] if len(set(votes)) == 1 and votes
                         else "ambiguous")
        if calls == ["female", "male"]:
            print("  -> the matcher is INVERTED on this clip: the man's button "
                  "carries the\n     woman's stem.  That is verbatim the user's "
                  "report.")
        elif calls == ["male", "female"]:
            print("  -> the shipped pairing is correct; the defect is elsewhere.")
        else:
            print("  -> the two channels do not read as two different voices, "
                  "so this is a\n     separation failure and the pairing "
                  "question is downstream of it.")


if __name__ == "__main__":
    main()
