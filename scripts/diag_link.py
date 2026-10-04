"""Which FACE owns the low voice?  Mixture pitch vs lips, with the matcher out.

``diag_ident.py`` established two distinct voices in the separator output:

    stem 0 (-> channel 1):  f0 146-149 Hz, F2 1371-1384 Hz,  8-9% voiced
    stem 1 (-> channel 0):  f0 200-202 Hz, F2 1600-1602 Hz, 25-28% voiced

An f0 ratio of 1.38 and an F2 ratio of 1.17 both point the same way, and 1.17 is
the textbook adult female/male F2 ratio.  So the low-f0 stem is the man's voice
on a biological prior.  Since ``assignment = [1, 0]`` sends track 0 -- the man,
confirmed on the annotated frame -- to stem 1, the pairing looks inverted.

But "looks inverted on a prior" is not a measurement, and the prior is the only
thing linking a voice to a face.  This measures that link, and the hard part is
avoiding circularity:

  * the matcher chose its assignment by correlating lip motion against **stem
    energy envelopes**.  So any test that labels audio frames by which stem is
    louder, or by which stem is voiced, reproduces the matcher's own statistic
    and can only ever agree with it.  Both are disqualified.
  * what is NOT circular is the **mixture's own pitch**: the separator never saw
    the video and the matcher never saw the pitch.  With the ceiling at 300 Hz
    the octave-ambiguous share of mixture frames is 7.7%, so it is now usable --
    which it was not at 400 Hz, and that is why every earlier attempt failed.

So: take the mixture frames whose octave-safe f0 sits confidently near one voice
and clearly away from the other, and ask which face's mouth is moving.  The
~175 Hz band where the two f0 distributions overlap is excluded rather than
guessed at.

  A  do the two stems genuinely differ in pitch, frame by frame, or only in the
     medians of two different frame sets?  Paired comparison on the frames where
     both are voiced, plus formants measured on that same matched set.
  B  the low-voice and high-voice frame sets from the mixture alone.
  C  lip motion in those two sets.  The statistic is a difference-in-differences
     between the two populations, so a face that simply moves more overall
     cannot produce it -- and per-track percentile ranks already force each
     track's marginal mean to 0.5.
  D  the resulting correct assignment, against the shipped one.
  E  and, as a by-product, how informative the lip signal actually is against a
     label that is not broken -- which decides whether the AV-VAD in item 3 of
     the bug report has anything to stand on.  The 49.8% figure I reported
     earlier was measured against the 303.9 Hz split and has to be redone.

Run::

    PYTHONPATH=. python scripts/diag_link.py runs/862bd92a01ac
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
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,    # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin               # noqa: E402
from diag_id import formants                               # noqa: E402

SUBNASALE = 2
FMAX_SPEECH = 300.0
F_LOW, F_HIGH = 146.0, 201.0                 # the two measured voices
LOW_MAX, HIGH_MIN = 165.0, 190.0             # exclude the overlap band
WIN_S = 0.4


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
    meta = json.loads((job / "meta.json").read_text())
    tj = json.loads((job / "tracks.json").read_text())
    fps, n_tr = float(tj["fps"]), len(tj["tracks"])
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)
    pre = np.asarray(np.load(CACHE_DIR / "diag_pose"
                             / f"{job.name}_stems_premask.npy"), dtype=np.float64)

    # ---- A: do the two stems really differ, frame by frame? ---------------- #
    print("=== A: paired per-frame comparison of the two stems ===")
    per = []
    for j in range(pre.shape[0]):
        seg, db, t = framify(pre[j], sr)
        f0, ape = yin(seg, sr, fmax=FMAX_SPEECH)
        ref = float(np.percentile(db, 95))
        v = (ape < YIN_THRESH) & np.isfinite(f0) & (db > ref - 25.0)
        per.append((f0, v, seg))
    both = per[0][1] & per[1][1]
    print(f"  frames voiced in BOTH stems: {int(both.sum())}")
    if both.sum() >= 20:
        d = per[1][0][both] - per[0][0][both]
        print(f"  stem1 f0 - stem0 f0: median {np.median(d):+6.1f} Hz, "
              f"stem1 higher in {float(np.mean(d > 0)) * 100:5.1f}% of them")
        # Formants on the SAME frames, so vowel content cannot differ between
        # the two measurements the way it can across two disjoint frame sets.
        idx = np.flatnonzero(both)
        for j in range(2):
            sub = np.concatenate([per[j][2][i] for i in idx])
            F = formants(sub, sr, n_form=3, order=16)
            print(f"  stem {j} on matched frames: "
                  + "  ".join(f"F{k + 1} {v:6.0f}" for k, v in enumerate(F)))

    # ---- B: the mixture's own confident frames ----------------------------- #
    seg, mdb, t = framify(mix, sr)
    f0, ape = yin(seg, sr, fmax=FMAX_SPEECH)
    conf = (ape < YIN_THRESH) & np.isfinite(f0)
    low = conf & (f0 <= LOW_MAX)
    high = conf & (f0 >= HIGH_MIN)
    print(f"\n=== B: mixture frames, separator-free and matcher-free ===")
    print(f"  ceiling {FMAX_SPEECH:g} Hz; overlap band "
          f"{LOW_MAX:g}-{HIGH_MIN:g} Hz excluded")
    print(f"  LOW  (<= {LOW_MAX:g} Hz): {int(low.sum()):4d} frames, "
          f"median f0 {np.median(f0[low]):6.1f} Hz")
    print(f"  HIGH (>= {HIGH_MIN:g} Hz): {int(high.sum()):4d} frames, "
          f"median f0 {np.median(f0[high]):6.1f} Hz")

    # ---- C: which mouth is moving in each set? ----------------------------- #
    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    aps = [aperture(lm[i]) for i in range(n_tr)]
    w_v = max(2, int(round(WIN_S * fps)))
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

    print("\n=== C: lip motion in those sets (track 0 = MAN, track 1 = WOMAN) ===")
    print("    per-track percentile ranks, so each track's marginal mean is 0.5")
    print("    and a face that simply moves more cannot fake a difference")
    stats = {}
    for tag, m in (("all frames", okv), ("LOW  voice", okv & low),
                   ("HIGH voice", okv & high)):
        if m.sum() < 20:
            print(f"  {tag}: {int(m.sum())} frames (too few)")
            continue
        d = r0[m] - r1[m]
        se = float(np.std(d, ddof=1) / np.sqrt(d.size))
        stats[tag] = (float(np.mean(d)), se, int(d.size))
        print(f"  {tag}: n={d.size:4d}   mean(rank_man - rank_woman) "
              f"{np.mean(d):+6.3f} +- {se:.3f}   "
              f"man ahead in {float(np.mean(d > 0)) * 100:5.1f}%")
    if "LOW  voice" in stats and "HIGH voice" in stats:
        (ml, sl, nl), (mh, sh, nh) = stats["LOW  voice"], stats["HIGH voice"]
        dd = ml - mh
        sd = float(np.sqrt(sl ** 2 + sh ** 2))
        print(f"\n  difference-in-differences: {dd:+.3f} +- {sd:.3f}  "
              f"({abs(dd) / max(sd, 1e-9):.1f} sigma)")
        if abs(dd) < 2 * sd:
            print("  -> no significant link between voice pitch and face. "
                  "The lip feature\n     cannot arbitrate ownership on this clip.")
            owner_low = None
        else:
            owner_low = 0 if dd > 0 else 1
            print(f"  -> the LOW voice belongs to track {owner_low} "
                  f"({'MAN' if owner_low == 0 else 'WOMAN'}), measured, "
                  f"not assumed")
    else:
        owner_low = None

    # ---- D: correct assignment vs shipped ---------------------------------- #
    print("\n=== D: the pairing ===")
    lo_stem = 0 if np.median(per[0][0][per[0][1]]) < np.median(
        per[1][0][per[1][1]]) else 1
    print(f"  the low-f0 stem is stem {lo_stem} "
          f"(f0 {np.median(per[lo_stem][0][per[lo_stem][1]]):.0f} Hz)")
    print(f"  shipped assignment {meta['assignment']}: track 0 -> stem "
          f"{meta['assignment'][0]}, track 1 -> stem {meta['assignment'][1]}")
    prior = 0            # the man owns the low voice, on the f0/F2 prior
    for src, who in (("measured (C)", owner_low), ("prior (f0+F2)", prior)):
        if who is None:
            print(f"  {src}: no call")
            continue
        want = [lo_stem, 1 - lo_stem] if who == 0 else [1 - lo_stem, lo_stem]
        print(f"  {src}: low voice -> track {who}, so the correct assignment is "
              f"{want}"
              + ("   MATCHES shipped" if want == list(meta["assignment"])
                 else "   <-- shipped assignment is INVERTED"))

    # ---- E: is the lip signal informative at all? -------------------------- #
    print("\n=== E: lip informativeness against a working label ===")
    print("    (the 49.8% I reported earlier used the broken 303.9 Hz split)")
    if owner_low is not None or True:
        # Score the prior's pairing: LOW frames should show the man ahead.
        for thr in (0.0, 0.2, 0.4):
            m = okv & (low | high) & (np.abs(r0 - r1) > thr)
            if m.sum() < 20:
                print(f"  |rank gap| > {thr}: {int(m.sum())} frames (too few)")
                continue
            pred_man = r0[m] > r1[m]
            is_low = low[m]
            acc = float(np.mean(pred_man == is_low))
            print(f"  |rank gap| > {thr}: n={int(m.sum()):4d}  "
                  f"'louder mouth owns the pitch' holds {acc * 100:5.1f}%"
                  + ("   <-- informative" if acc > 0.62 else
                     "   <-- chance" if acc < 0.58 else "   <-- weak"))


if __name__ == "__main__":
    main()
