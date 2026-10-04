"""Why did the matcher invert, and which audio feature fixes it?

Established: the separator produced two genuinely different voices --
stem 0 at 146-149 Hz (F2 1371) and stem 1 at 200-202 Hz (F2 1602), with stem 1
higher in 90.4% of the frames where both are voiced -- and ``assignment = [1, 0]``
handed the man's track the 200 Hz stem.  Two independent routes say the correct
assignment is ``[0, 1]``: the f0/F2 sex prior, and a matcher-free
difference-in-differences between mixture-pitch frame sets and lip motion
(+0.167 +- 0.079, 2.1 sigma).

The suspected mechanism is a property of the score, not a coding error.
``score[i, j] = corr(|d lip_i|, env_dB_j)`` rewards a stem for being loud
*whenever anyone speaks*.  Stem 1 holds 45.9% of the mixture energy on its own
against stem 0's 9.8%, so it carries the other speaker as leakage and its
envelope tracks total speech activity.  Both faces then correlate with it, the
Hungarian is forced to give it to whichever face moves more overall -- track 0,
ahead in 55.8% of frames -- and stem 0 falls to the other face by elimination.
That predicts ``[1, 0]`` exactly, and a near-zero margin: 0.0376.

So this script does three things.

**1. Reproduce the shipped numbers.**  Rebuild the score matrix from the cached
landmarks and the recomputed stems and check the confidence against the 0.0376 in
``meta.json``.  Until that matches, no variant measured here is comparable to
what shipped.

**2. Test candidate audio features against the known-correct answer.**  The
hypothesis says the defect is the *common mode* -- the part of every stem's
envelope that is merely "somebody is talking".  Removing it should leave identity:

    dB               what ships
    linear RMS       does the dB floor dominate the z-score?
    dB - crossmean   common-mode removed
    power share      10log10(p_j / sum_k p_k): the stem's share of each frame
    share, loud      the same, with silence frames dropped
    share, pre-mask  whether our own Wiener mask helps or hurts the matcher

**3. Settle item 1 with the corrected estimator.**  Per-chunk median f0 of each
stem over the four chunk interiors.  A flip at a boundary would show as the two
columns trading pitch ranges.  This is the octave-safe version of a test that was
run earlier with a 400 Hz ceiling, which is exactly the ceiling that produced
wholesale octave doubling on the mixture.

Run::

    PYTHONPATH=. python scripts/diag_feat.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dsp, media                                 # noqa: E402
from app.config import CACHE_DIR, CONFIG                   # noqa: E402
from app.matching import _zscore, energy_envelope          # noqa: E402
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,    # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace, resample_lip)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin               # noqa: E402

FMAX_SPEECH = 300.0
CORRECT = [0, 1]                 # track 0 (man) -> stem 0 (146 Hz)


def shipped_lip(lm: np.ndarray) -> np.ndarray:
    """``inner_lip_area / interocular**2`` -- exactly the shipped feature."""
    n = lm.shape[0]
    out = np.full(n, np.nan)
    for f in range(n):
        p = lm[f]
        if not np.isfinite(p[LEFT_EYE_OUTER, 0]):
            continue
        ring = p[INNER_LIP_RING, :2]
        if not np.isfinite(ring).all():
            continue
        inter = float(np.linalg.norm(p[LEFT_EYE_OUTER, :2]
                                     - p[RIGHT_EYE_OUTER, :2]))
        if inter < 1e-6:
            continue
        out[f] = _shoelace(ring.astype(np.float64)) / inter ** 2
    return out


def solve_and_conf(score: np.ndarray) -> tuple[list[int], float]:
    """The production pairing and its Murty-first-step margin, verbatim."""
    from scipy.optimize import linear_sum_assignment

    def _solve(m):
        r, c = linear_sum_assignment(-m)
        return list(r), list(c)

    rows, cols = _solve(score)
    s0 = float(sum(score[r, c] for r, c in zip(rows, cols)))
    span = float(score.max()) - float(score.min())
    pen = float(score.min()) - (span + 1.0) * (len(rows) + 1)
    second = -np.inf
    for r, c in zip(rows, cols):
        m = score.copy()
        m[r, c] = pen
        rr, cc = _solve(m)
        if any(a == r and b == c for a, b in zip(rr, cc)):
            continue
        second = max(second, float(sum(score[a, b] for a, b in zip(rr, cc))))
    conf = s0 - second if np.isfinite(second) else s0
    assign = [-1] * score.shape[0]
    for r, c in zip(rows, cols):
        assign[r] = int(c)
    return assign, max(0.0, float(conf))


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    meta = json.loads((job / "meta.json").read_text())
    tj = json.loads((job / "tracks.json").read_text())
    fps, n_tr = float(tj["fps"]), len(tj["tracks"])
    sr = CONFIG.audio.sample_rate
    mc = CONFIG.match

    mix = media.read_audio(CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav", sr)
    pre = np.asarray(np.load(CACHE_DIR / "diag_pose"
                             / f"{job.name}_stems_premask.npy"), dtype=np.float64)
    post = np.asarray(dsp.wiener_separate(
        pre.astype(np.float32), sample_rate=sr, nfft=CONFIG.audio.nfft,
        hop=CONFIG.audio.hop, exponent=CONFIG.gate.mask_exponent,
        floor=CONFIG.gate.mask_floor,
        cepstral_order=CONFIG.gate.cepstral_smooth_order), dtype=np.float64)

    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    lips_raw = [shipped_lip(lm[i]) for i in range(n_tr)]

    def score_matrix(stems: np.ndarray, mode: str, loud_only: bool = False,
                     differential: bool = False) -> np.ndarray:
        envs_db = [energy_envelope(s, sr, mc.env_frame_ms, mc.smooth_kernel)
                   for s in stems]
        n_frames = min(len(e) for e in envs_db)
        env_fps = 1000.0 / mc.env_frame_ms
        E = np.stack([e[:n_frames] for e in envs_db])
        if mode == "db":
            A = E
        elif mode == "linear":
            A = 10.0 ** (E / 20.0)
        elif mode == "db-crossmean":
            A = E - E.mean(axis=0, keepdims=True)
        elif mode == "share":
            p = 10.0 ** (E / 10.0)
            A = 10.0 * np.log10(p / np.maximum(p.sum(axis=0, keepdims=True), 1e-30))
        else:
            raise ValueError(mode)

        keep = np.ones(n_frames, dtype=bool)
        if loud_only:
            # Top half of the mixture's own dynamic range: silence frames carry
            # no identity information but plenty of floor variation.
            me = energy_envelope(mix, sr, mc.env_frame_ms,
                                 mc.smooth_kernel)[:n_frames]
            keep = me > np.median(me)

        lips = []
        for i in range(n_tr):
            r = resample_lip(list(lips_raw[i]), fps, n_frames, env_fps)
            lips.append(_zscore(np.abs(np.diff(r, prepend=r[:1])))[keep])

        if differential:
            dl, da = lips[0] - lips[1], _zscore(A[0][keep]) - _zscore(A[1][keep])
            sl, sa = float(np.std(dl)), float(np.std(da))
            r = (float(np.mean((dl - dl.mean()) * (da - da.mean())) / (sl * sa))
                 if sl > 1e-9 and sa > 1e-9 else 0.0)
            return np.array([[r, 0.0], [0.0, 0.0]])

        out = np.zeros((n_tr, stems.shape[0]))
        for i in range(n_tr):
            for j in range(stems.shape[0]):
                out[i, j] = float(np.dot(lips[i], _zscore(A[j][keep]))
                                  / max(int(keep.sum()), 1))
        return out

    # ---- 1: reproduce what shipped ----------------------------------------- #
    print("=== 1: reproduce the shipped matcher ===")
    s = score_matrix(post, "db")
    a, c = solve_and_conf(s)
    print("  score matrix (rows=tracks, cols=stems):")
    print("    " + np.array2string(s, precision=4).replace("\n", "\n    "))
    print(f"  reproduced: assignment {a}  confidence {c:.4f}")
    print(f"  meta.json : assignment {meta['assignment']}  confidence "
          f"{meta['confidence'][0]:.4f}"
          + ("   REPRODUCED"
             if a == list(meta["assignment"])
             and abs(c - meta["confidence"][0]) < 5e-3
             else "   <-- harness differs; variants below are only indicative"))

    # ---- 2: candidate features --------------------------------------------- #
    print(f"\n=== 2: candidate audio features (correct answer is {CORRECT}) ===")
    print(f"  {'feature':>18} {'t0->stem':>9} {'t1->stem':>9} {'conf':>8}  verdict")
    for name, stems, mode, loud in [
            ("db (shipped)", post, "db", False),
            ("linear RMS", post, "linear", False),
            ("db - crossmean", post, "db-crossmean", False),
            ("power share", post, "share", False),
            ("power share, loud", post, "share", True),
            ("share, pre-mask", pre, "share", True),
            ("db, pre-mask", pre, "db", False)]:
        s = score_matrix(stems, mode, loud)
        a, c = solve_and_conf(s)
        print(f"  {name:>18} {a[0]:9d} {a[1]:9d} {c:8.4f}  "
              + ("CORRECT" if a == CORRECT else "inverted")
              + ("" if c >= mc.min_confidence
                 else f"   (below min_confidence {mc.min_confidence})"))

    # ---- 2b: is the envelope even usable? ---------------------------------- #
    # `energy_envelope` is 20log10(rms + 1e-10), so a frame of near-silence
    # reads about -200 dB.  Post-Wiener with mask_floor=0.0 the quiet stem has a
    # great many such frames, and after z-scoring those outliers own the
    # variance -- the speech dynamics we meant to correlate get squeezed flat.
    print("\n=== 2b: envelope health (dB, before z-scoring) ===")
    print(f"  {'':>8} {'p1':>8} {'p50':>8} {'p95':>8} {'below p95-40':>13}")
    for j in range(post.shape[0]):
        e = energy_envelope(post[j], sr, mc.env_frame_ms, mc.smooth_kernel)
        q = np.percentile(e, [1, 50, 95])
        print(f"  stem {j:>2} {q[0]:8.1f} {q[1]:8.1f} {q[2]:8.1f} "
              f"{float(np.mean(e < q[2] - 40)) * 100:12.1f}%")

    # ---- 2c: the differential statistic ------------------------------------ #
    # Every variant above correlates ONE lip against ONE stem, so a stem that is
    # loud whenever anybody speaks correlates with both faces and the pairing is
    # decided by which face moves more overall.  The fix is to make the question
    # competitive on BOTH sides at once: does the lip-motion DIFFERENCE between
    # the two faces track the activity DIFFERENCE between the two stems?  Common
    # mode cancels in numerator and denominator, and with 2x2 the sign of a
    # single correlation settles the assignment.  This is the same structure as
    # the difference-in-differences in `diag_link`, which found +0.167 +- 0.079.
    print(f"\n=== 2c: differential -- corr(lip0-lip1, feat0-feat1) ===")
    print(f"  {'feature':>18} {'r':>8} {'implies':>10}  verdict")
    for name, stems, mode, loud in [
            ("db", post, "db", False),
            ("db, loud", post, "db", True),
            ("linear RMS", post, "linear", False),
            ("power share", post, "share", False),
            ("power share, loud", post, "share", True),
            ("share, pre-mask", pre, "share", False),
            ("db, pre-mask", pre, "db", False),
            ("db, pre-mask, loud", pre, "db", True)]:
        s = score_matrix(stems, mode, loud, differential=True)
        r = float(s[0, 0])
        a = CORRECT if r > 0 else CORRECT[::-1]
        print(f"  {name:>18} {r:+8.4f} {str(a):>10}  "
              + ("CORRECT" if a == CORRECT else "inverted"))

    # ---- 3: item 1, per-chunk stem identity -------------------------------- #
    print("\n=== 3: item 1 -- did the stems ever trade places at a boundary? ===")
    chunk = int(CONFIG.audio.chunk_s * sr)
    ov = min(int(CONFIG.audio.overlap_s * sr), chunk // 2)
    stride = chunk - ov
    n = pre.shape[1]
    starts = list(range(0, max(1, n - ov), stride))
    print(f"  chunk {CONFIG.audio.chunk_s:g}s overlap {CONFIG.audio.overlap_s:g}s"
          f" -> {len(starts)} chunks, {len(starts) - 1} boundaries at "
          + ", ".join(f"{st / sr:.0f}s" for st in starts[1:]))
    print(f"  {'chunk interior':>16}  "
          + "  ".join(f"{'stem ' + str(j) + ' (f0 / voiced / share)':>28}"
                      for j in range(pre.shape[0])))
    for k, st in enumerate(starts):
        # Interior only: skip the crossfade zones, so each reading is one chunk.
        a0 = st + (ov if k > 0 else 0)
        a1 = min(st + chunk, n) - (ov if k < len(starts) - 1 else 0)
        if a1 - a0 < int(0.5 * sr):
            continue
        # Share of the two stems' joint energy, so "few voiced frames" can be
        # told apart from "this whole chunk is quiet".
        tot = float(sum(np.sum(pre[j, a0:a1] ** 2) for j in range(pre.shape[0])))
        cells = []
        for j in range(pre.shape[0]):
            sh = 100.0 * float(np.sum(pre[j, a0:a1] ** 2)) / max(tot, 1e-30)
            seg, db, _ = framify(pre[j, a0:a1], sr)
            f0, ape = yin(seg, sr, fmax=FMAX_SPEECH)
            v = ((ape < YIN_THRESH) & np.isfinite(f0)
                 & (db > np.percentile(db, 95) - 25))
            head = (f"{np.median(f0[v]):6.1f} Hz  n={int(v.sum()):3d}"
                    if v.sum() >= 10 else "     too few      ")
            cells.append(f"{head}  {sh:4.1f}%")
        print(f"  {f'{a0 / sr:4.1f}-{a1 / sr:4.1f}s':>16}  "
              + "  ".join(f"{x:>28}" for x in cells))
    print("  a flip would show the two columns swapping pitch ranges")


if __name__ == "__main__":
    main()
