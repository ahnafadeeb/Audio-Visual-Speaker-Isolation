"""Is the lip signal measuring articulation, or head motion?

The pairing is settled: track 0 is the man (screen-left, centroid x 0.301, and
the annotated frame), stem 0 selectively carries the 146 Hz voice (+21.5 dB
paired swing, 95% CI [+11.3, +31.0], labelled from the mixture's own octave-safe
f0 with neither separator nor matcher in the loop), and channel 0 -- track 0's
button -- carries the HIGH voice.  So ``assignment = [1, 0]`` is inverted and
``[0, 1]`` is correct.

What is *not* settled is why.  Fifteen audio features fail to fix it: seven
absolute (dB, linear RMS, dB minus cross-stem mean, power share, share on loud
frames, share pre-mask, dB pre-mask) and eight differential
``corr(lip0-lip1, feat0-feat1)``, which cancels common mode on both sides.  All
fifteen land on ``[1, 0]``.  A feature bug does not survive that.  Two things
about the numbers say where to look instead:

  * every correlation is tiny, |r| <= 0.09, and the shipped confidence 0.0376 is
    already below ``min_confidence`` 0.05.  The matcher was not confidently
    wrong; it was operating at chance and reported so.
  * restricting to loud frames makes it *more* consistently wrong (r -0.028 ->
    -0.086).  Chance does not sharpen under conditioning.  Something systematic
    is pulling track 0 toward the woman's voice.

The candidate mechanism is rigid head motion.  The faces are 162 and 127 px in a
1920x1080 frame, so the inner-lip ring is ~20x11 px and its shoelace *area* is
quadratic in landmark noise.  Worse, the shipped feature normalises by
interocular distance, which **shrinks as the head yaws**, so a turn of the head
changes the feature with the mouth completely still.  Track 0's median yaw is
28.4 degrees against track 1's 18.8.  And a talk-show host turns toward his guest
precisely while she is talking -- which would manufacture a positive correlation
between his "lip motion" and her voice.

  A  do the head-motion proxies predict the shipped lip signal?
  B  does each track's head motion follow the OTHER speaker's voice?  Same
     mixture-f0 labels, difference-in-differences on per-track percentile ranks
     so a face that simply moves more cannot fake it.
  C  six candidate lip features, scored exactly as production scores them:

       area        shipped: shoelace(inner ring) / interocular^2
       gap         |p13 - p14| / interocular -- linear in noise, not quadratic
       gap_v       ... / (eye-midpoint to subnasale): a vertical normaliser does
                   not foreshorten under yaw, an interocular one does
       gap_proc    similarity-aligned to a per-track reference built from the
                   rigid points, so translation, in-plane rotation and scale are
                   removed by construction
       area_smooth smoothed before differencing, since differencing noise
                   amplifies it
       area_resid  shipped feature with head motion regressed out

Run::

    PYTHONPATH=. python scripts/diag_rigid.py runs/862bd92a01ac
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
from diag_feat import solve_and_conf                       # noqa: E402

FMAX_SPEECH = 300.0
LOW_MAX, HIGH_MIN = 165.0, 190.0
CORRECT = [0, 1]
UPPER_INNER, LOWER_INNER, SUBNASALE = 13, 14, 2
# Rigid points: eye corners and the nose bridge/tip move with the skull, not the
# jaw.  Deliberately no chin or lip landmark -- those are the signal.
RIGID = [33, 263, 133, 362, 168, 1, 6]


def _norm_rows(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a, axis=-1)


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray,
                                                              np.ndarray]:
    """Umeyama similarity (scale, rotation, translation) mapping src onto dst."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / src.shape[0]
    U, sv, Vt = np.linalg.svd(C)
    R = U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    var = float((S ** 2).sum() / src.shape[0])
    s = float(sv @ np.array([1.0, np.sign(np.linalg.det(U @ Vt))]) / var) \
        if var > 1e-12 else 1.0
    return s, R, mu_d - s * (R @ mu_s)


def lip_features(lm: np.ndarray) -> dict[str, np.ndarray]:
    """The candidate articulation signals for one track, per video frame."""
    n = lm.shape[0]
    out = {k: np.full(n, np.nan) for k in ("area", "gap", "gap_v", "gap_proc")}
    ok = np.isfinite(lm[:, LEFT_EYE_OUTER, 0]) & np.isfinite(lm[:, RIGID, 0]).all(1)

    # Reference for Procrustes: the median rigid configuration of this track.
    ref = np.nanmedian(lm[ok][:, RIGID, :2], axis=0) if ok.any() else None

    for f in np.flatnonzero(ok):
        p = lm[f]
        eL, eR = p[LEFT_EYE_OUTER, :2], p[RIGHT_EYE_OUTER, :2]
        inter = float(np.linalg.norm(eL - eR))
        vspan = float(np.linalg.norm((eL + eR) / 2.0 - p[SUBNASALE, :2]))
        ring = p[INNER_LIP_RING, :2]
        if inter < 1e-6 or vspan < 1e-6 or not np.isfinite(ring).all():
            continue
        gap = float(np.linalg.norm(p[UPPER_INNER, :2] - p[LOWER_INNER, :2]))
        out["area"][f] = _shoelace(ring.astype(np.float64)) / inter ** 2
        out["gap"][f] = gap / inter
        out["gap_v"][f] = gap / vspan
        if ref is not None:
            s, R, t = similarity_fit(p[RIGID, :2].astype(np.float64), ref)
            q = (s * (R @ p[[UPPER_INNER, LOWER_INNER], :2].T).T + t)
            out["gap_proc"][f] = float(abs(q[1, 1] - q[0, 1]))
    return out


def head_motion(boxes: list, lm: np.ndarray) -> dict[str, np.ndarray]:
    """Rigid-motion proxies, in face-widths so they are scale-free."""
    n = len(boxes)
    cx = np.full(n, np.nan); cy = np.full(n, np.nan); w = np.full(n, np.nan)
    for i, b in enumerate(boxes):
        if b is None:
            continue
        cx[i], cy[i], w[i] = b[0] + b[2] / 2, b[1] + b[3] / 2, b[2]
    yaw = np.full(n, np.nan)
    ok = np.isfinite(lm[:, LEFT_EYE_OUTER, 0])
    for f in np.flatnonzero(ok):
        p = lm[f]
        eL, eR = p[LEFT_EYE_OUTER, :2], p[RIGHT_EYE_OUTER, :2]
        inter = float(np.linalg.norm(eL - eR))
        if inter > 1e-6:
            yaw[f] = float((p[SUBNASALE, 0] - (eL[0] + eR[0]) / 2) / inter)
    d = lambda a: np.abs(np.diff(a, prepend=a[:1]))
    return {"translate": d(cx) / w + d(cy) / w,
            "scale": d(np.log(np.maximum(w, 1e-9))),
            "yaw": d(yaw)}


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

    feats = [lip_features(lm[i]) for i in range(n_tr)]
    hm = [head_motion(tj["tracks"][i]["boxes"], lm[i]) for i in range(n_tr)]

    # ---- A: does head motion explain the shipped lip signal? --------------- #
    print("=== A: head motion vs the shipped lip feature, per track ===")
    print("    corr of |d lip| against each rigid-motion proxy, and R^2 of all "
          "three\n    together.  A high R^2 means the 'lip' signal is largely "
          "the head.")
    for i in range(n_tr):
        dl = np.abs(np.diff(feats[i]["area"], prepend=feats[i]["area"][:1]))
        cols = [hm[i][k] for k in ("translate", "scale", "yaw")]
        ok = np.isfinite(dl) & np.all(np.isfinite(cols), axis=0)
        rs = [float(np.corrcoef(dl[ok], c[ok])[0, 1]) for c in cols]
        X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
        beta, *_ = np.linalg.lstsq(X, dl[ok], rcond=None)
        res = dl[ok] - X @ beta
        r2 = 1.0 - float(res.var() / max(dl[ok].var(), 1e-30))
        who = "MAN" if i == 0 else "WOMAN"
        print(f"  track {i} ({who:>5}): n={int(ok.sum()):4d}  "
              f"translate {rs[0]:+.3f}  scale {rs[1]:+.3f}  yaw {rs[2]:+.3f}"
              f"   R^2 {r2:.3f}")

    # ---- B: whose voice does each head follow? ----------------------------- #
    seg, mdb, t = framify(mix, sr)
    f0, ape = yin(seg, sr, fmax=FMAX_SPEECH)
    okf = (ape < YIN_THRESH) & np.isfinite(f0) & (mdb > np.percentile(mdb, 95) - 25)
    low, high = okf & (f0 <= LOW_MAX), okf & (f0 >= HIGH_MIN)
    vi = np.clip((t * fps).astype(int), 0, lm.shape[1] - 1)

    print("\n=== B: does each track's motion follow the OTHER voice? ===")
    print("    per-track percentile ranks, so each track's marginal mean is 0.5;")
    print("    LOW = the 146 Hz (male) voice, HIGH = the 221 Hz (female) voice")
    print(f"    {int(low.sum())} LOW frames, {int(high.sum())} HIGH frames")
    for sig in ("area", "translate", "yaw"):
        rk = []
        for i in range(n_tr):
            s = (np.abs(np.diff(feats[i]["area"], prepend=feats[i]["area"][:1]))
                 if sig == "area" else hm[i][sig])
            rk.append(_rank(s)[vi])
        m = np.isfinite(rk[0]) & np.isfinite(rk[1])
        d = rk[0] - rk[1]
        ml = float(d[m & low].mean()); mh = float(d[m & high].mean())
        sl = float(d[m & low].std(ddof=1) / np.sqrt(max((m & low).sum(), 1)))
        sh = float(d[m & high].std(ddof=1) / np.sqrt(max((m & high).sum(), 1)))
        sd = float(np.hypot(sl, sh))
        print(f"  {sig:>10}: mean(rank_man - rank_woman)  LOW {ml:+.3f}+-{sl:.3f}"
              f"   HIGH {mh:+.3f}+-{sh:.3f}   DiD {ml - mh:+.3f}+-{sd:.3f}"
              f"  ({abs(ml - mh) / max(sd, 1e-9):.1f}s)")
    print("    DiD > 0 means the man moves relatively more while the LOW voice")
    print("    speaks, i.e. the signal points the right way for [0, 1].")

    # ---- C: candidate features, scored exactly as production scores them --- #
    envs = [_zscore(energy_envelope(s, sr, mc.env_frame_ms, mc.smooth_kernel))
            for s in post]
    n_frames = min(len(e) for e in envs)
    env_fps = 1000.0 / mc.env_frame_ms
    E = np.stack([e[:n_frames] for e in envs])

    def evaluate(sigs: list[np.ndarray],
                 already_deriv: bool = False) -> tuple[list[int], float, float]:
        lips = []
        for i in range(n_tr):
            r = resample_lip(list(sigs[i]), fps, n_frames, env_fps)
            lips.append(_zscore(r if already_deriv
                                else np.abs(np.diff(r, prepend=r[:1]))))
        score = np.zeros((n_tr, post.shape[0]))
        for i in range(n_tr):
            for j in range(post.shape[0]):
                score[i, j] = float(np.dot(lips[i], E[j]) / n_frames)
        a, c = solve_and_conf(score)
        dl, da = lips[0] - lips[1], E[0] - E[1]
        r = float(np.corrcoef(dl, da)[0, 1])
        return a, c, r

    print(f"\n=== C: candidate lip features (correct answer is {CORRECT}) ===")
    print(f"  {'feature':>12} {'t0->stem':>9} {'t1->stem':>9} {'conf':>8} "
          f"{'diff r':>8}  verdict")
    cands: list[tuple[str, list[np.ndarray]]] = [
        (k, [feats[i][k] for i in range(n_tr)])
        for k in ("area", "gap", "gap_v", "gap_proc")]

    # Smoothed before differencing: a 5-frame (200 ms) boxcar over present frames.
    def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
        ker = np.ones(k) / k
        num = np.convolve(np.nan_to_num(x), ker, mode="same")
        cov = np.convolve(np.isfinite(x).astype(float), ker, mode="same")
        return np.where(cov > 0.5, num / np.maximum(cov, 1e-9), np.nan)

    cands.append(("area_smooth", [smooth(feats[i]["area"]) for i in range(n_tr)]))
    cands.append(("gap_v_smooth", [smooth(feats[i]["gap_v"]) for i in range(n_tr)]))

    # Head motion regressed out of the shipped feature's own derivative.  The
    # residual is "more mouth motion than the head alone predicts", already a
    # derivative and signed, so it is fed to `evaluate` without differencing.
    resid = []
    for i in range(n_tr):
        a = feats[i]["area"]
        dl = np.abs(np.diff(a, prepend=a[:1]))
        cols = [hm[i][k] for k in ("translate", "scale", "yaw")]
        ok = np.isfinite(dl) & np.all(np.isfinite(cols), axis=0)
        X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
        beta, *_ = np.linalg.lstsq(X, dl[ok], rcond=None)
        out = np.full(len(a), np.nan)
        out[ok] = dl[ok] - X @ beta
        resid.append(out)

    for name, sigs, deriv in ([(n, s, False) for n, s in cands]
                              + [("area_resid", resid, True)]):
        a, c, r = evaluate(sigs, already_deriv=deriv)
        print(f"  {name:>12} {a[0]:9d} {a[1]:9d} {c:8.4f} {r:+8.4f}  "
              + ("CORRECT" if a == CORRECT else "inverted")
              + ("" if c >= mc.min_confidence else "   (below min_confidence)"))
    print(f"  shipped meta: assignment {meta['assignment']} confidence "
          f"{meta['confidence'][0]:.4f}")


if __name__ == "__main__":
    main()
