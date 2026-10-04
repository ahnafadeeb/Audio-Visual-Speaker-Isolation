"""Does head yaw corrupt the lip feature, and does a yaw-invariant one fix it?

``diag_who.py`` established that on the real clip the matcher's evidence is a
coin flip *even against the correct pairing* (mean correlation +0.033 / +0.042,
positive in 4/8 and 5/8 windows), and that track 1's lip derivative has lag-1
autocorrelation +0.044 -- white noise, no speech-rate structure.  Track 1 is the
woman, who is in three-quarter view for nearly the whole clip.

There is a derivable mechanism.  The shipped feature is

    lip = inner_lip_area / interocular**2

Under a yaw of theta, the horizontal extent of anything on the face projects
with cos(theta) while vertical extent is unchanged.  So

    area          ~ cos(theta)        (one horizontal dimension, one vertical)
    interocular   ~ cos(theta)        (purely horizontal)
    interocular^2 ~ cos^2(theta)
    => lip        ~ 1 / cos(theta)

The feature is *amplified* by head rotation -- 2x at 60 degrees -- with no change
in actual mouth aperture.  Head turning therefore injects large multiplicative
noise into exactly the signal the matcher and the gate depend on.  This is the
same class of defect as the crop-normalised coordinate bug: a ruler that changes
with something other than what is being measured.

Candidate replacements, chosen so the yaw dependence cancels:

  A  area / interocular**2                 the shipped feature      ~ 1/cos
  B  gap(13,14) / vspan                    both vertical            ~ 1
  C  area / (interocular * vspan)          one h, one v, matched    ~ 1

``vspan`` is eye-midpoint to subnasale (landmark 2): vertical, rigid, and
crucially independent of the jaw -- using the chin would partly cancel the very
mouth opening being measured.

Run::

    PYTHONPATH=. python scripts/diag_pose.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dsp                                     # noqa: E402
from app.config import CACHE_DIR, CONFIG                # noqa: E402
from app.matching import energy_envelope                # noqa: E402
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,  # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace)

CACHE = CACHE_DIR / "diag_pose"
SUBNASALE = 2
NOSE_TIP = 1
N_LM = 478


def collect(job: Path, tracks, fps: float) -> np.ndarray:
    """(n_tracks, n_frames, N_LM, 3) landmarks in FRAME pixels, NaN where absent.

    Replicates ``FaceAnalyzer._lip_feature``'s crop and mapping exactly -- same
    0.15 pad, same 24 px floor, same ORIGINAL-crop-dimension mapping -- so any
    difference measured here is the feature formula, not the plumbing.
    """
    path = CACHE / f"{job.name}_lm.npy"
    if path.exists():
        a = np.load(path)
        print(f"  [cache] {path.name}: {a.shape}")
        return a

    import mediapipe as mp
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(str(job / "video.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_fr = min(len(t["boxes"]) for t in tracks)
    out = np.full((len(tracks), n_fr, N_LM, 3), np.nan, dtype=np.float32)

    for f in range(n_fr):
        ok, frame = cap.read()
        if not ok:
            break
        for i, t in enumerate(tracks):
            b = t["boxes"][f]
            if b is None:
                continue
            x, y, w, h = b
            pad = 0.15
            x0 = int(max(0, (x - w * pad) * W))
            y0 = int(max(0, (y - h * pad) * H))
            x1 = int(min(W, (x + w * (1 + pad)) * W))
            y1 = int(min(H, (y + h * (1 + pad)) * H))
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            crop_w, crop_h = x1 - x0, y1 - y0
            res = mesh.process(cv2.cvtColor(frame[y0:y1, x0:x1],
                                            cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            n = min(N_LM, len(lm))
            out[i, f, :n, 0] = [x0 + lm[k].x * crop_w for k in range(n)]
            out[i, f, :n, 1] = [y0 + lm[k].y * crop_h for k in range(n)]
            # z is in the same normalised units as x; scale it the same way so
            # the triple is a consistent (if weakly calibrated) 3-D point.
            out[i, f, :n, 2] = [lm[k].z * crop_w for k in range(n)]
        if f % 200 == 0:
            print(f"    frame {f}/{n_fr}")
    cap.release()
    mesh.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def features(lm: np.ndarray) -> dict[str, np.ndarray]:
    """lm: (n_frames, N_LM, 3) -> named per-frame features, NaN where invalid."""
    n = lm.shape[0]
    out = {k: np.full(n, np.nan) for k in ("A", "B", "C", "yaw", "yaw_z")}
    for f in range(n):
        p = lm[f]
        if not np.isfinite(p[LEFT_EYE_OUTER, 0]):
            continue
        eL, eR = p[LEFT_EYE_OUTER, :2], p[RIGHT_EYE_OUTER, :2]
        inter = float(np.linalg.norm(eL - eR))
        if inter < 1e-6:
            continue
        emid = (eL + eR) / 2.0
        vspan = float(np.linalg.norm(emid - p[SUBNASALE, :2]))
        ring = p[INNER_LIP_RING, :2]
        if not np.isfinite(ring).all() or vspan < 1e-6:
            continue
        area = _shoelace(ring.astype(np.float64))
        gap = float(np.linalg.norm(p[13, :2] - p[14, :2]))

        out["A"][f] = area / inter ** 2
        out["B"][f] = gap / vspan
        out["C"][f] = area / (inter * vspan)
        # Yaw proxy 1: nose-tip offset from the eye midline, in interocular
        # units.  Grows monotonically with |yaw| and needs no z.
        out["yaw"][f] = float(np.dot(p[NOSE_TIP, :2] - emid, (eR - eL) / inter)) / inter
        # Yaw proxy 2: the depth difference between the eye corners.  Signed,
        # and a real angle if MediaPipe's z is to be trusted at all.
        dz = float(p[RIGHT_EYE_OUTER, 2] - p[LEFT_EYE_OUTER, 2])
        dx = float(eR[0] - eL[0])
        out["yaw_z"][f] = np.degrees(np.arctan2(dz, abs(dx) + 1e-9))
    return out


def _dstats(x: np.ndarray) -> tuple[float, float, float]:
    """(lag1, lag2 autocorr of |diff|, p90/p50 of the feature)."""
    ok = x[np.isfinite(x)]
    if ok.size < 20:
        return np.nan, np.nan, np.nan
    d = np.abs(np.diff(ok))
    a1 = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if d.size > 3 else np.nan
    a2 = float(np.corrcoef(d[:-2], d[2:])[0, 1]) if d.size > 4 else np.nan
    p50, p90 = np.percentile(ok, [50, 90])
    return a1, a2, float(p90 / max(p50, 1e-12))


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps, tracks = float(tj["fps"]), tj["tracks"]
    n_tr = len(tracks)
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"{job.name}: collecting landmarks for {n_tr} tracks")
    lm = collect(job, tracks, fps)
    feats = [features(lm[i]) for i in range(n_tr)]

    print("\n=== yaw: is the premise of item 2 true on this clip? ===")
    for i in range(n_tr):
        y = np.abs(feats[i]["yaw"])
        yz = np.abs(feats[i]["yaw_z"])
        y, yz = y[np.isfinite(y)], yz[np.isfinite(yz)]
        if y.size == 0:
            print(f"  track {i}: no valid frames")
            continue
        print(f"  track {i}: |nose offset| p50 {np.percentile(y, 50):.3f} "
              f"p90 {np.percentile(y, 90):.3f}   "
              f"|yaw from z| p50 {np.percentile(yz, 50):5.1f} deg "
              f"p90 {np.percentile(yz, 90):5.1f} deg   "
              f"frames>45deg {float(np.mean(yz > 45)) * 100:5.1f}%")

    print("\n=== does the shipped feature track yaw instead of the mouth? ===")
    print("    corr(feature, |yaw proxy|) -- should be ~0 for a clean feature")
    for i in range(n_tr):
        y = np.abs(feats[i]["yaw"])
        for k in ("A", "B", "C"):
            x = feats[i][k]
            m = np.isfinite(x) & np.isfinite(y)
            c = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 20 else np.nan
            print(f"  track {i} feature {k}: corr with |yaw| = {c:+.4f}"
                  + ("   <-- shipped" if k == "A" else ""))

    print("\n=== signal quality per feature ===")
    print("    lag1/lag2 autocorr of |derivative| (white noise ~ 0; real mouth "
          "motion\n    at 25 fps must persist 3-6 frames), and p90/p50 dynamic "
          "range")
    for i in range(n_tr):
        for k in ("A", "B", "C"):
            a1, a2, dr = _dstats(feats[i][k])
            print(f"  track {i} feature {k}: lag1 {a1:+.3f}  lag2 {a2:+.3f}  "
                  f"p90/p50 {dr:5.2f}" + ("   <-- shipped" if k == "A" else ""))

    # ---- the payoff: does a yaw-invariant feature fix the ASSIGNMENT? ------ #
    raw, sr = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T
    n_ch = raw.shape[0]
    fr_ms = CONFIG.match.env_frame_ms
    envs = np.stack([energy_envelope(raw[j], sr, fr_ms,
                                     smooth=CONFIG.match.smooth_kernel)
                     for j in range(n_ch)])
    n_env = envs.shape[1]

    def _z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)

    # Established by diag_who tests 2-4 plus the user's report: track 0 is the
    # man and owns ch1; track 1 is the woman and owns ch0.
    TRUTH = [1, 0]
    print(f"\n=== does the feature choice fix the assignment?  truth = {TRUTH} ===")
    for k in ("A", "B", "C"):
        lipm = []
        for i in range(n_tr):
            v, ok = dsp.resample_hold(feats[i][k], fps, n_env, 1000.0 / fr_ms)
            d = np.abs(np.diff(v, prepend=v[:1]))
            d[~ok] = 0.0
            lipm.append(d)
        sc = np.array([[float(np.dot(_z(lipm[i]), _z(envs[j]))) / n_env
                        for j in range(n_ch)] for i in range(n_tr)])
        keep = sc[0, 0] + sc[1, 1]
        flip = sc[0, 1] + sc[1, 0]
        pick = [0, 1] if keep >= flip else [1, 0]
        # Per-window sign consistency: how often the window-local answer agrees
        # with the truth.  A rule that is right on average but 50/50 per window
        # is a coin flip that happened to land well.
        wn = max(1, int(4000.0 / fr_ms))
        agree = tot = 0
        for st in range(0, max(1, n_env - wn + 1), wn):
            sl = slice(st, st + wn)
            w = np.array([[float(np.dot(_z(lipm[i][sl]), _z(envs[j][sl]))) / wn
                           for j in range(n_ch)] for i in range(n_tr)])
            wk, wf = w[0, 0] + w[1, 1], w[0, 1] + w[1, 0]
            agree += ([0, 1] if wk >= wf else [1, 0]) == TRUTH
            tot += 1
        print(f"  feature {k}: score {np.round(sc, 5).tolist()}  keep {keep:+.5f} "
              f"flip {flip:+.5f}")
        print(f"             -> picks {pick} "
              f"{'CORRECT' if pick == TRUTH else 'WRONG  '}  "
              f"margin {abs(keep - flip):.5f}   "
              f"per-window agreement {agree}/{tot}"
              + ("   <-- shipped" if k == "A" else ""))


if __name__ == "__main__":
    main()
