"""Who is track 0?  The one fact this whole diagnosis rests on.

The chain built so far is:

  * pitch says ch0 is the HIGH voice (autocorrelation 183-281 Hz, HPS 199 Hz) and
    ch1 the LOW voice (105-176 Hz, HPS 168 Hz) -- two methods, same direction;
  * I read a 220 px contact sheet as "track 0 = the man";
  * therefore the true pairing is track0 -> ch1, track1 -> ch0, i.e. FLIP.

But ``diag_pose.py`` then found that *every* lip feature -- the shipped one and
both yaw-invariant replacements -- prefers KEEP, with ``score[0][0] = +0.075``
the largest element in the matrix.  Either the correlation is systematically
wrong, or the contact-sheet read was, and the second link in that chain is the
only one I established by eye.  So establish it properly.

Three tests, none of which is the disputed lip-vs-envelope correlation:

  I   **annotated frames.**  Full frames with each track's box drawn and
      labelled, plus large face crops, at times chosen for LOW yaw so the faces
      are actually readable.  Removes the ambiguity of a small crop.
  II  **visually decisive windows.**  Instead of conditioning on acoustic
      dominance and asking about the lips (test 3 of ``diag_who``), condition on
      the LIPS -- take only the windows where one track is clearly moving and the
      other is clearly still -- and ask which channel is loud.  Conditioning on
      the confident side of the evidence is the whole point.
  III **formants.**  Vocal-tract length, not f0, so it is independent of the
      pitch measurement rather than a restatement of it.  A shorter tract raises
      every formant, so F1/F2 separate the speakers even where f0 is ambiguous.

Run::

    PYTHONPATH=. python scripts/diag_id.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, CONFIG                 # noqa: E402
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,  # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace)

OUT = Path("runs/_diag")
SUBNASALE = 2
NOSE_TIP = 1


def per_frame(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(aperture, |yaw proxy|) per frame for one track; NaN where unmeasurable.

    ``aperture`` is feature C from ``diag_pose`` -- inner-lip area over
    (interocular * vertical span) -- which is the yaw-matched form.
    """
    n = lm.shape[0]
    ap = np.full(n, np.nan)
    yaw = np.full(n, np.nan)
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
        if vspan < 1e-6 or not np.isfinite(ring).all():
            continue
        ap[f] = _shoelace(ring.astype(np.float64)) / (inter * vspan)
        yaw[f] = abs(float(np.dot(p[NOSE_TIP, :2] - emid,
                                 (eR - eL) / inter)) / inter)
    return ap, yaw


def _rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank within the finite entries of ``x``; NaN elsewhere.

    Each track has its own face size and its own landmark noise, so raw
    magnitudes are not on a common scale and only ranks can be compared.
    """
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum() > 1:
        out[ok] = np.argsort(np.argsort(x[ok])) / (ok.sum() - 1)
    return out


def formants(x: np.ndarray, sr: int, *, n_form: int = 2, order: int = 16,
             frame_ms: float = 32.0, hop_ms: float = 16.0) -> list[float]:
    """Median F1..Fn over the loud frames, by LPC root-solving.

    Independent of f0: this measures vocal-tract resonances, so it cannot
    inherit an octave error from the pitch tracker.
    """
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if x.size < frame:
        return []
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    seg = x[idx]
    rms = np.sqrt((seg ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > max(np.percentile(rms, 95) * 10 ** (-15 / 20), 1e-6))
    if loud.size == 0:
        return []
    win = np.hamming(frame)
    got: list[list[float]] = [[] for _ in range(n_form)]
    for i in loud:
        s = seg[i] * win
        s = np.append(s[0], s[1:] - 0.97 * s[:-1])        # pre-emphasis
        r = np.correlate(s, s, mode="full")[frame - 1:frame + order]
        if r[0] <= 0:
            continue
        # Levinson-Durbin
        a = np.zeros(order + 1)
        a[0], e = 1.0, r[0]
        for k in range(1, order + 1):
            acc = r[k] + float(np.dot(a[1:k], r[k - 1:0:-1])) if k > 1 else r[1]
            lam = -acc / e
            a[1:k + 1] = a[1:k + 1] + lam * a[k - 1::-1][:k]
            e *= 1 - lam * lam
            if e <= 0:
                break
        if e <= 0:
            continue
        rt = np.roots(a)
        rt = rt[np.imag(rt) > 0]
        if rt.size == 0:
            continue
        f = np.sort(np.abs(np.angle(rt)) * sr / (2 * np.pi))
        bw = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(rt))
        f = np.sort([fv for fv, b in zip(f, bw[np.argsort(
            np.abs(np.angle(rt)) * sr / (2 * np.pi))]) if 90 < fv < 4000 and b < 500])
        for k in range(min(n_form, len(f))):
            got[k].append(float(f[k]))
    return [float(np.median(g)) if g else float("nan") for g in got]


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps, tracks = float(tj["fps"]), tj["tracks"]
    n_tr = len(tracks)
    OUT.mkdir(parents=True, exist_ok=True)

    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    aps, yaws = zip(*(per_frame(lm[i]) for i in range(n_tr)))
    n_fr = lm.shape[1]

    raw, sr = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T
    n_ch = raw.shape[0]

    # 10 ms channel level relative to each channel's own p95, as elsewhere.
    fr = int(sr * 0.01)
    n_af = raw.shape[1] // fr
    lvl = np.stack([10 * np.log10(
        (raw[j, :n_af * fr].reshape(n_af, fr) ** 2).mean(1) + 1e-12)
        for j in range(n_ch)])
    rel = lvl - np.percentile(lvl, 95, axis=1, keepdims=True)

    # ---- I: annotated frames at low yaw ----------------------------------- #
    print("=== test I: annotated frames (both tracks boxed and labelled) ===")
    both = np.isfinite(yaws[0]) & np.isfinite(yaws[1])
    worst = np.where(both, np.maximum(yaws[0], yaws[1]), np.inf)
    # Four readable frames, spread across the clip: within each quarter take the
    # frame whose WORST face is most frontal.
    picks = []
    for q in range(4):
        a, b = q * n_fr // 4, (q + 1) * n_fr // 4
        seg = worst[a:b]
        if np.isfinite(seg).any():
            picks.append(a + int(np.nanargmin(seg)))
    cap = cv2.VideoCapture(str(job / "video.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    COL = [(80, 220, 80), (80, 80, 240)]
    CELL = 420
    rows = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        full = frame.copy()
        cells = []
        for i in range(n_tr):
            b = tracks[i]["boxes"][min(f, len(tracks[i]["boxes"]) - 1)]
            if b is None:
                cells.append(np.zeros((CELL, CELL, 3), np.uint8))
                continue
            x, y, w_, h_ = b
            cv2.rectangle(full, (int(x * W), int(y * H)),
                          (int((x + w_) * W), int((y + h_) * H)), COL[i], 3)
            cv2.putText(full, f"track {i}", (int(x * W), max(18, int(y * H) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL[i], 3)
            cx, cy = (x + w_ / 2) * W, (y + h_ / 2) * H
            half = max(w_ * W, h_ * H) * 0.7
            crop = frame[int(max(0, cy - half)):int(min(H, cy + half)),
                         int(max(0, cx - half)):int(min(W, cx + half))]
            crop = (cv2.resize(crop, (CELL, CELL)) if crop.size
                    else np.zeros((CELL, CELL, 3), np.uint8))
            cv2.rectangle(crop, (0, 0), (CELL - 1, CELL - 1), COL[i], 6)
            cv2.putText(crop, f"track {i}  yaw {yaws[i][f]:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL[i], 2)
            cells.append(crop)
        strip = np.hstack(cells)
        full = cv2.resize(full, (strip.shape[1],
                                 int(H * strip.shape[1] / W)))
        lab = np.zeros((34, strip.shape[1], 3), np.uint8)
        cv2.putText(lab, f"t={f / fps:.2f}s  frame {f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.vstack([lab, full, strip]))
    cap.release()
    for k, r in enumerate(rows):
        p = OUT / f"{job.name}_id{k}.png"
        cv2.imwrite(str(p), r)
        print(f"  wrote {p}  ({r.shape[1]}x{r.shape[0]})")

    # ---- II: visually decisive windows ------------------------------------ #
    print("\n=== test II: condition on the LIPS, then ask which channel is loud ===")
    print("    windows where one track clearly moves and the other is clearly "
          "still.\n    If track i owns channel j, then track i moving alone must "
          "mean ch j is loud.")
    w_v = max(2, int(round(0.4 * fps)))                   # 400 ms in video frames
    mot = [np.abs(np.diff(a, prepend=a[:1])) for a in aps]
    # Per-window mean motion, NaN-aware, then rank within the track.
    ker = np.ones(w_v) / w_v
    wm = []
    for i in range(n_tr):
        num = np.convolve(np.nan_to_num(mot[i]), ker, mode="valid")
        cov = np.convolve(np.isfinite(mot[i]).astype(float), ker, mode="valid")
        wm.append(np.where(cov > 0.6, num / np.maximum(cov, 1e-9), np.nan))
    wr = [_rank(m) for m in wm]
    n_w = min(len(r) for r in wr)
    d = wr[0][:n_w] - wr[1][:n_w]                         # >0: track0 moves more

    # Audio dominance over the same windows.
    dom = rel[0] - rel[1] if n_ch == 2 else None
    a_w = max(2, int(round(0.4 / 0.01)))
    da = np.convolve(dom, np.ones(a_w) / a_w, mode="valid")
    t_w = (np.arange(n_w) + w_v / 2) / fps
    ia = np.clip((t_w * 100 - a_w / 2).astype(int), 0, len(da) - 1)
    da_w = da[ia]

    for thr in (0.5, 0.7, 0.85):
        m0 = np.isfinite(d) & (d > thr)                   # track0 moves, t1 still
        m1 = np.isfinite(d) & (d < -thr)
        if m0.sum() < 5 or m1.sum() < 5:
            print(f"  |rank gap| > {thr}: too few windows "
                  f"({int(m0.sum())}/{int(m1.sum())})")
            continue
        # Mean ch0-minus-ch1 dominance in each population.  If track0 owns ch0
        # (KEEP) the first is positive and the second negative; if track0 owns
        # ch1 (FLIP) the signs reverse.
        v0, v1 = float(np.mean(da_w[m0])), float(np.mean(da_w[m1]))
        p0 = float(np.mean(da_w[m0] > 0)) * 100
        p1 = float(np.mean(da_w[m1] < 0)) * 100
        print(f"  |rank gap| > {thr}:")
        print(f"    track0 moving alone ({int(m0.sum())} win): "
              f"mean (ch0-ch1) = {v0:+6.2f} dB, ch0 louder in {p0:5.1f}%")
        print(f"    track1 moving alone ({int(m1.sum())} win): "
              f"mean (ch0-ch1) = {v1:+6.2f} dB, ch1 louder in {p1:5.1f}%")
        print(f"    -> prefers {'KEEP [0,1]' if v0 > v1 else 'FLIP [1,0]'}"
              f"   separation {abs(v0 - v1):.2f} dB")

    # ---- III: formants ---------------------------------------------------- #
    print("\n=== test III: formants (vocal-tract length, independent of f0) ===")
    print("    a shorter tract raises every formant; typical F1/F2 for /a/-ish\n"
          "    speech run ~700/1200 Hz male vs ~850/1500 Hz female")
    for j in range(n_ch):
        f = formants(raw[j], sr)
        print(f"  ch{j}: F1 {f[0]:7.1f} Hz   F2 {f[1]:7.1f} Hz" if len(f) > 1
              else f"  ch{j}: insufficient voiced material")
    step = int(8.0 * sr)
    print("    per 8 s window:")
    for st in range(0, raw.shape[1] - step // 2, step):
        cells = []
        for j in range(n_ch):
            f = formants(raw[j, st:st + step], sr)
            cells.append(f"ch{j} {f[0]:6.0f}/{f[1]:6.0f}" if len(f) > 1
                         else f"ch{j}    n/a    ")
        print(f"    {st / sr:5.1f}s   " + "   ".join(cells))


if __name__ == "__main__":
    main()
