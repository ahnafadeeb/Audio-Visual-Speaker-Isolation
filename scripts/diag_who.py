"""Ground-truth the track<->voice pairing by LOOKING at the video.

Everything else in this diagnosis is a correlation, and the whole finding is that
the correlations on this clip are dominated by a common-mode component.  So the
pairing needs an anchor that is not a correlation at all.

Two independent ones:

  1. **exclusivity** -- for each (track, channel), the fraction of frames where
     the channel is loud but that track's lips are STILL.  The true owner of a
     channel cannot be still while it is loud, so the true owner should score
     LOW.  This is a veto-shaped test, not a correlation, so common-mode lip
     motion inflates both columns equally instead of choosing between them.
  2. **eyes** -- pick moments where exactly one channel is loud and the other is
     digitally silent, crop both faces, and write a contact sheet.  A human (or
     an image-reading model) can then just see whose mouth is open.

Run::

    PYTHONPATH=. python scripts/diag_who.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dsp, media                              # noqa: E402
from app.config import CONFIG                           # noqa: E402

SR = CONFIG.audio.sample_rate
OUT = Path("runs/_diag")


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps = float(tj["fps"])
    tracks = tj["tracks"]
    n_tr = len(tracks)

    raw, sr = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T
    n_ch = raw.shape[0]

    # ---- per-10 ms frame level, per channel ------------------------------- #
    fr = int(sr * 0.01)
    n_fr = raw.shape[1] // fr
    lvl = np.stack([
        10 * np.log10((raw[j, :n_fr * fr].reshape(n_fr, fr) ** 2).mean(1) + 1e-12)
        for j in range(n_ch)])
    ref = np.percentile(lvl, 95, axis=1, keepdims=True)
    rel = lvl - ref                                   # dB relative to own p95

    # ---- visual activity on the same 10 ms grid --------------------------- #
    from app.vision import FaceAnalyzer
    vt, _ = FaceAnalyzer(CONFIG.vision).analyze(str(job / "video.mp4"))
    vis, val = [], []
    for t in vt:
        v, ok = dsp.visual_activity(
            np.asarray(t.lip, dtype=np.float64), src_fps=fps, n_frames=n_fr,
            frame_ms=10.0, smooth_ms=CONFIG.gate.visual_smooth_ms,
            hold_ms=CONFIG.gate.visual_hold_ms,
            min_dynamic_range=CONFIG.gate.visual_min_dynamic_range)
        vis.append(v)
        val.append(ok)
    vis, val = np.stack(vis), np.stack(val)

    # ---- test 1: exclusivity --------------------------------------------- #
    print("=== test 1: exclusivity  P(track still | channel loud) ===")
    print("    the TRUE owner of a channel should be LOW in its column\n")
    LOUD, STILL = -8.0, 0.25
    print("            " + "  ".join(f"ch{j}" for j in range(n_ch)))
    excl = np.zeros((n_tr, n_ch))
    for i in range(n_tr):
        cells = []
        for j in range(n_ch):
            m = (rel[j] > LOUD) & val[i]
            excl[i, j] = float(np.mean(vis[i][m] < STILL)) if m.any() else np.nan
            cells.append(f"{excl[i, j] * 100:5.1f}%" if m.any() else "  n/a ")
        print(f"  track {i}  " + "  ".join(cells)
              + f"   (n loud&valid: "
                f"{[int(((rel[j] > LOUD) & val[i]).sum()) for j in range(n_ch)]})")
    if n_tr == 2 and n_ch == 2 and not np.isnan(excl).any():
        keep = excl[0, 0] + excl[1, 1]
        flip = excl[0, 1] + excl[1, 0]
        print(f"\n  keep(t0->ch0, t1->ch1) total still-while-loud = {keep * 100:5.1f}%")
        print(f"  flip(t0->ch1, t1->ch0) total still-while-loud = {flip * 100:5.1f}%")
        print(f"  -> exclusivity prefers "
              f"{'KEEP [0,1]' if keep < flip else 'FLIP [1,0]'} "
              f"(lower is better), margin {abs(keep - flip) * 100:.1f} points")

    # ---- test 1b: how much of the lip signal is COMMON to both tracks? ----- #
    print("\n=== test 1b: is the visual evidence common-mode? ===")
    both = val.all(axis=0)
    print(f"  frames where every track has a valid visual measurement: "
          f"{both.sum()}/{n_fr} ({both.mean() * 100:.1f}%)")
    if n_tr == 2 and both.sum() > 100:
        a, b = vis[0][both], vis[1][both]
        c = float(np.corrcoef(a, b)[0, 1])
        print(f"  corr(track0 activity, track1 activity) = {c:+.4f}")
        print(f"  mean activity: track0 {a.mean():.3f}  track1 {b.mean():.3f}")
        d = b - a
        print(f"  differential (v1 - v0): mean {d.mean():+.3f}  sd {d.std():.3f}"
              f"  |d|>0.3 in {float(np.mean(np.abs(d) > 0.3)) * 100:.1f}% of frames")
        print("  A high correlation with a small differential spread means the "
              "signal mostly says\n  'somebody is talking', not 'THIS one is'.")

    # ---- test 2: single-channel moments, ranked (no fixed threshold) ------- #
    print("\n=== test 2: best-separated moments, then a face contact sheet ===")
    OUT.mkdir(parents=True, exist_ok=True)
    picks: list[tuple[float, int]] = []
    w = 40                                             # 400 ms
    for j in range(n_ch):
        other = [k for k in range(n_ch) if k != j]
        # Rank by dominance instead of thresholding: the earlier version asked
        # for the other channel to be 22 dB down for 400 ms and no such span
        # exists on this clip, which is itself a finding -- the channels are
        # never cleanly exclusive.
        dom = rel[j] - np.max(np.stack([rel[k] for k in other]), axis=0)
        run = np.convolve(dom, np.ones(w) / w, mode="valid")
        idx = np.argsort(-run)
        sel: list[int] = []
        for c in idx:
            if all(abs(int(c) - s) * 0.01 > 4.0 for s in sel):
                sel.append(int(c))
            if len(sel) == 3:
                break
        for c in sel:
            t = (c + w / 2) * 0.01
            picks.append((t, j))
            print(f"  channel {j}: t={t:6.2f}s  mean dominance over 400 ms "
                  f"{run[c]:+.1f} dB")

    # Also sample the clip evenly, so the sheet identifies both faces even if no
    # moment is acoustically clean.
    for t in np.linspace(1.0, raw.shape[1] / sr - 1.0, 4):
        picks.append((float(t), -1))

    cap = cv2.VideoCapture(str(job / "video.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CELL = 220
    rows = []
    for t, j in picks:
        idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cells = []
        for i, tr in enumerate(tracks):
            b = tr["boxes"][min(idx, len(tr["boxes"]) - 1)]
            if b is None:
                b = next((x for x in reversed(tr["boxes"][:idx]) if x), None)
            if b is None:
                cells.append(np.zeros((CELL, CELL, 3), np.uint8))
                continue
            x, y, w_, h_ = b
            # Pad generously: the mouth is the point, and the detector box is
            # tight on the face.
            cx, cy = (x + w_ / 2) * W, (y + h_ / 2) * H
            half = max(w_ * W, h_ * H) * 0.75
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                crop = np.zeros((CELL, CELL, 3), np.uint8)
            crop = cv2.resize(crop, (CELL, CELL))
            cv2.putText(crop, f"track {i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cells.append(crop)
        strip = np.hstack(cells)
        lab = np.zeros((36, strip.shape[1], 3), np.uint8)
        cap_txt = (f"t={t:.2f}s   ch{j} dominant" if j >= 0
                   else f"t={t:.2f}s   (time sample)")
        cv2.putText(lab, cap_txt, (6, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.vstack([lab, strip]))
    cap.release()
    if rows:
        sheet = np.vstack(rows)
        p = OUT / f"{job.name}_who.png"
        cv2.imwrite(str(p), sheet)
        print(f"\n  wrote {p}  ({sheet.shape[1]}x{sheet.shape[0]})")

    # ---- test 3: lip motion during each channel's dominant windows --------- #
    # The quantitative form of the contact sheet.  Eyeballing three 220 px crops
    # gave contradictory reads, so score every window instead of three frames.
    #
    # Comparability is the whole difficulty: the two tracks have different face
    # sizes and different landmark noise, so their raw motion magnitudes are not
    # on one scale, and `visual_activity`'s p90 normalisation is exactly what
    # test 1b showed to be saturated.  So use each track's own PERCENTILE RANK:
    # "during the windows where channel j dominates, how far up its own motion
    # distribution does track i sit?"  Scale-free, and comparable across tracks.
    print("\n=== test 3: lip motion percentile during each channel's dominant "
          "windows ===")
    mot = []
    for t in vt:
        lip = np.asarray(t.lip, dtype=np.float64)
        v, ok = dsp.resample_hold(lip, fps, n_fr, 100.0)
        d = np.abs(np.diff(v, prepend=v[:1]))
        d[~ok] = np.nan
        mot.append(d)
    mot = np.stack(mot)

    w = 40
    ker = np.ones(w) / w
    n_w = n_fr - w + 1
    # Per-window mean motion, and its rank within that track's own windows.
    wm, wr = [], []
    for i in range(n_tr):
        m = np.convolve(np.nan_to_num(mot[i], nan=0.0), ker, mode="valid")
        cov = np.convolve((~np.isnan(mot[i])).astype(float), ker, mode="valid")
        m = np.where(cov > 0.5, m / np.maximum(cov, 1e-9), np.nan)
        wm.append(m)
        good = ~np.isnan(m)
        r = np.full(n_w, np.nan)
        r[good] = (np.argsort(np.argsort(m[good])) / max(1, good.sum() - 1))
        wr.append(r)
    wm, wr = np.stack(wm), np.stack(wr)

    doms = np.stack([
        np.convolve(rel[j] - np.max(np.stack([rel[k] for k in range(n_ch)
                                              if k != j]), axis=0), ker,
                    mode="valid") for j in range(n_ch)])

    print("     for each channel, the top-decile dominant windows:")
    sc3 = np.zeros((n_tr, n_ch))
    for j in range(n_ch):
        thr = np.nanpercentile(doms[j], 90)
        sel = doms[j] >= thr
        cells = []
        for i in range(n_tr):
            m = sel & ~np.isnan(wr[i])
            sc3[i, j] = float(np.nanmean(wr[i][m])) if m.any() else np.nan
            cells.append(f"track{i} rank {sc3[i, j]:.3f}")
        print(f"     ch{j}: {int(sel.sum())} windows (dominance >= "
              f"{thr:+.1f} dB)   " + "   ".join(cells))
    if n_tr == 2 and n_ch == 2 and not np.isnan(sc3).any():
        keep = sc3[0, 0] + sc3[1, 1]
        flip = sc3[0, 1] + sc3[1, 0]
        print(f"\n  keep(t0->ch0, t1->ch1) rank total = {keep:.3f}")
        print(f"  flip(t0->ch1, t1->ch0) rank total = {flip:.3f}")
        print(f"  -> lip motion prefers "
              f"{'KEEP [0,1]' if keep > flip else 'FLIP [1,0]'} "
              f"(higher is better), margin {abs(keep - flip):.3f}")

    # ---- test 4: octave-robust f0 (harmonic product spectrum) ------------- #
    # Plain autocorrelation can halve or double an octave, and a halving error
    # would invert the male/female call this whole diagnosis turns on.  HPS
    # multiplies energy at f0, 2f0, 3f0...: a subharmonic candidate only lands on
    # every other harmonic, so it loses.
    print("\n=== test 4: f0 by harmonic product spectrum (octave-robust) ===")
    step = int(4.0 * sr)
    print("    t0      " + "   ".join(f"ch{j}" for j in range(n_ch)))
    med: list[list[float]] = [[] for _ in range(n_ch)]
    for st in range(0, raw.shape[1] - step // 2, step):
        cells = []
        for j in range(n_ch):
            f = hps_f0(raw[j, st:st + step], sr)
            cells.append("   n/a  " if f is None else f"{f:6.1f}Hz")
            if f is not None:
                med[j].append(f)
        print(f"   {st / sr:5.1f}   " + "   ".join(cells))
    for j in range(n_ch):
        if med[j]:
            print(f"    ch{j}: median over windows = {np.median(med[j]):6.1f} Hz")

    # ---- test 5: WHY is the visual evidence weak?  Per-track signal quality - #
    # Tests 1/1b/3 all came back weak rather than wrong, which points at the
    # measurement rather than the decision rule.  In the contact sheet the woman
    # (track 1) is in three-quarter/profile view in almost every frame while the
    # man (track 0) is near-frontal -- i.e. exactly the head-rotation case in
    # item 2 of the bug report.  If her lip feature is noise-dominated, that
    # alone explains the mis-assignment, and item 2 is upstream of item 1.
    print("\n=== test 5: per-track lip signal quality ===")
    for i, t in enumerate(vt):
        lip = np.asarray(t.lip, dtype=np.float64)
        nan = float(np.mean(np.isnan(lip)))
        ok = lip[~np.isnan(lip)]
        if ok.size < 10:
            print(f"  track {i}: unusable ({nan * 100:.1f}% NaN)")
            continue
        d = np.abs(np.diff(ok))
        # lag-1 autocorrelation of the derivative: white landmark noise gives
        # ~0 (or negative, since differencing whitens); real mouth movement has
        # structure over several 40 ms frames.
        a1 = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if d.size > 3 else np.nan
        a2 = float(np.corrcoef(d[:-2], d[2:])[0, 1]) if d.size > 4 else np.nan
        p10, p50, p90 = np.percentile(ok, [10, 50, 90])
        print(f"  track {i}: NaN {nan * 100:5.1f}%   feature p10/p50/p90 "
              f"{p10:.4f}/{p50:.4f}/{p90:.4f}  (p90/p50 = {p90 / max(p50, 1e-9):.2f})")
        print(f"            |d| p50 {np.percentile(d, 50):.4f} p90 "
              f"{np.percentile(d, 90):.4f}   autocorr lag1 {a1:+.3f} "
              f"lag2 {a2:+.3f}")

    # Per-window correlation against the TRUE owner channel, now that tests
    # 2-4 plus the user's report have established track0<->ch1, track1<->ch0.
    true_ch = {0: 1, 1: 0}
    print("\n  per-window correlation of each track's lip derivative against "
          "its TRUE channel:")
    from app.matching import energy_envelope
    fr_ms = CONFIG.match.env_frame_ms
    envs = np.stack([energy_envelope(raw[j], sr, fr_ms,
                                     smooth=CONFIG.match.smooth_kernel)
                     for j in range(n_ch)])
    n_env = envs.shape[1]
    wn = max(1, int(4000.0 / fr_ms))
    for i in range(n_tr):
        if i not in true_ch or true_ch[i] >= n_ch:
            continue
        v, okm = dsp.resample_hold(np.asarray(vt[i].lip, dtype=np.float64), fps,
                                   n_env, 1000.0 / fr_ms)
        dd = np.abs(np.diff(v, prepend=v[:1]))
        dd[~okm] = 0.0
        cs = []
        for st in range(0, max(1, n_env - wn + 1), wn):
            sl = slice(st, st + wn)
            x, y = dd[sl], envs[true_ch[i]][sl]
            if x.std() < 1e-9 or y.std() < 1e-9:
                cs.append(np.nan)
                continue
            cs.append(float(np.corrcoef(x, y)[0, 1]))
        arr = np.array(cs)
        print(f"    track {i} vs ch{true_ch[i]}: "
              f"{np.round(arr, 3).tolist()}")
        print(f"       mean {np.nanmean(arr):+.3f}  positive in "
              f"{int(np.nansum(arr > 0))}/{int(np.sum(~np.isnan(arr)))} windows")


def hps_f0(x: np.ndarray, sr: int, *, fmin: float = 70.0, fmax: float = 320.0,
           frame_ms: float = 64.0, hop_ms: float = 32.0,
           n_harm: int = 5) -> float | None:
    """Median HPS f0 over the loud frames of ``x``, or ``None`` if none."""
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if x.size < frame:
        return None
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    seg = x[idx] * np.hanning(frame)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    ref = np.percentile(rms, 95)
    loud = np.flatnonzero(rms > max(ref * 10 ** (-20.0 / 20.0), 1e-6))
    if loud.size == 0:
        return None
    nfft = 1 << int(np.ceil(np.log2(frame * 4)))          # zero-pad for bin res
    mag = np.abs(np.fft.rfft(seg[loud], n=nfft, axis=1))
    df = sr / nfft
    cand = np.arange(int(fmin / df), int(fmax / df) + 1)
    out = []
    for row in mag:
        best_v, best_f = -np.inf, None
        lg = np.log(row + 1e-12)
        for c in cand:
            h = [c * k for k in range(1, n_harm + 1) if c * k < row.size]
            if len(h) < 3:
                continue
            v = float(np.sum(lg[h]))
            if v > best_v:
                best_v, best_f = v, c * df
        if best_f is not None:
            out.append(best_f)
    return float(np.median(out)) if out else None


if __name__ == "__main__":
    main()
