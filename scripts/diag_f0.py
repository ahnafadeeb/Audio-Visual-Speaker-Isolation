"""Octave-safe f0, and with it the answer to who owns which channel.

Three pitch estimators have now been used on this clip and two of them misled me:

  * plain autocorrelation (``diag_swap``) reported ch0 at 250-281 Hz and ch1 at
    105-176 Hz, which I read as "ch0 is the female".  But ch1 is 73.9% exact
    zero, so that was a real voice compared against separator residual.
  * harmonic product spectrum (``diag_who``, ``diag_mix``) reported the man at
    234 Hz and the woman at 390 Hz -- the latter pinned against the 400 Hz search
    ceiling.  HPS is robust against octave *halving* but biased toward octave
    *doubling*, because every harmonic of 2*f0 is also a harmonic of f0 and so
    scores just as well.  Halving both gives ~117 Hz and ~195 Hz.

YIN is the estimator built for this failure.  It takes the *first* dip of the
cumulative-mean-normalised difference function below an absolute threshold rather
than the global optimum, so a period of 2T is only chosen when T does not
explain the signal at all -- the bias runs toward the true, lowest period.  It
also reports aperiodicity, which is a usable "this frame is not voiced" signal
rather than an argmax that always returns something.

With a trustworthy f0 the question is direct.  Track 0 is the man and track 1 the
woman (annotated frames from ``diag_id``), and ``plan_channels`` makes channel i
the stem the matcher gave track i.  So measure, in the windows where exactly one
mouth is moving, the f0 of the mixture and of both channels:

    channel 0 carries the LOW voice  -> shipped pairing CORRECT
    channel 0 carries the HIGH voice -> shipped pairing INVERTED

Run::

    PYTHONPATH=. python scripts/diag_f0.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media                                    # noqa: E402
from app.config import CACHE_DIR, CONFIG                 # noqa: E402
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,  # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace)
from _job import pipeline_audio                          # noqa: E402

SUBNASALE = 2
FRAME_MS, HOP_MS = 64.0, 20.0
FMIN, FMAX = 60.0, 400.0
YIN_THRESH = 0.15


def yin(seg: np.ndarray, sr: int, *, thresh: float = YIN_THRESH,
        fmin: float = FMIN, fmax: float = FMAX
        ) -> tuple[np.ndarray, np.ndarray]:
    """(f0 per frame, aperiodicity per frame).  NaN f0 where unvoiced.

    ``seg`` is (n_frames, frame_len), unwindowed.  Difference function via
    autocorrelation so it is O(n log n), then the cumulative mean normalisation
    and the *first* sub-threshold dip -- that ordering is the octave defence.

    ``fmax`` matters more than it looks.  The default 400 Hz was chosen to give a
    female voice headroom, but it also admits the octave-DOUBLED reading of a
    150-200 Hz voice, and on a two-talker frame no candidate clears ``thresh`` at
    all, so the fallback ``argmin`` runs and lands wherever it likes.  Passing
    ``fmax=300`` excludes the doubled band for ordinary speech.
    """
    n_fr, W = seg.shape
    tau_max = min(W - 1, int(sr / fmin))
    tau_min = max(2, int(sr / fmax))

    x = seg - seg.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * W)))
    F = np.fft.rfft(x, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :tau_max + 1]
    # power of the two W-tau windows, via prefix sums, so d(tau) is exact
    cs = np.concatenate([np.zeros((n_fr, 1)), np.cumsum(x ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    p_head = cs[:, W - taus] - cs[:, 0:1]                  # sum x[0:W-tau]^2
    p_tail = cs[:, W:W + 1] - cs[:, taus]                  # sum x[tau:W]^2
    d = p_head + p_tail - 2 * ac

    # cumulative mean normalised difference; d'(0) := 1 by convention
    csd = np.cumsum(d[:, 1:], axis=1)
    dp = np.ones_like(d)
    dp[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(csd, 1e-12)

    f0 = np.full(n_fr, np.nan)
    ape = np.ones(n_fr)
    for i in range(n_fr):
        v = dp[i]
        pick = -1
        tau = tau_min
        while tau < tau_max:
            if v[tau] < thresh:
                # descend to the local minimum of this dip -- taking the first
                # sub-threshold sample rather than its minimum biases f0 high
                while tau + 1 < tau_max and v[tau + 1] < v[tau]:
                    tau += 1
                pick = tau
                break
            tau += 1
        if pick < 0:
            pick = int(np.argmin(v[tau_min:tau_max])) + tau_min
        # parabolic refinement on the difference function
        if 0 < pick < tau_max:
            a, b, c = v[pick - 1], v[pick], v[pick + 1]
            den = a + c - 2 * b
            shift = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        else:
            shift = 0.0
        per = pick + np.clip(shift, -1, 1)
        ape[i] = float(v[pick])
        if per > 0:
            f0[i] = sr / per
    return f0, ape


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


def framify(x: np.ndarray, sr: int):
    frame = int(sr * FRAME_MS / 1000.0)
    hop = int(sr * HOP_MS / 1000.0)
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    seg = x[idx]
    db = 10 * np.log10((seg ** 2).mean(axis=1) + 1e-12)
    return seg, db, hop * np.arange(n) / sr


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps, n_tr = float(tj["fps"]), len(tj["tracks"])
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        # NOT `job / "video.mp4"` as a fallback: normalize_video strips the audio
        # with -an, so that path fails in ffmpeg with "Output file does not
        # contain any stream".  pipeline_audio() finds a source that has one.
        mixp.parent.mkdir(parents=True, exist_ok=True)
        media.extract_audio(pipeline_audio(job), mixp, sr)
    mix = media.read_audio(mixp, sr)
    raw, _ = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T
    n_ch = raw.shape[0]

    srcs = {"mixture": mix, **{f"channel {j}": raw[j] for j in range(n_ch)}}
    est = {}
    for nm, x in srcs.items():
        seg, db, t = framify(x, sr)
        f0, ape = yin(seg, sr)
        est[nm] = (f0, ape, db, t)
    t = est["mixture"][3]

    print(f"{job.name}: YIN, {FRAME_MS:g} ms / {HOP_MS:g} ms hop, "
          f"threshold {YIN_THRESH}, range {FMIN:g}-{FMAX:g} Hz")

    # ---- sanity: does YIN agree with itself across sources? ---------------- #
    print("\n=== whole-clip f0 over voiced frames (aperiodicity < 0.15) ===")
    for nm, (f0, ape, db, _) in est.items():
        m = (ape < YIN_THRESH) & np.isfinite(f0) & (db > np.percentile(db, 95) - 25)
        if m.sum() < 20:
            print(f"  {nm:>10}: only {int(m.sum())} voiced frames")
            continue
        q = np.percentile(f0[m], [10, 25, 50, 75, 90])
        print(f"  {nm:>10}: voiced {m.mean() * 100:5.1f}%  f0 p10/p25/p50/p75/p90 "
              f"{q[0]:6.1f}/{q[1]:6.1f}/{q[2]:6.1f}/{q[3]:6.1f}/{q[4]:6.1f} Hz")

    # ---- the decisive conditioning ---------------------------------------- #
    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    aps = [aperture(lm[i]) for i in range(n_tr)]
    w_v = max(2, int(round(0.4 * fps)))
    ker = np.ones(w_v) / w_v
    wr = []
    for i in range(n_tr):
        m = np.abs(np.diff(aps[i], prepend=aps[i][:1]))
        num = np.convolve(np.nan_to_num(m), ker, mode="valid")
        cov = np.convolve(np.isfinite(m).astype(float), ker, mode="valid")
        wr.append(_rank(np.where(cov > 0.6, num / np.maximum(cov, 1e-9), np.nan)))
    n_w = min(len(r) for r in wr)
    wi = np.clip((t * fps - w_v / 2).astype(int), 0, n_w - 1)
    a, b = wr[0][:n_w][wi], wr[1][:n_w][wi]
    ok = np.isfinite(a) & np.isfinite(b)

    pops = {
        "MAN alone   (track 0)": ok & (a > .75) & (b < .35),
        "WOMAN alone (track 1)": ok & (b > .75) & (a < .35),
        "both still           ": ok & (a < .3) & (b < .3),
    }
    print("\n=== f0 in the windows where exactly one mouth is moving ===")
    print("    track 0 = MAN, track 1 = WOMAN; channel i = the stem the matcher")
    print("    gave track i.  Read the two 'alone' rows against each other.")
    for pn, pm in pops.items():
        print(f"\n  {pn}:")
        for nm, (f0, ape, db, _) in est.items():
            m = pm & (ape < YIN_THRESH) & np.isfinite(f0)
            if m.sum() < 6:
                print(f"    {nm:>10}: {int(m.sum())} voiced frames (too few)")
                continue
            q = np.percentile(f0[m], [25, 50, 75])
            print(f"    {nm:>10}: n={int(m.sum()):4d}  f0 "
                  f"{q[0]:6.1f}/{q[1]:6.1f}/{q[2]:6.1f} Hz   "
                  f"level median {np.median(db[pm]):+6.1f} dB   "
                  f"voiced {100 * m.sum() / max(pm.sum(), 1):5.1f}%")

    # ---- verdict ---------------------------------------------------------- #
    man = pops["MAN alone   (track 0)"]
    wom = pops["WOMAN alone (track 1)"]
    f0m, apm, _, _ = est["mixture"]
    vm = f0m[man & (apm < YIN_THRESH)]
    vw = f0m[wom & (apm < YIN_THRESH)]
    if vm.size > 5 and vw.size > 5:
        mm, mw = float(np.median(vm)), float(np.median(vw))
        print(f"\n=== verdict ===")
        print(f"  In the MIXTURE -- no separator, no matcher -- the man's windows "
              f"read {mm:.1f} Hz\n  and the woman's read {mw:.1f} Hz.")
        if mw <= mm:
            print("  These do not separate in the expected direction, so f0 cannot"
                  " arbitrate\n  the pairing on this clip.")
            return
        for j in range(n_ch):
            f0c, apc, _, _ = est[f"channel {j}"]
            cm = f0c[man & (apc < YIN_THRESH)]
            cw = f0c[wom & (apc < YIN_THRESH)]
            if cm.size < 6 or cw.size < 6:
                print(f"  channel {j}: too little voiced material to attribute")
                continue
            # Which reference is this channel's own pitch closer to?
            med = float(np.median(np.concatenate([cm, cw])))
            near = "MAN" if abs(med - mm) < abs(med - mw) else "WOMAN"
            print(f"  channel {j}: f0 {med:6.1f} Hz -> closer to the {near}"
                  f"  (|d_man| {abs(med - mm):5.1f}, |d_woman| {abs(med - mw):5.1f})")


if __name__ == "__main__":
    main()
