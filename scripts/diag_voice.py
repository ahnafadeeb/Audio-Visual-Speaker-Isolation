"""Condition on the LIPS, then measure VOICE IDENTITY.  This settles the pairing.

Every previous test was either circular or confounded:

  * ``plan_channels`` builds the export order from the assignment in *track*
    order, so ``stems_raw.wav`` channel *i* is by construction the stem the
    matcher gave track *i*.  Any keep-vs-flip correlation score computed in
    channel space therefore measures *agreement with the shipped assignment* --
    and the matcher chose that assignment to maximise exactly that correlation.
    ``diag_pose``'s and ``diag_swap``'s global "keep wins" results are circular.
  * ``diag_id`` test II is not circular but is baseline-confounded: channel 1 is
    73.9% exact zero, so "ch0 is louder" is the clip's default state, and the
    populations differ in loudness for reasons unrelated to ownership.
  * pitch and formants identify the two VOICES cleanly but say nothing about
    which FACE owns them.

So combine the two halves that are each solid: pick the windows where the visual
evidence is unambiguous (one track's mouth clearly moving, the other's clearly
still), and in exactly those windows ask *whose voice* each channel carries.  A
male speaker's f0 does not become female because the window was chosen badly.

The reference is the input mixture, measured in the same windows: it tells us
what f0 the person whose mouth is moving actually has, with no separator in the
path at all.

Run::

    PYTHONPATH=. python scripts/diag_voice.py runs/862bd92a01ac
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

SUBNASALE = 2
WIN_S = 0.4


def aperture(lm: np.ndarray) -> np.ndarray:
    """Yaw-matched inner-lip aperture per frame; NaN where unmeasurable."""
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


def hps_f0(x: np.ndarray, sr: int, *, fmin: float = 70.0, fmax: float = 330.0,
           frame_ms: float = 64.0, hop_ms: float = 32.0, n_harm: int = 5,
           floor_db: float = -25.0) -> float | None:
    """Median harmonic-product-spectrum f0 over the loud frames, else ``None``.

    HPS rather than autocorrelation because an octave error would invert the
    male/female call this whole question turns on, and a subharmonic candidate
    only lands on every other harmonic so it loses the product.
    """
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if x.size < frame:
        return None
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    seg = x[idx] * np.hanning(frame)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > max(np.percentile(rms, 95)
                                    * 10 ** (floor_db / 20.0), 1e-6))
    if loud.size == 0:
        return None
    nfft = 1 << int(np.ceil(np.log2(frame * 4)))
    mag = np.abs(np.fft.rfft(seg[loud], n=nfft, axis=1))
    df = sr / nfft
    cand = np.arange(int(fmin / df), int(fmax / df) + 1)
    out = []
    for row in mag:
        lg = np.log(row + 1e-12)
        best_v, best_f = -np.inf, None
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


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps = float(tj["fps"])
    n_tr = len(tj["tracks"])
    sr = CONFIG.audio.sample_rate

    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    aps = [aperture(lm[i]) for i in range(n_tr)]

    raw, _ = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T
    n_ch = raw.shape[0]

    # The mixture, with no separator in the path -- the identity reference.
    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)

    # ---- windows where exactly one mouth is clearly active ----------------- #
    w_v = max(2, int(round(WIN_S * fps)))
    ker = np.ones(w_v) / w_v
    wr = []
    for i in range(n_tr):
        m = np.abs(np.diff(aps[i], prepend=aps[i][:1]))
        num = np.convolve(np.nan_to_num(m), ker, mode="valid")
        cov = np.convolve(np.isfinite(m).astype(float), ker, mode="valid")
        wr.append(_rank(np.where(cov > 0.6, num / np.maximum(cov, 1e-9), np.nan)))
    n_w = min(len(r) for r in wr)
    r0, r1 = wr[0][:n_w], wr[1][:n_w]

    print(f"{job.name}: {n_w} windows of {WIN_S * 1000:.0f} ms, "
          f"{n_ch} channels, mixture {mix.size / sr:.2f} s")
    print("\nBy construction (plan_channels), channel i carries the stem the")
    print(f"matcher gave track i.  meta assignment = {json.loads((job / 'meta.json').read_text())['assignment']}"
          " in stem space.")
    print("Track 0 is the MAN and track 1 the WOMAN (confirmed on the annotated")
    print("frames written by diag_id).  So:")
    print("  if channel 0 carries the LOW voice  -> shipped pairing is CORRECT")
    print("  if channel 0 carries the HIGH voice -> shipped pairing is INVERTED")

    for hi, lo in ((0.75, 0.35), (0.85, 0.25)):
        # "track i alone": its own motion in its own top quantile while the
        # other's is in its own bottom quantile.  Per-track quantiles because the
        # two faces have different sizes and different landmark noise.
        sel = [np.isfinite(r0) & np.isfinite(r1) & (a > hi) & (b < lo)
               for a, b in ((r0, r1), (r1, r0))]
        print(f"\n=== active rank > {hi}, other < {lo} ===")
        for i in range(n_tr):
            idx = np.flatnonzero(sel[i])
            if idx.size == 0:
                print(f"  track {i} alone: no windows")
                continue
            # Per-window f0 of each source, over that window's audio only.
            rows: list[list[float | None]] = []
            for c in idx:
                t0 = (c) / fps
                a0 = int(t0 * sr)
                a1 = a0 + int(WIN_S * sr) + int(0.064 * sr)   # >= one HPS frame
                srcs = [mix[a0:a1]] + [raw[j, a0:a1] for j in range(n_ch)]
                rows.append([hps_f0(s, sr) for s in srcs])
            arr = np.array([[np.nan if v is None else v for v in r]
                            for r in rows], dtype=float)
            names = ["mixture"] + [f"channel {j}" for j in range(n_ch)]
            who = "MAN  " if i == 0 else "WOMAN"
            print(f"  track {i} ({who}) moving alone: {idx.size} windows")
            for k, nm in enumerate(names):
                col = arr[:, k]
                good = col[np.isfinite(col)]
                if good.size == 0:
                    print(f"    {nm:>10}: no measurable f0")
                    continue
                # Also report the energy actually present, so a channel that is
                # gated to silence in these windows is not read as an identity.
                e = []
                for c in idx:
                    a0 = int(c / fps * sr)
                    a1 = a0 + int(WIN_S * sr)
                    s = mix[a0:a1] if k == 0 else raw[k - 1, a0:a1]
                    e.append(10 * np.log10(float((s ** 2).mean()) + 1e-12))
                print(f"    {nm:>10}: f0 median {np.median(good):6.1f} Hz  "
                      f"p25 {np.percentile(good, 25):6.1f}  "
                      f"p75 {np.percentile(good, 75):6.1f}  "
                      f"(n={good.size})   level median {np.median(e):+6.1f} dB")

    # ---- sanity: the same measurement over the WHOLE clip ------------------ #
    print("\n=== whole-clip reference (no visual conditioning) ===")
    for nm, s in [("mixture", mix)] + [(f"channel {j}", raw[j])
                                       for j in range(n_ch)]:
        f = hps_f0(s, sr)
        print(f"  {nm:>10}: f0 {'n/a' if f is None else f'{f:6.1f} Hz'}")


if __name__ == "__main__":
    main()
