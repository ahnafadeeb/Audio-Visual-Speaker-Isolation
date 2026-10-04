"""What is actually IN this mixture?  Two assumptions need checking.

``diag_voice.py`` produced two results that break the chain of reasoning built so
far:

  1. the mixture's f0 reads ~195 Hz *whoever* is moving their mouth, and 187.5 Hz
     recurs as the p25 of three separate populations -- the signature of a
     persistent narrowband component, not of two alternating speakers;
  2. in the windows where the woman is the only one visibly moving, BOTH channels
     sit at -44 dB.  She is not audibly speaking there.

Both undermine the male/female call, which I made by comparing ch0's f0 against
ch1's.  Channel 1 is 73.9% exact zero, so that comparison was a real voice
against separator residual -- no identity information at all.

So characterise the mixture directly, with no separator and no matcher:

  A  **how many voices?**  Per-frame f0 over the loud frames.  Two speakers with
     different pitch give a bimodal histogram; one dominant speaker plus a tonal
     bed gives a unimodal one with a spike.
  B  **is there a continuous bed?**  The per-bin *temporal minimum* of the
     magnitude spectrum.  Speech is intermittent, so anything present in every
     single frame survives a minimum; a music or tone bed shows up as peaks that
     the long-term average hides.
  C  **f0 conditioned on the mouths**, including the both-still and both-moving
     populations that ``diag_voice`` did not measure.  If the both-still windows
     still read ~190 Hz, that component is not speech.
  D  **who talks, and how much?**  Voiced-frame count per f0 mode, so "the woman
     barely speaks" becomes a number instead of an impression.

Run::

    PYTHONPATH=. python scripts/diag_mix.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media                                    # noqa: E402
from app.config import CACHE_DIR, CONFIG                 # noqa: E402
from app.vision import (INNER_LIP_RING, LEFT_EYE_OUTER,  # noqa: E402
                        RIGHT_EYE_OUTER, _shoelace)

SUBNASALE = 2
FRAME_MS, HOP_MS = 64.0, 20.0


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


def frames(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(windowed frames, per-frame dB, frame start times)."""
    frame = int(sr * FRAME_MS / 1000.0)
    hop = int(sr * HOP_MS / 1000.0)
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    seg = x[idx]
    db = 10 * np.log10((seg ** 2).mean(axis=1) + 1e-12)
    return seg * np.hanning(frame), db, hop * np.arange(n) / sr


def hps_per_frame(seg: np.ndarray, sr: int, *, fmin: float = 60.0,
                  fmax: float = 400.0, n_harm: int = 5
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(f0 per frame, HPS peak-to-median contrast per frame).

    The contrast is the reliability signal: a frame with no harmonic structure
    still produces an argmax, and reporting it as an f0 is how a tonal bed or a
    burst of noise gets mistaken for a voice.
    """
    nfft = 1 << int(np.ceil(np.log2(seg.shape[1] * 4)))
    mag = np.abs(np.fft.rfft(seg, n=nfft, axis=1))
    lg = np.log(mag + 1e-12)
    df = sr / nfft
    cand = np.arange(max(1, int(fmin / df)), int(fmax / df) + 1)
    prod = np.stack([
        np.sum([lg[:, c * k] for k in range(1, n_harm + 1)
                if c * k < mag.shape[1]], axis=0) for c in cand], axis=1)
    best = np.argmax(prod, axis=1)
    med = np.median(prod, axis=1)
    mad = np.median(np.abs(prod - med[:, None]), axis=1) + 1e-9
    return cand[best] * df, (prod[np.arange(len(best)), best] - med) / mad


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tj = json.loads((job / "tracks.json").read_text())
    fps = float(tj["fps"])
    n_tr = len(tj["tracks"])
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)

    seg, db, t = frames(mix, sr)
    f0, contrast = hps_per_frame(seg, sr)
    ref = np.percentile(db, 95)
    loud = db > ref - 20.0
    print(f"{job.name}: mixture {mix.size / sr:.2f} s, {len(db)} frames of "
          f"{FRAME_MS:g} ms, p95 level {ref:+.1f} dB")
    print(f"  loud frames (> p95-20 dB): {int(loud.sum())}/{len(db)} "
          f"({loud.mean() * 100:.1f}%)")

    # ---- A: how many voices? ---------------------------------------------- #
    print("\n=== A: f0 histogram over loud, harmonically-structured frames ===")
    print("    two speakers with different pitch -> two modes")
    for cmin in (0.0, 3.0, 6.0):
        m = loud & (contrast > cmin)
        if m.sum() < 20:
            print(f"  contrast > {cmin}: only {int(m.sum())} frames")
            continue
        h, edges = np.histogram(f0[m], bins=np.arange(60, 405, 15))
        top = np.argsort(-h)[:4]
        print(f"  contrast > {cmin} ({int(m.sum())} frames): "
              f"median {np.median(f0[m]):6.1f} Hz")
        print("    " + "  ".join(
            f"{edges[i]:.0f}-{edges[i + 1]:.0f}Hz:{h[i]}" for i in sorted(top)))
        bars = "".join("#" if v > h.max() * 0.5 else ("+" if v > h.max() * 0.2
                       else ("." if v > 0 else " ")) for v in h)
        print(f"    60Hz [{bars}] 405Hz")

    # ---- B: is there a continuous bed? ------------------------------------ #
    print("\n=== B: temporal MINIMUM spectrum (what is present in EVERY frame) ===")
    print("    speech is intermittent, so a component surviving a per-bin min")
    print("    over all frames is not speech")
    nfft = 1 << int(np.ceil(np.log2(seg.shape[1] * 4)))
    mag = np.abs(np.fft.rfft(seg, n=nfft, axis=1))
    df = sr / nfft
    mn = 20 * np.log10(np.percentile(mag, 5, axis=0) + 1e-12)
    av = 20 * np.log10(mag.mean(axis=0) + 1e-12)
    band = slice(int(60 / df), int(1200 / df))
    rel = mn[band] - np.median(mn[band])
    pk = np.argsort(-rel)[:8]
    fr = (np.arange(nfft // 2 + 1) * df)[band]
    print("    strongest peaks in the 5th-percentile spectrum (60-1200 Hz):")
    for i in sorted(pk):
        print(f"      {fr[i]:7.1f} Hz   {rel[i]:+5.1f} dB above the p5 median"
              f"   (long-term average here {av[band][i]:+6.1f} dB)")
    print(f"    peak-to-median of the p5 spectrum = {rel.max():.1f} dB"
          + ("   <-- a continuous narrowband bed" if rel.max() > 12
             else "   <-- no strong continuous tone"))

    # ---- C/D: f0 conditioned on the mouths -------------------------------- #
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
    r0, r1 = wr[0][:n_w], wr[1][:n_w]
    # Map every audio frame to its video window.
    wi = np.clip((t * fps - w_v / 2).astype(int), 0, n_w - 1)
    ok = np.isfinite(r0[wi]) & np.isfinite(r1[wi])
    a, b = r0[wi], r1[wi]

    print("\n=== C: mixture f0 conditioned on which mouth is moving ===")
    pops = {
        "man alone   (r0>.75, r1<.35)": ok & (a > .75) & (b < .35),
        "woman alone (r1>.75, r0<.35)": ok & (b > .75) & (a < .35),
        "both moving (both  >.6)     ": ok & (a > .6) & (b > .6),
        "both still  (both  <.3)     ": ok & (a < .3) & (b < .3),
    }
    for nm, m in pops.items():
        for tag, mm in (("all  ", m), ("loud ", m & loud),
                        ("loud+struct", m & loud & (contrast > 6.0))):
            if mm.sum() < 8:
                print(f"  {nm} {tag}: {int(mm.sum())} frames (too few)")
                continue
            q = np.percentile(f0[mm], [25, 50, 75])
            print(f"  {nm} {tag}: n={int(mm.sum()):4d}  f0 "
                  f"{q[0]:6.1f}/{q[1]:6.1f}/{q[2]:6.1f} Hz  "
                  f"level median {np.median(db[mm]):+6.1f} dB")

    print("\n=== D: how much does each person actually speak? ===")
    print("    loud+structured frames attributable to each mouth being the "
          "active one:")
    tot = int((loud & (contrast > 6.0)).sum())
    for nm, m in list(pops.items())[:2]:
        mm = m & loud & (contrast > 6.0)
        print(f"  {nm}: {int(mm.sum()):4d} of {tot} "
              f"({100 * mm.sum() / max(tot, 1):5.1f}%)")


if __name__ == "__main__":
    main()
