"""Diagnose the -12 dB leakage knee in the silence chain.

Measured previously: ch0's exact-zero fraction during a B-only region is 100%
at -25 and -18 dB input leakage, but collapses to 67.9% at -12 dB.  This script
finds out *why* by instrumenting what the Schmitt gate actually sees, rather
than reasoning about it.

Run::

    python scripts/gating_knee.py

Pure numpy/scipy -- no torch, no model, no video.
"""

from __future__ import annotations

import numpy as np

from app import dsp
from app.config import CONFIG

SR = CONFIG.audio.sample_rate
DUR = 12.0
# A speaks, B speaks, A speaks.  The middle span is the one that must be
# digitally silent on channel 0.
A_TURNS = [(0.0, 4.0), (8.0, 12.0)]
B_TURNS = [(4.2, 7.8)]
B_ONLY = (4.5, 7.5)          # measured strictly inside B's turn


def syllables(t: np.ndarray, turns, rate: float, seed: int) -> np.ndarray:
    """Amplitude envelope with syllable-rate modulation and inter-word gaps."""
    r = np.random.default_rng(seed)
    env = np.zeros_like(t)
    for s, e in turns:
        m = (t >= s) & (t < e)
        local = t[m] - s
        # syllabic AM plus a slow prosodic contour; squared to deepen the gaps
        syl = (0.5 + 0.5 * np.sin(2 * np.pi * rate * local + r.uniform(0, 6.28))) ** 2
        pros = 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * local + r.uniform(0, 6.28))
        env[m] = syl * pros
    return env


def voice(turns, f0: float, rate: float, seed: int) -> np.ndarray:
    """Crude but adequate voiced-speech surrogate: harmonics + shaped noise."""
    r = np.random.default_rng(seed)
    t = np.arange(int(DUR * SR)) / SR
    sig = np.zeros_like(t)
    for k in range(1, 12):                       # harmonic stack, -6 dB/octave
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6.28))
    sig += 0.05 * r.normal(size=t.size)          # fricative-ish noise floor
    env = syllables(t, turns, rate, seed)
    out = (sig * env).astype(np.float32)
    return out / (np.abs(out).max() + 1e-12)


def frame_db(x: np.ndarray, frame_ms: float = 10.0) -> np.ndarray:
    frame = int(round(SR * frame_ms / 1000.0))
    n_frames = int(np.ceil(x.size / frame))
    padded = np.pad(x, (0, n_frames * frame - x.size))
    energy = (padded.reshape(n_frames, frame) ** 2).mean(axis=1)
    return 10.0 * np.log10(energy + 1e-12)


def lip_signal(turns, fps: float, rate: float, seed: int,
               dropout: tuple[float, float] | None = None) -> list[float]:
    """A lip-aperture surrogate for a speaker who talks during ``turns``.

    Mouth motion is syllabic like the audio but deliberately NOT a copy of it:
    a constant resting aperture plus per-frame jitter outside the turns (a
    closed mouth still moves), and a slow drift so the signal is not a clean
    on/off that would flatter the gate.  ``dropout`` inserts NaN -- the "face
    not visible" state -- to check the abstain path.
    """
    r = np.random.default_rng(1000 + seed)
    t = np.arange(int(DUR * fps)) / fps
    rest = 0.020                                    # mouth closed, not zero area
    lip = rest + 0.002 * r.normal(size=t.size)      # tracking jitter
    lip += 0.001 * np.sin(2 * np.pi * 0.15 * t)     # slow head/pose drift
    for s, e in turns:
        m = (t >= s) & (t < e)
        local = t[m] - s
        syl = (0.5 + 0.5 * np.sin(2 * np.pi * rate * local + r.uniform(0, 6.28))) ** 2
        lip[m] += 0.055 * syl                       # jaw opens on voiced syllables
    if dropout:
        lip[(t >= dropout[0]) & (t < dropout[1])] = np.nan
    return lip.tolist()


def measure(raw, g, lips, fps, fs, fe) -> tuple[float, float, float]:
    """Return (% exact zeros on ch0 during B-only, % of A's speech kept, veto)."""
    demo = dsp.apply_gate(raw, sample_rate=SR, cfg=g, lips=lips, video_fps=fps)
    seg = demo[0][fs:fe]
    zeros = 100.0 * np.count_nonzero(seg == 0.0) / seg.size

    # Retention: the gate must not buy silence by cutting the target.  Measure
    # over A's own turns, which is where a too-aggressive veto would show.
    keep_mask = np.zeros(demo.shape[1], dtype=bool)
    for s, e in A_TURNS:
        keep_mask[int(s * SR):min(int(e * SR), demo.shape[1])] = True
    kept = 100.0 * np.count_nonzero(demo[0][keep_mask] != 0.0) / max(keep_mask.sum(), 1)

    acoustic = dsp.apply_gate(raw, sample_rate=SR, cfg=g)
    veto = dsp.veto_cost(acoustic[0], demo[0], raw[1],
                         sample_rate=SR, frame_ms=g.frame_ms)
    return zeros, kept, veto


def main() -> None:
    import dataclasses

    A = voice(A_TURNS, f0=115.0, rate=4.5, seed=1)
    B = voice(B_TURNS, f0=190.0, rate=5.2, seed=2)

    g = CONFIG.gate
    fps = CONFIG.vision.target_fps
    fs, fe = int(B_ONLY[0] * SR), int(B_ONLY[1] * SR)
    fr = int(round(SR * g.frame_ms / 1000.0))
    qs, qe = fs // fr, fe // fr

    lips = [lip_signal(A_TURNS, fps, 4.5, 1), lip_signal(B_TURNS, fps, 5.2, 2)]
    acoustic = dataclasses.replace(g, visual_fusion=False)

    print(f"gate: open={g.open_db} close={g.close_db} ref_pct={g.ref_percentile} "
          f"min_on={g.min_on_ms}ms veto={g.visual_veto_db}dB")
    print("\n  zeros% = exact digital silence on ch0 during B's turn (want 100)")
    print("  kept%  = ch0 samples surviving during A's OWN turns (want ~100)")
    print(f"\n{'leak dB':>8}{'resid dB':>10}{'rel max':>9}{'>open?':>8}"
          f"{'A:zeros%':>10}{'A:kept%':>9}{'AV:zeros%':>11}{'AV:kept%':>10}{'veto':>8}")
    print("-" * 83)

    for leak_db in (-30, -25, -20, -18, -15, -12, -9, -6, -3):
        a = 10 ** (leak_db / 20.0)
        est0 = (A + a * B).astype(np.float32)      # channel 0: A, contaminated
        est1 = (B + a * A).astype(np.float32)

        raw = dsp.wiener_separate(
            np.stack([est0, est1]), sample_rate=SR,
            nfft=CONFIG.audio.nfft, hop=CONFIG.audio.hop,
            exponent=g.mask_exponent, floor=g.mask_floor,
            cepstral_order=g.cepstral_smooth_order)

        z_a, k_a, _ = measure(raw, acoustic, None, fps, fs, fe)
        z_v, k_v, veto = measure(raw, g, lips, fps, fs, fe)

        # what the gate saw on channel 0
        ldb = frame_db(raw[0], g.frame_ms)
        ref = float(np.percentile(ldb, g.ref_percentile))
        band = (ldb - ref)[qs:qe]
        resid = 10 * np.log10(np.mean(raw[0][fs:fe] ** 2) + 1e-20)

        print(f"{leak_db:>8}{resid:>10.1f}{band.max():>9.1f}"
              f"{'YES' if band.max() > g.open_db else 'no':>8}"
              f"{z_a:>10.1f}{k_a:>9.1f}{z_v:>11.1f}{k_v:>10.1f}{veto:>8.3f}")

    # --- the abstain path -------------------------------------------------- #
    # A face that vanishes must NOT be treated as a face that is silent.
    print("\nfailure modes at -9 dB leakage (the knee):")
    a = 10 ** (-9 / 20.0)
    raw = dsp.wiener_separate(
        np.stack([(A + a * B).astype(np.float32), (B + a * A).astype(np.float32)]),
        sample_rate=SR, nfft=CONFIG.audio.nfft, hop=CONFIG.audio.hop,
        exponent=g.mask_exponent, floor=g.mask_floor,
        cepstral_order=g.cepstral_smooth_order)

    cases = {
        "vision ok": lips,
        "A's face gone 1-3s": [lip_signal(A_TURNS, fps, 4.5, 1, dropout=(1.0, 3.0)),
                               lips[1]],
        "A's face never found": [None, lips[1]],
        "A's track frozen": [[0.02] * int(DUR * fps), lips[1]],
        "lips swapped (mismatch)": [lips[1], lips[0]],
    }
    print(f"{'case':>26}{'zeros%':>9}{'kept%':>8}{'veto':>8}{'alarm?':>8}")
    print("-" * 59)
    for name, lp in cases.items():
        z, k, veto = measure(raw, g, lp, fps, fs, fe)
        alarm = "FIRES" if veto > g.visual_disagree_alarm else "-"
        print(f"{name:>26}{z:>9.1f}{k:>8.1f}{veto:>8.3f}{alarm:>8}")

    print("\nInterpretation: 'rel max' is the residual's level relative to channel 0's")
    print("OWN 95th-percentile frame energy -- the quantity the Schmitt trigger")
    print("compares against open_db. Once it crosses, the acoustic gate opens on")
    print("the interferer and A:zeros% collapses. AV:zeros% is what vision buys,")
    print("and AV:kept% is the price -- it must stay high or we have traded one")
    print("failure for the other. 'veto' is the mis-assignment detector: it must")
    print("stay near 0 on every correctly-paired row and fire only on the swap.")


if __name__ == "__main__":
    main()
