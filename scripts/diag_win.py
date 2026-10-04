"""At what time scale is the audio-visual correlation real, if it is real at all?

Where this stands.  The pairing is settled: track 0 is the man, stem 0 carries
the 146 Hz voice (+21.5 dB paired swing, 95% CI [+11.3, +31.0], labelled from the
mixture's own f0 with neither separator nor matcher in the loop), so the shipped
``[1, 0]`` is inverted and ``[0, 1]`` is correct.  Two hypotheses for *why* are
dead:

  * the audio feature -- fifteen variants, seven absolute and eight differential,
    all invert (``diag_feat``).
  * rigid head motion contaminating the lip feature -- R^2 0.022 on both tracks,
    and no motion proxy follows either voice above 1.4 sigma (``diag_rigid``).

What did change the answer was time scale.  Smoothing the lip feature over 200 ms
*before* differencing flips both ``area`` and ``gap_v`` to the correct ``[0, 1]``,
and the aperture feature averaged over 400 ms gave the only significant visual
result this investigation has produced (+0.167 +- 0.079, 2.1 sigma in
``diag_link``).  That is the shape of a real effect buried in landmark noise:
differencing a noisy 20x11 px lip ring at 40 ms amplifies the noise, and the
articulation signal lives at the syllable rate.

But the flipped cells report r = +0.0045 and a margin of 0.011.  Tuning a window
until the answer comes out right, on one clip, with r that small, is exactly the
mistake of endorsing a gate fix that did nothing.  So every cell here is measured
against a null.

  A  **sweep** the integration window, applied identically to the lip activity
     and the audio envelope, and the lip pre-smoothing.  Report the assignment,
     the production confidence, and the differential r.
  B  **circular-shift null.**  Rotate the lip signal by a random lag and recompute
     r.  A real AV correlation dies under a shift; a spurious one does not.  This
     gives an honest two-sided p-value, and the fraction of shifts that land on
     ``[0, 1]`` gives the coin-flip baseline the assignment must beat.
  C  **split-half stability.**  Fit nothing, just ask whether the two halves of
     the clip agree on the assignment.  A window that only works on the whole
     clip is a window that works by luck.

Run::

    PYTHONPATH=. python scripts/diag_win.py runs/862bd92a01ac
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
from app.vision import resample_lip                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_feat import solve_and_conf                       # noqa: E402
from diag_rigid import lip_features                        # noqa: E402

CORRECT = [0, 1]
N_NULL = 4000
WINDOWS = (1, 3, 5, 9, 15, 25, 40)          # env frames; 40 ms each
PRE_SMOOTH = (1, 5)                          # video frames before differencing


def smooth_nan(x: np.ndarray, k: int) -> np.ndarray:
    """Boxcar that skips NaN, preserving NaN where the window is mostly empty."""
    if k <= 1:
        return x
    ker = np.ones(k) / k
    num = np.convolve(np.nan_to_num(x), ker, mode="same")
    cov = np.convolve(np.isfinite(x).astype(float), ker, mode="same")
    return np.where(cov > 0.5, num / np.maximum(cov, 1e-9), np.nan)


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    meta = json.loads((job / "meta.json").read_text())
    tj = json.loads((job / "tracks.json").read_text())
    fps, n_tr = float(tj["fps"]), len(tj["tracks"])
    sr = CONFIG.audio.sample_rate
    mc = CONFIG.match
    rng = np.random.default_rng(0)

    pre = np.asarray(np.load(CACHE_DIR / "diag_pose"
                             / f"{job.name}_stems_premask.npy"), dtype=np.float64)
    post = np.asarray(dsp.wiener_separate(
        pre.astype(np.float32), sample_rate=sr, nfft=CONFIG.audio.nfft,
        hop=CONFIG.audio.hop, exponent=CONFIG.gate.mask_exponent,
        floor=CONFIG.gate.mask_floor,
        cepstral_order=CONFIG.gate.cepstral_smooth_order), dtype=np.float64)
    lm = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_lm.npy")
    feats = [lip_features(lm[i]) for i in range(n_tr)]

    envs = [energy_envelope(s, sr, mc.env_frame_ms, mc.smooth_kernel) for s in post]
    n_frames = min(len(e) for e in envs)
    env_fps = 1000.0 / mc.env_frame_ms
    E0 = np.stack([e[:n_frames] for e in envs])

    def build(kind: str, pre_k: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        """(lip activity, audio activity), both (n, n_frames), z-scored."""
        L = []
        for i in range(n_tr):
            s = smooth_nan(feats[i][kind], pre_k)
            r = resample_lip(list(s), fps, n_frames, env_fps)
            L.append(_zscore(smooth_nan(np.abs(np.diff(r, prepend=r[:1])), W)))
        A = np.stack([_zscore(smooth_nan(E0[j], W)) for j in range(E0.shape[0])])
        return np.stack(L), A

    def stat(L: np.ndarray, A: np.ndarray) -> tuple[list[int], float, float]:
        score = (L @ A.T) / L.shape[1]
        a, c = solve_and_conf(score)
        dl, da = L[0] - L[1], A[0] - A[1]
        return a, c, float(np.corrcoef(dl, da)[0, 1])

    # ---- A: the sweep ------------------------------------------------------- #
    print(f"{job.name}: shipped meta assignment {meta['assignment']} "
          f"confidence {meta['confidence'][0]:.4f}; correct is {CORRECT}")
    print("\n=== A: integration window, applied to lip activity AND envelope ===")
    print(f"  {'feature':>8} {'pre':>4} {'W':>4} {'ms':>6} {'t0':>3} {'t1':>3} "
          f"{'conf':>8} {'diff r':>8}  verdict")
    cells = []
    for kind in ("area", "gap_v"):
        for pk in PRE_SMOOTH:
            for W in WINDOWS:
                L, A = build(kind, pk, W)
                a, c, r = stat(L, A)
                cells.append((kind, pk, W, a, c, r))
                print(f"  {kind:>8} {pk:4d} {W:4d} {W * mc.env_frame_ms:6.0f} "
                      f"{a[0]:3d} {a[1]:3d} {c:8.4f} {r:+8.4f}  "
                      + ("CORRECT" if a == CORRECT else "inverted"))

    # ---- B: circular-shift null -------------------------------------------- #
    print(f"\n=== B: circular-shift null, {N_NULL} shifts per cell ===")
    print("    p is the two-sided share of shifted |r| >= observed |r|;")
    print("    'shifts CORRECT' is the coin-flip baseline the assignment beats")
    print(f"  {'feature':>8} {'pre':>4} {'W':>4} {'diff r':>8} {'p':>7} "
          f"{'shifts CORRECT':>15}")
    # Every cell that landed on the right answer, plus the shipped configuration
    # as a control -- a null that clears the shipped cell too would prove nothing.
    picks = [c for c in cells if c[3] == CORRECT]
    picks += [c for c in cells if c[1] == 1 and c[2] == 1]
    for kind, pk, W, a, c, r in picks:
        L, A = build(kind, pk, W)
        dl, da = L[0] - L[1], A[0] - A[1]
        n = dl.size
        hits = 0
        corr_hits = 0
        for _ in range(N_NULL):
            sh = int(rng.integers(1, n))
            Ls = np.roll(L, sh, axis=1)
            rs = float(np.corrcoef(Ls[0] - Ls[1], da)[0, 1])
            hits += abs(rs) >= abs(r)
            corr_hits += solve_and_conf((Ls @ A.T) / n)[0] == CORRECT
        p = (hits + 1) / (N_NULL + 1)
        print(f"  {kind:>8} {pk:4d} {W:4d} {r:+8.4f} {p:7.3f} "
              f"{100 * corr_hits / N_NULL:14.1f}%"
              + ("   <-- survives" if p < 0.05 else "   <-- chance"))

    # ---- C: split-half ------------------------------------------------------ #
    print("\n=== C: do the two halves of the clip agree? ===")
    print(f"  {'feature':>8} {'pre':>4} {'W':>4} {'first half':>12} "
          f"{'second half':>12}")
    for kind, pk, W, a, c, r in picks:
        L, A = build(kind, pk, W)
        h = L.shape[1] // 2
        out = []
        for sl in (slice(0, h), slice(h, None)):
            Ls = np.stack([_zscore(L[i][sl]) for i in range(n_tr)])
            As = np.stack([_zscore(A[j][sl]) for j in range(A.shape[0])])
            out.append(solve_and_conf((Ls @ As.T) / Ls.shape[1])[0])
        print(f"  {kind:>8} {pk:4d} {W:4d} {str(out[0]):>12} {str(out[1]):>12}"
              + ("   agree" if out[0] == out[1] else "   DISAGREE")
              + ("" if out[0] != CORRECT else " (correct)"))


    # ---- D: is the visual signal dead, or alive but non-discriminative? ----- #
    # This decides what is worth building.  If each face's lip activity tracks
    # the MIXTURE envelope -- "somebody is speaking" -- then MediaPipe is working
    # and the failure is discrimination between two simultaneous talkers, which
    # no correlation feature fixes.  If it does not track even that, the lip
    # feature is noise and a better one has headroom.
    mix = media.read_audio(CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav", sr)
    menv = energy_envelope(mix, sr, mc.env_frame_ms, mc.smooth_kernel)[:n_frames]
    print("\n=== D: does each face track 'somebody is speaking' at all? ===")
    print(f"  {'feature':>8} {'pre':>4} {'W':>4} {'target':>10} "
          f"{'r(t0)':>8} {'p':>7} {'r(t1)':>8} {'p':>7}")
    for kind, pk, W in (("area", 1, 5), ("area", 5, 5), ("gap_v", 5, 5),
                        ("area", 5, 15)):
        L, A = build(kind, pk, W)
        tgts = [("mixture", _zscore(smooth_nan(menv, W)))]
        tgts += [(f"stem {j}", A[j]) for j in range(A.shape[0])]
        for tname, tgt in tgts:
            row = []
            for i in range(n_tr):
                r = float(np.corrcoef(L[i], tgt)[0, 1])
                n = L.shape[1]
                hits = sum(abs(float(np.corrcoef(np.roll(L[i], int(
                    rng.integers(1, n))), tgt)[0, 1])) >= abs(r)
                    for _ in range(N_NULL // 4))
                row.append((r, (hits + 1) / (N_NULL // 4 + 1)))
            print(f"  {kind:>8} {pk:4d} {W:4d} {tname:>10} "
                  f"{row[0][0]:+8.4f} {row[0][1]:7.3f} "
                  f"{row[1][0]:+8.4f} {row[1][1]:7.3f}"
                  + ("   both significant" if max(r for r, _ in row) and
                     all(p < 0.05 for _, p in row) else
                     "   one significant" if any(p < 0.05 for _, p in row)
                     else "   neither"))


if __name__ == "__main__":
    main()
