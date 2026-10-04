"""Are these two stems two speakers, or one near-mixture and one residual?

Eight audio features -- including the differential form that cancels common mode
on the lip side *and* the audio side -- all pair track 0 with stem 1.  A feature
bug would not survive that.  So one of the three premises is wrong:

  P1  track 0 is the man (read off an annotated frame -- my eyes)
  P2  stem 0 is the man, stem 1 the woman (f0 146 vs 200 Hz, F2 1371 vs 1602)
  P3  the shipped pairing is wrong

P2 is the weak one, and the per-chunk energy shares are why: stem 1 holds
87 / 87 / 62 / 81 % of the two stems' joint energy, and reads 242 and 273 Hz in
the first two chunks -- far above the 200 Hz whole-clip median and right where an
octave-doubled 121-137 Hz voice would land.  Both are the signature of a
**two-talker** signal, and this investigation has already established that pitch
and formants measured on a two-talker frame belong to neither speaker.  If stem 1
is a near-mixture then "stem 1 is the woman" was never a measurement, and the
matcher had no identity information to get right.

The test has to label audio frames without the separator and without the
matcher, so it uses the mixture's own octave-safe f0 (7.7% octave-ambiguous at a
300 Hz ceiling, against ~50% at 400) and asks a *paired* question:

  A  **selectivity.**  Within each stem, median level in LOW-voice frames minus
     median level in HIGH-voice frames.  Within-stem contrast is scale-free, so
     it survives SepFormer's unconstrained output magnitude.  The mixture's own
     contrast is the baseline -- one speaker simply being louder must not read as
     selectivity.
  B  **which stem wins each frame.**  Scale-free and paired: if the separation
     worked, LOW frames go to one stem and HIGH frames to the other.  A stem that
     wins both is a near-mixture, whatever its pitch says.  Bootstrapped, because
     the frame counts are in the hundreds.
  C  **the same two on the shipped channels**, which is what the user heard.
  D  **track geometry**, so P1 stops resting on my reading of a PNG: box
     centroids and sizes per track, printed next to the frame I annotated.

Run::

    PYTHONPATH=. python scripts/diag_sel.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media                                      # noqa: E402
from app.config import CACHE_DIR, CONFIG                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin               # noqa: E402

FMAX_SPEECH = 300.0
LOW_MAX, HIGH_MIN = 165.0, 190.0
N_BOOT = 2000


def frame_db(x: np.ndarray, sr: int, n: int) -> np.ndarray:
    _, db, _ = framify(x, sr)
    return db[:n]


def boot_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a bootstrap 95% interval."""
    d = float(a.mean() - b.mean())
    reps = np.empty(N_BOOT)
    for k in range(N_BOOT):
        reps[k] = (a[rng.integers(0, a.size, a.size)].mean()
                   - b[rng.integers(0, b.size, b.size)].mean())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return d, float(lo), float(hi)


def report(tag: str, sigs: list[np.ndarray], low: np.ndarray,
           high: np.ndarray, rng) -> None:
    print(f"\n  --- {tag} ---")
    print(f"  {'':>10} {'LOW dB':>9} {'HIGH dB':>9} {'contrast':>10}")
    for j, s in enumerate(sigs):
        ml, mh = float(np.median(s[low])), float(np.median(s[high]))
        print(f"  {'sig ' + str(j):>10} {ml:9.1f} {mh:9.1f} {ml - mh:+10.2f}")
    if len(sigs) != 2:
        return
    # Paired: which of the two is louder in this frame?  Scale-free apart from a
    # single global offset between the two, which the difference of fractions
    # cannot remove -- so the contrast above and this must agree to conclude.
    d = sigs[0] - sigs[1]
    w_low = (d[low] > 0).astype(float)
    w_high = (d[high] > 0).astype(float)
    diff, lo, hi = boot_ci(w_low, w_high, rng)
    print(f"  sig 0 is the louder one in {w_low.mean() * 100:5.1f}% of LOW frames "
          f"and {w_high.mean() * 100:5.1f}% of HIGH frames")
    print(f"  difference {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]", end="")
    if lo > 0:
        print("   -> sig 0 selectively carries the LOW voice")
    elif hi < 0:
        print("   -> sig 0 selectively carries the HIGH voice")
    else:
        print("   -> NO selectivity: these two do not split the speakers")
    # And the level difference itself, on the same paired frames.
    dl, dlo, dhi = boot_ci(d[low], d[high], rng)
    print(f"  mean(dB0 - dB1): LOW {float(d[low].mean()):+6.2f}  "
          f"HIGH {float(d[high].mean()):+6.2f}  "
          f"difference {dl:+.2f} dB  95% CI [{dlo:+.2f}, {dhi:+.2f}]")


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    meta = json.loads((job / "meta.json").read_text())
    tj = json.loads((job / "tracks.json").read_text())
    sr = CONFIG.audio.sample_rate
    rng = np.random.default_rng(0)

    mix = media.read_audio(CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav", sr)
    pre = np.asarray(np.load(CACHE_DIR / "diag_pose"
                             / f"{job.name}_stems_premask.npy"), dtype=np.float64)
    raw, _ = sf.read(job / "stems_raw.wav", dtype="float64", always_2d=True)
    raw = raw.T

    # ---- the label: mixture pitch only ------------------------------------- #
    seg, mdb, t = framify(mix, sr)
    f0, ape = yin(seg, sr, fmax=FMAX_SPEECH)
    ok = (ape < YIN_THRESH) & np.isfinite(f0) & (mdb > np.percentile(mdb, 95) - 25)
    low = ok & (f0 <= LOW_MAX)
    high = ok & (f0 >= HIGH_MIN)
    n = len(f0)
    print(f"{job.name}: mixture-only frame labels, ceiling {FMAX_SPEECH:g} Hz, "
          f"overlap band {LOW_MAX:g}-{HIGH_MIN:g} Hz excluded")
    print(f"  LOW  {int(low.sum()):4d} frames (median f0 {np.median(f0[low]):.1f} Hz)"
          f"   HIGH {int(high.sum()):4d} frames "
          f"(median f0 {np.median(f0[high]):.1f} Hz)")
    print(f"  mixture's own level: LOW {float(np.median(mdb[low])):+.1f} dB   "
          f"HIGH {float(np.median(mdb[high])):+.1f} dB   contrast "
          f"{float(np.median(mdb[low]) - np.median(mdb[high])):+.2f} dB")
    print("  ^ that contrast is the baseline: one speaker being louder is not "
          "selectivity")

    # ---- A/B: the raw separator output ------------------------------------- #
    report("pre-mask stems, STEM order (sig j = stem j)",
           [frame_db(pre[j], sr, n) for j in range(pre.shape[0])], low, high, rng)

    # ---- C: what the user actually heard ----------------------------------- #
    report(f"stems_raw.wav, CHANNEL order (assignment {meta['assignment']}; "
           f"channel 0 = track 0's button)",
           [frame_db(raw[j], sr, n) for j in range(raw.shape[0])], low, high, rng)

    # ---- D: track geometry, so P1 is not just my eyes ---------------------- #
    print("\n  --- track geometry (boxes are (x, y, w, h), normalised, frame "
          "space) ---")
    for tr in tj["tracks"]:
        v = np.asarray([b for b in tr["boxes"] if b is not None],
                       dtype=np.float64)
        cx, cy = v[:, 0] + v[:, 2] / 2, v[:, 1] + v[:, 3] / 2
        print(f"  track {tr['id']} (label {tr['label']!r}, channel {tr['channel']}): "
              f"present {v.shape[0]}/{len(tr['boxes'])}   "
              f"centroid x {np.median(cx):.3f} y {np.median(cy):.3f}   "
              f"box {np.median(v[:, 2]):.3f}x{np.median(v[:, 3]):.3f} "
              f"(~{np.median(v[:, 2]) * tj['width']:.0f}x"
              f"{np.median(v[:, 3]) * tj['height']:.0f} px)   "
              f"{'screen-LEFT' if np.median(cx) < 0.5 else 'screen-RIGHT'}")
    print("  cross-check against runs/_diag/862bd92a01ac_id0.png")


if __name__ == "__main__":
    main()
