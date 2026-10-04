"""Is the separator collapsing, and is our INPUT SCALE why?

``diag_source.py`` showed the collapse is present in the raw separator output, so
``dsp.wiener_separate`` is exonerated.  But two numbers in that same printout
undercut how it was measured:

  * the MAN label's longest run is 180 ms and its median 40 ms.  That is not
    turn-taking, it is frame-scale alternation -- the signature of two people
    talking *over each other*, which a TV debate show is full of.
  * if both speakers are active in nearly every frame, then "median level in MAN
    frames vs WOMAN frames" is flat for a CORRECT separation too, because both
    populations contain both voices.  The label only says who dominates the
    pitch.  So the -0.3 dB / -1.7 dB contrast is not yet evidence of collapse.

This measures collapse a way that overlap cannot fake, and then tests the most
likely wrapper defect.

  A  **levels.**  The raw stems came out at peak ~20 while the mixture is inside
     +-1.  SI-SNR is scale-invariant, so the model is under no obligation to
     match scale and a large factor is not by itself a fault -- but a factor of
     ~250 *between the two stems* needs to be on the record.
  B  **reconstruction.**  Least-squares fit of the mixture onto the two stems.
     If one stem explains the mixture nearly as well alone as the pair does, the
     other is a numerical residual rather than a source.
  C  **f0 purity -- the overlap-proof collapse test.**  Run YIN on each stem
     separately and ask what fraction of its OWN voiced frames fall below the
     male/female split.  A correct separation gives one stem ~all-low and the
     other ~all-high, whatever the levels and however much the two overlap in
     time.  A collapse gives both stems the same mixed fraction.  Scale-free,
     alignment-free, and it does not care that both people talk at once.
  D  **input-gain sweep.**  SepFormer's masking net is full of normalisation
     layers, so it is NOT scale-equivariant: feed it at a level far from what it
     trained on and it can degenerate.  We hand it the raw ffmpeg output with no
     normalisation at all (``separation.py`` line 182: ``np.asarray(mixture,
     dtype=np.float32)``).  Sweep the input gain over 60 dB on one 10 s excerpt
     and watch C.  If purity moves with gain, the fix is one line in the wrapper
     and has nothing to do with permutations or lip tracking.
  E  **single-speaker control.**  Feed an excerpt where one person clearly
     dominates.  A healthy model puts the voice on one stem and near-silence on
     the other.  If it duplicates the same voice onto both at comparable level,
     the model is failing on this material regardless of scale.

Run::

    PYTHONPATH=. python scripts/diag_scale.py runs/862bd92a01ac
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media, separation                          # noqa: E402
from app.config import CACHE_DIR, CONFIG                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin               # noqa: E402

F_MAN, F_WOM = 234.7, 393.5
SPLIT = float(np.sqrt(F_MAN * F_WOM))
EXCERPT_S = 10.0


def profile(x: np.ndarray, sr: int) -> dict:
    """Level and f0 composition of one signal, entirely on its own terms."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    out = {"peak_db": 20 * np.log10(peak + 1e-20),
           "rms_db": 20 * np.log10(rms + 1e-20),
           "n": 0, "p_man": float("nan"), "med": float("nan"), "voiced": 0.0}
    if x.size < int(0.1 * sr):
        return out
    seg, db, _ = framify(x, sr)
    f0, ape = yin(seg, sr)
    # Voiced AND within 25 dB of this signal's own p95, so a residual channel is
    # judged on its loudest material rather than on its noise floor.
    ref = float(np.percentile(db, 95))
    v = (ape < YIN_THRESH) & np.isfinite(f0) & (db > ref - 25.0)
    out["voiced"] = float(v.mean())
    if v.sum() >= 10:
        out["n"] = int(v.sum())
        out["p_man"] = float(np.mean(f0[v] < SPLIT))
        out["med"] = float(np.median(f0[v]))
    return out


def purity(stems: np.ndarray, sr: int) -> tuple[float, list[dict]]:
    """|p_man(stem0) - p_man(stem1)|, and the per-stem profiles behind it.

    1.0 means the two stems disagree completely about who they contain, i.e.
    perfect pitch-wise separation.  0.0 means they contain the same mix.
    """
    ps = [profile(stems[j], sr) for j in range(stems.shape[0])]
    if any(not np.isfinite(p["p_man"]) for p in ps) or len(ps) != 2:
        return float("nan"), ps
    return abs(ps[0]["p_man"] - ps[1]["p_man"]), ps


def _label(mix: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    seg, db, t = framify(mix, sr)
    f0, ape = yin(seg, sr)
    v = (ape < YIN_THRESH) & np.isfinite(f0)
    lab = np.zeros(len(f0), dtype=np.int8)
    lab[v & (f0 < SPLIT)] = 1
    lab[v & (f0 >= SPLIT)] = 2
    return lab, t


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)
    stems = np.load(CACHE_DIR / "diag_pose" / f"{job.name}_stems_premask.npy")
    stems = np.asarray(stems, dtype=np.float64)

    # ---- A: levels --------------------------------------------------------- #
    print(f"{job.name}: mixture {mix.size / sr:.2f} s at {sr} Hz")
    print("\n=== A: levels (dBFS) ===")
    pm = profile(mix, sr)
    print(f"  mixture : peak {pm['peak_db']:+7.1f}   rms {pm['rms_db']:+7.1f}   "
          f"dc {float(mix.mean()):+.5f}   clipped samples "
          f"{int(np.sum(np.abs(mix) >= 0.999))}")
    for j in range(stems.shape[0]):
        p = profile(stems[j], sr)
        print(f"  stem {j}  : peak {p['peak_db']:+7.1f}   rms {p['rms_db']:+7.1f}"
              f"   gain vs mixture {p['rms_db'] - pm['rms_db']:+7.1f} dB")
    d = (profile(stems[0], sr)["rms_db"] - profile(stems[1], sr)["rms_db"])
    print(f"  inter-stem level difference: {d:+.1f} dB")

    # ---- B: what explains the mixture? ------------------------------------- #
    print("\n=== B: least-squares reconstruction of the mixture ===")
    n = min(mix.size, stems.shape[1])
    M, S = mix[:n], stems[:, :n]
    e_mix = float(np.sum(M ** 2))
    for tag, rows in (("stem 0 alone", [0]), ("stem 1 alone", [1]),
                      ("both stems  ", [0, 1])):
        A = S[rows].T
        coef, *_ = np.linalg.lstsq(A, M, rcond=None)
        res = M - A @ coef
        ev = 1.0 - float(np.sum(res ** 2)) / e_mix
        print(f"  {tag}: explains {ev * 100:6.2f}% of mixture energy   "
              f"coeffs {np.array2string(coef, precision=4)}")
    print("    a stem that adds nothing over the other one is a residual, "
          "not a source")

    # ---- C: f0 purity, whole clip ------------------------------------------ #
    print("\n=== C: f0 purity -- immune to how much the two speakers overlap ===")
    print(f"    split at {SPLIT:.1f} Hz (man ref {F_MAN}, woman ref {F_WOM})")
    pmix = profile(mix, sr)
    print(f"  mixture: {pmix['n']:4d} voiced frames, {pmix['p_man'] * 100:5.1f}% "
          f"below split, median f0 {pmix['med']:6.1f} Hz")
    sc, ps = purity(stems, sr)
    for j, p in enumerate(ps):
        print(f"  stem {j} : {p['n']:4d} voiced frames "
              f"({p['voiced'] * 100:5.1f}% of frames), "
              f"{p['p_man'] * 100:5.1f}% below split, median f0 "
              f"{p['med']:6.1f} Hz")
    print(f"  -> purity |dp| = {sc:.3f}"
          + ("   <-- the two stems contain the SAME mix: COLLAPSE"
             if sc < 0.25 else
             "   <-- the stems do separate by voice" if sc > 0.5 else
             "   <-- partial separation"))

    # ---- pick the excerpts for D and E ------------------------------------- #
    lab, t = _label(mix, sr)
    w = int(EXCERPT_S * sr)
    hop_fr = int(round(EXCERPT_S / (t[1] - t[0]))) if t.size > 1 else 1
    best, best_k = None, -1
    for s in range(0, max(1, mix.size - w), int(sr)):          # 1 s grid
        f0i = int(s / sr / (t[1] - t[0]))
        seg_lab = lab[f0i:f0i + hop_fr]
        k = min(int(np.sum(seg_lab == 1)), int(np.sum(seg_lab == 2)))
        if k > best_k:
            best, best_k = s, k
    both = mix[best:best + w]
    print(f"\n  contested excerpt: {best / sr:.1f}-{(best + w) / sr:.1f} s "
          f"(min per-speaker voiced frames {best_k})")

    # A stretch where one speaker dominates the voiced frames.
    solo, solo_who, solo_frac = None, 0, 0.0
    for s in range(0, max(1, mix.size - w), int(sr)):
        f0i = int(s / sr / (t[1] - t[0]))
        seg_lab = lab[f0i:f0i + hop_fr]
        tot = int(np.sum(seg_lab > 0))
        if tot < 40:
            continue
        for k in (1, 2):
            fr = float(np.sum(seg_lab == k)) / tot
            if fr > solo_frac:
                solo, solo_who, solo_frac = s, k, fr
    print(f"  dominated excerpt: {solo / sr:.1f}-{(solo + w) / sr:.1f} s, "
          f"{'MAN' if solo_who == 1 else 'WOMAN'} holds "
          f"{solo_frac * 100:.0f}% of its voiced frames")

    # ---- D: input-gain sweep ----------------------------------------------- #
    print("\n=== D: does separation quality depend on INPUT SCALE? ===")
    print("    we feed the raw ffmpeg output unnormalised; SepFormer's masking")
    print("    net has normalisation layers, so scale-equivariance is not free")
    sep = separation.build_separator(
        CONFIG.runtime.separator, device="cuda",
        cache_dir=str(CACHE_DIR / "models"),
        chunk_s=EXCERPT_S + 1.0, overlap_s=CONFIG.audio.overlap_s)
    pe = profile(both, sr)
    print(f"    excerpt as-is: peak {pe['peak_db']:+.1f} dBFS, "
          f"{pe['p_man'] * 100:.1f}% of its voiced frames below split")
    print(f"  {'in gain':>9} {'in peak':>9} {'out rms 0':>10} {'out rms 1':>10} "
          f"{'p_man 0':>8} {'p_man 1':>8} {'purity':>7}")
    for g in (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0):
        est = np.asarray(sep.separate(both * g, sr), dtype=np.float64)
        sc_g, ps_g = purity(est, sr)
        print(f"  {g:9.4f} {20 * np.log10(np.max(np.abs(both * g)) + 1e-20):+9.1f} "
              f"{ps_g[0]['rms_db']:+10.1f} {ps_g[1]['rms_db']:+10.1f} "
              f"{ps_g[0]['p_man'] * 100:8.1f} {ps_g[1]['p_man'] * 100:8.1f} "
              f"{sc_g:7.3f}")

    # ---- E: single-speaker control ------------------------------------------ #
    print("\n=== E: control -- one dominant speaker in, what comes out? ===")
    print("    a healthy model emits the voice on one stem and near-silence on")
    print("    the other; duplicating it at equal level is a failure mode")
    est = np.asarray(sep.separate(mix[solo:solo + w], sr), dtype=np.float64)
    sc_s, ps_s = purity(est, sr)
    for j, p in enumerate(ps_s):
        print(f"  stem {j}: rms {p['rms_db']:+7.1f} dBFS   "
              f"{p['p_man'] * 100:5.1f}% below split   median f0 {p['med']:6.1f} Hz"
              f"   voiced {p['voiced'] * 100:5.1f}%")
    print(f"  inter-stem level difference "
          f"{ps_s[0]['rms_db'] - ps_s[1]['rms_db']:+.1f} dB   purity {sc_s:.3f}")
    sep.release()


if __name__ == "__main__":
    main()
