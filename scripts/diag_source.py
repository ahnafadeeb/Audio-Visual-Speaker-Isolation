"""Is the collapse the separator's, or our mask's?  And are the labels real?

``diag_collapse.py`` found that channel 0 -- the man's channel -- has *negative*
selectivity for its owner (-2.7 dB): it is louder during the woman's speech than
during his.  Channel 1 sits 32.8 dB lower and is effectively empty.  That is item
4 of the bug report, and it explains every symptom the user described.

Two things must be settled before any code changes.

**1. Are the f0 labels real?**  YIN has already misled me twice on this clip, and
the entire finding rests on labelling frames MAN/WOMAN from the mixture's pitch.
Two checks that do not use pitch:

  * **run-length structure.**  Conversation is turn-taking, so a correct label is
    temporally clustered.  Octave errors are frame-local, so a corrupted label is
    interleaved at the frame scale.
  * **spectral clustering.**  Two-means on the band-energy shape of the voiced
    frames -- which knows nothing about f0 -- should recover the same partition.

**2. Where does the collapse happen?**  ``stems_raw`` is
``wiener_separate(separator output)``, so the mask could be the culprit rather
than the model.  This re-runs the real separator and measures selectivity BEFORE
and AFTER the mask.  Which of the two moves the number decides whether the fix
belongs in ``separation.py`` or in ``dsp.py`` -- and guessing wrong here is
exactly the failure mode recorded in the gate-attribution lesson.

Run::

    PYTHONPATH=. python scripts/diag_source.py runs/862bd92a01ac
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dsp, media, separation                   # noqa: E402
from app.config import CACHE_DIR, CONFIG                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_f0 import YIN_THRESH, framify, yin             # noqa: E402

F_MAN, F_WOM = 234.7, 393.5


def labels(mix: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg, mdb, t = framify(mix, sr)
    f0, ape = yin(seg, sr)
    voiced = (ape < YIN_THRESH) & np.isfinite(f0)
    split = float(np.sqrt(F_MAN * F_WOM))
    lab = np.zeros(len(f0), dtype=np.int8)
    lab[voiced & (f0 < split)] = 1
    lab[voiced & (f0 >= split)] = 2
    return lab, seg, mdb


def runs(lab: np.ndarray) -> list[tuple[int, int]]:
    """(label, length) run-length encoding, ignoring unvoiced gaps <= 5 frames."""
    keep = np.flatnonzero(lab > 0)
    if keep.size == 0:
        return []
    out: list[tuple[int, int]] = []
    cur, n = int(lab[keep[0]]), 1
    for a, b in zip(keep, keep[1:]):
        if b - a <= 5 and lab[b] == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = int(lab[b]), 1
    out.append((cur, n))
    return out


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    meta = json.loads((job / "meta.json").read_text())
    sr = CONFIG.audio.sample_rate

    mixp = CACHE_DIR / "diag_pose" / f"{job.name}_mix.wav"
    if not mixp.exists():
        src = job / "input.mp4" if (job / "input.mp4").exists() else job / "video.mp4"
        media.extract_audio(src, mixp, sr)
    mix = media.read_audio(mixp, sr)
    lab, seg, mdb = labels(mix, sr)

    # ---- 1a: run-length structure ----------------------------------------- #
    print("=== label check 1: is the label temporally clustered? ===")
    rl = runs(lab)
    lens = {k: [n for c, n in rl if c == k] for k in (1, 2)}
    hop_ms = 20.0
    for k, nm in ((1, "MAN  "), (2, "WOMAN")):
        L = lens[k]
        if not L:
            print(f"  {nm}: no runs")
            continue
        print(f"  {nm}: {len(L)} runs, median {np.median(L):.0f} frames "
              f"({np.median(L) * hop_ms:.0f} ms), max {max(L)} "
              f"({max(L) * hop_ms:.0f} ms)")
    solo = sum(1 for _, n in rl if n == 1)
    print(f"  single-frame runs: {solo}/{len(rl)} ({100 * solo / max(len(rl), 1):.1f}%)"
          + ("   <-- frame-scale interleaving: labels are unreliable"
             if solo > 0.5 * len(rl) else
             "   <-- turn-taking structure: labels are trustworthy"))

    # ---- 1b: spectral clustering, which never sees f0 --------------------- #
    print("\n=== label check 2: does a pitch-blind spectral clustering agree? ===")
    v = np.flatnonzero(lab > 0)
    nfft = 1 << int(np.ceil(np.log2(seg.shape[1])))
    mag = np.abs(np.fft.rfft(seg[v] * np.hanning(seg.shape[1]), n=nfft, axis=1))
    edges = np.geomspace(120, sr / 2 - 200, 21)
    bins = (np.arange(nfft // 2 + 1) * sr / nfft)
    feat = np.stack([np.log(mag[:, (bins >= a) & (bins < b)].mean(axis=1) + 1e-12)
                     for a, b in zip(edges, edges[1:])], axis=1)
    feat -= feat.mean(axis=1, keepdims=True)              # remove level
    feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
    # Deterministic 2-means: seed on the two most distant frames.
    g = feat @ feat.T
    i0, i1 = np.unravel_index(np.argmin(g), g.shape)
    c = np.stack([feat[i0], feat[i1]])
    for _ in range(40):
        a = np.argmax(feat @ c.T, axis=1)
        for k in (0, 1):
            if (a == k).any():
                c[k] = feat[a == k].mean(axis=0)
                c[k] /= np.linalg.norm(c[k]) + 1e-12
    truth = lab[v]
    acc = max(float(np.mean((a == 0) == (truth == 1))),
              float(np.mean((a == 0) == (truth == 2))))
    print(f"  2-means on 20-band log spectra of {v.size} voiced frames agrees "
          f"with the f0 label {acc * 100:.1f}% of the time"
          + ("   <-- two genuinely different voices"
             if acc > 0.7 else "   <-- the partition is not reproducible"))

    # ---- 2: pre-mask vs post-mask selectivity ----------------------------- #
    print("\n=== where does the collapse happen: model, or our mask? ===")
    cache = CACHE_DIR / "diag_pose" / f"{job.name}_stems_premask.npy"
    if cache.exists():
        stems = np.load(cache)
        print(f"  [cache] pre-mask stems {stems.shape}")
    else:
        # Same construction the pipeline uses, so the stems are the shipped ones:
        # resolve_chunk_s is what actually ran on this 4 GB card.
        chunk_s = CONFIG.audio.resolve_chunk_s("cuda")
        print(f"  separating with chunk_s={chunk_s:.1f}s "
              f"overlap_s={CONFIG.audio.overlap_s:.1f}s ...")
        sep = separation.build_separator(
            CONFIG.runtime.separator, device="cuda",
            cache_dir=str(CACHE_DIR / "models"),
            chunk_s=chunk_s, overlap_s=CONFIG.audio.overlap_s)
        stems = sep.separate(mix, sr)
        sep.release()
        np.save(cache, stems)
        print(f"  separated: {stems.shape}")

    masked = dsp.wiener_separate(
        stems, sample_rate=sr, nfft=CONFIG.audio.nfft, hop=CONFIG.audio.hop,
        exponent=CONFIG.gate.mask_exponent, floor=CONFIG.gate.mask_floor,
        cepstral_order=CONFIG.gate.cepstral_smooth_order)

    n = min(len(lab), *(framify(s, sr)[1].size for s in stems))

    def report(tag: str, arr: np.ndarray) -> None:
        print(f"\n  {tag}  (STEM order, before plan_channels)")
        for j in range(arr.shape[0]):
            d = framify(arr[j], sr)[1][:n]
            a = float(np.median(d[lab[:n] == 1]))
            b = float(np.median(d[lab[:n] == 2]))
            print(f"    stem {j}: MAN {a:+7.1f} dB   WOMAN {b:+7.1f} dB   "
                  f"MAN-minus-WOMAN {a - b:+6.1f} dB")
        # A correct 2-source separation has the two stems preferring opposite
        # speakers, so the two differences must have opposite signs.
        ds = []
        for j in range(arr.shape[0]):
            d = framify(arr[j], sr)[1][:n]
            ds.append(float(np.median(d[lab[:n] == 1]))
                      - float(np.median(d[lab[:n] == 2])))
        if len(ds) == 2:
            ok = (ds[0] > 0) != (ds[1] > 0)
            print(f"    -> stems prefer {'OPPOSITE' if ok else 'THE SAME'} speaker"
                  f"   spread {abs(ds[0] - ds[1]):.1f} dB"
                  + ("" if ok else "   <-- COLLAPSE"))

    report("pre-mask (raw separator output)", stems)
    report("post-mask (what stems_raw holds)", masked)
    print(f"\n  meta assignment {meta['assignment']} (stem index per track), so "
          f"channel 0 = stem {meta['assignment'][0]}")


if __name__ == "__main__":
    main()
