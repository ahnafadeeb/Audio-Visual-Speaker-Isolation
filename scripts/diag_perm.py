"""Is the chunk stitcher actually getting the permutation right?

The user reports a mid-clip speaker swap at ~10 s on a real 2-speaker clip
(male + female), with the *other* button going completely silent until the
system "re-aligned" ~10 s later.  ``chunk_s = 10`` / ``overlap_s = 2`` puts a
boundary every 8 s, so the timing fits.  But ``meta.json`` for that very run
reports ``flipped: 0`` over 3 boundaries and ``held_boundaries: []`` -- i.e. the
stitcher believes it did nothing and was confident about it.  One of those two
stories is wrong.

GROUND TRUTH, two independent ways -- because the whole question is whether
overlap correlation is trustworthy, so a ground truth built out of overlap
correlation would beg it:

  * **pitch** (primary).  The clip is a male/female pair, so median f0 over
    voiced frames labels each raw row absolutely: no reference, no stitching, no
    chaining between chunks.  This is also exactly the axis item 4 of the bug
    report is about, so it does double duty.
  * **reference** (cross-check).  A second separation at the largest overlap the
    chunker allows (``overlap_s = 5`` -- ``_separate_once`` clamps overlap to
    ``chunk // 2``, which is what invalidated the first version of this script:
    it correlated an 8 s tail across a 5 s stride).  Ground truth for production
    chunk *c* = correlate its raw rows against that reference over the same span.

Agreement between the two is what makes either believable.

Then it scores three candidate decision rules per boundary:

  1. ``continuity`` -- overlap correlation.  What ships today.
  2. ``signature``  -- running energy-weighted log-band spectrum per identity.
     A cheap stand-in for a speaker embedding: no new dependency.
  3. ``anchor``     -- correlation of each row's energy envelope against each
     face track's lip motion over the whole chunk.  Independent modality, and
     unlike (1) it does not care whether the *overlap* held speech.

Run::

    PYTHONPATH=. python scripts/diag_perm.py runs/862bd92a01ac
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import media                                   # noqa: E402
from app.config import CACHE_DIR, CONFIG                # noqa: E402

SR = CONFIG.audio.sample_rate
CACHE = CACHE_DIR / "diag_perm"

CHUNK_S = 10.0
PROD_OVERLAP_S = 2.0
REF_OVERLAP_S = 5.0          # == CHUNK_S / 2, the most the chunker permits


def clamped_overlap(chunk_s: float, overlap_s: float) -> int:
    chunk = int(chunk_s * SR)
    return min(int(overlap_s * SR), chunk // 2)


def chunk_bounds(n: int, chunk_s: float, overlap_s: float) -> list[tuple[int, int]]:
    """Byte-for-byte the bounds logic in SepformerSeparator._separate_once."""
    chunk = int(chunk_s * SR)
    overlap = clamped_overlap(chunk_s, overlap_s)
    stride = chunk - overlap
    if n <= chunk:
        return [(0, n)]
    starts = list(range(0, max(1, n - overlap), stride))
    b = [(s, min(s + chunk, n)) for s in starts]
    return [x for x in b if x[1] > x[0]]


# --------------------------------------------------------------------------- #
# caching so the expensive parts run once
# --------------------------------------------------------------------------- #

def raw_chunks(mix: np.ndarray, chunk_s: float, overlap_s: float, tag: str
               ) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    """Per-chunk ``separate_batch`` output, UNALIGNED, cached to disk."""
    bounds = chunk_bounds(mix.size, chunk_s, overlap_s)
    path = CACHE / f"{tag}_c{chunk_s:g}_o{overlap_s:g}.npz"
    if path.exists():
        z = np.load(path)
        ests = [z[f"e{i}"] for i in range(len(bounds))]
        print(f"  [cache] {path.name}: {len(ests)} chunks")
        return bounds, ests

    import torch
    from app.separation import SepformerSeparator
    sep = SepformerSeparator(device=CONFIG.runtime.resolve_device(),
                             cache_dir=str(CACHE_DIR / "models"))
    sep.load()
    ests = []
    for i, (s, e) in enumerate(bounds):
        with torch.inference_mode():
            t = torch.from_numpy(mix[s:e].astype(np.float32)).unsqueeze(0).to(sep.device)
            est = sep._model.separate_batch(t)
            est = est.squeeze(0).transpose(0, 1).cpu().numpy()
            del t
        ests.append(np.asarray(est, dtype=np.float64)[:, : e - s])
        print(f"  chunk {i + 1}/{len(bounds)}  [{s / SR:6.2f}, {e / SR:6.2f}) s")
    sep.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"e{i}": a for i, a in enumerate(ests)})
    return bounds, ests


def face_tracks(video: Path, tag: str):
    """FaceAnalyzer output, cached.  Only ``lip`` and fps are needed."""
    path = CACHE / f"{tag}_tracks.npz"
    if path.exists():
        z = np.load(path)
        n = int(z["n"])
        print(f"  [cache] {path.name}: {n} tracks")
        return [z[f"lip{i}"] for i in range(n)], float(z["fps"])

    from app.vision import FaceAnalyzer
    fa = FaceAnalyzer(CONFIG.vision)
    tracks, vm = fa.analyze(str(video))
    lips = [np.asarray(t.lip, dtype=np.float64) for t in tracks]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n=len(lips), fps=vm["fps"],
                        **{f"lip{i}": a for i, a in enumerate(lips)})
    return lips, float(vm["fps"])


# --------------------------------------------------------------------------- #
# ground truth 1: pitch.  No stitching, no correlation, no chaining.
# --------------------------------------------------------------------------- #

def f0_stats(x: np.ndarray, *, fmin: float = 70.0, fmax: float = 330.0,
             frame_ms: float = 40.0, hop_ms: float = 10.0,
             clarity: float = 0.45) -> tuple[float | None, float, int]:
    """(median f0 over voiced frames, voiced fraction, n_voiced).

    Normalised autocorrelation, which is crude but adequate here: the question
    is only "is this row the ~120 Hz voice or the ~210 Hz voice", a decision with
    an octave of headroom, not a pitch-accuracy benchmark.  ``None`` means no
    voiced frames -- the contract this project uses everywhere for "no
    measurement window", never a sentinel number.
    """
    frame = int(SR * frame_ms / 1000.0)
    hop = int(SR * hop_ms / 1000.0)
    if x.size < frame:
        return None, 0.0, 0
    n_fr = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_fr)[:, None]
    seg = x[idx]
    seg = seg - seg.mean(axis=1, keepdims=True)

    rms = np.sqrt((seg ** 2).mean(axis=1))
    ref = np.percentile(rms, 95)
    loud = rms > max(ref * 10 ** (-25.0 / 20.0), 1e-6)      # 25 dB below p95

    lo, hi = int(SR / fmax), int(SR / fmin)
    f0s = []
    for i in np.flatnonzero(loud):
        s = seg[i]
        # Normalised autocorrelation over the candidate lag range only.
        best_r, best_l = 0.0, 0
        e0 = float(np.dot(s, s))
        if e0 <= 0:
            continue
        for lag in range(lo, min(hi + 1, frame - 1)):
            a, b = s[:-lag], s[lag:]
            d = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b))) + 1e-20
            r = float(np.dot(a, b)) / d
            if r > best_r:
                best_r, best_l = r, lag
        if best_r >= clarity and best_l > 0:
            f0s.append(SR / best_l)
    if not f0s:
        return None, 0.0, 0
    return float(np.median(f0s)), len(f0s) / n_fr, len(f0s)


# --------------------------------------------------------------------------- #
# the three candidate evidence terms
# --------------------------------------------------------------------------- #

def s_continuity(prev_tail: np.ndarray, cur: np.ndarray, k: int) -> np.ndarray:
    """abs normalised correlation, exactly as _align_permutation computes it."""
    a, b = prev_tail[:, -k:], cur[:, :k]
    n = a.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(a[i]) * np.linalg.norm(b[j]) + 1e-12
            out[i, j] = abs(float(np.dot(a[i], b[j])) / d)
    return out


N_BANDS = 24


def band_spectrum(x: np.ndarray, *, nfft: int = 512, hop: int = 256
                  ) -> np.ndarray | None:
    """Energy-weighted mean log spectrum in ``N_BANDS`` log-spaced bands.

    Energy weighting is the point: an unweighted mean over a chunk where this
    row is silent is a spectrum of the residual, which would poison the very
    identity it is supposed to define.
    """
    if x.size < nfft:
        return None
    n_fr = 1 + (x.size - nfft) // hop
    if n_fr < 4:
        return None
    win = np.hanning(nfft)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n_fr)[:, None]
    frames = x[idx] * win
    mag = np.abs(np.fft.rfft(frames, axis=1))

    edges = np.geomspace(80.0, SR / 2 - 100.0, N_BANDS + 1)
    bins = np.clip((edges / (SR / nfft)).astype(int), 0, mag.shape[1] - 1)
    band = np.stack([mag[:, bins[i]:max(bins[i] + 1, bins[i + 1])].mean(axis=1)
                     for i in range(N_BANDS)], axis=1)

    energy = (frames ** 2).mean(axis=1)
    thr = np.percentile(energy, 75) * 0.1
    w = energy * (energy > thr)
    if w.sum() <= 0:
        return None
    v = np.log(np.maximum((band * w[:, None]).sum(0) / w.sum(), 1e-10))
    v -= v.mean()                                   # kill overall level/gain
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-9 else None


def s_signature(sig: list[np.ndarray | None], cur: np.ndarray
                ) -> np.ndarray | None:
    n = cur.shape[0]
    have = [band_spectrum(cur[j]) for j in range(n)]
    if any(s is None for s in sig) or any(h is None for h in have):
        return None
    return np.array([[float(np.dot(sig[i], have[j])) for j in range(n)]
                     for i in range(n)])


def envelope_db(x: np.ndarray, frame: int) -> np.ndarray:
    n = int(np.ceil(x.size / frame))
    p = np.pad(x, (0, n * frame - x.size))
    return 10.0 * np.log10((p.reshape(n, frame) ** 2).mean(1) + 1e-12)


def _z(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def s_anchor(lips: list[np.ndarray], fps: float, cur: np.ndarray,
             s: int, e: int) -> np.ndarray | None:
    """``out[t, j]`` = corr(lip motion of TRACK t, energy envelope of ROW j).

    Note the axes: this is a row -> *track* score, an absolute question, unlike
    ``s_continuity`` which is a row -> previous-row score.  Keeping the two
    straight is the whole trick; conflating them is how a permutation bug hides.
    """
    from app.dsp import resample_hold
    n = cur.shape[0]
    if len(lips) < n:
        return None

    frame = int(SR * 0.04)                                   # 40 ms == 25 fps
    n_fr = int(np.ceil((e - s) / frame))
    env = np.stack([_z(envelope_db(cur[j], frame)[:n_fr]) for j in range(n)])

    rows = []
    for t in range(n):
        f0, f1 = int(s / SR * fps), int(np.ceil(e / SR * fps))
        seg = lips[t][f0:f1]
        if seg.size == 0 or np.all(np.isnan(seg)):
            return None
        v, ok = resample_hold(seg, fps, n_fr, 1000.0 / 40.0)
        d = np.abs(np.diff(v, prepend=v[:1]))
        d[~ok] = 0.0
        if d.std() < 1e-9:
            return None
        rows.append(_z(d))
    lipm = np.stack(rows)
    return np.array([[float(np.dot(lipm[t], env[j])) / n_fr for j in range(n)]
                     for t in range(n)])


# --------------------------------------------------------------------------- #

def best_two(score: np.ndarray) -> tuple[list[int], float, float]:
    """Best permutation ``o`` (``o[i]`` = column for row i), its total, runner-up."""
    n = score.shape[0]
    tot = sorted(((sum(score[i, p[i]] for i in range(n)), p)
                  for p in itertools.permutations(range(n))), key=lambda t: -t[0])
    return list(tot[0][1]), tot[0][0], tot[1][0]


def align_ref(bounds, ests, overlap):
    """Overlap-add with high-overlap correlation alignment, plus margin report."""
    n = bounds[-1][1]
    n_src = ests[0].shape[0]
    from app.separation import _fade_window
    out = np.zeros((n_src, n))
    w_acc = np.zeros(n)
    prev_tail, margins = None, []
    for idx, (s, e) in enumerate(bounds):
        est = ests[idx].copy()
        if prev_tail is not None:
            k = min(overlap, est.shape[1], prev_tail.shape[1])
            o, b1, b2 = best_two(s_continuity(prev_tail, est, k))
            margins.append(b1 - b2)
            est = est[o]
        w = _fade_window(e - s, overlap if idx > 0 else 0,
                         overlap if idx < len(bounds) - 1 else 0)
        out[:, s:e] += est * w
        w_acc[s:e] += w
        prev_tail = est[:, -overlap:]
    out /= np.maximum(w_acc, 1e-8)
    return out, margins


def main() -> None:
    job = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/862bd92a01ac")
    tag = job.name
    CACHE.mkdir(parents=True, exist_ok=True)

    wav = CACHE / f"{tag}_mix.wav"
    if not wav.exists():
        media.extract_audio(job / "input.mp4", wav, SR)
    mix = media.read_audio(wav, SR)
    print(f"{tag}: {mix.size / SR:.2f} s @ {SR} Hz")

    print(f"\nreference separation (overlap {REF_OVERLAP_S:g} s "
          f"-> stride {CHUNK_S - REF_OVERLAP_S:g} s)")
    ref_ov = clamped_overlap(CHUNK_S, REF_OVERLAP_S)
    ref_bounds, ref_ests = raw_chunks(mix, CHUNK_S, REF_OVERLAP_S, tag)
    ref, ref_margins = align_ref(ref_bounds, ref_ests, ref_ov)
    print(f"  overlap used: {ref_ov / SR:g} s;  boundary margins "
          f"min={min(ref_margins):.4f} median={np.median(ref_margins):.4f} "
          f"n={len(ref_margins)}")
    if min(ref_margins) < 0.20:
        print("  !! a reference boundary is NOT decisive -- cross-check is weak")

    print(f"\nproduction separation (shipped: chunk {CHUNK_S:g} s, "
          f"overlap {PROD_OVERLAP_S:g} s)")
    pro_bounds, pro_ests = raw_chunks(mix, CHUNK_S, PROD_OVERLAP_S, tag)
    pro_ov = clamped_overlap(CHUNK_S, PROD_OVERLAP_S)
    n_src = pro_ests[0].shape[0]

    print("\nface tracks")
    lips, fps = face_tracks(job / "video.mp4", tag)
    print(f"  {len(lips)} tracks @ {fps} fps, present frames: "
          f"{[int(np.sum(~np.isnan(l))) for l in lips]}")

    # ---- ground truth A: pitch --------------------------------------------- #
    print("\n=== ground truth A: pitch (no stitching, no correlation) ===")
    print("  whole-clip reference rows, to establish the two f0 modes:")
    ref_f0 = []
    for i in range(n_src):
        f, frac, nv = f0_stats(ref[i])
        ref_f0.append(f)
        print(f"    reference row {i}: median f0 "
              + (f"{f:6.1f} Hz" if f else "  none  ")
              + f"  voiced {frac * 100:5.1f}% ({nv} frames)")

    print("  production raw rows, per chunk:")
    pitch_lab = []
    for idx, (s, e) in enumerate(pro_bounds):
        f = []
        for j in range(n_src):
            v, frac, nv = f0_stats(pro_ests[idx][j])
            f.append((v, frac, nv))
        # Label: which row is the LOWER-pitched voice?  lab[k] = row holding
        # pitch-identity k, k=0 being the low voice.
        if all(x[0] is not None for x in f):
            order = sorted(range(n_src), key=lambda j: f[j][0])
            lab = list(order)
            sep_hz = abs(f[order[-1]][0] - f[order[0]][0])
        else:
            lab, sep_hz = None, 0.0
        pitch_lab.append((lab, sep_hz))
        desc = "  ".join(
            f"row{j}={'  n/a ' if f[j][0] is None else f'{f[j][0]:6.1f}Hz'}"
            f"/v{f[j][1] * 100:4.1f}%" for j in range(n_src))
        print(f"    chunk {idx} [{s / SR:5.2f},{e / SR:5.2f}) {desc}"
              f"   -> low-first {lab}  gap {sep_hz:5.1f} Hz"
              + ("   <-- rows within 30 Hz: NOT separable by pitch"
                 if 0 < sep_hz < 30 else ""))

    # ---- ground truth B: correlation against the reference ----------------- #
    print("\n=== ground truth B: correlation against the high-overlap reference ===")
    truth = []
    for idx, (s, e) in enumerate(pro_bounds):
        sc = np.zeros((n_src, n_src))
        for i in range(n_src):                       # reference identity
            for j in range(n_src):                   # raw production row
                a, b = ref[i, s:e], pro_ests[idx][j]
                m = min(a.size, b.size)
                d = np.linalg.norm(a[:m]) * np.linalg.norm(b[:m]) + 1e-12
                sc[i, j] = abs(float(np.dot(a[:m], b[:m])) / d)
        o, b1, b2 = best_two(sc)
        truth.append({"order": o, "margin": b1 - b2, "score": sc})
        print(f"  chunk {idx} [{s / SR:5.2f},{e / SR:5.2f}) identity<-row {o} "
              f"margin {b1 - b2:.4f}"
              + ("   <-- ambiguous" if b1 - b2 < 0.2 else "")
              + f"   corr {np.round(sc, 3).tolist()}")

    # Do A and B agree?  Both name a row ordering per chunk; they may use
    # opposite label conventions, so compare the *relative* pattern.
    print("\n  do A and B agree on which chunks are flipped relative to chunk 0?")
    for idx in range(len(pro_bounds)):
        a = pitch_lab[idx][0]
        b = truth[idx]["order"]
        fa = None if a is None or pitch_lab[0][0] is None else (a != pitch_lab[0][0])
        fb = b != truth[0]["order"]
        mark = "agree" if fa is None or fa == fb else "*** DISAGREE ***"
        print(f"    chunk {idx}: pitch says flipped={fa}, reference says "
              f"flipped={fb}   {mark}")

    # ---- replay the shipped stitcher, and every candidate rule ------------- #
    print("\n=== what the shipped stitcher decides, vs what it should ===")
    from scipy.optimize import linear_sum_assignment
    sig: list[np.ndarray | None] = [None] * n_src
    prev_tail = None
    n_wrong = 0
    rows = []

    for idx, (s, e) in enumerate(pro_bounds):
        est = pro_ests[idx].copy()
        want = truth[idx]["order"]      # want[i] = raw row holding identity i

        rule = {}
        if prev_tail is not None:
            k = min(pro_ov, est.shape[1], prev_tail.shape[1])
            sc = s_continuity(prev_tail, est, k)
            o, b1, b2 = best_two(sc)
            rms_a = float(np.sqrt((prev_tail[:, -k:] ** 2).mean(1)).max())
            rms_b = float(np.sqrt((est[:, :k] ** 2).mean(1)).max())
            rule["continuity"] = (o, b1 - b2, rms_a > 1e-3 and rms_b > 1e-3, sc)
            sg = s_signature(sig, est)
            if sg is not None:
                o, b1, b2 = best_two(sg)
                rule["signature"] = (o, b1 - b2, True, sg)
            an = s_anchor(lips, fps, est, s, e)
            if an is not None:
                o, b1, b2 = best_two(an)
                rule["anchor"] = (o, b1 - b2, True, an)

        # --- replay the SHIPPED decision exactly ---------------------------- #
        applied = list(range(n_src))
        shipped_margin = 0.0
        if prev_tail is not None:
            k = min(pro_ov, est.shape[1], prev_tail.shape[1])
            sc = s_continuity(prev_tail, est, k)
            if (np.sqrt((prev_tail[:, -k:] ** 2).mean(1)).max() >= 1e-3
                    and np.sqrt((est[:, :k] ** 2).mean(1)).max() >= 1e-3):
                _, o = linear_sum_assignment(-sc)
                o = list(o)
                shipped_margin = float(
                    sum(sc[i, o[i]] for i in range(n_src)) - np.trace(sc))
                if shipped_margin >= 0.15:
                    applied = o
        est = est[applied]
        # applied[i] = raw row placed at output i; want[m] = raw row holding
        # identity m.  So output row i holds identity m with want[m]==applied[i].
        inv_want = {r: m for m, r in enumerate(want)}
        holds = [inv_want[applied[i]] for i in range(n_src)]
        ok = holds == list(range(n_src))
        n_wrong += not ok
        rows.append((idx, s / SR, applied, holds, ok, shipped_margin, rule))

        for i in range(n_src):
            b = band_spectrum(est[i])
            if b is not None:
                sig[i] = b if sig[i] is None else _renorm(0.7 * sig[i] + 0.3 * b)
        prev_tail = est[:, -pro_ov:]

    for idx, t0, applied, holds, ok, sm, rule in rows:
        tags = "  ".join(f"{k}={v[0]}/{v[1]:+.3f}{'' if v[2] else '(no-ev)'}"
                         for k, v in rule.items())
        print(f"  chunk {idx} @{t0:5.2f}s applied={applied} (shipped margin "
              f"{sm:+.3f}) -> output holds {holds} {'OK ' if ok else 'WRONG'}"
              f"   {tags}")

    print(f"\n  chunks whose output rows hold the wrong identity: "
          f"{n_wrong}/{len(pro_bounds)}")
    bad = [r for r in rows if not r[4]]
    if bad:
        span = []
        for idx, t0, *_ in bad:
            s, e = pro_bounds[idx]
            span.append(f"[{s / SR:.1f},{e / SR:.1f})")
        print(f"  wrong over spans: {' '.join(span)}")

    print("\nrule accuracy at boundaries (does the rule name the true "
          "row->identity map, up to its own fixed label convention?)")
    for name in ("continuity", "signature", "anchor"):
        got = [(idx, r[6][name]) for r in rows for idx in [r[0]]
               if name in r[6]]
        if not got:
            continue
        # A rule may label identities in its own order; what matters is whether
        # it tracks the CHANGES.  Score both readings and report both.
        direct = sum(v[0] == truth[i]["order"] for i, v in got)
        n_ = len(got)
        noev = sum(not v[2] for _, v in got)
        marg = [v[1] for _, v in got]
        print(f"  {name:11s} {direct}/{n_} correct   "
              f"margins {np.round(marg, 3).tolist()}   ({noev} with no evidence)")
        for i, v in got:
            print(f"      chunk {i}: order {v[0]} vs truth {truth[i]['order']}"
                  f"   matrix {np.round(v[3], 3).tolist()}")

    # ---- item 4: did one speaker collapse onto both rows? ----------------- #
    print("\n=== item 4 -- channel collapse? ===")
    for idx, (s, e) in enumerate(pro_bounds):
        sc = truth[idx]["score"]
        print(f"  chunk {idx} [{s / SR:5.2f},{e / SR:5.2f}) best corr per "
              f"identity {np.round(sc.max(axis=1), 3).tolist()}   "
              f"rms {np.round(np.sqrt((pro_ests[idx] ** 2).mean(1)), 5).tolist()}")
    b0, b1 = band_spectrum(ref[0]), band_spectrum(ref[1])
    if b0 is not None and b1 is not None:
        print(f"  cosine(band spectrum of ref row 0, row 1) = "
              f"{float(np.dot(b0, b1)):+.4f}   "
              f"(near +1 => the log-band signature cannot tell them apart)")

    meta = json.loads((job / "meta.json").read_text())
    print(f"\nwhat meta.json claimed: {meta['alignment']}")


def _renorm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


if __name__ == "__main__":
    main()
