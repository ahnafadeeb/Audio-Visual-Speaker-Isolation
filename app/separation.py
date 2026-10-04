"""Separation backends behind one swappable interface.

The interface is deliberately narrow::

    stems = separator.separate(mixture, sample_rate)   # -> (n_sources, n_samples)

Everything downstream (DSP, muxing, the UI) depends only on that shape.

``AVTSESeparator`` deliberately does not implement it. It needs the faces, so
it has its own entry point -- see its docstring and ARCHITECTURE_V3.md §6.

Backends:
  * ``PassthroughSeparator``  -- no model, for building the UI (Step 3).
  * ``SepformerSeparator``    -- speechbrain/sepformer-whamr16k, chunked.
  * ``AVTSESeparator``        -- AV-MossFormer2, one call per face. v3 default.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol, Sequence

import numpy as np

from .channels import greedy_one_to_one

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


class Separator(Protocol):
    n_sources: int

    def separate(
        self, mixture: np.ndarray, sample_rate: int, progress: ProgressFn | None = ...
    ) -> np.ndarray:
        ...


# --------------------------------------------------------------------------- #
# Passthrough -- lets the whole app be built and demoed with zero ML risk
# --------------------------------------------------------------------------- #

class PassthroughSeparator:
    """Emits the mixture on every channel.

    Not a joke backend: it is how you build and rehearse the entire delivery
    path (mux -> browser -> splitter -> crossfade) before the model is in play.
    A/B switching sounds identical, but every other property -- sync, timing,
    click-freeness, artefact layout -- is exercised for real.
    """

    def __init__(self, n_sources: int = 2) -> None:
        self.n_sources = n_sources

    def separate(self, mixture, sample_rate, progress=None):
        if progress:
            progress(1.0, "passthrough")
        return np.tile(np.asarray(mixture, dtype=np.float32), (self.n_sources, 1))


# --------------------------------------------------------------------------- #
# SepFormer
# --------------------------------------------------------------------------- #

class SepformerSeparator:
    """speechbrain/sepformer-whamr16k with chunking and permutation alignment.

    Two things here are load-bearing and easy to get wrong:

    **1. ``separate_batch``, never ``separate_file``.**  ``separate_file`` ends
    with ``est / est.abs().max(dim=1, keepdim=True)[0]`` -- it peak-normalises
    each source *independently*.  A -25 dB residual channel gets rescaled to
    peak 1.0, i.e. boosted ~25 dB into clear audibility, and worst during
    pauses where the target's own peak is smallest.  A genuinely silent channel
    becomes 0/0 -> NaN.  This single line is the most likely cause of the
    "ghostly whispers".  ``separate_batch`` returns raw decoder output.

    **2. Cross-chunk permutation alignment.**  Each call is independently
    permuted; SepFormer carries no speaker identity between calls.  Without
    alignment the speakers trade places mid-sentence, which sounds *exactly*
    like bleed-through and will send you debugging the wrong stage.
    """

    MODEL_ID = "speechbrain/sepformer-whamr16k"

    # Retry floor.  Below ~1 s a chunk is shorter than SepFormer's own dual-path
    # segment and quality falls off a cliff; if 1 s still OOMs, the card cannot
    # run this model at all and CPU is the honest answer.
    MIN_CHUNK_S = 1.0

    def __init__(self, device: str = "cpu", cache_dir: str | None = None,
                 chunk_s: float = 10.0, overlap_s: float = 2.0) -> None:
        self.device = device
        self.cache_dir = cache_dir
        self.chunk_s = chunk_s
        self.overlap_s = overlap_s
        self.n_sources = 2
        self._model = None
        # Set here as well as in each pass so a caller reading these after a
        # failed run gets an empty list rather than AttributeError.
        self.last_alignment: list[dict] = []
        self.last_chunk_s = chunk_s

    # -- lifecycle ---------------------------------------------------------- #

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy

        log.info("loading %s on %s", self.MODEL_ID, self.device)
        # COPY, not the default SYMLINK: Windows refuses symlinks (WinError
        # 1314) unless Developer Mode is on or Python runs elevated, so a fresh
        # demo machine fails its first download.  Costs one duplicate of the
        # weights on disk.
        self._model = SepformerSeparation.from_hparams(
            source=self.MODEL_ID,
            savedir=self.cache_dir,
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )
        self._model.eval()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def release(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- inference ---------------------------------------------------------- #

    def separate(self, mixture, sample_rate, progress=None):
        """Separate, halving ``chunk_s`` and retrying on CUDA OOM.

        SepFormer's dual-path attention is quadratic in the number of intra-chunk
        segments, so peak activation memory grows roughly with ``chunk_s**2``.
        That makes OOM a *predictable, recoverable* condition rather than a bug:
        the same clip that dies at 10 s fits comfortably at 4 s, and the only
        cost is a few more chunk boundaries for ``_align_permutation`` to stitch.

        Without this, a 4 GB card that is also driving the desktop raises
        ``torch.cuda.OutOfMemoryError`` from inside ``separate_batch`` -- minutes
        into a job, in front of an audience, with the artefacts half-written.
        The retry converts that into a slower run and a log line.

        The retry is deliberately NOT a fallback to CPU on the first failure:
        CPU inference for this model is ~30x slower, so silently taking it would
        turn a 40 s job into a 20 minute one and look like a hang.  We exhaust
        the cheap option (smaller chunks) first and only then let the error out.
        """
        import torch

        chunk_s = self.chunk_s
        while True:
            try:
                return self._separate_once(mixture, sample_rate, chunk_s, progress)
            except torch.cuda.OutOfMemoryError:
                # Drop the model's activations AND the cached blocks before
                # measuring or retrying; without empty_cache the allocator holds
                # the very segments whose absence caused the failure, so the
                # retry can fail at a size that would otherwise have fit.
                torch.cuda.empty_cache()
                nxt = chunk_s / 2.0
                if nxt < self.MIN_CHUNK_S:
                    free, total = torch.cuda.mem_get_info()
                    log.error(
                        "CUDA OOM at chunk_s=%.1fs, the minimum (%.1fs) -- "
                        "%.2f of %.2f GB free. This card cannot run %s; "
                        "re-run with --device cpu.",
                        chunk_s, self.MIN_CHUNK_S, free / 2**30, total / 2**30,
                        self.MODEL_ID)
                    raise
                log.warning("CUDA OOM at chunk_s=%.1fs -- retrying at %.1fs",
                            chunk_s, nxt)
                chunk_s = nxt

    def _separate_once(self, mixture, sample_rate, chunk_s, progress=None):
        import torch

        self.load()
        x = np.asarray(mixture, dtype=np.float32)
        n = x.size

        self.last_chunk_s = chunk_s
        chunk = int(chunk_s * sample_rate)
        overlap = int(self.overlap_s * sample_rate)
        # A retry can drive chunk_s below overlap_s, at which point stride <= 0
        # and the `starts` range below spins forever (or emits one chunk and
        # silently truncates the clip).  Keep the overlap a fixed fraction of
        # the chunk instead, so the crossfade and the permutation-alignment
        # window shrink with it rather than inverting it.
        overlap = min(overlap, chunk // 2)
        stride = chunk - overlap
        if n <= chunk:
            bounds = [(0, n)]
        else:
            starts = list(range(0, max(1, n - overlap), stride))
            bounds = [(s, min(s + chunk, n)) for s in starts]
            bounds = [b for b in bounds if b[1] > b[0]]

        out = np.zeros((self.n_sources, n), dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        prev_tail: np.ndarray | None = None
        self.last_alignment: list[dict] = []
        # Accumulated across every chunk, so a boundary that lands on a shared
        # pause can still be decided by who these two voices have been so far.
        anchor = _IdentityAnchor(self.n_sources)

        for idx, (s, e) in enumerate(bounds):
            seg = x[s:e]
            with torch.inference_mode():
                t = torch.from_numpy(seg).unsqueeze(0).to(self.device)
                est = self._model.separate_batch(t)          # (1, T, n_src) -- NOT separate_file
                est = est.squeeze(0).transpose(0, 1).cpu().numpy()   # (n_src, T)
                # `est` is already back on the host by here, but `t` is not: it
                # stays bound for the whole next iteration, so the input tensor
                # of chunk N+1 is allocated while chunk N's is still resident.
                # One chunk of 16 kHz mono is small; the activations graphed off
                # it are not, and on a 4 GB card that overlap is the difference
                # between fitting and not.
                del t

            est = np.asarray(est, dtype=np.float64)[:, : e - s]

            if prev_tail is not None and overlap > 0:
                est, decision = _align_permutation(
                    est, prev_tail, overlap, anchor=anchor,
                    sample_rate=sample_rate)
                decision["boundary_s"] = round(s / sample_rate, 2)
                self.last_alignment.append(decision)
                if not decision["confident"]:
                    log.warning(
                        "chunk boundary at %.1fs: no evidence for permutation "
                        "(margin %.4f, id_margin %s) -- holding previous speaker "
                        "order.  If the speakers sound swapped from here on, this "
                        "is where it happened.",
                        decision["boundary_s"], decision["margin"],
                        decision.get("id_margin", "n/a"))
                elif decision["basis"] == "identity":
                    log.info(
                        "chunk boundary at %.1fs: overlap uninformative "
                        "(margin %.4f); accumulated speaker identity chose "
                        "order %s (id_margin %.4f)",
                        decision["boundary_s"], decision["margin"],
                        decision["order"], decision.get("id_margin", 0.0))

            # Observe AFTER aligning, so profile i always accumulates channel i
            # in its final, stitched orientation.  Observing before would fold
            # whichever source SepFormer happened to emit first into profile 0
            # and destroy the very thing the anchor is for.
            anchor.observe(est, sample_rate)

            w = _fade_window(e - s, overlap if idx > 0 else 0,
                             overlap if idx < len(bounds) - 1 else 0)
            out[:, s:e] += est * w
            weight[s:e] += w
            prev_tail = est[:, -overlap:] if overlap and est.shape[1] >= overlap else None

            if progress:
                progress((idx + 1) / len(bounds), f"separating {idx + 1}/{len(bounds)}")

        np.divide(out, np.maximum(weight, 1e-8), out=out)
        return out.astype(np.float32)


# Below this normalised-correlation margin, the winning permutation is not
# meaningfully better than the best alternative to it.  Uncorrelated noise over a
# 2 s window scores ~1/sqrt(k) ~ 0.006 per pair, so 0.15 sits two orders of
# magnitude above chance while real speech clears it by ~1.8.
PERM_MARGIN = 0.15
# RMS below which a source contributes no evidence at all (-60 dBFS).
PERM_SILENCE_RMS = 1e-3
# Two conditions the accumulated spectral identity must satisfy TOGETHER before
# it is allowed to decide a boundary the waveform could not.  Neither works
# alone, and the reason is worth keeping: each covers the other's blind spot.
#
#   PERM_ID_MARGIN  the pairing margin on the cosine matrix (Murty, as above).
#                   Kills the SIMILAR-VOICES case.  Two same-sex voices give a
#                   cosine matrix that is high everywhere, so the margin
#                   collapses: measured median 0.00-0.02 across 20/10/5/0 dB,
#                   against 0.27-0.34 for a clearly-different pair.
#   PERM_ID_COS     the smallest cosine among the winning pairs.  Kills the
#                   NOISE case.  Profiles are L2-normalised, so a profile
#                   accumulated from noise is a unit vector in a random
#                   direction and its 2x2 margin is LARGE by amplification --
#                   measured null median 0.287, p90 0.670, max 1.383 over 400
#                   draws, i.e. the noise margin EXCEEDS the 0.573 a real
#                   distinct pair scores.  A margin threshold alone is therefore
#                   unbuildable at any value.  The absolute cosine is what
#                   separates them: null max 0.428 over 800 draws against
#                   0.547-1.000 for real voices down to 0 dB.
#
# Measured jointly (60 seeds x 4 SNRs x 3 voice-similarity conditions):
# distinct voices -> accuracy 1.00 in every cell where it acted; similar and
# near-identical voices -> abstains 100% of the time; pure noise -> 0/400
# false actions.  At 0 dB residual SNR it abstains 53% of the time and is still
# 100% correct when it acts, which is the behaviour wanted from a tie-breaker.
#
# 0.45 sits in a genuinely narrow window -- above the 0.428 noise ceiling, below
# the 0.47 median a real distinct pair scores at 0 dB.  Do not raise it without
# re-measuring both ends; there is not much room.
PERM_ID_MARGIN = 0.15
PERM_ID_COS = 0.45
# Low-order real-cepstrum coefficients kept as the identity profile.  c0 is
# dropped on purpose: it is the frame's overall log-level, which says nothing
# about who is speaking and everything about how loud they were in this chunk.
PERM_ID_COEF = 20


def _cepstral_profile(x: np.ndarray, sample_rate: int, *,
                      n_coef: int = PERM_ID_COEF,
                      frame_ms: float = 32.0,
                      hop_ms: float = 16.0) -> np.ndarray | None:
    """Mean low-order real cepstrum over this signal's loudest frames.

    A cheap, dependency-free stand-in for a speaker embedding.  The real cepstrum
    of the log magnitude spectrum separates the spectral *envelope* (low-order
    coefficients: vocal tract shape) from the excitation (high-order: pitch), and
    keeping coefficients 1..n_coef therefore measures roughly what an LPC formant
    analysis measures -- vocal tract length -- while being immune to the octave
    errors that make pitch a treacherous speaker label.

    Two deliberate choices:

    * **Level is removed twice, on purpose.**  Each frame's magnitude spectrum is
      divided by its own mean *before* the log, and c0 is then dropped after it.
      The first step is what actually makes the profile scale-free (see the
      comment at the division); the second costs nothing and removes any residual
      constant.  SepFormer's output magnitude is unconstrained -- it trains on
      scale-invariant SI-SNR -- so a profile that tracked level would drift chunk
      to chunk for reasons that have nothing to do with identity.
    * **Only the loudest frames count.**  A cepstrum computed over a pause
      describes the noise floor, and the noise floor is the same on both
      channels -- averaging it in pulls the two profiles together exactly when
      they most need to be distinguishable.

    Returns ``None`` when no frame carries enough level to measure, which the
    caller must treat as "no evidence" rather than as a zero vector.
    """
    x = np.asarray(x, dtype=np.float64)
    n = max(8, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if x.size < n:
        return None
    n_frames = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n)[None, :]

    rms = np.sqrt((frames ** 2).mean(axis=1))
    # Loud AND above the absolute floor: a percentile alone would happily select
    # "the loudest silence" on a channel that is silent throughout.
    keep = (rms > PERM_SILENCE_RMS)
    if keep.sum() >= 4:
        keep &= rms >= np.percentile(rms[keep], 50.0)
    if keep.sum() < 2:
        return None

    mag = np.abs(np.fft.rfft(frames[keep], axis=1))
    # Per-frame scale removal BEFORE the log, not after.  Dropping c0 alone
    # leaves a residual level dependence, because `log(a*|X| + eps)` is not
    # `log a + log(|X| + eps)` wherever |X| approaches eps -- i.e. at spectral
    # nulls, which is most of the bins.  Measured: 1.0e-5 of profile drift for a
    # 4x attenuation, small but structural, and it grows as the level falls.
    # Normalising each frame by its own mean magnitude makes the argument of the
    # log scale-free, so the epsilon sees identical numbers at any input level.
    mag /= mag.mean(axis=1, keepdims=True) + 1e-20
    cep = np.fft.irfft(np.log(mag + 1e-12), axis=1)
    prof = cep[:, 1:1 + n_coef].mean(axis=0)
    nrm = float(np.linalg.norm(prof))
    return prof / nrm if nrm > 1e-12 else None


class _IdentityAnchor:
    """Per-channel spectral identity, accumulated over every chunk so far.

    This is the "global speaker tracking" half of cross-chunk alignment, and it
    exists because waveform correlation over the overlap window has a specific,
    common blind spot: when the overlap lands on a shared pause, there is no
    waveform to correlate and the decision is a coin flip that then propagates
    to the end of the clip.  A profile accumulated over *all previous chunks*
    still has evidence at that moment, because it was built when people were
    talking.

    Accumulation is a weighted running mean, weighted by how much loud audio the
    chunk contributed, so a chunk in which a speaker barely spoke cannot drag
    their profile toward whoever was actually talking.

    Deliberately NOT a visual anchor, which is what the original bug report
    proposed: the lip signal on the test clip fails a 4000-shift circular null at
    every time scale (docs/DIAG_MATCHER.md), so anchoring stem order to it would
    lock in a coin flip rather than break a tie.  Spectral identity at least
    measures a property of the voice being separated.
    """

    def __init__(self, n_src: int, n_coef: int = PERM_ID_COEF) -> None:
        self.profiles = np.zeros((n_src, n_coef), dtype=np.float64)
        self.mass = np.zeros(n_src, dtype=np.float64)

    def observe(self, stems: np.ndarray, sample_rate: int) -> None:
        """Fold an aligned chunk's channels into their profiles."""
        for i in range(min(len(self.mass), stems.shape[0])):
            prof = _cepstral_profile(stems[i], sample_rate)
            if prof is None:
                continue
            w = float(np.sqrt(np.mean(stems[i] ** 2)))       # loudness as weight
            total = self.mass[i] + w
            if total <= 0:
                continue
            self.profiles[i] = (self.profiles[i] * self.mass[i] + prof * w) / total
            self.mass[i] = total

    def ready(self) -> bool:
        return bool(np.all(self.mass > 0))

    def score(self, cur: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """``(n_src, n_src)`` cosine similarity of accumulated i to new channel j.

        ``None`` when either side has nothing measurable, so the caller holds
        rather than acting on a zero matrix that would look like a perfect tie.
        """
        if not self.ready():
            return None
        n = self.profiles.shape[0]
        news = [_cepstral_profile(cur[j], sample_rate) for j in range(min(n, cur.shape[0]))]
        if any(p is None for p in news) or len(news) < n:
            return None
        ref = self.profiles / (np.linalg.norm(self.profiles, axis=1, keepdims=True) + 1e-12)
        return np.asarray([[float(ref[i] @ news[j]) for j in range(n)]
                           for i in range(n)])


def _best_pairing(score: np.ndarray) -> list[int]:
    """Columns maximising the total, as a permutation of rows."""
    n_src = score.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
        _, order = linear_sum_assignment(-score)
        return list(int(c) for c in order)
    except Exception:                                    # pragma: no cover
        # NOT np.argmax(score, axis=1).  `order` is used as `cur[order]`, so it
        # must be a permutation; argmax can repeat a column, which duplicates
        # one source onto two channels and drops the other entirely -- one
        # speaker heard on both buttons, the other gone from the mix.
        pairs = greedy_one_to_one(score)
        return [c for _, c in pairs]


def _pairing_margin(score: np.ndarray) -> tuple[list[int], float]:
    """``(order, margin)`` where margin beats the best FEASIBLE alternative.

    The previous version computed ``best - trace``, which is 0.0 by construction
    whenever the identity order wins -- and identity wins at most boundaries, so
    the reported margin was 0.0 nearly always and ``meta.alignment.min_margin``
    could not tell a decisive hold from a coin flip.  Worse, ``confident`` was
    then true *at* margin 0.0, so a boundary with no evidence at all logged as
    confident and never raised the warning it was written to raise.

    The margin that means something is the winner's total minus the runner-up
    *pairing's* total, which is well defined in both directions: large when one
    ordering clearly fits, near zero when the two orderings are interchangeable.
    Found by forbidding each winning pair in turn and re-solving -- the first
    step of Murty's algorithm, the same construction ``matching.pairing_margin``
    uses for stem/face pairing.
    """
    n_src = score.shape[0]
    order = _best_pairing(score)
    s0 = float(sum(score[i, order[i]] for i in range(n_src)))
    span = float(score.max()) - float(score.min())
    pen = float(score.min()) - (span + 1.0) * (n_src + 1)
    second = -np.inf
    for i in range(n_src):
        m = score.copy()
        m[i, order[i]] = pen
        alt = _best_pairing(m)
        if alt == order:
            continue                    # unavoidable pair: no alternative exists
        second = max(second, float(sum(score[k, alt[k]] for k in range(n_src))))
    margin = s0 - second if np.isfinite(second) else s0
    return order, max(0.0, float(margin))


def _align_permutation(cur: np.ndarray, prev_tail: np.ndarray,
                       overlap: int, *,
                       anchor: "_IdentityAnchor | None" = None,
                       sample_rate: int = 16000) -> tuple[np.ndarray, dict]:
    """Reorder ``cur``'s sources to match ``prev_tail`` over the overlap region.

    Each ``separate_batch`` call is independently permuted -- SepFormer carries
    no speaker identity between calls -- so consecutive chunks must be stitched
    by correlation.

    **The decision must be gated on evidence.**  Measured: when the overlap
    window lands on a shared pause (both speakers quiet, only converter noise),
    a bare argmax/Hungarian on the correlation matrix preserves identity in
    16/40 trials -- a coin flip.  And because ``prev_tail`` is taken from the
    already-aligned chunk, one bad call propagates: the speakers stay swapped
    for the rest of the clip.  Shared pauses are ubiquitous in conversation and
    a boundary occurs every ``chunk_s - overlap_s`` seconds, so this fires
    often, and the result is indistinguishable by ear from bleed-through -- it
    sends you debugging the gate instead of the stitcher.

    So the decision is a cascade, and which rung decided is recorded in
    ``basis``:

      ``waveform``  the overlap carries a decisive correlation.  Best evidence
                    there is -- it is literally the same audio on both sides of
                    the boundary -- so it is consulted first and alone.
      ``identity``  the overlap was uninformative, but the accumulated spectral
                    profiles (``_IdentityAnchor``) are decisive on BOTH of their
                    conditions -- see ``PERM_ID_COS`` for why one is not enough.
                    This is the rung that survives a shared pause, because the
                    profiles were built while people were talking.  Measured to
                    be 100% correct when it fires on acoustically distinct
                    speakers and to abstain 100% of the time on similar ones.
      ``hold``      neither.  Keep the existing order: holding a possibly-wrong
                    order costs nothing extra (it is already 50/50), while acting
                    on noise throws away a correct one.

    The old code had no third rung and no ``identity`` rung: a boundary with a
    sub-threshold margin held the order *and reported itself confident*, so a
    genuine flip arriving at a quiet boundary stayed wrong for the rest of the
    clip with nothing in the artifacts to show it.

    One case the identity rung cannot detect, stated rather than papered over:
    if the chunk contains speakers the anchor has never heard (a scene cut), the
    cosines are high and the margin can clear the gate, so it will pick an order
    with no basis for it.  Measured: margin 0.292, min cosine 0.721 on two unseen
    voices.  It is left alone because at such a boundary there is no prior
    identity to preserve -- holding is exactly as arbitrary as flipping -- so the
    error is symmetric and costs nothing relative to the alternative.
    """
    n_src = cur.shape[0]
    identity = {"order": list(range(n_src)), "margin": 0.0, "flip_gain": 0.0,
                "confident": False, "flipped": False, "basis": "hold"}
    k = min(overlap, cur.shape[1], prev_tail.shape[1])
    if k <= 0:
        return cur, identity

    a = prev_tail[:, -k:]
    b = cur[:, :k]

    # Evidence gate: a correlation computed where nobody is speaking is a
    # correlation of noise, however confident the resulting number looks.
    rms_a = np.sqrt(np.mean(a * a, axis=1))
    rms_b = np.sqrt(np.mean(b * b, axis=1))
    quiet = rms_a.max() < PERM_SILENCE_RMS or rms_b.max() < PERM_SILENCE_RMS

    score = np.zeros((n_src, n_src))
    if not quiet:
        for i in range(n_src):
            for j in range(n_src):
                ai, bj = a[i], b[j]
                denom = np.linalg.norm(ai) * np.linalg.norm(bj) + 1e-12
                score[i, j] = abs(float(np.dot(ai, bj)) / denom)

    order, margin = _pairing_margin(score) if not quiet else (list(range(n_src)), 0.0)
    keep_total = float(np.trace(score))
    best_total = float(sum(score[i, order[i]] for i in range(n_src)))
    decision = {"order": list(order), "margin": round(margin, 5),
                # What acting would gain over doing nothing.  Distinct from
                # `margin` once n_src > 2, and it is this that must clear the
                # threshold before a REORDER is applied -- `margin` only says how
                # well separated the winner is from its runner-up.
                "flip_gain": round(best_total - keep_total, 5),
                "confident": bool(margin >= PERM_MARGIN),
                "flipped": list(order) != list(range(n_src)),
                "basis": "waveform"}

    if margin >= PERM_MARGIN:
        if not decision["flipped"]:
            return cur, decision
        return cur[order], decision

    # ---- the waveform had nothing to say: ask the accumulated identity ---- #
    id_score = anchor.score(cur, sample_rate) if anchor is not None else None
    if id_score is not None:
        id_order, id_margin = _pairing_margin(id_score)
        id_cos = min(float(id_score[i, id_order[i]]) for i in range(n_src))
        decision.update({"id_margin": round(id_margin, 5),
                         "id_cos": round(id_cos, 5)})
        # BOTH conditions, for the reasons measured at PERM_ID_COS.  The margin
        # alone admits noise; the cosine alone admits acoustically similar
        # speakers and any pair the anchor has never heard.
        if id_margin >= PERM_ID_MARGIN and id_cos >= PERM_ID_COS:
            decision.update({
                "order": list(id_order), "basis": "identity",
                "confident": True, "flipped": list(id_order) != list(range(n_src)),
            })
            return (cur[id_order] if decision["flipped"] else cur), decision

    decision.update({"order": list(range(n_src)), "flipped": False,
                     "confident": False, "basis": "hold"})
    return cur, decision


def _fade_window(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Raised-cosine crossfade window for overlap-add of adjacent chunks."""
    w = np.ones(length, dtype=np.float64)
    fi = min(fade_in, length // 2)
    fo = min(fade_out, length // 2)
    if fi > 0:
        w[:fi] = 0.5 * (1.0 - np.cos(np.pi * np.arange(fi) / fi))
    if fo > 0:
        w[length - fo:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fo) / fo))
    return w


# --------------------------------------------------------------------------- #
# AV-TSE -- audio-visual target speaker extraction (ARCHITECTURE_V3.md)
# --------------------------------------------------------------------------- #

class AVTSESeparator:
    """Extracts one voice per face, conditioned on that face's mouth ROI.

    Deliberately NOT a :class:`Separator`. The protocol above takes audio
    alone, which is exactly the limitation that made a separate matching stage
    necessary -- and ``docs/DIAG_MATCHER.md`` measured that stage at chance
    (0.0679 on the reference clip; 0.0683 on white noise). Widening the
    protocol to carry video would preserve the shape of a design whose shape
    is the problem. This class takes the faces as an argument instead, so
    "which voice belongs to which face" is an input, never an inference.

    Consequences, all of which delete code rather than add it:

    * no permutation alignment -- chunk k and chunk k+1 are conditioned on the
      same face, so they cannot disagree about who they are following;
    * no channel map -- face *i* produces channel *i* by construction;
    * no 2-speaker limit -- N faces means N calls.

    Chunking keeps a ``context_s`` margin on each side of every body and
    throws it away. The model has least context at its own edges, so this puts
    those edges outside what we keep, instead of crossfading two equally
    edge-damaged estimates over a long window.

    Windows are short (see ``AVTSEConfig.chunk_s`` for the measurement) and
    snapped to whole video frames, so every window's audio and lips start on
    the same instant; they run ``batch`` at a time to pay for being short.

    With two or more faces, ``refine_passes`` re-extracts every face from the
    mixture minus the other faces' estimates -- see :func:`refinement_inputs`.
    """

    def __init__(
        self,
        device: str = "cuda",
        *,
        chunk_s: float = 1.0,
        context_s: float = 0.5,
        fade_s: float = 0.032,
        batch: int = 8,
        refine_passes: int = 0,
        refine_shared_weight: float = 0.5,
        refine_smooth: tuple[int, int] = (9, 5),
        model=None,
        adapter=None,
    ) -> None:
        self.device = device
        self.adapter = adapter
        self.chunk_s = float(chunk_s)
        self.context_s = float(context_s)
        self.fade_s = float(fade_s)
        self.batch = max(1, int(batch))
        self.refine_passes = max(0, int(refine_passes))
        self.refine_shared_weight = float(refine_shared_weight)
        self.refine_smooth = refine_smooth
        self._model = model

    # -- lazy load so constructing the app never touches the GPU ----------- #
    @property
    def model(self):
        if self._model is None:
            from .avtse import load_model
            self._model = load_model(self.device, adapter=self.adapter)
        return self._model

    @property
    def base_model(self):
        """The released weights, for faces an adapter was not trained on."""
        if self.adapter is None:
            return self.model
        from .avtse import load_model
        return load_model(self.device)

    def _run(self, mix: np.ndarray, roi: np.ndarray, model=None) -> np.ndarray:
        """One span: ``(L,)`` audio + ``(Tv, 112, 112)`` ROI -> ``(L,)``."""
        return self._run_batch(np.asarray(mix, np.float32)[None], np.asarray(roi)[None], model)[0]

    def _run_batch(self, mixes: np.ndarray, rois: np.ndarray, model=None) -> np.ndarray:
        """``(B, L)`` audio + ``(B, Tv, 112, 112)`` ROIs -> ``(B, L)`` float32."""
        import torch

        model = model if model is not None else self.model

        dev = torch.device(self.device)
        a = torch.from_numpy(np.ascontiguousarray(mixes, dtype=np.float32)).to(dev)
        v = torch.from_numpy(np.ascontiguousarray(rois, dtype=np.float32)).to(dev)
        # TF32 for this model only (restored after): 17% faster on the RTX
        # 5060, output 63 dB below the signal away from fp32 -- i.e. identical.
        # bf16 autocast was 28% faster but only 33 dB away, which is audible
        # in principle on a stem that must not be altered, so not used.
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        try:
            with torch.inference_mode():
                est = model(a, v)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        # [B, 1, T] per the model docstring; reshape rather than squeeze so a
        # batch of one cannot collapse to 1-D and be mis-fitted downstream.
        est = est.float().cpu().numpy().reshape(len(mixes), -1)
        return np.stack([_fit(e, mixes.shape[1]) for e in est])

    def _run_windows(self, n_windows: int, window, progress: ProgressFn | None,
                     needed: Sequence[bool] | None = None, model=None) -> np.ndarray:
        """Run ``window(k) -> (audio, roi)`` for every k, ``self.batch`` at a time.

        Windows are built per batch rather than all up front: at 49 frames of
        112x112 float32 per window, a 5-minute clip's full ROI stack would be
        ~0.8 GB of host memory for no benefit.  An OOM halves the batch and
        retries the same windows, so the first-guess size never kills a job.
        """
        import torch

        # Windows nobody needs (the face is off screen for the whole body) are
        # never run: their output would be muted anyway. On an edited talk show
        # that is most of a guest's windows.
        todo = [j for j in range(n_windows) if needed is None or needed[j]]
        length = len(window(0)[0])
        out = np.zeros((n_windows, length), dtype=np.float32)
        k = 0
        while k < len(todo):
            b = min(self.batch, len(todo) - k)
            ids = todo[k:k + b]
            pairs = [window(j) for j in ids]
            try:
                est = self._run_batch(np.stack([p[0] for p in pairs]),
                                      np.stack([p[1] for p in pairs]), model)
            except torch.cuda.OutOfMemoryError:
                if b == 1:
                    raise
                self.batch = max(1, b // 2)
                torch.cuda.empty_cache()
                log.warning("avtse: OOM at batch %d, retrying at %d", b, self.batch)
                continue
            out[ids] = est
            k += b
            if progress:
                progress(k / len(todo), f"avtse {k}/{len(todo)}")
        if progress and not todo:
            progress(1.0, "avtse: face never on screen")
        return out

    def separate_one(
        self,
        mixture: np.ndarray,
        roi: np.ndarray,
        sample_rate: int,
        progress: ProgressFn | None = None,
        active: np.ndarray | None = None,
        model=None,
    ) -> np.ndarray:
        """Extract the speaker whose mouth is in ``roi``.

        ``active``, optional, is a per-video-frame bool: False where this face
        is off screen. Windows whose kept body is entirely inactive are not run
        and come back as silence. ``model`` overrides :attr:`model` (see
        ``adapted`` in :meth:`separate_for_faces`).

        Args:
            mixture: ``(n_samples,)`` mono at ``sample_rate``.
            roi: ``(n_frames, 112, 112)`` normalised, from :mod:`app.roi`.
                ``n_frames`` must be ``n_video_frames_for(n_samples)``.
            sample_rate: must be 16000; the checkpoint is 16 kHz-only.

        Returns:
            ``(n_samples,)`` float32, same length as ``mixture``.
        """
        from .avtse.config import SAMPLE_RATE, VIDEO_FPS

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"AV-TSE is {SAMPLE_RATE} Hz only; got {sample_rate}. Resample "
                "upstream -- the visual/audio frame correspondence is baked "
                "into the checkpoint."
            )

        mixture = np.asarray(mixture, dtype=np.float32)
        n = len(mixture)
        roi = np.asarray(roi)

        # Everything below is counted in VIDEO FRAMES (640 samples each), so a
        # window's audio and its lips start on the same instant. The previous
        # sample-based layout rounded the visual start per window, which could
        # put the lips up to half a frame (20 ms) off the audio they condition.
        spf = sample_rate // VIDEO_FPS
        body_f = max(1, int(round(self.chunk_s * VIDEO_FPS)))
        ctx_f = max(0, int(round(self.context_s * VIDEO_FPS)))
        fade_f = max(1, int(np.ceil(self.fade_s * VIDEO_FPS)))
        win_f = body_f + 2 * ctx_f
        n_f = -(-n // spf)

        # Short clip: one call, no seams at all.
        if n_f <= win_f:
            out = self._run(mixture, roi[: max(1, int(round(n / spf)))], model)
            if progress:
                progress(1.0, "avtse")
            return _fit(out, n)

        # Bodies overlap by `fade_f` frames so there is something to crossfade
        # over. Stop at the window that reaches the end rather than striding
        # past it, so there is never a runt the previous one already covered.
        step_f = max(1, body_f - fade_f)
        starts = [0]
        while starts[-1] + body_f < n_f:
            starts.append(starts[-1] + step_f)

        # Pad so every window is full length and the batch is rectangular:
        # silence for audio (it lands in the discarded context), held edge
        # frames for the lips (a face does not vanish at the clip boundary).
        total_f = starts[-1] + win_f
        audio = np.zeros(total_f * spf, dtype=np.float32)
        audio[ctx_f * spf : ctx_f * spf + n] = mixture
        right = max(0, total_f - ctx_f - len(roi))
        lips = np.concatenate([np.repeat(roi[:1], ctx_f, 0), roi,
                               np.repeat(roi[-1:], right, 0)])[:total_f]

        def window(k: int) -> tuple[np.ndarray, np.ndarray]:
            s = starts[k]
            return audio[s * spf : (s + win_f) * spf], lips[s : s + win_f]

        needed = None
        if active is not None:
            act = np.asarray(active, dtype=bool)
            needed = [bool(act[s:s + body_f].any()) for s in starts]
        est = self._run_windows(len(starts), window, progress, needed, model)

        body = body_f * spf
        fade = (body_f - step_f) * spf
        out = np.zeros(total_f * spf, dtype=np.float64)
        wsum = np.zeros_like(out)
        for k, s in enumerate(starts):
            # Discard the context margins: keep only the body.
            keep = est[k, ctx_f * spf : ctx_f * spf + body]
            w = _fade_window(body, fade if k > 0 else 0,
                             fade if k < len(starts) - 1 else 0)
            out[s * spf : s * spf + body] += keep * w
            wsum[s * spf : s * spf + body] += w

        np.divide(out, np.maximum(wsum, 1e-8), out=out)
        return out[:n].astype(np.float32)

    def separate_for_faces(
        self,
        mixture: np.ndarray,
        rois: Sequence[np.ndarray],
        sample_rate: int,
        progress: ProgressFn | None = None,
        presence: np.ndarray | None = None,
        adapted: Sequence[bool] | None = None,
    ) -> np.ndarray:
        """One extraction per face, then refinement. Returns ``(n_faces, n_samples)``.

        Channel *i* is face *i*. That identity is the whole point -- see the
        class docstring.

        ``presence`` is an optional ``(n_faces, n_samples)`` 0..1 envelope,
        0 where that face is off screen. It only shapes what refinement
        subtracts: an off-screen face's estimate is conditioned on frozen lips
        and is not evidence of anything, so it must not be taken out of
        anyone else's input.

        ``adapted``, with an adapter loaded, says per face whether to use it;
        False faces are extracted by the released weights. An adapter is
        trained on specific people and makes strangers worse (app/adapt.py).
        """
        n_faces = len(rois)
        if n_faces == 0:
            raise ValueError("no faces to extract")

        passes = 1 + (self.refine_passes if n_faces >= 2 else 0)
        total = n_faces * passes
        spf = sample_rate // 25
        actives = [None] * n_faces if presence is None else \
            [np.asarray(p)[::spf] > 0 for p in presence]

        models = [None if adapted is None or adapted[i] else self.base_model
                  for i in range(n_faces)]

        def extract(inputs: Sequence[np.ndarray], p: int) -> np.ndarray:
            out = np.zeros((n_faces, len(mixture)), dtype=np.float32)
            for i, roi in enumerate(rois):
                def sub(f: float, m: str, _i=i) -> None:
                    if progress:
                        tag = f"pass {p + 1}/{passes}, " if passes > 1 else ""
                        progress((p * n_faces + _i + f) / total,
                                 f"{tag}face {_i + 1}/{n_faces}: {m}")
                out[i] = self.separate_one(inputs[i], roi, sample_rate, sub,
                                           active=actives[i], model=models[i])
            return out

        stems = extract([mixture] * n_faces, 0)
        for p in range(1, passes):
            basis = stems if presence is None else stems * presence
            inputs = refinement_inputs(
                mixture, basis, sample_rate=sample_rate,
                shared_weight=self.refine_shared_weight,
                smooth_frames=self.refine_smooth[0],
                smooth_bins=self.refine_smooth[1])
            stems = extract(inputs, p)
        return stems


def refinement_inputs(
    mixture: np.ndarray,
    stems: np.ndarray,
    *,
    sample_rate: int,
    shared_weight: float = 0.5,
    nfft: int = 512,
    hop: int = 128,
    smooth_frames: int = 9,
    smooth_bins: int = 5,
) -> np.ndarray:
    """Per-face mixtures with the OTHER faces' estimates taken out.

    Feeding AV-TSE ``mixture - (other faces)`` gives it a cleaner input, and on
    two similar voices reading over each other that is worth a lot (face 1's
    WER 36% -> 24% on one clip, 7.6% -> 5.4% on another). But the other
    estimates are not clean either. Where face *i* speaks and face *j* is
    quieter, AV-TSE conditioned on *j* partly returns *i* -- and subtracting
    that cancels face *i*'s own voice. Plain subtraction took one clip's face
    0 from 2.3% to 30.2% WER for exactly this reason.

    The leaked part is recognisable: it is the same waveform as face *i*'s own
    estimate, so it is phase-coherent with it. Per STFT bin we regress the
    other estimate onto this face's estimate over a small neighbourhood,

        a = <S_j, S_i> / <S_i, S_i>        (|a| clipped to 1)

    and subtract ``S_j - (1 - shared_weight) * a * S_i``: everything that is
    NOT this face in full, and the shared part only at ``shared_weight``. The
    shared part cannot be attributed from the audio alone -- it may be *j*'s
    voice leaking into *i* rather than the reverse -- so it is split, not
    given away. 0.5 was the best of {0, 0.5, 1} on both phone clips.

    Returns ``(n_faces, n_samples)`` float32.
    """
    from scipy.ndimage import uniform_filter

    from .dsp import _stft_engine

    stems = np.asarray(stems, dtype=np.float32)
    n_faces, n = stems.shape
    sft = _stft_engine(nfft, hop, sample_rate)
    specs = [sft.stft(s).astype(np.complex64) for s in stems]
    size = (smooth_bins, smooth_frames)

    def local(z: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(z):
            return uniform_filter(z.real, size) + 1j * uniform_filter(z.imag, size)
        return uniform_filter(z, size)

    out = np.empty((n_faces, n), dtype=np.float32)
    for i in range(n_faces):
        own = specs[i]
        own_pow = local(np.abs(own) ** 2) + 1e-12
        remove = np.zeros_like(own)
        for j in range(n_faces):
            if j == i:
                continue
            a = local(specs[j] * np.conj(own)) / own_pow
            mag = np.abs(a)
            # Never attribute more of S_j to face i than face i's estimate holds.
            a = np.where(mag > 1.0, a / np.maximum(mag, 1e-12), a)
            remove += specs[j] - (1.0 - shared_weight) * a * own
        y = sft.istft(remove, k1=n)[:n]
        out[i] = mixture - np.asarray(y, dtype=np.float32)
    return out


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Force ``x`` to exactly ``n`` samples (the model pads to its own stride)."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


# --------------------------------------------------------------------------- #

def build_separator(name: str, *, device: str, cache_dir: str | None = None,
                    chunk_s: float = 10.0, overlap_s: float = 2.0) -> Separator:
    if name == "passthrough":
        return PassthroughSeparator()
    if name == "sepformer":
        return SepformerSeparator(device=device, cache_dir=cache_dir,
                                  chunk_s=chunk_s, overlap_s=overlap_s)
    if name == "avtse":
        raise ValueError(
            "avtse does not implement the audio-only Separator protocol -- it "
            "needs mouth ROIs. Construct AVTSESeparator directly and call "
            "separate_for_faces(); see ARCHITECTURE_V3.md §6."
        )
    raise ValueError(
        f"unknown separator {name!r}; expected 'sepformer', 'avtse' or "
        f"'passthrough'"
    )
