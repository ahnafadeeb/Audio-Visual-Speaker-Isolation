"""Assign separated audio stems to face tracks.

The signal is correlation between an audio stem's energy envelope and a face's
lip-aperture signal, resolved globally with the Hungarian algorithm so the
assignment is one-to-one.

Note this stage inherits a genuine failure mode: when it mis-assigns, the user
clicks face A and hears speaker B -- a loud, obvious, on-camera failure.  That
is the strongest argument for the AV-TSE upgrade path (ARCHITECTURE_V2.md §3),
where the model is conditioned on one face and emits one stream, so no
assignment step exists at all.

Until then, TWO numbers are reported, and the second one is the one to trust.

``confidence`` is the margin of the winning one-to-one pairing over the best
*feasible* alternative -- a property of the whole assignment, not of one track,
because that is what the Hungarian solver actually decides.  Every track in the
pairing therefore shares one number: with two faces and two stems, if one face
is confidently placed the other is forced, however ambiguous its own row looks.

``significance`` exists because that margin turned out to be **uncalibrated**,
and the failure was silent.  It divides by ``n_frames`` as though smoothed
envelope frames were independent samples, so it grows with the smoothing window
while the answer does not improve.  Measured on the real two-speaker clip
(docs/DIAG_MATCHER.md), integration window against reported margin:

    40 ms (shipped) 0.0679 | 200 ms 0.1870 | 600 ms 0.4125 | 1600 ms 0.6654

-- and the pairing is *wrong* at every one of those windows.  So a threshold on
the margin cannot protect the user: a wider kernel ships the same wrong answer
with 13x the confidence.  ``significance`` is a permutation p-value for the same
margin against a circular-shift null (see ``pairing_significance``), which is
invariant to that inflation because the null statistic inflates identically.

On the clip that motivated this, the honest number is p ~ 0.9: nine shifted lip
signals in ten produce a margin at least as large as the real one, and the
pairing the real signal chose was in fact inverted.  A coin flip that landed
wrong, reported at a confidence above threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import greedy_one_to_one
from .vision import FaceTrack, resample_lip


def energy_envelope(x: np.ndarray, sample_rate: int, frame_ms: float,
                    smooth: int = 7) -> np.ndarray:
    """Short-time RMS envelope in dB, smoothed."""
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n = int(np.ceil(len(x) / frame))
    padded = np.pad(np.asarray(x, dtype=np.float64), (0, n * frame - len(x)))
    rms = np.sqrt((padded.reshape(n, frame) ** 2).mean(axis=1))
    env = 20.0 * np.log10(rms + 1e-10)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        env = np.convolve(env, k, mode="same")
    return env


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def _solve(m: np.ndarray) -> tuple[list[int], list[int]]:
    """Optimal one-to-one rows->cols maximising the total."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-m)
        return list(r), list(c)
    except Exception:                                    # pragma: no cover
        # NOT np.argmax(m, axis=1): argmax has no one-to-one constraint, so two
        # tracks can claim the same stem.  channels.claims now arbitrates that
        # centrally, but the assignment should not manufacture the ambiguity.
        # See channels.greedy_one_to_one.
        pairs = greedy_one_to_one(m)
        return [p[0] for p in pairs], [p[1] for p in pairs]


def pairing_margin(score: np.ndarray) -> tuple[list[int], list[int], float]:
    """``(rows, cols, margin)`` for the optimal one-to-one pairing.

    The margin is the winning total minus the total of the best FEASIBLE
    alternative, because that is the number that decides the pairing.  On the
    test clip Speaker A preferred stem 0 by 0.35 while Speaker B's row was
    nearly a tie (0.072 vs 0.0099): ``[A->0, B->1]`` is then the only consistent
    pairing and wins by 0.29, yet any per-track marginal measure reports B at
    ~0.0 and flags a match that is forced to be right.  Per-row margin is also
    what the earlier top-2 bug computed, and it failed the opposite way -- a
    track placed on its SECOND choice by the global optimum read a positive
    margin arguing for the stem it did NOT get.

    "Feasible" is the load-bearing word.  Moving one track to its best other
    stem is not an alternative assignment: that stem is already taken, so the
    relaxed total can exceed the optimum and the margin collapses to 0.  The
    true second best differs from the optimum in at least one pair, so
    forbidding each winning pair in turn and re-solving finds it (the first step
    of Murty's algorithm).  At most ``max_faces`` solves of a <=4x4 matrix.

    Read ``significance``, not this, when deciding whether to trust a pairing:
    this number is a scale, not a probability, and it inflates with smoothing.
    """
    rows, cols = _solve(score)
    if not rows:
        return rows, cols, 0.0
    s0 = float(sum(score[r, c] for r, c in zip(rows, cols)))
    span = float(score.max()) - float(score.min())
    pen = float(score.min()) - (span + 1.0) * (len(rows) + 1)
    second = -np.inf
    for r, c in zip(rows, cols):
        m = score.copy()
        m[r, c] = pen
        rr, cc = _solve(m)
        if any(a == r and b == c for a, b in zip(rr, cc)):
            continue              # unavoidable pair: no alternative exists
        second = max(second, float(sum(score[a, b] for a, b in zip(rr, cc))))
    # second == -inf means the pairing is forced (a single stem, a single
    # track): there is no competing hypothesis to beat, so fall back to the raw
    # score rather than inventing an infinite margin.
    margin = s0 - second if np.isfinite(second) else s0
    return rows, cols, max(0.0, float(margin))


def pairing_significance(lips: np.ndarray, envs: np.ndarray, *,
                        n_shifts: int = 400,
                        seed: int = 0) -> tuple[float, float]:
    """``(p_value, null_agreement)`` for the pairing these signals imply.

    **Why a shift and not a shuffle.**  The question is whether the *temporal
    alignment* between a face and a stem carries speaker identity.  Rolling the
    lip block by a common lag destroys exactly that alignment while preserving
    every other property of the signals: each lip signal keeps its own
    autocorrelation, its marginal distribution, and its relationship to the
    other face's lip signal.  An i.i.d. shuffle would destroy the
    autocorrelation too, making the null far easier to beat and the p-value
    flattering -- smoothed envelopes are strongly autocorrelated, so a shuffled
    null is not the right comparison.

    A **common** lag, not one per track, for the same reason: both faces are
    displaced together, so lip-vs-lip structure survives and only the
    audio-visual link is broken.  That is the narrowest null that still answers
    the question, and therefore the hardest one to pass.

    ``p_value`` is the share of shifts whose margin reaches the observed margin,
    with the usual +1/+1 correction so it can never be 0 (400 shifts cannot
    justify p < 1/401).  This is the number to threshold.

    ``null_agreement`` is the share of shifts that reproduce the observed
    pairing, and it must be read carefully: for a 2x2 problem a null that has
    destroyed all alignment information picks either pairing about half the time,
    so **~0.5 is the null's own generic value and says nothing by itself** -- a
    correct pairing scores ~0.5 here too.  It is diagnostic only when it is
    *high*: an agreement near 1.0 means the pairing survives destroying the
    alignment, i.e. it is being decided by some shift-invariant asymmetry between
    the two rows rather than by who is speaking when.  Report it as the baseline
    it is; let ``p_value`` carry the verdict.

    Deterministic given the same signals: the shifts come from a fixed seed, so
    two runs of the same job report the same p rather than a number that wanders.
    """
    n_frames = lips.shape[1]
    if n_frames < 4 or lips.shape[0] == 0 or envs.shape[0] == 0:
        return 1.0, 1.0
    rows, cols, observed = pairing_margin((lips @ envs.T) / n_frames)
    pairing = sorted(zip(rows, cols))
    rng = np.random.default_rng(seed)
    # Exclude lag 0 and n (both are the identity) so no "shift" is the observed
    # arrangement re-scored -- that would count the alternative hypothesis as
    # evidence for the null.
    lags = rng.integers(1, n_frames, size=int(n_shifts))
    hits = 0
    agree = 0
    for lag in lags:
        r, c, m = pairing_margin((np.roll(lips, int(lag), axis=1) @ envs.T)
                                 / n_frames)
        hits += m >= observed
        agree += sorted(zip(r, c)) == pairing
    return (hits + 1) / (int(n_shifts) + 1), agree / max(int(n_shifts), 1)


@dataclass
class MatchResult:
    """What the matcher knows, including how much of it is trustworthy."""

    assignment: list[int]        # assignment[track] = stem index, or -1
    confidence: list[float]      # the pairing margin, shared by every track
    score: np.ndarray            # (n_tracks, n_stems) correlation matrix
    significance: float          # p-value of `confidence` under a shift null
    null_agreement: float        # share of shifts reproducing this pairing;
    #                              ~0.5 is the 2x2 null's own value, so only a
    #                              HIGH number is informative.  See
    #                              pairing_significance.

    def trustworthy(self, cfg) -> bool:
        """Whether the pairing beat its own null.

        Both conditions, because they fail independently: a large margin that a
        shifted lip signal reproduces just as easily is not evidence, and a
        small p-value on a margin below ``min_confidence`` is a well-resolved
        measurement of nothing.
        """
        return (self.significance <= getattr(cfg, "max_p_value", 0.05)
                and (max(self.confidence, default=0.0)
                     >= getattr(cfg, "min_confidence", 0.05)))


def match_stems_to_tracks(
    stems: np.ndarray,
    tracks: list[FaceTrack],
    *,
    sample_rate: int,
    video_fps: float,
    cfg,
) -> MatchResult:
    """Pair each face track with an audio stem, and say how much to believe it."""
    n_tracks, n_stems = len(tracks), stems.shape[0]
    if n_tracks == 0 or n_stems == 0:
        return MatchResult([], [], np.zeros((0, 0)), 1.0, 1.0)

    envs = [_zscore(energy_envelope(s, sample_rate, cfg.env_frame_ms, cfg.smooth_kernel))
            for s in stems]
    n_frames = min(len(e) for e in envs)
    env_fps = 1000.0 / cfg.env_frame_ms

    lips = []
    for t in tracks:
        r = resample_lip(t.lip, video_fps, n_frames, env_fps)
        # Correlate on the *derivative*: mouth opening/closing tracks speech
        # onsets far better than absolute aperture, and it is immune to a
        # speaker who simply rests with their mouth open -- the exact failure
        # that killed the absolute-position gate.
        lips.append(_zscore(np.abs(np.diff(r, prepend=r[:1]))))

    L = np.stack(lips) if lips else np.zeros((0, n_frames))
    E = np.stack([e[:n_frames] for e in envs])
    score = (L @ E.T) / n_frames if n_frames else np.zeros((n_tracks, n_stems))

    rows, cols, margin = pairing_margin(score)

    assignment = [-1] * n_tracks
    confidence = [0.0] * n_tracks
    for r, c in zip(rows, cols):
        assignment[r] = int(c)
        confidence[r] = margin

    # Any stem left unclaimed goes to the highest-scoring unassigned track, and
    # its confidence stays 0.0 deliberately: a backfilled pairing was not
    # chosen against a competing hypothesis, so it has no margin to report.
    # 0.0 < min_confidence, so the UI labels it low-confidence and the audio is
    # still playable.  See docs/REVIEW_CONTRACTS.md "Not a defect".
    for j in range(n_stems):
        if j not in assignment:
            free = [i for i in range(n_tracks) if assignment[i] == -1]
            if free:
                best = max(free, key=lambda i: score[i, j])
                assignment[best] = j

    p, agree = pairing_significance(
        L, E, n_shifts=getattr(cfg, "null_shifts", 400))
    return MatchResult(assignment, confidence, score, p, agree)
