"""Identity verification: keep the voice that belongs to the face.

AV-TSE is an extractor, not a detector. Conditioned on a face, it emits a
voice on every frame -- including frames where that person is not speaking.
If someone who is never on screen is talking then, the model has nothing
else to reach for, and that off-screen person comes out instead. Measured on
the reference clip, this costs face 0 about 8 of 32 seconds.

No audio-only energy statistic can catch it. The intruder is a loud, clean,
perfectly good voice; ARCHITECTURE_V3 section 4 proposed a stream-to-mixture
energy ratio, and that test passes happily on an intruder. What distinguishes
them is *who they are*, so this module asks that question directly.

The discriminative score is deliberately relative::

    s(w) = cos(w, ref_mine) - max over other refs of cos(w, ref_other)

Absolute similarity to one's own reference does not separate: measured on the
reference clip it gives a *negative* margin (own 5th percentile 0.371 against
intruder 95th percentile 0.600) because the on-screen man and the off-screen
intruder sound alike -- their centroids sit at cosine +0.74. The difference
separates on the same data: margin +0.185, AUC 1.000. Comparing two similar
voices to each other is a much easier question than describing either one in
the absolute.

Read those last two numbers narrowly. The references come from the clustering
that also supplies the labels, so a perfect AUC is what a self-consistent
clustering reports whether or not it is *right* -- it says the decision
boundary is crisp, not that the identities are real. The evidence for the
latter is external and lives elsewhere: the mixture attestation in
``_choose_k`` (a cluster absent from the room is not a person), and face 1 of
the reference clip as a negative control -- a genuinely single-speaker stem,
which comes back purity 1.000 with 0% muted rather than being carved up to
justify the machinery.

"Other refs" spans both unclaimed identities (intruders) and the other tracked
faces, so one formula covers off-screen intrusion and face-to-face bleed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "IdentityConfig",
    "IdentityPlan",
    "covered",
    "enroll",
    "embed",
    "gate_intruders",
    "score_samples",
    "load_encoder",
]

#: speechbrain ECAPA-TDNN, 192-d, trained on VoxCeleb1+2. Non-gated.
ENCODER_REPO = "speechbrain/spkrec-ecapa-voxceleb"

_LOCK = threading.Lock()
_CACHE: dict = {}


@dataclass(frozen=True)
class IdentityConfig:
    """Measured on ``runs/862bd92a01ac``; see ``scripts/fit_identity.py``."""

    #: Embedding window. 1.0 s was measured to be too short -- even face 1,
    #: which holds a single speaker for the whole clip, split 91/34 under
    #: 2-means at that length, i.e. the embedding noise exceeded the between-
    #: speaker distance. 2.0 s is the shortest that was stable.
    win_s: float = 2.0
    hop_s: float = 0.5

    #: Windows quieter than (this stem's loud level - gate_db) are not embedded:
    #: an embedding of near-silence describes the room, not a person, and
    #: pooling those invents a spurious "speaker" made of noise.
    gate_db: float = 12.0

    #: Schmitt trigger on s(w). Zero is the natural decision point (equally
    #: close to two references), and the measured distributions clear it by a
    #: wide margin on both sides, so the hysteresis band is set at +-0.05
    #: rather than fitted tightly to one clip.
    on_score: float = 0.05
    off_score: float = -0.05

    #: A verdict must persist this long to take effect. Stops a single noisy
    #: window from punching a hole in an otherwise good stretch.
    min_on_s: float = 0.40
    min_off_s: float = 0.40

    #: A cluster holding less than this share of windows is not a speaker --
    #: applied both to the pooled windows and, decisively, to the mixture's.
    #: See ``_choose_k``: the reference clip's spurious cluster holds 0% of
    #: the mixture against 5% for the real off-screen speaker, so this is not
    #: a tight fit.
    min_share: float = 0.04

    #: ...but a share is a fraction, and a fraction of a small number stops
    #: being evidence. A 13.7 s clip yields 25 mixture windows, where one
    #: window IS 4% -- so the share test above passed a cluster attested by a
    #: single observation, and k ran to its ceiling of 6 on a 3-face clip
    #: (purity 0.36, two thirds of a face muted). An absolute floor is what
    #: the conservation argument actually needed: one window is one
    #: observation and cannot be told from an embedding excursion, whereas at
    #: hop_s=0.5 three windows span a real second and a half of the room.
    #: Both tests must pass, so neither a short clip nor a long one can let a
    #: cluster in on less evidence than the other would require.
    min_mix_windows: int = 3
    #: Identities beyond the tracked faces to look for.
    max_extra: int = 3


@dataclass
class IdentityPlan:
    """Per-face reference embeddings plus everything the report needs."""

    refs: np.ndarray                       # (n_faces, D) unit vectors
    others: np.ndarray                     # (n_other, D) intruders + rivals
    n_clusters: int = 0
    #: Per-face share of its own claimed cluster, for the preflight report.
    purity: list[float] = field(default_factory=list)
    #: Cosine between each face's reference and its nearest rival. Small means
    #: the two voices genuinely sound alike and the gate is doing real work.
    nearest_rival: list[float] = field(default_factory=list)

    def score(self, emb: np.ndarray, face: int) -> np.ndarray:
        """s(w) for one face, given (N, D) unit-norm window embeddings."""
        mine = emb @ self.refs[face]
        rivals = np.concatenate(
            [self.others, np.delete(self.refs, face, axis=0)], axis=0
        ) if len(self.refs) > 1 else self.others
        if len(rivals) == 0:
            # Nothing to be confused with. Report a constant pass rather than
            # inventing a threshold the caller would then gate on.
            return np.full(len(emb), 1.0)
        return mine - (emb @ rivals.T).max(axis=1)


# --------------------------------------------------------------------------- #

def load_encoder(device: str = "cpu"):
    """Load ECAPA once per (device). Thread-safe; mirrors avtse.load_model."""
    from pathlib import Path
    import os

    key = str(device)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        home = os.environ.get("HF_HOME") or str(
            Path(__file__).resolve().parents[1] / ".cache" / "hf")
        # speechbrain wants "cuda:0"; a bare "cuda" trips its device parser.
        dev = "cuda:0" if device == "cuda" else str(device)
        # COPY: the default SYMLINK needs Developer Mode on Windows (WinError
        # 1314).  See SepformerSeparator.load.
        enc = EncoderClassifier.from_hparams(
            source=ENCODER_REPO,
            savedir=str(Path(home) / "ecapa"),
            run_opts={"device": dev},
            local_strategy=LocalStrategy.COPY,
        )
        _CACHE[key] = enc
        return enc


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def embed(
    x: np.ndarray,
    sr: int,
    cfg: IdentityConfig,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the loud windows of one signal. Returns ``(t_centres, (N, D))``."""
    import torch

    enc = load_encoder(device)
    n, h = int(cfg.win_s * sr), int(cfg.hop_s * sr)
    starts = list(range(0, max(len(x) - n, 1), h))
    # Anchor a final window flush with the end. The stride leaves up to one
    # hop uncovered, and ``gate_intruders`` closes whatever it has not
    # verified -- so without this the last fraction of a second is muted on
    # every stem, for no reason but arithmetic.
    if len(x) > n and starts[-1] != len(x) - n:
        starts.append(len(x) - n)
    segs = [x[s : s + n] for s in starts]
    if not segs:
        return np.zeros(0), np.zeros((0, 192), dtype=np.float32)

    loud = float(np.percentile([_rms_db(s) for s in segs], 90))
    keep = [i for i, s in enumerate(segs) if _rms_db(s) > loud - cfg.gate_db]
    if not keep:
        return np.zeros(0), np.zeros((0, 192), dtype=np.float32)

    batch = torch.from_numpy(np.stack([segs[i] for i in keep]).astype(np.float32))
    with torch.no_grad():
        e = enc.encode_batch(batch.to(next(enc.mods.parameters()).device))
    e = e.squeeze(1).cpu().numpy()
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    t = np.array([(starts[i] + n / 2) / sr for i in keep])
    return t, e


def _kmeans(X: np.ndarray, k: int, iters: int = 80, seed: int = 0):
    """Cosine k-means on unit vectors, k-means++ seeded.

    The seeding matters here rather than being boilerplate: a rare intruder
    holding a handful of windows is exactly what we are hunting, and random
    initialisation routinely swallows it into a neighbouring cluster.
    """
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        C.append(X[np.argmax(1.0 - (X @ np.stack(C).T).max(axis=1))])
    C = np.stack(C)
    lab = np.full(len(X), -1)
    for _ in range(iters):
        new = np.argmax(X @ C.T, axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].mean(axis=0)
                C[j] = c / (np.linalg.norm(c) + 1e-12)
    return lab, C


def _choose_k(X: np.ndarray, n_faces: int, cfg: IdentityConfig,
              mix_emb: np.ndarray | None = None):
    """Grow k while every identity is still attested in the mixture.

    The mixture is the arbiter because it is the only recording of the room;
    the stems are model outputs and can be split by artifacts. A cluster that
    owns part of a stem but no part of the mixture cannot be a person, since
    the stem was derived from the mixture. On the reference clip that test is
    not close:

        k=3   stem/mixture shares  25/25  70/70   5/5     -- all attested
        k=4                        18/20  70/70  48/0    10/10  <- c2 absent

    Cluster c2 holds 48% of face 0's stem and nothing at all in the room. It
    is face 0's own voice in an acoustic state the extractor renders
    differently, and adopting it as a rival is what made the gate mute 55% of
    face 0 at k=4 and 38% of a single-speaker stem at k=5.

    The share test alone was not enough, and a second clip showed why. A
    13.7 s clip gives 25 mixture windows, so one window is exactly 4% and the
    ``<`` comparison let it through; k climbed to its ceiling of 6 and the
    gate muted 43/67/61% of three faces. ``min_mix_windows`` adds the absolute
    floor the argument was always relying on implicitly -- a cluster must be
    seen in the room more than once to count as a person in it.

    Two alternatives were measured and rejected. A threshold on centroid
    cosine has to fit inside the gap between two different men (+0.741) and
    two halves of one man (+0.785) -- 0.044 wide, on one clip. Split-half
    stability is degenerate: it scores k=2 at 1.000 and would pick the k that
    detects no intruder at all.

    Without a mixture there is no arbiter, so k stays at ``n_faces``: an
    unverifiable third identity is not evidence of a third person.
    """
    base = _kmeans(X, max(n_faces, 1))
    if mix_emb is None or len(mix_emb) == 0:
        return base
    best = base
    for k in range(max(n_faces, 1) + 1, n_faces + cfg.max_extra + 1):
        if k > len(X):
            break
        lab, C = _kmeans(X, k)
        if np.bincount(lab, minlength=k).min() / len(X) < cfg.min_share:
            break
        # Attestation is judged on the mixture alone, not on the pooled data,
        # so a cluster cannot be kept alive by the stems that invented it.
        # Counted as well as shared: see ``min_mix_windows``. On a short clip
        # the share alone degenerates to "at least one window", which is not
        # a conservation argument, it is a rounding result.
        mix_count = np.bincount(np.argmax(mix_emb @ C.T, axis=1), minlength=k)
        mix_share = mix_count / len(mix_emb)
        log.debug("identity: k=%d mixture shares %s counts %s", k,
                  [f"{s:.2f}" for s in mix_share], list(mix_count))
        if mix_share.min() < cfg.min_share or mix_count.min() < cfg.min_mix_windows:
            break
        best = (lab, C)
    return best


def enroll(
    stems: Sequence[np.ndarray],
    sr: int,
    cfg: IdentityConfig | None = None,
    *,
    mixture: np.ndarray | None = None,
    device: str = "cpu",
    k: int | None = None,
    cached: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[IdentityPlan, list[tuple[np.ndarray, np.ndarray]]]:
    """Work out who is in these stems, and which of them belongs to each face.

    Clusters every stem's windows *jointly* rather than per stem. Per-stem
    clustering was measured to be unstable -- the split moved from t=10-18 to
    t=2-6 as the window length changed, and a clean single-speaker stem split
    anyway. Pooling gives every cluster more evidence and lets mutual
    exclusivity be enforced, so two faces cannot both claim one voice.

    Args:
        stems: one ``(n_samples,)`` stem per face, in face order.
        mixture: the input audio. Included in the clustering but claimed by no
            face, and used by :func:`_choose_k` as the arbiter of which
            identities are real. Intruder detection is off without it.
        k: force the number of identities instead of choosing it. For fitting
            and diagnostics only -- production should let ``_choose_k`` decide.
        cached: pre-computed ``(t, embeddings)`` per stem, then mixture. Lets a
            sweep re-cluster without paying for the encoder each time.

    Returns:
        ``(plan, per_face_windows)`` where ``per_face_windows[i]`` is the
        ``(t, embeddings)`` pair for face *i*, reusable for scoring without
        re-embedding.
    """
    from scipy.optimize import linear_sum_assignment

    cfg = cfg or IdentityConfig()
    if cached is not None:
        per_face = list(cached[: len(stems)])
        mix_emb = cached[len(stems)][1] if len(cached) > len(stems) else None
    else:
        per_face = [embed(np.asarray(s, dtype=np.float32), sr, cfg, device)
                    for s in stems]
        mix_emb = None
        if mixture is not None:
            _, mix_emb = embed(np.asarray(mixture, dtype=np.float32), sr, cfg, device)

    pool = [e for _, e in per_face if len(e)]
    owner = [np.full(len(e), i) for i, (_, e) in enumerate(per_face) if len(e)]
    if mix_emb is not None and len(mix_emb):
        pool.append(mix_emb)
        owner.append(np.full(len(mix_emb), -1))
    if not pool:
        raise ValueError("no audible windows in any stem")

    X = np.concatenate(pool)
    owner = np.concatenate(owner)
    lab, C = _kmeans(X, k) if k else _choose_k(X, len(stems), cfg, mix_emb)
    k = len(C)

    # Each face claims the cluster it dominates, exclusively.
    share = np.stack([
        np.bincount(lab[owner == i], minlength=k) / max((owner == i).sum(), 1)
        for i in range(len(stems))
    ])
    rows, cols = linear_sum_assignment(-share)
    claim = {int(r): int(c) for r, c in zip(rows, cols)}
    unclaimed = [j for j in range(k) if j not in claim.values()]

    refs = np.stack([C[claim[i]] for i in range(len(stems))])
    others = C[unclaimed] if unclaimed else np.zeros((0, C.shape[1]))
    purity = [float(share[i, claim[i]]) for i in range(len(stems))]
    rival = []
    for i in range(len(stems)):
        pool_r = np.concatenate([others, np.delete(refs, i, axis=0)], axis=0) \
            if len(refs) > 1 else others
        rival.append(float((pool_r @ refs[i]).max()) if len(pool_r) else 0.0)

    log.info("identity: k=%d, claims=%s, unclaimed=%s, purity=%s",
             k, claim, unclaimed, [f"{p:.2f}" for p in purity])
    plan = IdentityPlan(refs=refs, others=others, n_clusters=k,
                        purity=purity, nearest_rival=rival)
    return plan, per_face


def score_samples(
    t: np.ndarray,
    s: np.ndarray,
    n_samples: int,
    sr: int,
) -> np.ndarray:
    """Lift a per-window score onto the sample grid.

    Linear interpolation, held flat outside the window centres. Windows
    overlap by design, so the score is already smooth in time; interpolating
    keeps the gate's ramps from landing on a staircase.
    """
    if len(t) == 0:
        return np.zeros(n_samples)
    grid = np.arange(n_samples) / sr
    return np.interp(grid, t, s, left=s[0], right=s[-1])


def covered(t: np.ndarray, n_samples: int, sr: int, win_s: float) -> np.ndarray:
    """Which samples lie within half a window of a verified window centre.

    Everything outside this was skipped by the loudness gate, so no identity
    was ever measured there. Callers need it for two different reasons -- the
    gate closes it, and the report must not count it as a verdict -- so it
    lives here rather than being rewritten at each site.
    """
    if len(t) == 0:
        return np.zeros(n_samples, dtype=bool)
    grid = np.arange(n_samples) / sr
    j = np.searchsorted(t, grid)
    lo, hi = np.clip(j - 1, 0, len(t) - 1), np.clip(j, 0, len(t) - 1)
    return np.minimum(np.abs(grid - t[lo]), np.abs(grid - t[hi])) <= win_s / 2


def gate_intruders(
    stem: np.ndarray,
    t: np.ndarray,
    s: np.ndarray,
    sr: int,
    cfg: IdentityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mute the stretches of ``stem`` that are somebody else.

    Reuses ``dsp._run_schmitt`` and ``dsp._ramp`` rather than reimplementing
    them: the dwell-time and raised-cosine logic there is what makes the
    closed state reach *exactly* 0.0 without clicking, and having two copies
    of that guarantee is how one of them drifts.

    The state machine runs at window rate, not sample rate. An identity
    verdict is not meaningful at finer resolution than the window it was
    measured over, and pretending otherwise would just let interpolation
    noise chatter the gate.

    Returns:
        ``(gated, envelope)``. The envelope is exactly 0.0 where muted and
        exactly 1.0 where open, with raised-cosine ramps between.
    """
    from .dsp import _ramp, _run_schmitt

    cfg = cfg or IdentityConfig()
    stem = np.asarray(stem, dtype=np.float32)
    if len(t) == 0:
        # Nothing audible was ever verified. Silence is the safe answer: an
        # unverified stem is exactly the case this module exists to catch.
        return np.zeros_like(stem), np.zeros(len(stem), dtype=np.float32)

    per_win = max(cfg.hop_s, 1e-6)
    state = _run_schmitt(
        np.asarray(s, dtype=np.float64),
        np.full(len(s), cfg.on_score),
        np.full(len(s), cfg.off_score),
        max(1, int(round(cfg.min_on_s / per_win))),
        max(1, int(round(cfg.min_off_s / per_win))),
    )

    env = score_samples(t, state.astype(np.float64), len(stem), sr) > 0.5

    # Only audio near a verified window may pass. Elsewhere nothing was
    # measured -- the loudness gate skipped it -- and ``score_samples`` would
    # otherwise hold the nearest verdict outward, reporting an extrapolation
    # as a decision. It can extrapolate in either direction: on the reference
    # clip face 0's first verified window is at 3.5 s and negative, so the
    # first 3.5 s inherited a mute; had it been positive, 3.5 s of unverified
    # residue would have been passed instead. Closing is the answer that
    # matches the rest of the module, and it costs nothing real: an
    # unverified region is by construction >= gate_db below the stem's own
    # speech, and ``stems_raw`` keeps it ungated for the metrics.
    ok = covered(t, len(stem), sr, cfg.win_s)

    env = _ramp((env & ok).astype(np.float32), sr, ramp_ms=20.0)
    return (stem * env).astype(np.float32), env

