"""The silence chain: Wiener masking (A3) and the Schmitt-trigger gate (A4).

This module is pure numpy/scipy.  No torch, no model, no I/O -- so it is fast to
test and it is the one part of the pipeline you can trust unit tests about.

Read ARCHITECTURE_V2.md §2 before changing any constant in here.  Three findings
are counter-intuitive and were arrived at by measurement:

  1. Mask the *estimates*, not the mixture.
  2. Never floor the mask.
  3. Exact zeros are only achievable in the time domain, after the ISTFT.

Deliberately NOT implemented: mixture-consistency projection.  It is a good
technique but it is wrong for whamr16k, whose targets are anechoic while its
input is reverberant+noisy -- so s1+s2 != mixture *by design*, and forcing
consistency re-injects the exact noise the model was trained to discard.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import ShortTimeFFT, get_window

__all__ = [
    "wiener_separate",
    "schmitt_gate",
    "apply_gate",
    "gate_mask",
    "resample_hold",
    "visual_activity",
    "veto_cost",
    "whisper_db",
    "measure_leakage_db",
    "silence_stats",
]


# --------------------------------------------------------------------------- #
# Stage A3 -- Wiener TF mask
# --------------------------------------------------------------------------- #

def _stft_engine(nfft: int, hop: int, fs: int) -> ShortTimeFFT:
    win = get_window("hann", nfft, fftbins=True)
    return ShortTimeFFT(win=win, hop=hop, fs=fs, fft_mode="onesided", scale_to="magnitude")


def wiener_separate(
    stems: np.ndarray,
    *,
    sample_rate: int,
    nfft: int = 512,
    hop: int = 128,
    exponent: float = 2.0,
    floor: float = 0.0,
    cepstral_order: int = 24,
) -> np.ndarray:
    """Cross-suppress a set of separated stems with a Wiener-style power ratio.

    Parameters
    ----------
    stems
        ``(n_sources, n_samples)`` float32.  These are *estimates* from the
        separator, already at their true relative levels (no per-source peak
        normalisation -- see Bug 1/Bug 2 in the architecture doc).
    exponent
        Mask sharpness ``p``.  p=1 measured -19.4 dB leakage, p=2 -35.8 dB,
        p=3 -41.1 dB but with 0.6 dB more distortion and audible musical noise.
        p=2 is the knee.
    floor
        Minimum mask value.  **Keep at 0.0.**  See module docstring.
    cepstral_order
        Liftering order for log-gain smoothing.  Suppresses musical noise
        without passing a constant fraction of the interferer.  0 disables.

    Returns
    -------
    ``(n_sources, n_samples)`` float32, same length as the input.
    """
    stems = np.atleast_2d(np.asarray(stems, dtype=np.float32))
    n_src, n_samples = stems.shape
    if n_src < 2:
        return stems.copy()

    sft = _stft_engine(nfft, hop, sample_rate)
    specs = np.stack([sft.stft(s) for s in stems])          # (n_src, freq, time)

    power = np.abs(specs) ** exponent
    total = power.sum(axis=0, keepdims=True) + 1e-12
    masks = power / total                                    # sums to 1 across sources

    if floor > 0.0:
        masks = np.maximum(masks, floor)
        masks /= masks.sum(axis=0, keepdims=True) + 1e-12

    if cepstral_order > 0:
        masks = np.stack([_smooth_log_gain(m, cepstral_order) for m in masks])

    out = np.empty_like(stems)
    for i in range(n_src):
        y = sft.istft(masks[i] * specs[i], k1=n_samples)
        out[i] = np.asarray(y[:n_samples], dtype=np.float32)
    return out


def _smooth_log_gain(mask: np.ndarray, order: int) -> np.ndarray:
    """Cepstrally smooth a TF gain surface.

    Operating on log-gain (rather than linear gain) keeps the smoothing
    perceptually uniform, and truncating the cepstrum removes the rapid
    frame-to-frame bin flicker that is heard as musical noise -- without
    lifting the floor, which is what a mask floor would do.
    """
    log_gain = np.log(np.maximum(mask, 1e-10))
    n_freq = log_gain.shape[0]
    # Real cepstrum along frequency, per frame.
    cep = np.fft.irfft(log_gain, axis=0, n=2 * (n_freq - 1))
    lifter = np.zeros(cep.shape[0], dtype=np.float64)
    lifter[:order] = 1.0
    lifter[-(order - 1):] = 1.0          # keep it symmetric -> real result
    cep *= lifter[:, None]
    smoothed = np.fft.rfft(cep, axis=0, n=2 * (n_freq - 1))[:n_freq].real
    return np.clip(np.exp(smoothed), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Strict isolation for per-face extractions (AV-TSE)
# --------------------------------------------------------------------------- #

def _local(z: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Box-average over a (bins, frames) neighbourhood; complex-safe."""
    from scipy.ndimage import uniform_filter
    if np.iscomplexobj(z):
        return uniform_filter(z.real, size) + 1j * uniform_filter(z.imag, size)
    return uniform_filter(z, size)


def strict_isolation(
    stems: np.ndarray,
    *,
    sample_rate: int,
    nfft: int = 512,
    hop: int = 128,
    smooth_bins: int = 5,
    smooth_frames: int = 9,
    ownership_power: float = 1.0,
    mask_exponent: float = 0.0,
) -> np.ndarray:
    """Push each face's channel towards containing only that face.

    Two steps, both needing every face's estimate, so this only runs with two
    or more faces:

    1. **Coherent leak cancellation.** What leaks into channel *i* from face
       *j* is a copy of face *j*'s waveform, so it is phase-coherent with
       channel *j*. Per STFT bin, regress channel *i* onto channel *j* over a
       small neighbourhood and subtract that component -- weighted by how much
       louder *j* is locally, ``w = 1 / (1 + (P_i / P_j) ** ownership_power)``,
       because a shared component belongs to whichever channel holds it
       loudest. Independent speech (face *i* talking over *j*) is not coherent
       with *j* and is left alone, which is why this costs far less voice
       quality than a power mask alone.
    2. **Optional ratio mask across faces** (``mask_exponent > 0``): each
       time-frequency cell goes (almost) entirely to the face that owns it.
       OFF by default -- see below.

    Measured on a ground-truth bench built from the team's phone clip (two
    similar voices reading over each other), channel-to-interferer ratio, and
    on the real clips as ECAPA same-instant cosine between channels (lower =
    more distinct) and Whisper WER against the read scripts:

                          bench SIR     real cosine   real WER (4 channels)
        refined AV-TSE    10.4 / 8.1    0.47 / 0.45   4.3 24.1 3.5 5.4
        + cancellation    16.3 / 13.4   0.31 / 0.30   15.2 27.6 5.8 6.5
        + mask p=4        18.4 / 15.9   0.25 / 0.31   19.6 32.8 23.3 9.8

    The mask buys ~2 dB on the bench and nothing reliable on the real clips,
    for a large loss of clarity, and on the talk show (distinct voices) it
    made the channels LESS distinct (0.13 -> 0.20) where cancellation alone
    is neutral (0.13). Cancelling twice, or over-subtracting 1.5x, also cost
    more WER than it bought. Hence cancellation only.
    """
    stems = np.atleast_2d(np.asarray(stems, dtype=np.float32))
    n_src, n = stems.shape
    if n_src < 2:
        return stems.copy()

    sft = _stft_engine(nfft, hop, sample_rate)
    size = (smooth_bins, smooth_frames)
    specs = [sft.stft(s).astype(np.complex64) for s in stems]
    power = [_local(np.abs(s) ** 2, size) + 1e-20 for s in specs]

    cleaned = []
    for i in range(n_src):
        y = specs[i].copy()
        for j in range(n_src):
            if j == i:
                continue
            a = _local(specs[i] * np.conj(specs[j]), size) / power[j]
            mag = np.abs(a)
            a = np.where(mag > 1.0, a / np.maximum(mag, 1e-12), a)
            w = 1.0 / (1.0 + (power[i] / power[j]) ** ownership_power)
            y -= w * a * specs[j]
        cleaned.append(y)

    if mask_exponent > 0:
        # Normalise by the per-cell max before raising to the exponent, so a
        # high exponent cannot underflow float32 in quiet cells (p=4 on raw
        # power did).
        pw = [np.abs(y) ** 2 for y in cleaned]
        ref = np.maximum.reduce(pw) + 1e-30
        pw = [(q / ref) ** (mask_exponent / 2.0) for q in pw]
        total = sum(pw)
        cleaned = [(pw[i] / total) * cleaned[i] for i in range(n_src)]

    out = np.empty_like(stems)
    for i in range(n_src):
        y = sft.istft(cleaned[i], k1=n)
        out[i] = np.asarray(y[:n], dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
# Visual voice activity -- the modality the acoustic gate cannot reach
# --------------------------------------------------------------------------- #

def resample_hold(signal, src_fps: float, n_out: int, out_fps: float,
                  max_hold_s: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Resample a NaN-punctuated per-frame signal onto another frame grid.

    NaN means "face not visible", which is a *real state* -- interpolating
    across it invents a mouth that was never on screen.  Short dropouts
    (<= ``max_hold_s``) hold the last valid value; longer ones go to 0 and are
    reported invalid.

    Returns ``(values, valid)``, both length ``n_out``.  The ``valid`` mask is
    the part callers must not ignore: a face that is absent is not a face that
    is silent, and treating the two alike is what would mute a speaker who
    briefly turned away.
    """
    a = np.asarray(signal, dtype=np.float64)
    if a.size == 0 or n_out <= 0:
        return np.zeros(max(n_out, 0)), np.zeros(max(n_out, 0), dtype=bool)

    ok = ~np.isnan(a)
    if not ok.any():
        return np.zeros(n_out), np.zeros(n_out, dtype=bool)

    max_hold = int(max_hold_s * src_fps)
    filled, valid = a.copy(), ok.copy()
    last, held = np.nan, 0
    for i, v in enumerate(a):
        if np.isnan(v):
            held += 1
            usable = (not np.isnan(last)) and held <= max_hold
            filled[i] = last if usable else 0.0
            valid[i] = usable
        else:
            last, held = v, 0
    filled = np.nan_to_num(filled, nan=0.0)

    t_src = np.arange(filled.size) / max(src_fps, 1e-6)
    t_out = np.arange(n_out) / max(out_fps, 1e-6)
    values = np.interp(t_out, t_src, filled, left=0.0, right=0.0)
    # Nearest-neighbour the validity: it is a state, not a quantity to blend.
    idx = np.clip(np.round(t_out * src_fps).astype(int), 0, valid.size - 1)
    out_valid = valid[idx]
    # Anything past the last source frame is extrapolation, not observation.
    # np.interp pads it with 0.0, which looks exactly like "mouth perfectly
    # still" -- so without this a video one frame shorter than its audio would
    # veto the tail of the clip and clip the last word.
    out_valid &= t_out <= t_src[-1] + 1.0 / max(src_fps, 1e-6)
    return values, out_valid


def visual_activity(lip, *, src_fps: float, n_frames: int, frame_ms: float,
                    smooth_ms: float = 90.0, hold_ms: float = 250.0,
                    min_dynamic_range: float = 1e-4
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-gate-frame visual voice activity in ``[0, 1]``, plus a validity mask.

    Built on the **derivative** of lip aperture, for the same reason the matcher
    is: mouth opening/closing tracks speech onsets, and it is immune to a
    speaker who simply rests with their mouth open -- the exact failure that
    killed the absolute-position gate (ARCHITECTURE_V2.md §1).

    Normalisation is per-track and relative (90th percentile of the track's own
    motion), because the feature is an area ratio whose absolute scale depends
    on face size and camera. A track whose motion never exceeds
    ``min_dynamic_range`` is reported **invalid rather than silent**: a static
    face is indistinguishable from a tracking failure that emits a constant,
    and guessing wrong in that direction mutes a real speaker.

    ``hold_ms`` dilates the result in **both** directions -- the visual analogue
    of ``min_off_ms``.  Without it the veto punches holes in continuous speech:
    the mouth is briefly still during sustained vowels and nasals, and at 25 fps
    a single still frame is 40 ms of audio.  Measured on the synthetic bench,
    dilation recovered target retention from 94.1% to 97.7% -- identical to the
    pure-acoustic gate -- at no cost to the silence.  Backward dilation is
    also physiologically right: lips begin moving *before* voicing (anticipatory
    coarticulation), so the veto must lift ahead of the onset.
    """
    out_fps = 1000.0 / frame_ms
    lip_f, valid = resample_hold(lip, src_fps, n_frames, out_fps)
    if not valid.any():
        return np.zeros(n_frames), np.zeros(n_frames, dtype=bool)

    d = np.abs(np.diff(lip_f, prepend=lip_f[:1]))
    d[~valid] = 0.0

    k = max(1, int(round(smooth_ms / frame_ms)))
    if k > 1:
        d = np.convolve(d, np.ones(k) / k, mode="same")

    ref = float(np.percentile(d[valid], 90)) if valid.any() else 0.0
    if ref < min_dynamic_range:
        # No usable motion anywhere -> abstain, do not veto.
        return np.zeros(n_frames), np.zeros(n_frames, dtype=bool)

    v = np.clip(d / ref, 0.0, 1.0)
    hold = int(round(hold_ms / frame_ms))
    if hold > 1:
        v = _dilate(v, hold)
    return v, valid


def _dilate(v: np.ndarray, width: int) -> np.ndarray:
    """Symmetric running maximum -- grey dilation over ``width`` frames."""
    try:
        from scipy.ndimage import maximum_filter1d
        return maximum_filter1d(v, size=width, mode="nearest")
    except Exception:                                   # pragma: no cover
        half = width // 2
        pad = np.pad(v, half, mode="edge")
        return np.stack([pad[i:i + v.size] for i in range(width)]).max(axis=0)


# --------------------------------------------------------------------------- #
# Stage A4 -- time-domain Schmitt-trigger gate
# --------------------------------------------------------------------------- #

def gate_mask(
    x: np.ndarray,
    *,
    sample_rate: int,
    open_db: float = -30.0,
    close_db: float = -40.0,
    ref_percentile: float = 95.0,
    min_on_ms: float = 60.0,
    min_off_ms: float = 30.0,
    ramp_ms: float = 10.0,
    frame_ms: float = 10.0,
    lookahead_ms: float = 20.0,
    visual: np.ndarray | None = None,
    visual_valid: np.ndarray | None = None,
    visual_veto_db: float = 25.0,
    visual_rival: np.ndarray | None = None,
    visual_rival_valid: np.ndarray | None = None,
    visual_symmetric: bool = False,
) -> np.ndarray:
    """Return a sample-rate gain envelope in ``[0, 1]`` with exact zeros.

    Gates on **the stem's own energy**, never on dominance over the other
    stem.  Measured dominance on target-active frames has a 10th percentile of
    -4.5 dB: the interferer is legitimately louder during 10% of the target's
    own speech, so every dominance-driven configuration cut 40-70% of genuine
    target frames.

    There is no exception to that, and one was attempted: a rival-relative leak
    *floor* (shut the gate where this stem sits more than M dB below the loudest
    rival) looked justified on paper and in an offline sweep, and still did
    nothing end-to-end.  See the refutation note further down in this function,
    at the point where it used to be applied.

    ``rel`` being intra-stem does mean the gate cannot distinguish "my quiet
    speech" from "their loud residual" -- both are simply energy in this stem --
    and the reported "if the other person is loud, the loud person's voice leaks"
    is real.  But it does not arrive through ``open_db``.  Measured on every
    leaking frame of the real clip, ZERO were opened by the open threshold; they
    are held open by the hysteresis band and ``min_off_ms``, or added by
    lookahead dilation.  The fix is there instead: see
    ``config.GateConfig.min_off_ms``.

    The hysteresis band (``open_db`` > ``close_db``) plus the min-duration
    state machine is what stops the gate chattering on breath and word gaps.

    **Audio-visual fusion.**  ``visual`` is a per-frame activity in ``[0, 1]``
    (see :func:`visual_activity`) for the speaker this stem belongs to, and
    ``visual_rival`` is the same signal for the most active *other* speaker.
    Both thresholds shift *up* where the rival's mouth is moving more than this
    stem's owner's:

        effective_open = open_db + visual_veto_db * clip(v_rival - v_self, 0, 1)

    This exists because of a measured wall.  A p=2 Wiener mask suppresses the
    interferer by roughly 3x its input leakage in dB, so the residual tracks
    the separator's quality; once it crosses ``open_db`` relative to the stem's
    own 95th-percentile energy, the gate opens on the *other* speaker and the
    exact-zero fraction collapses (measured: 100% at -12 dB input leakage, 70%
    at -9 dB, 23% at -6 dB).  No audio-domain threshold fixes this -- lowering
    ``open_db`` to chase the residual starts cutting genuine quiet speech,
    which is the failure that killed the earlier attempts.  Vision is
    independent of acoustic level, so it breaks the tie no amount of gain
    staging can.

    **Why the difference and not ``veto_db * (1 - v_self)``.**  The absolute
    form was tried first and measured *worse on both axes* than no fusion at
    all: on a real 640x360 clip it gave silence 54.2% / retain 90.7% at
    ``open_db = -30``, where the pure-acoustic gate at ``-18`` gave 57.8% /
    99.1%.  The reason is that ``v_self`` is not calibrated.  Its separability
    between "this speaker talking" and "the other one talking" is only
    AUC 0.597, and its mean on the target's own speech frames is 0.69 rather
    than ~1.0 -- so the absolute form charged the target 25*(1-0.69) = 7.7 dB
    on average, and 22.4 dB at the 90th percentile, *against its own words*.
    That lifts ``effective_open`` from -30 to -8 dB on a tenth of real speech.
    Only the 0.104 target-vs-interferer gap in ``v`` carried information; the
    rest was common-mode -- camera shake, global lighting, encoder noise and
    the detector's own box jitter, all of which move both faces together and
    therefore cancel in a difference.  Measured AUC for the differential form
    is 0.623 against 0.597, and it is symmetric across stems (0.623/0.623 vs
    0.607/0.586), so it is not merely favouring the better-tracked face.

    The differential form is also **retain-safe by construction**, which the
    absolute form was not: a stem is penalised only when some *other* face is
    visibly more active, so a speaker is never charged for their own stillness.
    A quiet-but-speaking talker with a barely-moving mouth keeps their acoustic
    gate untouched as long as nobody else is moving more.

    Both thresholds shift together, preserving the hysteresis band width.  The
    shift is zero wherever *either* face is invalid: an absent face abstains
    rather than vetoes, and a missing rival leaves nothing to compare against.
    With no rival supplied at all the fusion is inert and this is the pure
    acoustic gate -- the correct fallback for a single-speaker clip.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    n_frames = int(np.ceil(n / frame))

    padded = np.pad(x, (0, n_frames * frame - n))
    energy = (padded.reshape(n_frames, frame) ** 2).mean(axis=1)
    ldb = 10.0 * np.log10(energy + 1e-12)

    ref = float(np.percentile(ldb, ref_percentile))
    rel = ldb - ref

    open_th = np.full(n_frames, open_db, dtype=np.float64)
    close_th = np.full(n_frames, close_db, dtype=np.float64)
    if visual is not None and visual_veto_db:
        v = _fit(np.asarray(visual, dtype=np.float64), n_frames)
        if visual_valid is None:
            use = np.ones(n_frames, dtype=bool)
        else:
            use = _fit(np.asarray(visual_valid, dtype=np.float64), n_frames) > 0.5
        if visual_rival is None:
            # No rival face -> nothing to compare against.  This is the pure
            # acoustic gate; do not reach for the absolute form, it measured
            # worse than nothing (see the docstring).
            vv = 0.0
        else:
            vr = _fit(np.asarray(visual_rival, dtype=np.float64), n_frames)
            if visual_rival_valid is None:
                ruse = np.ones(n_frames, dtype=bool)
            else:
                ruse = _fit(np.asarray(visual_rival_valid, dtype=np.float64),
                            n_frames) > 0.5
            use &= ruse
            # clip(0, 1): the rival moving LESS than this speaker never
            # penalises this stem; only genuine excess motion vetoes.
            # visual_symmetric lifts that restriction and lets the shift go
            # NEGATIVE -- lowering the threshold where this speaker is visibly
            # the active one, which protects quiet speech instead of only
            # punishing loud residual.
            lo = -1.0 if visual_symmetric else 0.0
            vv = np.clip(vr - v, lo, 1.0)
        shift = np.where(use, visual_veto_db * vv, 0.0)
        open_th += shift
        close_th += shift

    # Rival-relative leak floor: TRIED AND REFUTED, do not resurrect.
    #
    # The idea was to raise open_th where this stem sits more than M dB below
    # the loudest rival -- a strictly weaker test than the dominance gating
    # refuted above (a floor asks the target not to lose badly, not to win), and
    # an offline sweep supported it: at M = 20, 64.2% of leak frames satisfied
    # the condition while only 0.37% of genuine target speech did.
    #
    # End-to-end it changed NOTHING (silence/retain identical to disabled, at
    # both M = 20 and with the visual veto active).  The reason is structural,
    # not a matter of picking a better M: attributing each leaking frame to the
    # mechanism that held the gate open gives ZERO frames opened by open_th.
    # They are held by hysteresis/min_off, or added by lookahead dilation.  A
    # constraint on the OPEN threshold cannot close a gate that open_th never
    # opened, so no value of M could have worked.
    #
    # Applying the same floor to close_th did move silence, but it was dominated
    # on both axes by plain min_off_ms -- which uses no rival information at all
    # (M = 15 gave 80.2%/92.5% where min_off_ms = 10 gave 82.4%/96.4%).  The
    # cross-stem coupling bought nothing over a dwell change.  See
    # config.GateConfig.min_off_ms for the attribution table and the fix.

    open_frames = max(1, int(round(min_on_ms / frame_ms)))
    close_frames = max(1, int(round(min_off_ms / frame_ms)))

    state = _run_schmitt(rel, open_th, close_th, open_frames, close_frames)

    # Lookahead: open the gate slightly before the frame that triggered it, so
    # the ramp lands in the silence *before* the onset rather than eating it.
    look = max(0, int(round(lookahead_ms / frame_ms)))
    state = _advance(state, look)

    env = np.repeat(state.astype(np.float32), frame)[:n]
    return _ramp(env, sample_rate, ramp_ms)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    """Length-match a per-frame auxiliary signal to the gate's frame count."""
    if a.size == n:
        return a
    if a.size == 0:
        return np.zeros(n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, a.size), a)


def _run_schmitt(
    rel_db: np.ndarray,
    open_db: np.ndarray,
    close_db: np.ndarray,
    min_on: int,
    min_off: int,
) -> np.ndarray:
    """Two-threshold state machine with minimum dwell times.

    ``open_db``/``close_db`` are per-frame arrays so the audio-visual fusion can
    raise them where the speaker is visibly not talking.
    """
    n = rel_db.size
    state = np.zeros(n, dtype=bool)
    is_open = False
    dwell = 0

    for i in range(n):
        v = rel_db[i]
        if is_open:
            # Only close after the signal has been below close_db long enough.
            if v < close_db[i]:
                dwell += 1
                if dwell >= min_off:
                    is_open = False
                    dwell = 0
            else:
                dwell = 0
        else:
            if v > open_db[i]:
                dwell += 1
                if dwell >= min_on:
                    is_open = True
                    # Retroactively open across the frames that justified it,
                    # otherwise the onset is truncated by min_on frames.
                    state[max(0, i - min_on + 1): i + 1] = True
                    dwell = 0
            else:
                dwell = 0
        state[i] = is_open or state[i]
    return state


def _advance(state: np.ndarray, look: int) -> np.ndarray:
    """Shift gate openings earlier by ``look`` frames (dilate to the left)."""
    if look <= 0:
        return state
    out = state.copy()
    for k in range(1, look + 1):
        out[:-k] |= state[k:]
    return out


def _ramp(env: np.ndarray, sample_rate: int, ramp_ms: float) -> np.ndarray:
    """Replace hard 0->1 / 1->0 edges with raised-cosine ramps.

    A step change in gain is a click.  A raised cosine over 10 ms is inaudible
    and still reaches *exactly* 0 and *exactly* 1 at its endpoints, which is
    what preserves the digital-silence guarantee.
    """
    L = max(1, int(round(sample_rate * ramp_ms / 1000.0)))
    if L <= 1 or env.size == 0:
        return env

    fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(L) / L)).astype(np.float32)
    fade_out = fade_in[::-1]

    out = env.copy()
    edges = np.flatnonzero(np.diff(env)) + 1
    for e in edges:
        if env[e] > env[e - 1]:                      # rising: ramp up *before* e
            s = max(0, e - L)
            out[s:e] = np.maximum(out[s:e], fade_in[L - (e - s):])
        else:                                        # falling: ramp down after e
            t = min(env.size, e + L)
            out[e:t] = np.maximum(out[e:t], fade_out[: t - e])
    return np.clip(out, 0.0, 1.0)


def schmitt_gate(x: np.ndarray, *, sample_rate: int, **kw) -> np.ndarray:
    """Convenience: compute the envelope and apply it."""
    return (x * gate_mask(x, sample_rate=sample_rate, **kw)).astype(np.float32)


def apply_gate(stems: np.ndarray, *, sample_rate: int, cfg,
               lips: list | None = None, video_fps: float = 25.0,
               carriers: np.ndarray | None = None) -> np.ndarray:
    """Gate every stem using a :class:`GateConfig`, with a cross-stem veto.

    ``carriers``, when given, is what the gate is APPLIED to; ``stems`` is only
    what it LISTENS to. The AV path decides open/closed on its unmasked stems
    and applies the result to the strictly-isolated ones: masking thins the
    target's own quiet syllables, so a gate listening to the masked audio cut
    2-17% of word time, where the same gate on the unmasked audio cut 0.05-1.6%.

    ``lips`` is an optional per-stem list of raw lip-aperture signals at
    ``video_fps`` (``None`` for a stem with no matched face).  When present and
    ``cfg.visual_fusion`` is on, each stem's gate is biased by how much its
    speaker's mouth is moving *relative to the other speakers'* -- see
    :func:`gate_mask` for why the comparison is relative and not absolute.

    Note the stems are therefore **not** gated independently: the decision for
    stem i reads the visual activity of every other stem.  They remain
    independent in the audio domain, which is the property that matters -- no
    stem's *energy* influences another's gate, because dominance-driven gating
    measured 40-70% loss of genuine target frames.

    A stem whose entry is ``None``, or whose track yields no usable motion, or
    which has no rival with usable motion, falls back to the pure-acoustic
    gate.  Degrading to the old behaviour is always preferable to muting a
    speaker on bad vision.
    """
    stems = np.atleast_2d(np.asarray(stems, dtype=np.float32))
    n_src = stems.shape[0]
    fuse = bool(getattr(cfg, "visual_fusion", False)) and lips is not None
    frame = max(1, int(round(sample_rate * cfg.frame_ms / 1000.0)))

    # Pass 1: every speaker's visual activity, on a COMMON frame grid.
    #
    # The veto is differential (see gate_mask), so a stem cannot be gated until
    # the other speakers' activity is known -- hence two passes rather than the
    # single loop this used to be.  The grid is sized from the longest stem so
    # that rival signals are directly comparable frame-for-frame; stems are
    # equal length in practice, and _fit absorbs any residual mismatch.
    n_common = int(np.ceil(max((s.size for s in stems), default=0) / frame))
    V: list[np.ndarray | None] = [None] * n_src
    OK: list[np.ndarray | None] = [None] * n_src
    for i, s in enumerate(stems):
        lip = lips[i] if fuse and i < len(lips) else None
        if lip is not None and s.size:
            V[i], OK[i] = visual_activity(
                lip,
                src_fps=video_fps,
                n_frames=n_common,
                frame_ms=cfg.frame_ms,
                smooth_ms=getattr(cfg, "visual_smooth_ms", 220.0),
                hold_ms=getattr(cfg, "visual_hold_ms", 120.0),
                min_dynamic_range=getattr(cfg, "visual_min_dynamic_range", 1e-4),
            )

    # Pass 2: gate each stem against the most active rival.
    out = []
    for i, s in enumerate(stems):
        vis, vis_ok = V[i], OK[i]
        rival = rival_ok = None
        others = [j for j in range(n_src) if j != i and V[j] is not None]
        if vis is not None and others:
            # max over the others: with >2 speakers the veto must answer "is
            # ANYONE else more active", not "is the average of them".
            rival = np.maximum.reduce([V[j] for j in others])
            rival_ok = np.logical_or.reduce(
                [OK[j] if OK[j] is not None else np.ones(n_common, dtype=bool)
                 for j in others])
        env = gate_mask(
            s,
            sample_rate=sample_rate,
            open_db=cfg.open_db,
            close_db=cfg.close_db,
            ref_percentile=cfg.ref_percentile,
            min_on_ms=cfg.min_on_ms,
            min_off_ms=cfg.min_off_ms,
            ramp_ms=cfg.ramp_ms,
            frame_ms=cfg.frame_ms,
            lookahead_ms=cfg.lookahead_ms,
            visual=vis,
            visual_valid=vis_ok,
            visual_veto_db=(getattr(cfg, "visual_veto_db", 0.0)
                            if vis is not None and rival is not None else 0.0),
            visual_rival=rival,
            visual_rival_valid=rival_ok,
            visual_symmetric=bool(getattr(cfg, "visual_symmetric", False)),
        )
        src = s if carriers is None else np.asarray(carriers[i], dtype=np.float32)
        out.append((src * env).astype(np.float32))
    return np.stack(out) if n_src else stems


# --------------------------------------------------------------------------- #
# Measurement -- so claims about silence are checkable, not asserted
# --------------------------------------------------------------------------- #

def veto_cost(acoustic: np.ndarray, fused: np.ndarray,
              other: np.ndarray | None = None, *, sample_rate: int,
              frame_ms: float = 10.0, ref_db: float = 20.0) -> float:
    """Fraction of the acoustic gate's output that the visual veto removed.

    **This is the runtime mis-assignment detector** the matcher never had.

    When a stem is paired with the right face, the veto only lifts thresholds
    where the speaker is already silent, so it removes ~nothing: measured 0.000
    on the synthetic bench at every leakage level where the separator still
    works.  When the pairing is wrong, vision and audio disagree constantly and
    the veto eats the target -- measured 0.36 with the two lip signals swapped.

    ``other`` removes a confound.  When the separator is bad, the *acoustic*
    gate itself opens on the other speaker's residual (that is the measured
    -9 dB knee), and the veto correctly kills it -- that is the fusion doing
    its job, but it shows up as "lost" acoustic output.  Counting only samples
    where ``other`` is weak (at least ``ref_db`` below its own 95th-percentile
    reference) leaves the regions where the veto can only be removing this
    stem's own speech -- which is precisely the mis-assignment signature.
    With the exclusion, correctly-paired veto stays ~0.000 even at -6 dB
    leakage, while the swap still fires (measured 0.36 vs alarm 0.25).

    Returns 0.0 when the acoustic gate passed nothing (no denominator).
    """
    a = np.asarray(acoustic) != 0.0
    f = np.asarray(fused) != 0.0

    if other is not None and a.shape == f.shape == np.asarray(other).shape:
        o = np.asarray(other, dtype=np.float32)
        frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
        n_frames = int(np.ceil(o.size / frame))
        padded = np.pad(o, (0, n_frames * frame - o.size))
        ldb = 10.0 * np.log10((padded.reshape(n_frames, frame) ** 2).mean(axis=1) + 1e-12)
        ref = float(np.percentile(ldb, 95.0))
        weak = np.repeat(ldb < ref - ref_db, frame)[: o.size]
        a = a & weak
        f = f | ~weak          # excluded regions are invisible to both counts

    kept = int(a.sum())
    if kept == 0:
        return 0.0
    lost = int(np.count_nonzero(a & ~f))
    return lost / kept


def whisper_db(out: np.ndarray, own: np.ndarray, other: np.ndarray, *,
               sample_rate: int, frame_ms: float = 10.0, active_db: float = 20.0
               ) -> float | None:
    """Residual level in ``out`` while the OTHER speaker holds the floor.

    This is the ghost whisper, and it is a different measurement from
    :func:`measure_leakage_db` -- which needs a ground-truth interferer signal
    and therefore only works on the synthetic bench.  At runtime no such signal
    exists: all we have is the output channel and the two raw stems.  So
    measure the thing the listener actually hears instead.

    Restricted to frames where ``other`` is clearly active and ``own`` is
    clearly not -- the frames where any energy in ``out`` is audible as another
    person whispering under the selected speaker.  Referenced to ``out``'s own
    speech level, so it reads as "the whisper is N dB below the voice you
    selected" and is level- and gain-independent.

    Returns ``None`` when no such frame exists (continuous overlap, or one
    speaker never talks): that is "not measurable", not "measured and perfect".
    Same contract as the other metrics here -- see :mod:`app.serialization`.
    """
    n = max(1, int(sample_rate * frame_ms / 1000.0))
    if out.size < n or own.size < n or other.size < n:
        return None

    def _fr(v: np.ndarray) -> np.ndarray:
        m = v.size // n * n
        return v[:m].reshape(-1, n)

    eo, eu, ex = (np.sqrt((_fr(v) ** 2).mean(1)) for v in (out, own, other))
    k = min(eo.size, eu.size, ex.size)
    eo, eu, ex = eo[:k], eu[:k], ex[:k]
    if k == 0:
        return None

    # "Clearly active" against each stem's OWN 95th percentile, so the test does
    # not care how the two speakers are balanced in the mix.
    thr = 10.0 ** (-active_db / 20.0)
    win = (ex > np.percentile(ex, 95) * thr) & ~(eu > np.percentile(eu, 95) * thr)
    if not win.any():
        return None

    leak = float(np.mean(eo[win] ** 2))
    if leak <= 0.0:
        return None                     # bit-exact silence: no whisper at all
    ref = float(np.mean(eo[eo > 0] ** 2)) + 1e-12
    return 10.0 * np.log10(leak / ref + 1e-20)


def measure_leakage_db(target: np.ndarray, other: np.ndarray, *, sample_rate: int
                       ) -> float | None:
    """Energy of ``other`` during ``target``-silent regions, relative to target.

    ``other`` must be the INTERFERER COMPONENT, not the neighbouring output
    channel.  Pass it the ground-truth interferer waveform, which in practice
    means this is a synthetic-bench metric (see scripts/check_fusion.py).

    Handing it the adjacent gated stem measures something that sounds similar
    and is nearly opposite: with correct turn-taking the other channel is *loud*
    exactly where this one is silent, so the result climbs toward 0 dB and above
    as separation gets BETTER.  A real job was reporting +4.4 dB from precisely
    that mistake while the gate was working.  For runtime use, call
    :func:`whisper_db`, which needs no ground truth.

    Lower is better.

    Returns ``None`` when there is no measurement window -- ``target`` never
    goes silent, or is empty -- because that is "not measurable", not "measured
    and perfect".  Reporting ``-inf`` here would read as the *best possible*
    leakage while actually meaning the opposite: the gate never closed.  It is
    also not JSON-representable; see :mod:`app.serialization`.
    """
    env = gate_mask(target, sample_rate=sample_rate)
    silent = env < 1e-6
    if not silent.any() or target.size == 0:
        return None
    leak = float(np.mean(other[silent] ** 2))
    ref = float(np.mean(target**2)) + 1e-12
    return 10.0 * np.log10(leak / ref + 1e-20)


def silence_stats(x: np.ndarray) -> dict[str, float | None]:
    """Fraction of samples that are *exactly* zero, and the residual floor.

    ``nonzero_floor_db`` is ``None`` for a channel with no nonzero samples at
    all.  That is the *success* case -- a perfectly muted channel -- and it has
    no floor to report rather than a floor of ``-inf``.  Callers that render
    this must handle ``None``; the UI prints it as "none (silent)".
    """
    n = x.size or 1
    zeros = int(np.count_nonzero(x == 0.0))
    nz = x[x != 0.0]
    floor = float(20.0 * np.log10(np.abs(nz).min() + 1e-20)) if nz.size else None
    return {
        "exact_zero_fraction": zeros / n,
        "nonzero_floor_db": floor,
        "peak": float(np.abs(x).max()) if x.size else 0.0,
    }
