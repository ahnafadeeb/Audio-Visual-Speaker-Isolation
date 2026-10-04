"""Central configuration.

Every tunable number in the pipeline lives here.  The DSP constants in
``GateConfig`` came out of synthetic measurements (see ARCHITECTURE_V2.md §2);
their *relative orderings* transfer to real audio but the **absolute values must
be re-fitted** against real SepFormer output during Step 1 of the build order.
Treat them as initial conditions, not constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
CACHE_DIR = PROJECT_ROOT / ".cache"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
#: Fine-tuned AV-TSE adapters (python -m app.adapt) and the noise they train
#: against. ``active.txt`` names the one new jobs use.
ADAPTERS_DIR = PROJECT_ROOT / "adapters"

# Keep the multi-GB HF model tree off C: and inside the project.  Must be set
# before huggingface_hub / speechbrain are imported, hence the module-level
# placement -- app.main imports this first.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
# The Windows CUDA allocator does not implement expandable segments; setting it
# there only earns a UserWarning on every run.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000       # sepformer-whamr16k is a 16 kHz model
    n_speakers: int = 2

    # STFT for the Wiener stage.  nfft=512 / hop=128 (75% overlap) with a Hann
    # window is COLA/NOLA-satisfying and round-trips to ~1e-6.
    nfft: int = 512
    hop: int = 128

    # Chunked inference.  Not a memory necessity on an 8 GB card for short
    # clips -- it exists so that clip duration stops being a variable that can
    # OOM you.
    chunk_s: float = 10.0
    overlap_s: float = 2.0

    # --- VRAM-adaptive chunking, fitted to MEASURED data ------------------- #
    #
    # These come from scripts/bench_vram.py on a GTX 1050 Ti (4 GB, 3.28 GB
    # free), running real separate_batch calls:
    #
    #     chunk_s   1     2     3     4     5     6     8    10    12
    #     peak MB  257   383   509   632   758   883  1134  1385  1636
    #
    # >>> The scaling is LINEAR in this range: chunk_s^0.93 measured. <<<
    #
    # This corrects an assumption that was wrong and load-bearing.  SepFormer's
    # dual-path stack costs S*K^2 (intra-segment attention) + K*S^2 (inter-
    # segment attention), where K is a FIXED hyperparameter (250) and S is the
    # segment count, which grows with chunk_s.  The K*S^2 term is the quadratic
    # one, but it only dominates once S > K -- about 16 s of 16 kHz audio.
    # Below that the linear S*K^2 term rules, which is why doubling chunk_s here
    # doubles memory rather than quadrupling it.
    #
    # Why the correction matters beyond tidiness: believing it was quadratic
    # argued for the SMALLEST chunk that works.  It is the opposite.  Every
    # chunk boundary is a _align_permutation decision, and a wrong one swaps the
    # speakers for the rest of the clip -- audibly identical to the bleed-through
    # Objective A exists to eliminate.  So the correct chunk_s is the LARGEST
    # that safely fits, and over-shrinking buys nothing while adding risk.
    #
    # Fit: peak_MB ~= 133 + 126 * chunk_s   (108 MB weights + ~25 MB fixed)
    vram_fixed_mb: float = 133.0
    vram_per_chunk_s_mb: float = 126.0
    # Fraction of FREE VRAM we are willing to occupy.  0.5 because the benchmark
    # had the card nearly to itself and a live demo does not: the browser holds a
    # decoded video, the compositor holds the desktop, and the allocator
    # fragments over a long job.
    vram_budget_fraction: float = 0.5
    # Never go below this.  Under ~2 s the chunk approaches SepFormer's own
    # 250-frame dual-path segment and both quality and boundary count degrade
    # faster than the memory saving is worth.
    min_chunk_s: float = 2.0

    def resolve_chunk_s(self, device: str = "cuda") -> float:
        """The largest chunk length that fits the card actually present.

        Resolved at call time rather than baked in, so the SAME checkout runs on
        a 4 GB Pascal and an 8 GB Ada laptop with no edit to undo before a demo.

        Measured against FREE memory, not total: on a machine whose display runs
        off the same GPU several hundred MB are gone before the process starts,
        and that is exactly the margin an OOM eats.

        Never returns MORE than the configured ``chunk_s`` -- a big card gets the
        tuned value, not an extrapolation into the region where the S^2 term
        starts to bite.
        """
        if device != "cuda":
            return self.chunk_s
        try:
            import torch
            if not torch.cuda.is_available():
                return self.chunk_s
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            # Never let a probe failure block a job; the default is the safe
            # value on the machine this ships to.
            return self.chunk_s

        budget_mb = (free / 2**20) * self.vram_budget_fraction
        fits = (budget_mb - self.vram_fixed_mb) / self.vram_per_chunk_s_mb
        # Floor first, then cap: a card too small for even min_chunk_s still gets
        # min_chunk_s and is allowed to OOM honestly, where the retry loop in
        # SepformerSeparator.separate can see it and react.
        return float(min(self.chunk_s, max(self.min_chunk_s, fits)))


# --------------------------------------------------------------------------- #
# The silence chain
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GateConfig:
    """Stage A3 (Wiener) + Stage A4 (Schmitt gate)."""

    # --- A3: Wiener TF mask ------------------------------------------------ #
    # Applied to the ESTIMATES, not the mixture.  Mixture-domain masking
    # measured 12-17 dB worse on leakage because it throws away SepFormer's
    # work and degenerates into a crude magnitude comparator.
    mask_exponent: float = 2.0      # p=2 is the measured knee
    # NEVER raise this.  A conventional -20 dB floor collapsed leakage
    # suppression from -35.8 dB to -1.6 dB: a floor passes a fixed fraction of
    # the interferer at all times, which *is* the ghost whisper.
    mask_floor: float = 0.0
    # Musical noise is controlled by smoothing the log-gain instead of by a
    # floor.  Cepstral liftering order; 0 disables.
    cepstral_smooth_order: int = 24

    # --- A4: time-domain Schmitt trigger ----------------------------------- #
    # Thresholds are relative to the 95th-percentile frame energy of *this*
    # stem, so they are level-independent.
    # Re-fitted on real SepFormer residual by scripts/fit_gate.py; the old
    # -30/-40 came off the synthetic bench, where "silence" was a literal zero
    # region.  Real residual is not like that: at -30 dB, 75% of BOTH stems'
    # frames sat above the threshold, the gate held both channels open through
    # 55% of the clip, and the interferer stayed audible at a median -33.7 dB.
    # That is the ghost whisper -- a gate that never closed.
    #
    # Measured trade curve at visual_veto_db = 4 (silence / retain):
    #     -30  29.7% / 99.5%      -20  55.7% / 98.8%
    #     -25  42.0% / 99.3%      -18  61.2% / 98.4%   <- most silence
    #     -22  45.6% / 99.3%      -16  67.1% / 90.2%   <- cliff
    #
    # -18 maximises silence, but it is one grid step from the retain cliff in
    # BOTH open_db and visual_veto_db, and the fit rests on a single clip (the
    # two available jobs are byte-identical, so there was no held-out clip to
    # confirm where the cliff sits).  -20 gives up 5.5 points of silence to buy
    # a step of margin on the axis that fails audibly -- a clipped word is a
    # worse demo defect than a quieter whisper.  Re-run fit_gate.py on a second
    # real clip before moving to -18.
    # Fitted on a real clip by scripts/fit_gate.py; see the band note below,
    # which is the part that actually mattered.  -20 rather than the sweep's
    # silence-maximising -18 because scripts/diag_sir.py measured the target's
    # own quiet speech at p10 = -16.2 dB relative to its p95: -18 leaves 1.8 dB
    # of margin before the gate starts cutting real words, -20 leaves 3.8 dB,
    # and the whole fit rests on ONE clip.  The two settings differ by 3.5
    # points of silence and nothing in retain.
    open_db: float = -20.0          # above this -> speech
    # The HYSTERESIS BAND is the load-bearing number, not open_db.
    #
    # This was -40 (a 10 dB band) carried over from the synthetic fit, and it
    # was the single biggest cause of the ghost whisper -- bigger than open_db,
    # which is where three rounds of tuning went.  diag_sir.py measured the
    # interferer's residual inside a stem spanning p50 -31.4 to p90 -21.5 dB
    # relative to that stem's own p95.  A 10 dB band starting at -20 therefore
    # covers the residual distribution almost exactly: the gate opens on the
    # residual's loud tail and hysteresis then holds it open across the bulk of
    # the residual, so silence stalled near 55% even with open_db sitting
    # inside the measured 5.3 dB of headroom.  Narrowing the band lets the gate
    # fall shut again between residual peaks.
    #
    # Measured at open_db = -20 (silence / retain):
    #     band 10 dB -> 55.7% / 98.8%      band  4 dB -> 64.6% / 97.9%
    #     band  6 dB -> 61.6% / 97.9%      band  3 dB -> 66.0% / 97.9%
    #                                      band  2 dB -> 67.7% / 97.9%
    # 8-12 points of silence for ~1 point of retain, a far better exchange rate
    # than open_db offers anywhere on its range.
    #
    # 3 dB rather than 2 because a narrow band makes the gate quicker to CLOSE,
    # and the risk that carries is on a clip with more dynamic speech than this
    # one -- retain is flat at 97.9% across 2/3/4 dB here, so the extra dB is
    # bought for nothing measurable.
    #
    # Chatter is not the constraint a narrow band would suggest: frame-level
    # gate transitions are 2.8-3.9/s against the 22.2/s ceiling that
    # min_on_ms + min_off_ms already imposes, and bands of 2, 3 and 4 dB are
    # temporally identical.  Those duration limits, not the band width, are
    # what suppress chatter here.
    close_db: float = -23.0         # below this -> silence (hysteresis band)
    ref_percentile: float = 95.0
    min_on_ms: float = 60.0         # min duration of an open state
    # min_off_ms and lookahead_ms are where the ghost whisper actually lived.
    #
    # Reported as "if the other non selected person is loud then the loud
    # person's voice leaks" plus "the selected speaker's sound is a bit
    # suppressed in the process".  The obvious reading -- the residual crosses
    # open_db -- is WRONG, and measuring it is what found the real cause.
    # Attributing every leaking frame (rival clearly active, this stem not) to
    # the mechanism holding the gate open:
    #
    #     opened by open_th ................  0 frames  (both stems)
    #     held open by hysteresis/min_off ... 54 / 34
    #     added by lookahead dilation ....... 26 / 18
    #
    # ZERO leak frames are opened by open_db.  The gate opens legitimately on
    # real target speech, the target stops, the rival keeps talking, and the
    # residual settles INSIDE the hysteresis band (rel p90 = -20.6 against
    # close_db = -23): too low to re-open, too high to ever satisfy min_off.
    # So no threshold change can reach it -- only the dwell and the dilation.
    #
    # lookahead_ms = 10 is a correction, not a fit: _run_schmitt already opens
    # retroactively across the min_on frames that justified the decision
    # (dsp.py, state[i-min_on+1:i+1] = True), so onsets do not depend on
    # _advance at all.  The lookahead only has to cover the ramp, and ramp_ms is
    # 10.  The extra 10 ms bought no onset protection and leaked.
    #
    # min_off_ms = 10 is one frame, the minimum, and it is safe for a structural
    # reason -- though not the obvious one.  Short min_off does NOT "refill"
    # brief dropouts: a frame genuinely below close_db closes the gate, and
    # should.  What makes it cheap is that _run_schmitt opens RETROACTIVELY
    # across the min_on frames that justified the decision, so re-opening costs
    # no min_on delay.  Without that fill, a 1-frame min_off would lose
    # (min_on - 1) frames = 50 ms of speech at EVERY re-onset, and retain would
    # collapse.  With it, the gate simply tracks the signal's own gaps.
    # Measured end-to-end through apply_gate with the visual veto active
    # (silence / retain / whisper dB / gate events per second, real clip):
    #     min_off 30 look 20   66.7% / 98.1% / -23.2 / 22.1   <- was shipped
    #     min_off 30 look 10   71.2% / 98.1%
    #     min_off 20 look 10   75.1% / 98.1% / -23.8 / 12.3   <- conservative
    #     min_off 10 look 10   80.5% / 97.9% / -25.6 / 11.1   <- shipped
    # Better on every axis at once, chatter included: closing promptly produces
    # FEWER transitions than hanging open inside the band (22.1 -> 11.1 per s),
    # which is the opposite of what a faster dwell is assumed to cost.
    #
    # Fitted on ONE clip, so prefer min_off_ms = 20 if a second clip ever shows
    # word holes; it keeps retain at the baseline 98.1% and still gains 8.4.
    min_off_ms: float = 10.0        # min duration of a closed state
    ramp_ms: float = 10.0           # raised-cosine anti-click ramp
    frame_ms: float = 10.0          # energy analysis frame
    lookahead_ms: float = 10.0      # == ramp_ms; see min_off_ms above

    # --- A4b: audio-visual fusion ------------------------------------------ #
    # The acoustic gate hits a wall: a p=2 Wiener mask suppresses the
    # interferer by roughly 3x its input leakage in dB, so once the separator
    # is mediocre the residual crosses open_db and the gate opens on the wrong
    # speaker.  Measured exact-zero fraction vs input leakage: 100% @ -12 dB,
    # 70% @ -9 dB, 23% @ -6 dB.  Lowering open_db to chase the residual starts
    # cutting genuine quiet speech -- that is the failure that killed the two
    # earlier gate attempts.  Vision is independent of acoustic level, so it is
    # the only tie-breaker that does not trade one failure for the other.
    #
    # visual_veto_db shifts BOTH thresholds up where a RIVAL face is moving
    # more than this stem's own speaker, preserving the hysteresis band width:
    #     effective_open = open_db + visual_veto_db * clip(v_rival - v_self, 0, 1)
    #
    # The absolute form -- veto_db * (1 - v_self) -- was tried first and
    # measured worse on BOTH axes than no fusion at all.  On a real 640x360
    # clip: fused at open_db -30 gave silence 54.2% / retain 90.7%, where the
    # pure-acoustic gate at -18 gave 57.8% / 99.1%.  v_self is simply not
    # calibrated: its mean on the target's own speech frames is 0.69, not ~1.0,
    # so the absolute form charged real words 7.7 dB on average and 22.4 dB at
    # p90 -- lifting effective_open from -30 to -8 on a tenth of genuine
    # speech.  Only the 0.104 target-vs-interferer gap carried information; the
    # rest was common-mode motion that cancels in a difference.
    #
    # 8 dB, not 25.  The differential signal separates at AUC 0.62-0.66 -- real
    # information, but weak, so it is sized to break ties the acoustic gate
    # cannot, not to override it.  The old 25 dB was sized off a synthetic
    # bench where the lip signal was clean; on a conference-resolution face it
    # is roughly three times the authority the evidence supports.
    # 4 dB, measured at open_db = -20 (silence / retain):
    #     0 -> 47.7% / 99.3%     8 -> 57.3% / 98.1%
    #     4 -> 55.7% / 98.8%    12 -> 63.5% / 91.4%   <- cliff
    #
    # 8 is tempting and only costs 0.7 retain, but it sits one step from that
    # cliff, and the veto's authority scales with lip-signal quality -- which
    # is the most clip-variable quantity in the system, since it tracks face
    # size.  A clip with larger faces yields a sharper signal, so the SAME
    # visual_veto_db pushes further toward the cliff.  4 leaves margin for that.
    visual_veto_db: float = 4.0
    # MEASURED AND REJECTED -- kept because the idea is a natural one to have
    # again.  Letting the differential shift go negative (lowering thresholds
    # where this speaker is visibly the active one) sounds strictly better: it
    # uses the full discriminative range instead of half of it, and protects
    # quiet speech rather than only punishing loud residual.  It measured
    # WORSE.  At the chosen operating point, one-sided gave silence 61.2% at
    # retain 98.4%; symmetric gave 58.0% at the same retain, and the retain
    # column was unchanged across the whole grid.  The negative half re-opens
    # the gate on frames the positive half correctly silenced, and buys back no
    # retain because retain was already ~99% wherever the veto was small enough
    # to matter.  There is nothing to protect at that operating point.
    visual_symmetric: bool = False
    # Smoothing on the lip-motion derivative.  Measured AUC of the differential
    # veto against smooth_ms, at hold_ms 120: 0.619 @ 90, 0.645 @ 150, 0.654 @
    # 220, 0.656 @ 300, 0.657 @ 400 -- a plateau from ~220 on.  220 takes the
    # knee rather than the maximum: the extra 0.003 past it costs responsiveness
    # at turn boundaries, and turns in real dialogue run about 2 s.
    #
    # The rise from 90 ms is itself a finding: per-frame lip motion at this
    # resolution is noise-dominated, and only its slow envelope carries speaker
    # information.  A sharper feature would prefer a shorter window.
    visual_smooth_ms: float = 220.0
    # Symmetric dilation of the activity signal -- the visual min_off_ms.  The
    # mouth is briefly still mid-utterance (sustained vowels, nasals) and at
    # 25 fps one still frame is 40 ms of audio; without this the veto punches
    # holes in continuous speech.
    #
    # 120, not the old 250.  That figure came off the synthetic bench, whose
    # turns were long and clean; a 250 ms symmetric dilation is a 500 ms
    # window, and at real conversational turn rates it reaches across the
    # boundary and paints the rival's frames with this speaker's motion.
    # Measured relative AUC at smooth 220: 0.654 @ hold 120, 0.654 @ 250,
    # 0.590 @ 400.  120 sits at the top of the plateau with the most margin
    # before that collapse.
    visual_hold_ms: float = 120.0
    # A track whose lip motion never exceeds this is reported INVALID rather
    # than silent -- a static face and a tracking failure look identical, and
    # guessing "silent" mutes a real speaker.
    visual_min_dynamic_range: float = 1e-4
    # Master switch.  False = pure-acoustic gate (the A4 behaviour), which is
    # what the no-face / no-match path falls back to anyway.
    visual_fusion: bool = True
    # If the visual veto removes more than this fraction of the acoustic gate's
    # own output, the stem is almost certainly paired with the wrong face.
    # Measured (with the interferer-active exclusion in veto_cost): correctly
    # paired 0.000 down to -12 dB leakage, 0.015 at -9, 0.102 at -6, 0.227 at
    # -3 -- versus 0.405 for a swapped pairing.  0.30 sits in the middle of
    # that gap.  Note the -3 dB row is a separator that has essentially failed;
    # the margin there is genuinely thin and the alarm may miss a swap on a
    # clip that is already unusable.
    visual_disagree_alarm: float = 0.30


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VisionConfig:
    # ClearVoice's video pipeline hardcodes 25 fps in three places; normalising
    # to 25 here keeps the AV-TSE upgrade path (§3) a drop-in.
    target_fps: float = 25.0
    detect_every: int = 5           # run detection every Nth frame, track between
    max_faces: int = 4
    min_track_frames: int = 12      # discard blips
    iou_match_threshold: float = 0.3
    #: Appearance re-identification (app/reid.py). Without it the tracker
    #: assumed faces move continuously, and on a 196 s talk-show excerpt with
    #: camera cuts it kept each person for 0.2-1.6% of the clip.
    reid: bool = True
    #: A track unseen for longer than this is only re-joined by appearance,
    #: never by position: after a cut, the same spot may hold someone else.
    reid_gap_frames: int = 12
    #: Cosine below which a position match is refused as a different person
    #: (a cut between two close-ups). Deliberately below reid.SAME_PERSON:
    #: continuing a track needs only "not clearly someone else".
    reid_veto: float = 0.20

    # --- crop upscaling (DISABLED) ------------------------------------------ #
    #
    # FaceMesh is trained on roughly 192x192 face crops.  On a real 640x360
    # two-person clip each face box is 0.7-0.9% of the frame (~45x45 px): the
    # landmarks still return (no NaN, so nothing looks broken), but the inner
    # lip ring spans ~8 px and the aperture is coarsely quantised.
    #
    # Upscaling the crop to 192 px was tried (min_face_px = 192) and then
    # disabled, because the measurements did not support keeping it:
    #
    #   * It does NOT help the gate.  visual_activity normalises per track
    #     against that track's own 90th-percentile motion, so the amplitude
    #     gain the upscale produces (lip std 0.0088 -> 0.0111) is divided out.
    #     A/B on the real clip, band 3 dB:  silence 66.7% with the upscale,
    #     66.3% without -- the same within noise, marginally WORSE with.
    #   * It DOES hurt the matcher.  The matcher correlates lip-derivative vs
    #     stem envelope, and upscaling a 45 px crop makes the correlation
    #     flatter (more uniform across stems).  Speaker B's scores went
    #     [0.072, 0.0099] -> [0.0529, 0.0463] -- discrimination collapsed,
    #     confidence dropped below match.min_confidence (0.05).
    #
    # The gate-level gain the upscale was assumed to buy back was inferred from
    # the matcher's confidence, and that inference was itself wrong: the old
    # confidence formula reported the row's top-2 gap even when the solver
    # placed the track on its second choice, so the "0.1406 -> 0.2223"
    # improvement was partly that artefact (see matching.match_stems_to_tracks).
    # With the honest formula the upscale still helps the matcher on that
    # axis, but not enough to justify its cost to the discriminator.
    min_face_px: int = 0
    # Never upscale beyond this factor.  A face 20 px across is genuinely gone,
    # and a 10x upscale produces confident-looking landmarks on interpolated
    # mush -- worse than an honest NaN, because the veto would then act on it.
    max_upscale: float = 4.0


@dataclass(frozen=True)
class MatchConfig:
    """Cross-modal assignment of audio stems to face tracks."""
    env_frame_ms: float = 40.0      # 25 fps -> one envelope sample per video frame
    smooth_kernel: int = 7
    min_confidence: float = 0.05    # below this, report the match as unreliable

    # --- the calibration min_confidence turned out to need ------------------ #
    #
    # min_confidence alone CANNOT protect the user, and this is measured, not
    # argued (docs/DIAG_MATCHER.md).  The margin it thresholds divides by
    # n_frames as though smoothed envelope frames were independent, so it grows
    # with the smoothing window while the answer stays wrong:
    #
    #     window   40 ms   200 ms   600 ms   1600 ms
    #     margin  0.0679   0.1870   0.4125    0.6654      <- all WRONG pairings
    #
    # A wider smooth_kernel would therefore have shipped the same inverted
    # pairing at 13x the confidence, comfortably clear of any threshold.  The
    # fix is a permutation p-value on that same margin against a circular-shift
    # null, which is invariant to the inflation because the null statistic
    # inflates identically.  See matching.pairing_significance.
    #
    # 400 shifts: the smallest p this can resolve is 1/401 = 0.0025, an order of
    # magnitude below the threshold, and the whole test costs ~0.2 s because
    # each shift is one small matmul plus <=k+1 solves of a <=4x4 matrix.  More
    # shifts buy resolution nobody reads.
    null_shifts: int = 400
    # 0.05, the conventional line, and on the real clip the honest answer is
    # p ~ 0.9 with the null reproducing the pairing ~50% of the time -- so this
    # threshold correctly marks that clip's pairing untrustworthy.  Raising it
    # to make a demo look confident would be re-introducing exactly the defect
    # this field exists to expose.
    max_p_value: float = 0.05


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AVTSEConfig:
    """Audio-visual target speaker extraction (ARCHITECTURE_V3.md).

    One model call per face, conditioned on that face's mouth ROI. There is no
    permutation to align and no matcher to get wrong, which changes what
    chunking is FOR: under SepFormer the dominant risk at a boundary was a
    speaker swap, so the rule was "largest chunk that fits". Here a boundary
    can only cost a little transformer context, so the rule is instead
    "comfortable chunk, generous discarded margin".
    """

    # Measured with scripts/bench_vram.py --backend avtse on a GTX 1050 Ti
    # (4 GB), real forward passes, peak allocated:
    #
    #     chunk_s     2     3     5     8
    #     peak MiB   526   648   900  1268
    #
    # Fit: peak_MiB ~= 276 + 124 * chunk_s  (263 MiB of that is the weights).
    # Reproduces every measured point to within 2 MiB. Linear across the whole
    # usable range -- masknet_numlayers is 1, so there is no S^2 inter-segment
    # term of the kind AudioConfig documents for SepFormer.
    vram_fixed_mb: float = 276.0
    vram_per_chunk_s_mb: float = 124.0
    vram_budget_fraction: float = 0.5

    #: Body length per model call. SHORT, because the checkpoint is: upstream
    #: decodes in 3 s windows (``one_time_decode_length: 3``) and that is the
    #: regime it was trained in. The earlier 12 s default was chosen for
    #: "more transformer context", and on a clip where two similar voices
    #: overlap continuously it made both faces' outputs near-copies of the
    #: mixture. Measured 2026-09-24 as the ECAPA cosine between the two faces'
    #: channels at the same instant (0 = two different people, 1 = the same
    #: audio), decoded length = body + 2 * context:
    #:
    #:     decoded     14 s    6 s    5 s    3 s    2 s
    #:     phone 20 s  0.864  0.781  0.668  0.639  0.595
    #:     phone 37 s  0.705    --     --   0.533  0.515
    #:     talk show   0.123    --     --   0.125  0.113
    #:
    #: Monotone on the overlapped clips and flat on the turn-taking one, so
    #: nothing pays for the short window except compute, which batching
    #: (``batch_budget_fraction``) buys back.
    chunk_s: float = 1.0
    min_chunk_s: float = 1.0

    #: Extra audio decoded on each side of a body and then DISCARDED. The
    #: model has least context at its own edges, so the fix is to place those
    #: edges outside the region we keep -- rather than crossfading two equally
    #: edge-damaged estimates together, which is what v2 did.
    context_s: float = 0.5

    #: Seam crossfade between adjacent bodies. Short on purpose: with context
    #: margins discarded, adjacent bodies already agree closely, so this only
    #: has to mask a residual sub-sample discontinuity. v2's 2 s crossfade made
    #: 20% of output a blend of two independent inferences, which comb-filters.
    #: Rounded up to whole video frames at run time so every window starts on
    #: a frame boundary and lips never slip against audio.
    fade_s: float = 0.032

    #: Share of FREE VRAM the batched windows may use. Short windows mean many
    #: calls; running them one at a time made a 37 s clip ~4x slower.
    batch_budget_fraction: float = 0.5

    #: Refinement passes (N >= 2 faces). Pass k+1 re-extracts each face from
    #: the mixture MINUS the other faces' pass-k estimates, so the model gets
    #: a cleaner input. Plain subtraction is wrong where the other face's
    #: estimate contains THIS face's voice (AV-TSE fills a pause with whoever
    #: is talking): it cancels the target. So the part of the other estimate
    #: that is locally phase-coherent with this face's own estimate is
    #: subtracted only at ``refine_shared_weight``. Measured by Whisper WER
    #: against the read scripts, face 0 / face 1:
    #:
    #:                        phone 20 s       phone 37 s
    #:     1 pass            10.9 / 36.2 %    2.3 / 7.6 %
    #:     plain subtract    17.4 / 19.0 %   30.2 / 19.6 %   (target cancelled)
    #:     shared x 0.5       2.2 / 24.1 %    3.5 / 5.4 %
    #:
    #: A second refinement pass did not improve either clip.
    refine_passes: int = 1
    refine_shared_weight: float = 0.5
    #: Neighbourhood for the local coherence estimate, in STFT frames x bins
    #: at nfft 512 / hop 128 (72 ms x 156 Hz). Large enough that two
    #: independent voices do not look coherent by chance (bias ~ 1/sqrt(45)).
    refine_smooth_frames: int = 9
    refine_smooth_bins: int = 5

    #: Silence gate for this path. GateConfig's -20/-23 dB, 10 ms hold was
    #: tuned to squeeze SepFormer's residual; on AV-TSE stems of continuous
    #: overlapped reading it closed inside words. Measured against Whisper
    #: word timings on the two phone clips (4 channels):
    #:
    #:     open/close  hold    words muted        between-word gaps zeroed
    #:     -20/-23     10 ms   12.2 - 17.7 %      37 - 72 %
    #:     -30/-40    120 ms   0.05 - 1.6 %       0.1 - 29 %
    #:
    #: The gaps are not silence there -- they hold the OTHER reader, a few dB
    #: under this one -- so no threshold zeroes them without cutting words.
    #: This setting only zeroes real silence and never chops speech. A
    #: coherent leak canceller between channels was also tried: it lowered the
    #: gap level 2-4 dB but raised WER on every channel, so it is not used.
    gate_open_db: float = -30.0
    gate_close_db: float = -40.0
    gate_min_off_ms: float = 120.0

    #: Strict isolation (dsp.strict_isolation): coherent leak cancellation
    #: between faces, with 2+ faces. The web player gets BOTH versions in one
    #: buffer -- strict and natural -- and a switch between them, because
    #: strict removes more of the other voice (bench 10.4/8.1 -> 16.3/13.4 dB;
    #: in the target's pauses down to -26..-40 dB) at some cost in clarity
    #: (WER 3.5 -> 5.8% on the 37 s clip). A cross-face mask on top
    #: (strict_mask_exponent > 0) is available but measured not worth it --
    #: see the dsp docstring. The silence gate for the strict version listens
    #: to the natural one: listening to processed audio cut 2-17% of words.
    strict_isolation: bool = True
    strict_mask_exponent: float = 0.0
    strict_ownership_power: float = 1.0

    #: The ECAPA identity gate (app/identity.py). OFF by default: on the
    #: team's own phone recordings it could not tell the two on-screen voices
    #: apart (leave-one-out accuracy 45-53%, i.e. chance), chose k=5 and muted
    #: 45-78% of CORRECT speech; on the reference talk show its k decision
    #: flips with chunk length. It is the only defence against an off-screen
    #: speaker, so enable it for clips that have one -- and check purity in
    #: meta.json when you do.
    identity_gate: bool = False

    #: Speaker/room adapter over the released weights -- see app/adapt.py.
    #: "auto" uses the one named in adapters/active.txt (none if that file is
    #: absent), "none" forces the released model, anything else is an adapter
    #: name in adapters/ or a path. Read per job, so ``python -m app.adapt
    #: use NAME`` takes effect on the next upload without a restart.
    adapter: str = "auto"
    #: An adapter only helps the people it was trained on (it lowered a
    #: stranger clip's isolation by 4-9 dB), so it is applied per face, to
    #: faces whose SFace cosine to one of its training faces is at least this.
    #: Same person across takes: 0.86-0.96; different people: <= 0.34. 0 = apply
    #: to every face.
    adapter_face_match: float = 0.5

    def resolve_adapter(self) -> Path | None:
        """The adapter file new jobs should load, or None for the base model."""
        name = self.adapter.strip()
        if name.lower() == "auto":
            try:
                name = (ADAPTERS_DIR / "active.txt").read_text(encoding="utf-8").strip()
            except OSError:
                return None
        if not name or name.lower() == "none":
            return None
        path = Path(name)
        if not path.suffix:
            path = ADAPTERS_DIR / f"{name}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"adapter {name!r} not found at {path}. Run `python -m app.adapt list`, "
                f"or `python -m app.adapt use none` to go back to the released model.")
        return path

    def resolve_chunk_s(self, device: str = "cuda") -> float:
        """Largest body length that fits the card actually present.

        Same contract and rationale as :meth:`AudioConfig.resolve_chunk_s`:
        measured against FREE memory, floored at ``min_chunk_s``, never
        extrapolated above the configured ``chunk_s``.
        """
        if device != "cuda":
            return self.chunk_s
        try:
            import torch
            if not torch.cuda.is_available():
                return self.chunk_s
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            return self.chunk_s

        budget_mb = (free / 2**20) * self.vram_budget_fraction
        # The context margins are decoded too, so they cost memory: a body of
        # B seconds actually runs B + 2*context_s through the model.
        fits = (budget_mb - self.vram_fixed_mb) / self.vram_per_chunk_s_mb - 2 * self.context_s
        return float(min(self.chunk_s, max(self.min_chunk_s, fits)))

    def resolve_batch(self, chunk_s: float, device: str = "cuda") -> int:
        """How many windows to run per forward pass on the card present.

        Same linear VRAM model as :meth:`resolve_chunk_s`, applied to B
        windows of ``chunk_s + 2 * context_s`` seconds each. An OOM at run
        time halves it anyway, so this only has to be a good first guess.
        """
        if device != "cuda":
            return 4
        try:
            import torch
            if not torch.cuda.is_available():
                return 4
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            return 4
        budget_mb = (free / 2**20) * self.batch_budget_fraction - self.vram_fixed_mb
        per_window = self.vram_per_chunk_s_mb * (chunk_s + 2 * self.context_s)
        return int(max(1, min(64, budget_mb // per_window)))


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    # "avtse" | "sepformer" | "passthrough".  AV-TSE is the default because on
    # the team's showcase recording (two similar voices reading over each
    # other) SepFormer's channels transcribe at 80-128% WER -- both texts
    # interleaved -- while AV-TSE's read at 2-5%.  Blind separation cannot
    # split those two voices; only the lips can.
    separator: str = "avtse"
    max_upload_mb: int = 512
    max_duration_s: float = 300.0

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    avtse: AVTSEConfig = field(default_factory=AVTSEConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIG = Config()


def ensure_dirs() -> None:
    for d in (RUNS_DIR, CACHE_DIR, CACHE_DIR / "hf"):
        d.mkdir(parents=True, exist_ok=True)
