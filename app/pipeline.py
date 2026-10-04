"""The offline pipeline: video in, artefact directory out.

Runnable without the server::

    python -m app.pipeline input.mp4 --out runs/demo --separator sepformer

Emits BOTH stem sets on every run:

  * ``stems_demo.wav`` -- A1 + A3 + A4, hard-gated, what the UI plays.
  * ``stems_raw.wav``  -- A1 + A3 only, ungated, what the metrics script reads.

They differ by one cheap time-domain multiply, so writing both always is nearly
free -- and it means the SI-SDR number and the demo audio provably come from
the same run.  Gating *destroys* objective scores (a hard gate zeroes frames the
reference still has signal in, and SI-SDR punishes that heavily), so scoring the
demo mix would badly understate the system.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import dsp, media
from . import roi as roi_mod
from .channels import lips_by_stem, plan_channels
from .config import CONFIG, Config
from .matching import match_stems_to_tracks
from .separation import build_separator
from .serialization import dumps as json_dumps
from .vision import FaceAnalyzer
log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]

# Stage weights, by measured share of wall clock.  A progress bar that sits at
# 20% for 40 s reads as a hang; one that moves proportionally reads as work.
STAGE_WEIGHTS = {
    "prepare":  0.05,
    "faces":    0.35,
    "separate": 0.40,
    "dsp":      0.06,
    "match":    0.05,
    "gate":     0.04,
    "export":   0.05,
}

#: The four stages the AV path collapses into one.  It reports a single
#: fraction across their combined weight, because their individual weights
#: describe a pipeline it does not run.
AVTSE_STAGES = ("separate", "dsp", "match", "gate")

#: Reliability thresholds for the AV path, used only for the UI's "low
#: confidence" tag -- the identity gate itself never consults them.  Both sit
#: at the point where "this channel is this face" stops being true of the
#: majority of the channel, rather than being fitted to a clip: on the
#: reference clip face 0 reads purity 0.75 / muted 0.33 and face 1 reads
#: 1.00 / 0.00, so neither face is anywhere near the line.
MIN_IDENTITY_PURITY = 0.5
MAX_IDENTITY_MUTED = 0.5


@dataclass
class PipelineResult:
    job_dir: Path
    meta: dict


class ClipTooLongError(RuntimeError):
    """Raised when the input exceeds ``runtime.max_duration_s``."""


class Pipeline:
    def __init__(self, cfg: Config = CONFIG, separator=None):
        self.cfg = cfg
        self._separator = separator

    def separator(self):
        if self._separator is None:
            from .config import CACHE_DIR
            device = self.cfg.runtime.resolve_device()
            # chunk_s is resolved against the card actually present, not the
            # one this was tuned on -- a 4 GB Pascal needs a shorter chunk than
            # the 8 GB Ada target, and attention memory grows with chunk_s^2.
            chunk_s = self.cfg.audio.resolve_chunk_s(device)
            if chunk_s != self.cfg.audio.chunk_s:
                log.info("chunk_s %.1fs -> %.1fs (limited VRAM)",
                         self.cfg.audio.chunk_s, chunk_s)
            self._separator = build_separator(
                self.cfg.runtime.separator,
                device=device,
                cache_dir=str(CACHE_DIR / "models"),
                chunk_s=chunk_s,
                overlap_s=self.cfg.audio.overlap_s,
            )
        return self._separator

    # ------------------------------------------------------------- avtse -- #

    def _run_avtse(self, video_path: Path, mixture: np.ndarray, sr: int,
                   tracks, vmeta: dict, emit) -> tuple[np.ndarray, np.ndarray, dict]:
        """Extract one stem per face, then verify each stem's identity.

        ``emit(frac, msg)`` spans this whole method, because it collapses four
        of the audio-only path's stages into one and their separate weights no
        longer describe anything.

        The deletions are the point:

        * **no separation/matching split.** The model is conditioned on face
          *i*'s mouth, so channel *i* is face *i* by construction. There is
          nothing to match, so ``match_stems_to_tracks`` -- measured at chance
          on the reference clip, p=0.9 -- is not consulted.
        * **no permutation alignment.** Every chunk of channel *i* follows the
          same face, so consecutive chunks cannot disagree about who they are.
        * **no Wiener masking.** It assumes the stems partition the mixture;
          AV-TSE emits independent estimates that do not sum to it, and
          forcing a partition on them alters timbre for no measured gain. The
          user's constraint is that the voice must not be altered.
        * **no visual veto.** It existed to catch a bad pairing. There is no
          pairing to get wrong.

        What replaces them is :mod:`app.identity`, which answers the question
        conditioning cannot: the model emits a voice on every frame, so when
        the target is silent it returns whoever else is audible -- including
        someone who is never on screen. Measured at ~8 s of 32 on this clip.
        """
        from .identity import IdentityConfig, covered, enroll, gate_intruders
        from .roi import boxes_to_array, extract_roi, n_video_frames_for
        from .separation import AVTSESeparator

        n_vf = n_video_frames_for(len(mixture))
        size = (vmeta["width"], vmeta["height"])
        emit(0.02, f"cropping mouths for {len(tracks)} faces")
        # Keep the whole RoiTrack, not just .roi: face size and coverage are
        # what predict whether the extraction can work at all, and discarding
        # them here is why a clip with two barely-visible faces came back
        # 67% muted with nothing in meta.json to explain it.
        roi_tracks = [
            extract_roi(str(video_path), boxes_to_array(t.boxes), n_vf, size)
            for t in tracks
        ]
        rois = [r.roi for r in roi_tracks]

        device = self.cfg.runtime.resolve_device()
        acfg = self.cfg.avtse
        chunk_s = acfg.resolve_chunk_s(device)
        # A fine-tuned adapter (app/adapt.py) is applied only to the faces it
        # was trained on, recognised by their SFace embedding; everyone else
        # gets the released weights, which it measurably beats on strangers.
        adapter = acfg.resolve_adapter()
        adapted = match = None
        if adapter is not None:
            from .adapt import match_faces
            match = match_faces(adapter, [t.emb() for t in tracks])
            if match is not None and acfg.adapter_face_match > 0:
                adapted = [m >= acfg.adapter_face_match for m in match]
                log.info("adapter %s: face match %s", adapter.stem,
                         [round(m, 3) for m in match])
                if not any(adapted):
                    adapter = adapted = None
        sep = AVTSESeparator(device, chunk_s=chunk_s, adapter=adapter,
                             context_s=acfg.context_s,
                             fade_s=acfg.fade_s,
                             batch=acfg.resolve_batch(chunk_s, device),
                             refine_passes=acfg.refine_passes,
                             refine_shared_weight=acfg.refine_shared_weight,
                             refine_smooth=(acfg.refine_smooth_frames,
                                            acfg.refine_smooth_bins))
        id_on = acfg.identity_gate
        span = 0.72 if id_on else 0.87
        # This path's own silence thresholds -- see AVTSEConfig.gate_open_db.
        gate_cfg = dataclasses.replace(
            self.cfg.gate, open_db=acfg.gate_open_db,
            close_db=acfg.gate_close_db, min_off_ms=acfg.gate_min_off_ms)
        # Where each face is actually on screen, at sample rate, with 20 ms
        # raised-cosine edges. Off screen, the model saw frozen lips and its
        # output is not this person's voice -- it is muted, and it is kept out
        # of refinement (see AVTSESeparator.separate_for_faces).
        spf = sr // 25
        presence = np.stack([
            dsp._ramp(np.repeat((~r.absent).astype(np.float32), spf)[: len(mixture)]
                      if r.absent is not None else np.ones(len(mixture), np.float32),
                      sr, ramp_ms=20.0)
            for r in roi_tracks])
        if presence.shape[1] < len(mixture):
            presence = np.pad(presence, ((0, 0), (0, len(mixture) - presence.shape[1])),
                              mode="edge")

        stems_raw = sep.separate_for_faces(
            mixture, rois, sr, progress=lambda f, m: emit(0.05 + span * f, m),
            presence=presence, adapted=adapted)

        # Strict isolation needs every face's estimate, so it exists only with
        # two or more faces. Both versions ship, strict first: channels
        # [0, N) are strict and [N, 2N) natural, and tracks.json says so via
        # `modes`. The gate always LISTENS to the natural stems (word-safe) and
        # is applied to both -- see dsp.apply_gate(carriers=).
        n_faces = stems_raw.shape[0]
        strict = None
        if acfg.strict_isolation and n_faces >= 2:
            emit(0.90, "removing cross-talk")
            strict = dsp.strict_isolation(
                stems_raw, sample_rate=sr,
                mask_exponent=acfg.strict_mask_exponent,
                ownership_power=acfg.strict_ownership_power)

        def both(natural_env_gated: np.ndarray, env: np.ndarray | None = None):
            """Stack [strict, natural] once the natural stems are gated, and
            mute every face while it is off screen."""
            natural_env_gated = natural_env_gated * presence
            if strict is None:
                return natural_env_gated
            gated_strict = dsp.apply_gate(stems_raw, sample_rate=sr, cfg=gate_cfg,
                                          carriers=strict)
            if env is not None:
                gated_strict = gated_strict * env
            return np.concatenate([gated_strict * presence, natural_env_gated])

        facts = {
            "chunk_s": chunk_s,
            "refine_passes": acfg.refine_passes if len(rois) >= 2 else 0,
            "modes": ({"strict": 0, "natural": n_faces} if strict is not None
                      else None),
            # Why a face extracted badly, when it did. Reported next to the
            # identity numbers they explain, because on their own those look
            # like a gate defect and the cause is upstream of the model.
            "face_px": [round(r.median_face_px, 1) for r in roi_tracks],
            "coverage": [round(r.coverage, 4) for r in roi_tracks],
            "usable": [bool(r.usable) for r in roi_tracks],
            "off_screen_frac": [round(float(1.0 - p.mean()), 4) for p in presence],
            # Which fine-tuned adapter ran (app/adapt.py), None = released
            # model; per face, whether it was used and the face-match cosine.
            "adapter": adapter.stem if adapter is not None else None,
            "adapter_faces": (adapted if adapted is not None
                              else [adapter is not None] * len(rois)),
            "adapter_match": [round(m, 3) for m in match] if match is not None else None,
        }

        if not id_on:
            # Silence gate only. See AVTSEConfig.identity_gate for why the
            # identity stage is off by default: on the showcase recording it
            # muted most of each speaker's correct speech.
            emit(0.94, "gating silence")
            stems_demo = both(dsp.apply_gate(stems_raw, sample_rate=sr, cfg=gate_cfg))
            emit(1.0, "done")
            return stems_raw, stems_demo, {"enabled": False, **facts}

        # -- identity ------------------------------------------------------- #
        # The mixture is passed in because it is the arbiter of which voices
        # are real: a cluster that owns part of a stem but none of the room
        # cannot be a person. Without it there is no intruder detection.
        emit(0.80, "enrolling speaker identities")
        icfg = IdentityConfig()
        plan, per_face = enroll(list(stems_raw), sr, icfg, mixture=mixture,
                                device=device)
        emit(0.92, f"{plan.n_clusters} identities, "
                   f"{len(plan.others)} off-screen")

        # -- silence -------------------------------------------------------- #
        # The identity gate answers "is this somebody else?".  It does not
        # answer "is anybody speaking?", and those are different questions:
        # without this stage face 1's channel never reaches a digital zero.
        # Measured on the reference clip, its quietest 1% of frames sit 44 dB
        # below its own speech and are still audible on a 40 dB amplification
        # -- a faint whisper, which is the one thing the brief rules out.
        #
        # Pure acoustic, no visual fusion, and the margin is why.  The gate's
        # thresholds are relative to each stem's own p95, and v2 needed lip
        # motion because SepFormer's residual interferer sat at p50 = -31 dB
        # relative to p95, i.e. INSIDE the -20/-23 dB hysteresis band, where no
        # threshold can separate it from quiet speech.  AV-TSE has already
        # removed the interferer, so what survives a pause is 44 dB down with
        # ~20 dB of clearance below the band.  Adding lip motion here would buy
        # nothing measurable and introduce an unmonitored way to clip real
        # words -- unmonitored because the veto that detected exactly that
        # failure is one of the stages this path deletes.
        emit(0.94, "gating silence")
        quiet = dsp.apply_gate(stems_raw, sample_rate=sr, cfg=gate_cfg)

        stems_demo = np.zeros_like(stems_raw)
        envs = np.zeros_like(stems_raw)
        muted, wrong, verified = [], [], []
        for i, (t, E) in enumerate(per_face):
            s = plan.score(E, i)
            # Identity is scored on the UNGATED stem: the question is who this
            # is, and silence carries no answer either way.  Only the product
            # is what ships.
            stems_demo[i], env = gate_intruders(quiet[i], t, s, sr, icfg)
            envs[i] = env
            off = env < 0.5
            muted.append(float(off.mean()))
            # Split the muted time by cause, so meta.json does not report an
            # extrapolation over never-verified audio as an identity verdict.
            cov = covered(t, len(env), sr, icfg.win_s)
            wrong.append(float((off & cov).mean()))
            verified.append(float(cov.mean()))
            if muted[-1] > 0.5:
                log.warning(
                    "face %d: %.0f%% of its stem is somebody else -- the "
                    "conditioning is not holding on this face (purity %.2f)",
                    i, 100 * muted[-1], plan.purity[i])
        stems_demo = both(stems_demo, envs)
        emit(1.0, "identity verified")

        ident = {
            "enabled": True,
            "n_clusters": plan.n_clusters,
            "n_offscreen": int(len(plan.others)),
            "purity": [round(p, 4) for p in plan.purity],
            # Cosine to the nearest other identity. Near 1.0 would mean the
            # clustering split one voice in two and the gate is scoring a face
            # against itself; +0.74 on the reference clip is two men who
            # genuinely sound alike, which is the gate doing real work.
            "nearest_rival": [round(r, 4) for r in plan.nearest_rival],
            "muted_frac": [round(m, 4) for m in muted],
            "muted_wrong_identity": [round(w, 4) for w in wrong],
            "verified_frac": [round(v, 4) for v in verified],
            **facts,
        }
        return stems_raw, stems_demo, ident

    # ---------------------------------------------------------------- run -- #

    def run(self, input_path: Path, job_dir: Path,
            progress: ProgressFn | None = None) -> PipelineResult:
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        done = 0.0

        def emit(stage: str, frac: float, msg: str) -> None:
            if progress:
                progress(stage, min(1.0, done + STAGE_WEIGHTS[stage] * frac), msg)

        def finish(stage: str) -> None:
            nonlocal done
            done += STAGE_WEIGHTS[stage]

        sr = self.cfg.audio.sample_rate
        fps = self.cfg.vision.target_fps

        # -- prepare -------------------------------------------------------- #
        emit("prepare", 0.1, "demuxing")
        video_path = job_dir / "video.mp4"
        audio_path = job_dir / "mixture.wav"

        # Enforce the duration cap BEFORE transcoding: normalize_video re-encodes
        # the whole file with libx264, so a 40-minute upload would burn minutes
        # of CPU before anything noticed.  runtime.max_duration_s existed as a
        # config field but was never read by anything -- an hour-long clip under
        # the 512 MB size limit went straight through.
        cap = self.cfg.runtime.max_duration_s
        probed = media.probe_duration(input_path)
        if cap and probed > cap:
            raise ClipTooLongError(
                f"clip is {probed:.0f}s; the limit is {cap:.0f}s. "
                f"Trim it, or raise runtime.max_duration_s.")

        media.normalize_video(input_path, video_path, fps)
        media.extract_audio(input_path, audio_path, sr)
        mixture = media.read_audio(audio_path, sr)
        duration = len(mixture) / sr

        # Authoritative re-check.  probe_duration returns 0.0 when ffprobe is
        # missing -- and imageio-ffmpeg bundles ffmpeg WITHOUT ffprobe, which is
        # exactly the fallback path a machine with no system ffmpeg takes.  So
        # the cheap probe above is an optimisation, not the guarantee; this is
        # the guarantee, measured off the decoded samples.
        if cap and duration > cap:
            raise ClipTooLongError(
                f"clip is {duration:.0f}s; the limit is {cap:.0f}s. "
                f"Trim it, or raise runtime.max_duration_s.")

        emit("prepare", 1.0, f"{duration:.1f}s @ {sr} Hz")
        finish("prepare")

        # -- faces ---------------------------------------------------------- #
        emit("faces", 0.0, "detecting faces")
        analyzer = FaceAnalyzer(self.cfg.vision,
                                progress=lambda f, m: emit("faces", f, m))
        tracks, vmeta = analyzer.analyze(video_path)
        log.info("found %d tracks (%s backend)", len(tracks), vmeta["backend"])
        if not tracks and self.cfg.runtime.separator == "avtse":
            # Said in the user's terms: this reaches the upload panel verbatim.
            raise RuntimeError(
                "No faces were found in this video. Isolation follows each "
                "speaker's lips, so the speakers' faces need to be visible.")
        finish("faces")

        # -- separate / dsp / match / gate ----------------------------------- #
        # Two paths that differ in kind, not in degree.  The audio-only one
        # separates blind and then *infers* which face each stem belongs to;
        # the AV one is told, and spends the four saved stages on verifying
        # that the model returned the face it was asked for.
        #
        # Both branches must finish() all four stages, or the progress bar
        # stops short of 1.0 on whichever path skipped one.
        avtse = self.cfg.runtime.separator == "avtse"
        alignment: list[dict] = []
        used_chunk_s = None
        ident: dict = {}

        if avtse:
            span = sum(STAGE_WEIGHTS[s] for s in AVTSE_STAGES)
            at = done

            def av_emit(frac: float, msg: str) -> None:
                if progress:
                    progress("separate", min(1.0, at + span * frac), msg)

            av_emit(0.0, "loading model")
            stems_raw, stems_demo, ident = self._run_avtse(
                video_path, mixture, sr, tracks, vmeta, av_emit)
            used_chunk_s = ident.pop("chunk_s")
            for stage in AVTSE_STAGES:
                finish(stage)

            # Face i conditioned channel i, so the assignment is the identity
            # permutation and there is nothing to reorder.  `confidence` is not
            # a matcher margin here -- there was no matcher -- so it carries the
            # identity purity when that was measured: the share of this channel
            # that really is this face.  With the identity stage off, the only
            # evidence left is how much of the clip the face was actually seen
            # for, which is what the conditioning had to work with.
            assignment = list(range(len(tracks)))
            confidence = list(ident["purity"] if ident["enabled"] else ident["coverage"])
            channel_of: list[int | None] = list(range(len(tracks)))
            # No p-value, because no null test was run.  Stating one would be
            # inventing evidence; `basis` says how the pairing was established
            # instead, and the UI phrases its note from that.
            pairing = {"basis": "conditioning", "trustworthy": True}
            visual = {
                # Not an optional fusion on this path: the mouth ROI is what
                # the model consumes, so every stem is conditioned by
                # construction.  There is no veto_cost either -- the veto
                # existed to catch a bad pairing, and `identity` is what
                # detects failure now.
                "enabled": True,
                "conditioned_stems": int(stems_raw.shape[0]),
                "total_stems": int(stems_raw.shape[0]),
            }
        else:
            emit("separate", 0.0, "loading model")
            separator = self.separator()
            stems = separator.separate(
                mixture, sr, progress=lambda f, m: emit("separate", f, m))
            # Chunk-boundary permutation decisions.  A speaker swap mid-clip is
            # audibly indistinguishable from bleed-through, so when Objective A
            # appears to regress, check `held_boundaries` here BEFORE touching
            # the gate -- that is the stage that actually failed.
            alignment = list(getattr(separator, "last_alignment", []) or [])
            # The chunk size the separator ACTUALLY ran at, which is not
            # necessarily the configured one: an OOM retry halves it in-flight.
            # Recorded so a slow run has a visible cause in meta.json rather
            # than being mystery wall clock.
            used_chunk_s = getattr(separator, "last_chunk_s", None)
            finish("separate")

            # -- dsp -------------------------------------------------------- #
            emit("dsp", 0.2, "wiener masking")
            stems_raw = dsp.wiener_separate(
                stems, sample_rate=sr,
                nfft=self.cfg.audio.nfft, hop=self.cfg.audio.hop,
                exponent=self.cfg.gate.mask_exponent,
                floor=self.cfg.gate.mask_floor,
                cepstral_order=self.cfg.gate.cepstral_smooth_order,
            )
            emit("dsp", 1.0, "masked")
            finish("dsp")

            # -- match ------------------------------------------------------ #
            # Runs BEFORE the gate, not after: the gate is audio-visual now, so
            # it needs to know which face belongs to which stem.  Matching
            # consumes `stems_raw` (ungated) either way -- correlating against
            # gated audio would feed the matcher a signal the gate already
            # shaped, which is circular.
            emit("match", 0.3, "matching voices to faces")
            match = match_stems_to_tracks(
                stems_raw, tracks, sample_rate=sr,
                video_fps=vmeta["fps"], cfg=self.cfg.match)
            assignment, confidence = match.assignment, match.confidence
            # The margin alone is not evidence -- it inflates with smoothing
            # while the answer stays wrong (docs/DIAG_MATCHER.md).  This is the
            # number that decides whether the pairing beat its own null.
            if not match.trustworthy(self.cfg.match):
                log.warning(
                    "stem/face pairing is not distinguishable from chance "
                    "(p=%.3f, %.0f%% of shifted-lip nulls reproduce it, margin "
                    "%.4f) -- the UI will offer the swap control; do not trust "
                    "the channel labels on this clip",
                    match.significance, 100 * match.null_agreement,
                    max(confidence, default=0.0))
            pairing = {
                # The pairing's own honesty report.  `confidence` is a scale
                # that inflates with match.smooth_kernel; `significance` is the
                # p-value of that scale under a circular-shift null and does
                # not.  When `trustworthy` is false the channel labels are a
                # coin flip -- the UI offers the swap control and says so.
                "basis": "null-test",
                "trustworthy": bool(match.trustworthy(self.cfg.match)),
                "significance": round(match.significance, 4),
                "null_agreement": round(match.null_agreement, 4),
                "null_shifts": self.cfg.match.null_shifts,
            }
            finish("match")

            # -- gate ------------------------------------------------------- #
            emit("gate", 0.2, "audio-visual gating")
            lips = lips_by_stem(assignment, tracks, stems_raw.shape[0])
            stems_demo = dsp.apply_gate(stems_raw, sample_rate=sr,
                                        cfg=self.cfg.gate, lips=lips,
                                        video_fps=vmeta["fps"])

            # How much of the acoustic gate's output the visual veto removed,
            # per stem.  ~0 when the face/voice pairing is right; large when it
            # is wrong.  This is the runtime mis-assignment detector matching.py
            # notes it does not have -- and it is measured on the FINAL signals,
            # not on the statistic the assignment was chosen to maximise.
            veto: list[float] = [0.0] * stems_raw.shape[0]
            if self.cfg.gate.visual_fusion and any(lp is not None for lp in lips):
                emit("gate", 0.7, "checking face/voice agreement")
                acoustic = dsp.apply_gate(stems_raw, sample_rate=sr,
                                          cfg=self.cfg.gate)
                for j in range(stems_raw.shape[0]):
                    # The loudest competing stem, sample-wise.  veto_cost uses
                    # it to ignore stretches where the acoustic gate was open on
                    # someone else's residual -- there, the veto closing is the
                    # fusion working, not evidence of a bad pairing.
                    rest = np.delete(stems_raw, j, axis=0)
                    other = np.abs(rest).max(axis=0) if rest.shape[0] else None
                    veto[j] = dsp.veto_cost(acoustic[j], stems_demo[j], other,
                                            sample_rate=sr,
                                            frame_ms=self.cfg.gate.frame_ms)
                    if veto[j] > self.cfg.gate.visual_disagree_alarm:
                        log.warning(
                            "stem %d: visual veto removed %.0f%% of gated "
                            "speech -- the face matched to it is probably the "
                            "wrong one", j, 100 * veto[j])
            finish("gate")

            # Reorder stems so channel i belongs to track i -- the browser then
            # needs no indirection: track index == audio channel index.
            order, channel_of = plan_channels(assignment, len(tracks),
                                              stems_raw.shape[0])
            stems_raw = stems_raw[order]
            stems_demo = stems_demo[order]
            veto = [veto[j] for j in order]  # keep it indexed by CHANNEL too
            visual = {
                # Which stems the gate actually had vision for.  If
                # `fused_stems` is 0 on a clip with visible faces, the gate
                # silently ran pure-acoustic and Objective A is back to its
                # measured -9 dB knee -- check the matcher, not the gate.
                "enabled": bool(self.cfg.gate.visual_fusion),
                "fused_stems": sum(1 for lp in lips if lp),
                "total_stems": len(lips),
                # Per CHANNEL, post-reorder.  ~0 is healthy; anything above
                # gate.visual_disagree_alarm means that face is probably paired
                # with the wrong voice.
                "veto_cost": [round(c, 4) for c in veto],
            }

        # -- export --------------------------------------------------------- #
        emit("export", 0.2, "writing stems")
        # ONE gain across both files.  Normalising each on its own peak makes
        # the demo and raw exports differ by an arbitrary scale factor, which
        # is not A/B comparable and moves PESQ (SI-SDR is scale-invariant, so
        # it would not have shown this).
        gain = media.peak_gain(stems_demo, stems_raw)
        if gain != 1.0:
            # Apply in memory so the stats below describe the audio that
            # actually ships, not the pre-normalisation array.
            stems_demo = stems_demo * gain
            stems_raw = stems_raw * gain
        media.write_multichannel_wav(stems_demo, job_dir / "stems_demo.wav", sr, gain=1.0)
        media.write_multichannel_wav(stems_raw, job_dir / "stems_raw.wav", sr, gain=1.0)

        emit("export", 0.6, "writing tracks")

        def reliable(i: int) -> bool:
            """Whether the UI should present this face's audio without a caveat.

            The two paths can fail in unrelated ways, so they are checked on
            unrelated evidence -- there is no shared number to threshold.

            On the AV path nothing can be *mispaired*: channel *i* was
            extracted from face *i*'s own mouth.  What can go wrong is that the
            conditioning did not hold and the model spent part of the clip on
            somebody else, so the checks are the two things
            :mod:`app.identity` measures:

              * **purity** -- of the windows in this stem, the share belonging
                to the identity this face claimed.  Below 0.5 the stem is more
                somebody else than it is this face.
              * **muted fraction** -- how much the identity gate had to remove.
                A channel that is mostly silence is not usable even if what
                survives is pure.

            On the audio-only path everything is an inference, so there are
            three independent ways to fail:

              * the matcher's margin -- was the assignment well-separated?
              * that margin's **p-value** against a circular-shift null.  The
                margin on its own is not evidence: it grows with the smoothing
                window while the answer stays wrong, so on the real clip it read
                0.0679 for an inverted pairing and would have read 0.6654 with a
                wider kernel (docs/DIAG_MATCHER.md).  This check is the one that
                actually catches that clip.
              * the visual veto cost -- does the final audio agree with this
                face?  Measured on the shipped signals rather than on the
                statistic the assignment was chosen to maximise.
            """
            ch = channel_of[i]
            if ch is None:
                return False
            if avtse:
                if not ident["enabled"]:
                    return ident["usable"][i]
                return (ident["purity"][i] >= MIN_IDENTITY_PURITY
                        and ident["muted_frac"][i] <= MAX_IDENTITY_MUTED)
            conf = confidence[i] if i < len(confidence) else 0.0
            if conf < self.cfg.match.min_confidence:
                return False
            if pairing["significance"] > self.cfg.match.max_p_value:
                return False
            return veto[ch] <= self.cfg.gate.visual_disagree_alarm

        def caveat(i: int) -> str | None:
            """Why this face's audio is not trustworthy, in the user's terms.

            ``reliable`` says *that* something is wrong; this says *what*, and
            the distinction earns its keep because the causes call for
            different actions. A face the model could not condition on is a
            property of the clip -- no setting fixes it, and the honest
            message is "this speaker is barely on screen". A face whose stem
            came back impure is a model outcome on a face that WAS visible,
            which is the case where re-running or swapping might help.

            Ordered most-upstream first, so the message names the root cause
            rather than the symptom it produced downstream.
            """
            ch = channel_of[i]
            if ch is None:
                return "no stem"
            if avtse:
                cov = ident["coverage"][i]
                if cov < roi_mod.MIN_COVERAGE:
                    return f"on screen {cov:.0%} of the clip"
                if ident["face_px"][i] < roi_mod.MIN_FACE_PX:
                    return f"face too small ({ident['face_px'][i]:.0f}px)"
                if not ident["enabled"]:
                    return None
                if ident["purity"][i] < MIN_IDENTITY_PURITY:
                    return "voice not consistent"
                if ident["muted_frac"][i] > MAX_IDENTITY_MUTED:
                    return f"{ident['muted_frac'][i]:.0%} removed as someone else"
                return None
            conf = confidence[i] if i < len(confidence) else 0.0
            if conf < self.cfg.match.min_confidence:
                return "weak face-voice correlation"
            if pairing["significance"] > self.cfg.match.max_p_value:
                return "pairing not statistically significant"
            if veto[ch] > self.cfg.gate.visual_disagree_alarm:
                return "audio disagrees with the lips"
            return None

        tracks_json = {
            "fps": vmeta["fps"],
            "width": vmeta["width"],
            "height": vmeta["height"],
            "n_frames": vmeta["n_frames"],
            # Duplicated from meta.json deliberately: the browser fetches
            # tracks.json to draw the face boxes and needs to know in the SAME
            # payload whether the channel labels can be trusted, so it can offer
            # the swap control without a second round trip.
            "pairing": pairing,
            # Channel offsets of each isolation version in stems_demo.wav, when
            # there is more than one (AV path, 2+ faces): face i plays channel
            # `channel + modes[mode]`. Absent means one version, offset 0.
            "modes": ident.get("modes") if avtse else None,
            "adapter": ident.get("adapter") if avtse else None,
            "adapter_faces": ident.get("adapter_faces") if avtse else None,
            "tracks": [
                {
                    "id": t.track_id,
                    "label": t.label(),
                    "channel": channel_of[i],
                    "confidence": round(confidence[i], 4) if i < len(confidence) else 0.0,
                    "reliable": reliable(i),
                    "caveat": caveat(i),
                    "boxes": [list(np.round(b, 4)) if b is not None else None
                              for b in t.boxes],
                }
                for i, t in enumerate(tracks)
            ],
        }
        (job_dir / "tracks.json").write_text(json_dumps(tracks_json), encoding="utf-8")

        # Computed AFTER the export gain is applied above, so `peak` describes
        # the shipped WAV rather than an intermediate array.
        stats = {
            f"channel_{i}": dsp.silence_stats(stems_demo[i])
            for i in range(stems_demo.shape[0])
        }
        if stems_demo.shape[0] >= 2:
            # whisper_db, NOT measure_leakage_db.  The latter wants the
            # ground-truth interferer component; handing it the neighbouring
            # OUTPUT channel -- which is what this used to do -- measures how
            # loud the other speaker is while this one is silent, and that
            # number RISES as separation improves.  Real jobs were reporting
            # +4.4 dB from that inversion while the gate was working correctly.
            #
            # One entry per channel: with the demo mix the user hears exactly
            # one channel at a time, so "how loud is the ghost in the channel I
            # selected" is the per-channel question, not a clip-wide scalar.
            # stems_demo may hold several isolation versions of the same faces
            # (tracks.json `modes`), so demo channel i is face i % n_raw.
            n_raw = stems_raw.shape[0]
            stats["whisper_db"] = {
                f"channel_{i}": dsp.whisper_db(
                    stems_demo[i], stems_raw[i % n_raw],
                    np.abs(np.delete(stems_raw, i % n_raw, axis=0)).max(axis=0),
                    sample_rate=sr)
                for i in range(stems_demo.shape[0])
            } if n_raw >= 2 else {}
        stats["export_gain"] = round(gain, 6)

        meta = {
            "sample_rate": sr,
            "duration": duration,
            "fps": vmeta["fps"],
            "width": vmeta["width"],
            "height": vmeta["height"],
            "n_speakers": int(stems_raw.shape[0]),
            "n_tracks": len(tracks),
            "vision_backend": vmeta["backend"],
            "separator": self.cfg.runtime.separator,
            "device": self.cfg.runtime.resolve_device(),
            "chunk_s": used_chunk_s,
            "assignment": assignment,
            "confidence": [round(c, 4) for c in confidence],
            "matching": pairing,
            # Per-face identity verification, AV path only.  This is what
            # replaced `matching` as the thing that can actually be wrong: the
            # pairing is given, so the open question is whether the model held
            # onto the face it was conditioned on.  `muted_frac` counts ONLY
            # what identity removed -- silence removed by the acoustic gate is
            # in `silence.channel_N.exact_zero_fraction`, and the two causes are
            # kept apart on purpose.  It then splits into
            # `muted_wrong_identity` (a measured verdict) plus the remainder
            # (audio too quiet to verify, which the gate closes rather than
            # guesses about) -- do not read the total as a detection rate.
            "identity": ident or None,
            "silence": stats,
            # Chunk-boundary permutation decisions.  All zero on the AV path,
            # which is not a degenerate reading: channel i follows face i in
            # every chunk, so no boundary has a permutation to decide.
            "alignment": {
                "boundaries": len(alignment),
                "flipped": sum(1 for d in alignment if d.get("flipped")),
                "held_boundaries": [d["boundary_s"] for d in alignment
                                    if not d.get("confident")],
                # Murty margin: winner minus best FEASIBLE alternative pairing.
                # Read it as evidence strength, and note that it is NOT the old
                # `best - trace`, which was structurally 0.0 whenever the
                # identity order won -- i.e. at almost every boundary -- and so
                # could not tell a decisive hold from a coin flip.
                "min_margin": round(min((d["margin"] for d in alignment),
                                        default=0.0), 5),
                # How each boundary was actually decided.  `identity` means the
                # overlap window was uninformative (typically a shared pause)
                # and the accumulated cepstral speaker profiles broke the tie;
                # `hold` means nothing could, and that boundary is a coin flip
                # whose outcome propagates to the end of the clip.
                "basis": {b: sum(1 for d in alignment if d.get("basis") == b)
                          for b in ("waveform", "identity", "hold")},
            },
            "visual": visual,
            "elapsed_s": round(time.time() - t_start, 2),
            "config": self.cfg.to_dict(),
        }
        # json_dumps: meta["silence"] carries a non-finite residual floor for
        # any perfectly-muted channel, and bare json.dumps would write the
        # non-standard `-Infinity` token.  Python's json.loads accepts it on the
        # way back in (so adopt_existing would never notice), but it is not
        # valid JSON and no other tool will read it.
        (job_dir / "meta.json").write_text(json_dumps(meta, indent=2, default=str),
                                           encoding="utf-8")
        emit("export", 1.0, "done")
        finish("export")

        try:
            audio_path.unlink()
        except OSError:
            pass

        return PipelineResult(job_dir=job_dir, meta=meta)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="AV speaker isolation, offline")
    ap.add_argument("input")
    ap.add_argument("--out", default="runs/cli")
    ap.add_argument("--separator", default=None,
                    choices=["sepformer", "avtse", "passthrough"])
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--identity-gate", action="store_true",
                    help="AV path: also mute voices ECAPA says are not this face "
                         "(for clips with an off-screen speaker; see AVTSEConfig)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    import dataclasses
    cfg = CONFIG
    rt = cfg.runtime
    if args.separator or args.device:
        rt = dataclasses.replace(
            rt,
            separator=args.separator or rt.separator,
            device=args.device or rt.device)
        cfg = dataclasses.replace(cfg, runtime=rt)
    if args.identity_gate:
        cfg = dataclasses.replace(
            cfg, avtse=dataclasses.replace(cfg.avtse, identity_gate=True))

    def show(stage: str, pct: float, msg: str) -> None:
        print(f"\r[{pct * 100:5.1f}%] {stage:<9} {msg:<44}", end="", flush=True)

    result = Pipeline(cfg).run(Path(args.input), Path(args.out), progress=show)
    print("\n")
    print(json.dumps(result.meta["silence"], indent=2))
    print(f"\nartefacts -> {result.job_dir}")


if __name__ == "__main__":
    main()
