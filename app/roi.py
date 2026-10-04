"""Mouth-ROI extraction for AV-MossFormer2.

Turns a face-box track into the exact tensor the vendored visual frontend was
trained on. Getting this wrong does not crash -- it silently degrades
extraction quality -- so every constant here is traceable to upstream's
``video_process.py`` and is covered by ``scripts/check_roi.py``.

The pipeline, per frame::

    bs  = max(box_w, box_h) / 2                 # "detection box size"
    pad the frame by int(bs * 1.8) with gray 110
    crop  [my - bs : my + 1.8*bs,  mx - 1.4*bs : mx + 1.4*bs]   # 2.8bs square
    resize -> 224x224
    BGR -> gray,  /255
    centre-crop [56:168, 56:168] -> 112x112
    (x - 0.4161) / 0.1688

Net effect: a square of side ``1.4*bs`` centred ``0.4*bs`` BELOW the box
centre -- i.e. the mouth and chin, not the eyes. The downward bias is
deliberate and is why a naive "centre crop of the face" underperforms.

One intentional deviation from upstream: upstream round-trips the crops
through a lossy XVID AVI before reading them back. We keep the frames in
memory. That is strictly better -- codec noise in the visual cue can only
hurt -- and it is the one place we are *not* bit-faithful to the reference
implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .avtse.config import ROI_SIZE, SAMPLE_RATE, VIDEO_FPS

__all__ = [
    "CROP_SCALE",
    "ROI_MEAN",
    "ROI_STD",
    "MIN_FACE_PX",
    "MIN_COVERAGE",
    "RoiTrack",
    "boxes_to_array",
    "extract_roi",
    "n_video_frames_for",
]

log = logging.getLogger(__name__)

#: Upstream ``cropScale``. Sets how much context around the box is kept.
CROP_SCALE = 0.40

#: Fill value for the constant border, matching upstream's ``constant_values=(110, 110)``.
PAD_VALUE = 110

#: Dataset statistics the frontend expects. Not ImageNet numbers -- these are
#: the LRS/VoxCeleb mouth-crop mean/std in [0,1] units.
ROI_MEAN = 0.4161
ROI_STD = 0.1688

#: Median filter width over the box track, in frames (upstream ``kernel_size=13``).
#: At 25 fps this is ~0.5 s, which kills detector jitter without lagging real
#: head motion enough to smear the mouth.
SMOOTH_FRAMES = 13

#: Below this native face size the mouth ROI is mostly upscaling artefacts and
#: extraction quality falls off. See ARCHITECTURE_V3.md section 5 -- this is a
#: data requirement on source clips, not something code can fix.
MIN_FACE_PX = 150

#: Below this fraction of frames actually carrying a detected box, the ROI
#: stack is mostly held and padded geometry rather than observed mouths, and
#: the extraction is not conditioned on much.
#:
#: Measured across three clips, coverage predicts purity almost exactly:
#:
#:     coverage   100%   94%   56%   13%    7%
#:     purity     1.00   0.74  0.98  0.44  0.50
#:
#: The two low-coverage faces are from an edited clip where the speakers
#: appear in separate shots -- the detector is not failing, the face is
#: genuinely absent, and no detector setting recovers it (model_selection,
#: min_detection_confidence down to 0.2, and 2x upscaling were each measured
#: and none ever finds a third face in a frame). 0.35 sits below the 56% case
#: that works and well above the 13% case that does not.
MIN_COVERAGE = 0.35

#: Longest detector dropout bridged by holding the box geometry, in frames
#: (0.48 s). Shorter gaps are a missed detection on a face that is still
#: there. Longer ones are usually a camera cut: the face is gone and the held
#: box now frames someone else's mouth or the background, which conditions the
#: model on the wrong lips. Those frames get the last real mouth, frozen --
#: still lips make AV-TSE go quiet (measured: frozen ROI -> output -59 dB) --
#: and are reported in ``RoiTrack.absent`` so the pipeline can mute them.
MAX_HOLD_FRAMES = 12


def _long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Frames of ``mask`` that sit in a True run longer than ``min_len``."""
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


@dataclass(frozen=True)
class RoiTrack:
    """A per-face mouth-ROI stack plus the diagnostics preflight needs."""

    #: (T, 112, 112) float32, normalised. Ready for the model.
    roi: np.ndarray
    #: Native face-box side length per frame, in pixels, before any resizing.
    face_px: np.ndarray
    #: Frames where the box was missing and had to be filled by hold/pad.
    filled: np.ndarray
    #: Frames inside a dropout longer than MAX_HOLD_FRAMES: the face is not on
    #: screen, its ROI is a frozen copy, and its audio should be muted there.
    absent: np.ndarray | None = None

    @property
    def median_face_px(self) -> float:
        return float(np.median(self.face_px)) if self.face_px.size else 0.0

    @property
    def undersized(self) -> bool:
        """True when this face is too small for reliable extraction."""
        return self.median_face_px < MIN_FACE_PX

    @property
    def coverage(self) -> float:
        """Fraction of frames whose ROI came from an actually-detected box."""
        return float(1.0 - self.filled.mean()) if self.filled.size else 0.0

    @property
    def sparse(self) -> bool:
        """True when the face is absent for most of the clip.

        Distinct from :attr:`undersized`, and the distinction is the point: a
        small face is a resolution problem the user could fix by re-shooting,
        while a sparse one usually means the clip is edited and this person is
        simply not on screen. Neither is a code defect, and both make the
        extraction unreliable in ways the user should be told about rather
        than left to discover by hearing silence.
        """
        return self.coverage < MIN_COVERAGE

    @property
    def usable(self) -> bool:
        return not (self.undersized or self.sparse)

    def __len__(self) -> int:
        return int(self.roi.shape[0])


def boxes_to_array(boxes: list) -> np.ndarray:
    """Parse a ``tracks.json`` box list into a dense (F, 4) array.

    Frames where the detector found nothing are stored as ``null`` and become
    rows of NaN, which :func:`extract_roi` fills by holding the last good
    geometry. Track 1 of the reference clip drops 54 of 809 frames, so this is
    the normal case, not an edge case.
    """
    out = np.full((len(boxes), 4), np.nan, dtype=np.float64)
    for i, b in enumerate(boxes):
        if b is not None and len(b) == 4:
            out[i] = b
    return out


def n_video_frames_for(n_samples: int) -> int:
    """Visual frame count implied by an audio length.

    Upstream derives the visual length from the audio, not from the video
    file: ``int(n_samples / 16000 * 25)``. We match that exactly, because the
    separator interpolates the visual stream onto the audio frame grid and a
    one-frame disagreement shifts every lip against its phoneme.
    """
    return int(n_samples / SAMPLE_RATE * VIDEO_FPS)


def _median_smooth(track: np.ndarray, k: int = SMOOTH_FRAMES) -> np.ndarray:
    """Column-wise median filter with edge replication."""
    if track.shape[0] < 2 or k < 3:
        return track
    k = min(k | 1, track.shape[0] | 1)  # odd, and no longer than the track
    half = k // 2
    padded = np.pad(track, ((half, half), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)
    return np.median(win, axis=-1).astype(np.float64)


def _crop_one(frame_bgr: np.ndarray, cx: float, cy: float, bs: float) -> np.ndarray:
    """Crop, resize, grey and centre-crop a single frame. Returns (112,112) uint8-ish float."""
    bsi = int(bs * (1 + 2 * CROP_SCALE))
    padded = cv2.copyMakeBorder(
        frame_bgr, bsi, bsi, bsi, bsi, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    my, mx = cy + bsi, cx + bsi

    y0, y1 = int(my - bs), int(my + bs * (1 + 2 * CROP_SCALE))
    x0, x1 = int(mx - bs * (1 + CROP_SCALE)), int(mx + bs * (1 + CROP_SCALE))
    face = padded[y0:y1, x0:x1]
    if face.size == 0:  # degenerate box; caller records this frame as filled
        face = np.full((2, 2, 3), PAD_VALUE, np.uint8)

    face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
    grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    off = ROI_SIZE // 2
    return grey[112 - off : 112 + off, 112 - off : 112 + off]


def extract_roi(
    video_path: str,
    boxes: np.ndarray,
    n_frames: int,
    frame_size: tuple[int, int],
) -> RoiTrack:
    """Build the (T, 112, 112) mouth-ROI stack for one tracked face.

    Args:
        video_path: source video; read sequentially, decoded once.
        boxes: (F, 4) normalised ``[x, y, w, h]`` in [0,1], one row per video
            frame. Rows of NaN mark frames where the face was not detected.
        n_frames: target length T, from :func:`n_video_frames_for`. The stack
            is edge-padded or truncated to exactly this.
        frame_size: ``(width, height)`` in pixels.

    Returns:
        A :class:`RoiTrack`. ``roi`` is normalised and model-ready.
    """
    width, height = frame_size
    boxes = np.asarray(boxes, dtype=np.float64)

    # Absolute pixel geometry: centre and half-side, in the detector's terms.
    cx = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    cy = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
    bs = np.maximum(boxes[:, 2] * width, boxes[:, 3] * height) / 2.0

    missing = ~np.isfinite(cx) | ~np.isfinite(cy) | ~np.isfinite(bs) | (bs <= 0)
    if missing.all():
        raise ValueError("track has no valid boxes")
    if missing.any():
        # Hold the last good geometry across dropouts rather than interpolating
        # through them -- a face that vanishes behind a hand has not moved to
        # the midpoint of where it was and where it reappears.
        idx = np.where(~missing, np.arange(len(missing)), -1)
        idx = np.maximum.accumulate(idx)
        idx[idx < 0] = np.flatnonzero(~missing)[0]
        cx, cy, bs = cx[idx], cy[idx], bs[idx]

    geom = _median_smooth(np.stack([cx, cy, bs], axis=1))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video_path}")

    rois: list[np.ndarray] = []
    face_px: list[float] = []
    try:
        for i in range(len(geom)):
            ok, frame = cap.read()
            if not ok:
                break
            gx, gy, gbs = geom[i]
            rois.append(_crop_one(frame, gx, gy, gbs))
            face_px.append(2.0 * gbs)
    finally:
        cap.release()

    if not rois:
        raise ValueError(f"decoded no frames from {video_path}")

    stack = np.stack(rois).astype(np.float32)
    sizes = np.asarray(face_px, dtype=np.float32)
    filled = missing[: len(stack)].copy()

    # Long absences: freeze the last real mouth instead of cropping whatever
    # the held box now frames (see MAX_HOLD_FRAMES). A leading absence has no
    # "last" mouth, so it takes the first one instead.
    absent = _long_runs(filled, MAX_HOLD_FRAMES)
    if absent.any():
        present = np.flatnonzero(~absent)
        src = np.where(~absent, np.arange(len(absent)), -1)
        src = np.maximum.accumulate(src)
        src[src < 0] = present[0]
        stack = stack[src]

    # Match the audio-derived length exactly (upstream pads with mode='edge').
    if len(stack) < n_frames:
        pad = n_frames - len(stack)
        stack = np.pad(stack, ((0, pad), (0, 0), (0, 0)), mode="edge")
        sizes = np.pad(sizes, (0, pad), mode="edge")
        filled = np.pad(filled, (0, pad), constant_values=True)
        # A short tail pad is the video ending a frame early, not an absence.
        absent = np.pad(absent, (0, pad), mode="edge")
    elif len(stack) > n_frames:
        stack, sizes, filled = stack[:n_frames], sizes[:n_frames], filled[:n_frames]
        absent = absent[:n_frames]

    stack = (stack - ROI_MEAN) / ROI_STD

    med = float(np.median(sizes))
    if med < MIN_FACE_PX:
        log.warning(
            "face is %.0f px (median); below the %d px floor -- extraction "
            "quality will suffer. Use a higher-resolution source clip.",
            med, MIN_FACE_PX,
        )
    track = RoiTrack(roi=stack, face_px=sizes, filled=filled, absent=absent)
    if track.sparse:
        log.warning(
            "face is visible in only %.0f%% of frames (floor %.0f%%) -- the "
            "rest is held geometry, so extraction is conditioned on very "
            "little. Usually an edited clip where this speaker appears in a "
            "separate shot.",
            100 * track.coverage, 100 * MIN_COVERAGE,
        )
    return track
