"""Face detection, tracking, and a lip-aperture feature that is actually valid.

**The bug this module exists to fix.**  The prototype computed lip aperture as
the distance between MediaPipe landmarks 13 and 14 *in the coordinate space of
the per-face crop it fed to FaceMesh*.  MediaPipe normalises landmarks to the
image it was given, so those coordinates are fractions of the **crop**, not the
frame.  Each face gets its own crop, and the crop resizes every frame as the
box tracks the face -- so the "lip opening" signal was measured with a ruler
whose length changed continuously.

Cosine similarity is scale-invariant, which is why cross-modal *matching* still
worked and hid the defect.  But every *threshold* -- the absolute-position gate
and the velocity gate, both of which failed -- was compared against that moving
ruler.  No fixed threshold could ever have worked.

Two independent fixes are applied here:

1. **Coordinates are mapped back to frame pixels.**  ``x_px = crop_x0 +
   lm.x * crop_w``.  Every landmark returned by this module is in absolute
   frame pixels, always.

2. **The feature itself is made scale-invariant.**  Instead of a raw distance,
   we use the *area of the inner-lip polygon* (shoelace over the inner-lip
   ring) divided by the *squared inter-ocular distance*.  Both scale as the
   square of apparent face size, so the ratio is dimensionless and comparable
   across speakers, distances, and frames.  A raw pixel distance would still
   drift as a subject leans toward the camera.

If mediapipe is unavailable, a Haar-cascade + mouth-region-motion backend is
used instead.  It is a worse feature, but it keeps the pipeline runnable, and
it is normalised the same way so nothing downstream changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .dsp import resample_hold

log = logging.getLogger(__name__)

# MediaPipe FaceMesh indices.
# Inner lip ring, ordered around the polygon -- required for the shoelace area.
INNER_LIP_RING = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


@dataclass
class FaceTrack:
    """A face followed across frames.  ``boxes[i] is None`` where absent."""
    track_id: int
    boxes: list[tuple[float, float, float, float] | None] = field(default_factory=list)
    lip: list[float] = field(default_factory=list)      # NaN where absent
    # Re-identification state (app/reid.py). Running sum of unit embeddings
    # rather than a single reference, so one bad frame cannot redefine who
    # this track is.
    emb_sum: np.ndarray | None = None
    last_seen: int = -1
    last_box: tuple[float, float, float, float] | None = None

    def emb(self) -> np.ndarray | None:
        if self.emb_sum is None:
            return None
        n = float(np.linalg.norm(self.emb_sum))
        return self.emb_sum / n if n > 0 else None

    def add_emb(self, e: np.ndarray | None) -> None:
        if e is not None:
            self.emb_sum = e.copy() if self.emb_sum is None else self.emb_sum + e

    @property
    def n_present(self) -> int:
        return sum(b is not None for b in self.boxes)

    def label(self) -> str:
        return f"Speaker {chr(ord('A') + self.track_id)}"


# --------------------------------------------------------------------------- #

class FaceAnalyzer:
    """Detect faces, track them, and emit a valid lip-aperture signal."""

    def __init__(self, cfg, *, progress=None):
        self.cfg = cfg
        self.progress = progress
        self._mesh = None
        self._detector = None
        self._backend = "none"
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            import mediapipe as mp
            if not hasattr(mp, "solutions"):
                raise AttributeError("mp.solutions removed in this mediapipe build")
            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,                 # one call per crop
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            self._backend = "mediapipe"
        except Exception as exc:
            log.warning("mediapipe unavailable (%s); using OpenCV fallback", exc)
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._backend = "opencv"

    @property
    def backend(self) -> str:
        return self._backend

    # -- main loop ---------------------------------------------------------- #

    def analyze(self, video_path: str) -> tuple[list[FaceTrack], dict]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg.target_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        tracks: list[FaceTrack] = []
        frame_idx = 0
        last_dets: list[tuple[float, float, float, float]] = []

        embedder = None
        if getattr(self.cfg, "reid", False):
            from .reid import FaceEmbedder
            embedder = FaceEmbedder()
            if not embedder.available:
                embedder = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            embs = None
            if frame_idx % self.cfg.detect_every == 0 or not last_dets:
                last_dets = self._detect(frame, width, height)
                # Identity is only measured on detection frames: between them
                # the boxes are the same ones, and so are their owners.
                if embedder is not None:
                    embs = [embedder.embed(frame, d) for d in last_dets]
            dets = last_dets

            assignment = _assign(tracks, dets, self.cfg.iou_match_threshold,
                                 frame_idx, embs=embs,
                                 gap=(self.cfg.reid_gap_frames if embedder else None),
                                 veto=self.cfg.reid_veto)

            for t in tracks:                       # pad every track to this frame
                while len(t.boxes) < frame_idx:
                    t.boxes.append(None)
                    t.lip.append(float("nan"))

            for track, box in assignment:
                lip = self._lip_feature(frame, box, width, height)
                while len(track.boxes) < frame_idx:
                    track.boxes.append(None)
                    track.lip.append(float("nan"))
                track.boxes.append(box)
                track.lip.append(lip)

            for t in tracks:
                while len(t.boxes) <= frame_idx:
                    t.boxes.append(None)
                    t.lip.append(float("nan"))

            frame_idx += 1
            if self.progress and total and frame_idx % 25 == 0:
                self.progress(frame_idx / total, f"faces {frame_idx}/{total}")

        cap.release()

        if embedder is not None:
            n_before = len(tracks)
            tracks = _merge_same_person(tracks)
            log.info("tracks: %d fragments -> %d people after re-identification",
                     n_before, len(tracks))

        tracks = [t for t in tracks if t.n_present >= self.cfg.min_track_frames]
        tracks.sort(key=lambda t: -t.n_present)
        tracks = tracks[: self.cfg.max_faces]
        for i, t in enumerate(tracks):
            t.track_id = i

        meta = {"fps": float(fps), "width": width, "height": height,
                "n_frames": frame_idx, "backend": self._backend}
        return tracks, meta

    # -- detection ---------------------------------------------------------- #

    def _detect(self, frame, width, height) -> list[tuple[float, float, float, float]]:
        """Return boxes as (x, y, w, h) normalised to [0, 1] in FRAME space."""
        if self._backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._detector.process(rgb)
            out = []
            if res.detections:
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    out.append((max(0.0, bb.xmin), max(0.0, bb.ymin),
                                min(1.0, bb.width), min(1.0, bb.height)))
            return out

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        return [(x / width, y / height, w / width, h / height) for x, y, w, h in faces]

    # -- the fixed lip feature ---------------------------------------------- #

    def _lip_feature(self, frame, box, width, height) -> float:
        """Scale-invariant lip aperture, or NaN.

        Returns ``inner_lip_area / interocular_distance**2``, computed from
        landmarks mapped back into **frame pixel** coordinates.
        """
        x, y, w, h = box
        # Pad the crop: FaceMesh needs some margin, and a tight box clips the chin.
        pad = 0.15
        x0 = int(max(0, (x - w * pad) * width))
        y0 = int(max(0, (y - h * pad) * height))
        x1 = int(min(width, (x + w * (1 + pad)) * width))
        y1 = int(min(height, (y + h * (1 + pad)) * height))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return float("nan")

        crop = frame[y0:y1, x0:x1]
        crop_w, crop_h = x1 - x0, y1 - y0

        if self._backend != "mediapipe":
            return _mouth_motion_proxy(crop)

        # Optional: upscale small crops before landmarking.  DISABLED by
        # default (min_face_px = 0) -- measured on a real 640x360 clip it did
        # not improve the gate (visual_activity normalises per track, so the
        # amplitude gain is divided out) and it flattened the matcher's
        # cross-modal correlation.  See VisionConfig.min_face_px for the
        # numbers.  Kept because it is the right lever for a genuinely
        # low-resolution source; landmarks come back crop-NORMALISED either
        # way, so the mapping below is unaffected by the resize -- which is the
        # whole reason to do it here rather than upscaling the frame.
        want = getattr(self.cfg, "min_face_px", 0)
        short = min(crop_w, crop_h)
        if want and short > 0 and short < want:
            f = min(float(getattr(self.cfg, "max_upscale", 4.0)), want / short)
            if f > 1.01:
                crop = cv2.resize(crop, (int(round(crop_w * f)), int(round(crop_h * f))),
                                  interpolation=cv2.INTER_CUBIC)

        res = self._mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return float("nan")
        lm = res.multi_face_landmarks[0].landmark

        # >>> THE FIX <<<  Map crop-normalised coords back to frame pixels.
        # lm.x is a fraction of crop_w, NOT of the frame.  Multiplying by
        # crop_w and adding the crop origin recovers absolute pixels; without
        # this every downstream number is scaled by an arbitrary, time-varying
        # crop size.
        #
        # Note this uses the ORIGINAL crop_w/crop_h, not the upscaled crop's
        # dimensions, and that is deliberate rather than an oversight: lm.x is a
        # fraction of whatever image was submitted, so a 2x upscale leaves it
        # unchanged.  Scaling by the resized width here would reintroduce
        # precisely the bug this comment marks -- a resolution-dependent ruler.
        def px(i: int) -> tuple[float, float]:
            return (x0 + lm[i].x * crop_w, y0 + lm[i].y * crop_h)

        ring = np.array([px(i) for i in INNER_LIP_RING], dtype=np.float64)
        area = _shoelace(ring)                                  # pixels^2

        eL, eR = np.array(px(LEFT_EYE_OUTER)), np.array(px(RIGHT_EYE_OUTER))
        interocular = float(np.linalg.norm(eL - eR))            # pixels
        if interocular < 1e-6:
            return float("nan")

        # Both numerator and denominator scale as (apparent face size)^2, so
        # the ratio is dimensionless: comparable across speakers and distances.
        return float(area / (interocular ** 2))


def _shoelace(poly: np.ndarray) -> float:
    """Absolute polygon area via the shoelace formula."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _mouth_motion_proxy(crop: np.ndarray) -> float:
    """Fallback feature: vertical-gradient energy in the lower-middle face.

    Crude, but normalised by the crop area so it does not inherit the very bug
    this module is about.
    """
    h, w = crop.shape[:2]
    mouth = crop[int(h * 0.60): int(h * 0.90), int(w * 0.25): int(w * 0.75)]
    if mouth.size == 0:
        return float("nan")
    g = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))))


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / (aw * ah + bw * bh - inter + 1e-12)


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a @ b)


def _assign(tracks: list[FaceTrack], dets, thresh: float, frame_idx: int, *,
            embs: list | None = None, gap: int | None = None, veto: float = 0.2):
    """Associate this frame's detections with tracks.

    1. **Position**: IoU against each track's last box, best pairs first, but
       only for tracks seen within ``gap`` frames, and refused when the
       embedding says it is clearly someone else (``veto``) -- a cut between
       two centred close-ups keeps the box and changes the person.
    2. **Appearance**: a detection left over is matched to the most similar
       track not already taken this frame (``reid.SAME_PERSON``), however long
       that track has been gone. This is what re-attaches a face after a cut.
    3. Anything still unmatched starts a new track. There is no cap here:
       capping at ``max_faces`` DURING the scan let early false detections use
       up the slots and every later face was dropped. ``analyze`` caps the
       final list instead.

    ``gap=None`` (no embedder) keeps the old position-only behaviour.
    """
    from .reid import SAME_PERSON

    out, used_d, used_t = [], set(), set()
    embs = embs if embs is not None else [None] * len(dets)

    pairs = []
    for ti, t in enumerate(tracks):
        if t.last_box is None:
            continue
        if gap is not None and frame_idx - t.last_seen > gap:
            continue
        for di, d in enumerate(dets):
            v = _iou(t.last_box, d)
            if v <= thresh:
                continue
            c = _cos(embs[di], t.emb())
            if c is not None and c < veto:
                continue
            pairs.append((v, ti, di))
    for v, ti, di in sorted(pairs, reverse=True):
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti); used_d.add(di)
        out.append((tracks[ti], dets[di], embs[di]))

    for di, d in enumerate(dets):
        if di in used_d or embs[di] is None:
            continue
        best, best_c = None, SAME_PERSON
        for ti, t in enumerate(tracks):
            if ti in used_t:
                continue
            c = _cos(embs[di], t.emb())
            if c is not None and c > best_c:
                best, best_c = ti, c
        if best is not None:
            used_t.add(best); used_d.add(di)
            out.append((tracks[best], d, embs[di]))

    for di, d in enumerate(dets):
        if di in used_d:
            continue
        t = FaceTrack(track_id=len(tracks))
        t.boxes = [None] * frame_idx
        t.lip = [float("nan")] * frame_idx
        tracks.append(t)
        out.append((t, d, embs[di]))

    for t, d, e in out:
        t.last_box, t.last_seen = d, frame_idx
        t.add_emb(e)
    return [(t, d) for t, d, _ in out]


def _merge_same_person(tracks: list[FaceTrack], max_overlap: float = 0.05) -> list[FaceTrack]:
    """Join fragments of one person that online re-ID left apart.

    Two tracks are one person when they look alike (``SAME_PERSON``) and are
    almost never on screen at the same time -- a person cannot be in two
    places, so heavy co-presence vetoes the merge whatever the embeddings say.
    Largest tracks absorb smaller ones first.
    """
    from .reid import SAME_PERSON

    tracks = sorted(tracks, key=lambda t: -t.n_present)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(tracks):
            pa = np.array([b is not None for b in a.boxes])
            for j in range(i + 1, len(tracks)):
                b = tracks[j]
                c = _cos(a.emb(), b.emb())
                if c is None or c < SAME_PERSON:
                    continue
                pb = np.array([x is not None for x in b.boxes])
                n = min(len(pa), len(pb))
                both = int((pa[:n] & pb[:n]).sum())
                if both > max_overlap * max(1, min(pa.sum(), pb.sum())):
                    continue
                for k in range(n):
                    if a.boxes[k] is None and b.boxes[k] is not None:
                        a.boxes[k], a.lip[k] = b.boxes[k], b.lip[k]
                if b.emb_sum is not None:
                    a.add_emb(b.emb_sum)
                tracks.pop(j)
                merged = True
                break
            if merged:
                break
    return tracks


def resample_lip(lip: list[float], src_fps: float, n_out: int, out_fps: float) -> np.ndarray:
    """Resample a lip signal onto the audio-envelope grid, NaN-aware.

    Thin wrapper over :func:`app.dsp.resample_hold`, which is the single
    implementation of the NaN/hold contract: NaN means "face not visible" (a
    real state, not zero aperture), short dropouts (<= 200 ms) hold the last
    valid value, longer ones go to 0.

    This wrapper drops the validity mask because the matcher does not need it
    -- it correlates, and an invalid stretch simply contributes nothing.  The
    **gate** does need it (an absent face must abstain, not veto), so callers
    on that path use ``resample_hold`` / ``visual_activity`` directly.
    """
    values, _valid = resample_hold(lip, src_fps, n_out, out_fps)
    return values
