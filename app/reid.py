"""Face re-identification for the tracker: who is this face, across cuts?

IoU tracking alone assumes a face moves continuously. Edited video breaks
that twice over: after a camera cut a returning face lands somewhere new (so
IoU cannot find its old track), and a cut between two close-ups puts a
DIFFERENT person in the same spot (so IoU happily continues the wrong track).
Measured on a 196 s Bengali talk-show excerpt with cuts, the IoU-only tracker
kept each person for 0.2-1.6% of the clip.

An appearance embedding answers both: OpenCV's SFace (128-d, cosine; OpenCV's
own same-identity threshold is 0.363), with YuNet supplying the five
landmarks SFace's alignment needs. Both ship with opencv-contrib, so this adds
~39 MB of ONNX weights and no packages. Weights come from the ``opencv`` org on
the Hugging Face hub and land in HF_HOME like every other model here.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["SAME_PERSON", "FaceEmbedder"]

#: OpenCV's published cosine threshold for SFace "same identity".
SAME_PERSON = 0.363

_SFACE = ("opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx")
_YUNET = ("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx")

_LOCK = threading.Lock()


def _fetch(repo_file: tuple[str, str]) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(*repo_file)


class FaceEmbedder:
    """Embed a face given the frame and its (normalised xywh) box.

    Not thread-safe per instance (the OpenCV nets hold state); the analyser
    owns one. ``available`` is False when the weights cannot be fetched, and
    the tracker then falls back to IoU-only -- degraded, never broken.
    """

    def __init__(self) -> None:
        self.available = False
        try:
            with _LOCK:
                sface, yunet = _fetch(_SFACE), _fetch(_YUNET)
            self._rec = cv2.FaceRecognizerSF.create(sface, "")
            self._det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6, 0.3, 5)
            self.available = True
        except Exception as exc:                          # pragma: no cover
            log.warning("face re-identification unavailable (%s: %s); "
                        "tracking by position only", type(exc).__name__, exc)

    def embed(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
        """Unit-norm 128-d embedding, or None if no face is found in the box.

        YuNet runs on a padded crop around the box rather than the whole
        frame: it is only asked for landmarks of a face we already found,
        which is both cheaper and immune to picking a different face.
        """
        if not self.available:
            return None
        H, W = frame.shape[:2]
        x, y, w, h = box
        pad = 0.35
        x0 = int(max(0, (x - w * pad) * W)); y0 = int(max(0, (y - h * pad) * H))
        x1 = int(min(W, (x + w * (1 + pad)) * W)); y1 = int(min(H, (y + h * (1 + pad)) * H))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        crop = frame[y0:y1, x0:x1]
        # YuNet is fastest and most reliable around 160-320 px; scale the crop
        # to that rather than asking it to find a 600 px face.
        s = 240.0 / max(crop.shape[:2])
        if s < 1.0:
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)
        self._det.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self._det.detect(crop)
        if faces is None or len(faces) == 0:
            return None
        # The face nearest the crop centre is the one the box was about.
        cx, cy = crop.shape[1] / 2, crop.shape[0] / 2
        f = min(faces, key=lambda r: (r[0] + r[2] / 2 - cx) ** 2 + (r[1] + r[3] / 2 - cy) ** 2)
        aligned = self._rec.alignCrop(crop, f)
        e = self._rec.feature(aligned).reshape(-1).astype(np.float32)
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else None
