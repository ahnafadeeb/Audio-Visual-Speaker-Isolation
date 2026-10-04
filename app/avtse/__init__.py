"""Vendored AV-MossFormer2 target-speaker-extraction model.

See VENDORED.md for provenance and the (two) local edits.
"""

from .av_mossformer2 import AV_MossFormer2_TSE_16K, av_mossformer2
from .config import (
    ROI_SIZE,
    SAMPLE_RATE,
    VIDEO_FPS,
    AVTSEModelArgs,
    default_args,
)
from .loader import (
    CHECKPOINT_FILE,
    REPO_ID,
    apply_adapter,
    clear_cache,
    download_checkpoint,
    load_model,
    read_adapter,
)

__all__ = [
    "AV_MossFormer2_TSE_16K",
    "av_mossformer2",
    "AVTSEModelArgs",
    "default_args",
    "SAMPLE_RATE",
    "VIDEO_FPS",
    "ROI_SIZE",
    "load_model",
    "download_checkpoint",
    "clear_cache",
    "REPO_ID",
    "CHECKPOINT_FILE",
    "read_adapter",
    "apply_adapter",
]
