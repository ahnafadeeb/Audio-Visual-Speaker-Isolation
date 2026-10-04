"""Model hyper-parameters for the vendored AV-MossFormer2 TSE network.

Replaces clearvoice's YAML + argparse plumbing. Values transcribed verbatim
from upstream ``clearvoice/config/inference/AV_MossFormer2_TSE_16K.yaml``
(retrieved 2026-09-19) -- see VENDORED.md.

These describe the *checkpoint*, not our runtime. They must match the released
weights exactly or ``load_state_dict`` will fail on shape. Runtime knobs
(chunking, device, batch) live in ``app/config.py`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["NetworkReferenceArgs", "NetworkAudioArgs", "AVTSEModelArgs", "default_args"]

#: The checkpoint is 16 kHz-only. Upstream asserts this; so do we.
SAMPLE_RATE = 16000

#: Visual frames per second the model was trained at. A data contract, not a
#: preference: the visual stream is interpolated onto the audio frame grid in
#: ``Separator.forward``, so a wrong fps silently misaligns lips against audio.
VIDEO_FPS = 25

#: Mouth ROI side length. Structurally pinned -- ``VisualFrontend`` ends in
#: ``AvgPool2d(kernel_size=(4,4))`` after four stride-2 stages, so 112 -> 4 -> 1
#: is the only input that makes ``reshape(B, -1, 512)`` valid.
ROI_SIZE = 112


@dataclass(frozen=True)
class NetworkReferenceArgs:
    """Visual (reference / cue) branch."""

    cue: str = "lip"
    backbone: str = "resnet18"
    emb_size: int = 256


@dataclass(frozen=True)
class NetworkAudioArgs:
    """Audio branch: encoder, MossFormer2 masknet, decoder."""

    backbone: str = "mossformer2"

    # Learned time-domain encoder/decoder. Stride is L//2 = 8 samples = 0.5 ms.
    encoder_kernel_size: int = 16
    encoder_out_nchannels: int = 512
    encoder_in_nchannels: int = 1

    # TSE emits ONE stream -- this is the whole point of the architecture and
    # the reason there is no permutation to align and no matcher to get wrong.
    masknet_numspks: int = 1
    masknet_chunksize: int = 250
    masknet_numlayers: int = 1
    masknet_norm: str = "ln"
    masknet_useextralinearlayer: bool = False
    masknet_extraskipconnection: bool = True

    intra_numlayers: int = 24
    intra_nhead: int = 8
    intra_dffn: int = 1024
    intra_dropout: float = 0.0
    intra_use_positional: bool = True
    intra_norm_before: bool = True


@dataclass
class AVTSEModelArgs:
    """Top-level args bundle handed to ``av_mossformer2``.

    Deliberately NOT frozen: ``av_mossformer2.__init__`` assigns ``args.causal
    = 0`` on construction. Left mutable rather than patched out, to keep the
    vendored diff at the two edits recorded in VENDORED.md.
    """

    network_reference: NetworkReferenceArgs = field(default_factory=NetworkReferenceArgs)
    network_audio: NetworkAudioArgs = field(default_factory=NetworkAudioArgs)

    #: 0 = non-causal (offline). Changes padding in the 3D frontend and the
    #: visual adaptor, so it must match how the checkpoint was trained.
    causal: int = 0

    sampling_rate: int = SAMPLE_RATE


def default_args() -> AVTSEModelArgs:
    """Args matching the released ``AV_MossFormer2_TSE_16K`` checkpoint."""
    return AVTSEModelArgs()
