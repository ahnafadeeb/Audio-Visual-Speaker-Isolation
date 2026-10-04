"""Checkpoint loading for the vendored AV-MossFormer2 TSE model.

Ours, not upstream. Replaces clearvoice's checkpoint plumbing with something
that (a) works on CPU, (b) never silently tolerates a key mismatch, and
(c) caches the built model so the web app pays the load cost once, and
(d) applies an optional *adapter* -- a small file of fine-tuned tensors for
the last few layers, made by ``python -m app.adapt`` -- on top of the
released weights.
"""

from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

import torch

from .av_mossformer2 import av_mossformer2
from .config import AVTSEModelArgs, default_args

__all__ = ["REPO_ID", "CHECKPOINT_FILE", "download_checkpoint", "load_model", "clear_cache",
           "read_adapter", "apply_adapter"]

log = logging.getLogger(__name__)

REPO_ID = "alibabasglab/AV_MossFormer2_TSE_16K"

#: The repo also carries ``last_best_checkpoint_old.pt`` (a second ~700 MB
#: copy) and an extension-less ``last_best_checkpoint``. Name the file
#: explicitly and use ``hf_hub_download`` -- a ``git clone`` pulls all three.
CHECKPOINT_FILE = "last_best_checkpoint.pt"

_CACHE: dict[tuple[str, str], av_mossformer2] = {}
_LOCK = threading.Lock()


def download_checkpoint(revision: str | None = None) -> Path:
    """Fetch the checkpoint into ``HF_HOME`` and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, CHECKPOINT_FILE, revision=revision))


def _strip_ddp(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the ``module.`` prefix left by DistributedDataParallel training.

    The released checkpoint was saved from a DDP-wrapped model, so every key
    carries the prefix. Stripping is unconditional-but-guarded: if a future
    checkpoint is saved unwrapped, the keys pass through untouched.
    """
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_model(
    device: torch.device | str = "cpu",
    args: AVTSEModelArgs | None = None,
    checkpoint: Path | str | None = None,
    adapter: Path | str | None = None,
) -> av_mossformer2:
    """Build the network and load the released weights.

    With ``adapter``, returns a separate copy of that model with the adapter's
    tensors loaded over it (see :func:`apply_adapter`); the base model stays
    cached and untouched, so switching adapters never needs a reload from disk.

    Loaded with ``strict=True`` on purpose. A silently-partial load produces a
    model that runs and returns plausible-sounding garbage, which is far more
    expensive to debug than an exception at startup.

    Returns a module in ``eval()`` mode with gradients disabled.
    """
    if adapter is not None:
        return _load_adapted(device, args, checkpoint, Path(adapter))
    device = torch.device(device)
    path = Path(checkpoint) if checkpoint is not None else download_checkpoint()
    key = (str(path), str(device))

    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        log.info("loading AV-MossFormer2 TSE from %s onto %s", path.name, device)

        # weights_only=False: this is a full training checkpoint (model,
        # optimizer, epoch, ...), not a bare tensor dict. Safe because the
        # file comes from a pinned, non-gated HF repo we control the id of.
        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = _strip_ddp(raw["model"] if "model" in raw else raw)

        model = av_mossformer2(args or default_args())
        model.load_state_dict(state, strict=True)

        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        _CACHE[key] = model
        return model


def read_adapter(path: Path | str) -> dict:
    """Load an adapter file and check it was made for these weights.

    ``weights_only=True``: an adapter is a plain dict of tensors and strings,
    and unlike the released checkpoint it is a file people copy between
    machines, so nothing in it is allowed to execute on load.
    """
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("state"), dict):
        raise ValueError(f"{path}: not an adapter file (no 'state' dict)")
    if raw.get("base") != REPO_ID:
        raise ValueError(f"{path}: adapter was made for {raw.get('base')!r}, not {REPO_ID!r}")
    return raw


def apply_adapter(model: av_mossformer2, state: dict[str, torch.Tensor]) -> None:
    """Copy adapter tensors over the matching parameters, strictly.

    Same policy as the base load: every key must name an existing parameter
    of the same shape, or nothing is changed and this raises.
    """
    params = dict(model.named_parameters())
    bad = [k for k, v in state.items() if k not in params or params[k].shape != v.shape]
    if bad:
        raise ValueError(f"adapter does not fit this model: {len(bad)} bad keys, e.g. {bad[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def _load_adapted(device, args, checkpoint, path: Path) -> av_mossformer2:
    # Keyed on mtime too, so re-training an adapter under the same name is
    # picked up by the next job without restarting the server.
    key = (f"adapter:{path.resolve()}:{path.stat().st_mtime_ns}", str(torch.device(device)))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
    base = load_model(device, args, checkpoint)
    raw = read_adapter(path)
    log.info("applying adapter %s (%d tensors, last %s layers)", path.name,
             len(raw["state"]), raw.get("last_layers", "?"))
    model = copy.deepcopy(base)
    apply_adapter(model, raw["state"])
    with _LOCK:
        # One adapted copy resident at a time: each is another ~260 MB of VRAM.
        for k in [k for k in _CACHE if k[0].startswith("adapter:")]:
            del _CACHE[k]
        _CACHE[key] = model
    return model


def clear_cache() -> None:
    """Drop cached models (frees VRAM). Used by tests and the VRAM bench."""
    with _LOCK:
        _CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
