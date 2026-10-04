# Vendored: AV-MossFormer2 TSE (16 kHz)

## Provenance

| | |
|:--|:--|
| Upstream | https://github.com/modelscope/ClearerVoice-Studio |
| Path | `clearvoice/clearvoice/models/av_mossformer2_tse/` |
| Branch | `main` |
| Retrieved | 2026-09-19 |
| Licence | Apache-2.0 (`visual_frontend.py` is MIT, © 2020 Smeet Shah) |
| Weights | `alibabasglab/AV_MossFormer2_TSE_16K` → `last_best_checkpoint.pt` |

## Why vendored rather than `pip install clearvoice`

The `clearvoice` package pins `numpy<2.0`, `opencv-python==4.10.0.84`,
`librosa==0.10.2.post1` and `scenedetect==0.6.6`. Those pins belong to the
*package*, not to the model. The complete import closure of the seven files
below is:

```
torch  torchaudio  numpy  einops  rotary_embedding_torch
```

`einops` declares no dependencies; `rotary-embedding-torch` declares only
`einops>=0.8, torch>=2.4`. Neither constrains numpy. Vendoring therefore lets
the whole project live in **one** virtualenv and removes the two-venv +
subprocess design in `ARCHITECTURE_V2.md` §5.5, along with its file-based IPC
and its `.cuda()` monkey-patch.

Verified on install: `pip install einops rotary-embedding-torch` resolved with
no changes to numpy, opencv, or any existing package.

## Files

| Vendored path | Upstream path | Bytes |
|:--|:--|--:|
| `av_mossformer2.py` | `av_mossformer2.py` | 7114 |
| `visual_frontend.py` | `visual_frontend.py` | 6065 |
| `utils/Transformer.py` | `mossformer/utils/Transformer.py` | 14993 |
| `utils/one_path_flash_fsmn.py` | `mossformer/utils/one_path_flash_fsmn.py` | 22605 |
| `utils/conv_module.py` | `mossformer/utils/conv_module.py` | 3487 |
| `utils/fsmn.py` | `mossformer/utils/fsmn.py` | 3341 |
| `utils/normalization.py` | `mossformer/utils/normalization.py` | 2599 |

## Local edits

Exactly two, both in `av_mossformer2.py`, both marked `VENDOR-EDIT` in-line:

1. **Line 9 — import path.** `from .mossformer.utils.one_path_flash_fsmn import ...`
   → `from .utils.one_path_flash_fsmn import ...`, because the `mossformer/`
   level was flattened away (it contained nothing else we use).

2. **Line 177 — hardcoded `.cuda()`.** In `overlap_and_add`:

   ```python
   frame = signal.new_tensor(frame).long().cuda()   # upstream
   frame = signal.new_tensor(frame).long().to(signal.device)   # here
   ```

   Upstream crashes on a CPU-only machine despite the adjacent comment claiming
   "signal may in GPU or CPU". `new_tensor` already inherits `signal`'s device,
   so the `.cuda()` was both redundant and wrong; `.to(signal.device)` is
   written explicitly rather than deleted so the intent survives a re-vendor.

`config.py` and `loader.py` in this directory are **ours**, not upstream — they
replace clearvoice's YAML/argparse plumbing.

## Re-vendoring

`scripts/vendor_avtse.py` re-downloads the seven files and re-applies both
edits, failing loudly if an anchor string has moved upstream.
