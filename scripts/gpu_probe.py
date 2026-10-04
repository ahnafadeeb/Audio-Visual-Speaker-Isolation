"""Does this torch wheel actually carry machine code for the installed GPU?

    .venv\\Scripts\\python.exe scripts\\gpu_probe.py

Run this FIRST on any new machine, before preflight and long before a demo.

`torch.cuda.is_available()` is not the check people think it is.  It proves a
driver and a CUDA runtime exist; it says nothing about whether the wheel was
compiled for the installed GPU's compute capability.  A wheel built only for
newer architectures imports fine, reports CUDA available, allocates memory
happily -- and then dies at the first real kernel launch with::

    RuntimeError: CUDA error: no kernel image is available for execution on the device

which surfaces minutes into a job, from deep inside SepFormer, looking like a
model bug.  PTX for an *earlier* arch can be JIT-compiled forward, never
backward, so a too-new wheel has no path to run at all.

This project targets two different cards:
  * GTX 1050 Ti  -- Pascal, sm_61, 4 GB  (the development machine)
  * RTX 4050/4060 -- Ada,    sm_89, 6-8 GB (the demo machine)

sm_61 is the one at risk: it is old enough that recent CUDA builds have begun
dropping it.  So this script does not stop at metadata -- it launches the actual
kernel families SepFormer uses (matmul, fp16 matmul, conv1d, SDPA) and reports
peak memory, which is the other thing that decides whether a 4 GB card copes.
"""

from __future__ import annotations

import sys
import time

try:
    import torch
except Exception as exc:                                    # pragma: no cover
    print(f"FAIL: cannot import torch: {exc}")
    raise SystemExit(1)

print("=" * 70)
print("GPU probe")
print("=" * 70)
print(f"torch           : {torch.__version__}")
print(f"built with CUDA : {torch.version.cuda}")
print(f"cuda available  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nFAIL: no CUDA device visible. Either the driver is missing or a "
          "CPU-only wheel got installed (the classic symptom of forgetting "
          "--index-url on the torch install).")
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
sm = f"sm_{props.major}{props.minor}"
free, total = torch.cuda.mem_get_info()

print(f"device          : {props.name}")
print(f"capability      : {props.major}.{props.minor}   ({sm})")
print(f"total VRAM      : {props.total_memory / 2**30:.2f} GB")
print(f"free VRAM now   : {free / 2**30:.2f} GB "
      f"({(total - free) / 2**30:.2f} GB already in use)")

archs = torch.cuda.get_arch_list()
print(f"wheel archs     : {' '.join(archs)}")

if sm in archs:
    print(f"VERDICT (static): {sm} IS in this wheel -- native SASS, no JIT.")
else:
    ptx = [a for a in archs if a.startswith("compute_")]
    usable = [p for p in ptx
              if int(p.split("_")[1]) <= props.major * 10 + props.minor]
    if usable:
        print(f"VERDICT (static): {sm} absent, but PTX {usable} can JIT forward "
              f"to it (slow first launch, then cached).")
    else:
        print(f"VERDICT (static): {sm} NOT in this wheel and no PTX can reach "
              f"it. Kernels will fail at launch.")

# --------------------------------------------------------------------------- #
# The part that actually settles it.  Everything above is metadata.
# --------------------------------------------------------------------------- #

print("\n-- launching real kernels " + "-" * 44)

failures: list[str] = []


def stage(name: str, fn) -> None:
    t0 = time.time()
    try:
        fn()
        torch.cuda.synchronize()
        print(f"  [ ok ] {name:<14} {(time.time() - t0) * 1000:7.0f} ms")
    except Exception as exc:
        first = str(exc).strip().splitlines()[0]
        print(f"  [FAIL] {name:<14} {first}")
        failures.append(name)


x = None


def _alloc() -> None:
    global x
    x = torch.randn(2048, 2048, device="cuda")


stage("alloc", _alloc)
stage("fp32 matmul", lambda: (x @ x).sum().item())
stage("fp16 matmul", lambda: (lambda h: (h @ h).sum().item())(
    torch.randn(512, 512, device="cuda", dtype=torch.float16)))
# conv1d and SDPA are SepFormer's own kernel mix: the encoder/decoder are 1-D
# convolutions and the dual-path blocks are attention.
stage("conv1d", lambda: torch.nn.functional.conv1d(
    torch.randn(1, 8, 16_000, device="cuda"),
    torch.randn(8, 8, 33, device="cuda"), padding=16).sum().item())
stage("sdpa", lambda: torch.nn.functional.scaled_dot_product_attention(
    *(torch.randn(1, 4, 512, 64, device="cuda"),) * 3).sum().item())

print(f"\npeak allocated  : {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

if failures:
    print(f"\nNOT USABLE -- {len(failures)} kernel family/families failed: "
          f"{', '.join(failures)}")
    print("Reinstall torch from an index that still builds "
          f"{sm}, or run on CPU (--device cpu).")
    sys.exit(1)

print("\nUSABLE -- every kernel family SepFormer needs launched on this GPU.")
