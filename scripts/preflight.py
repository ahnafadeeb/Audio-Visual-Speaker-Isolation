"""Pre-flight: prove the machine can run a job BEFORE the demo starts.

    python scripts/preflight.py            # check
    python scripts/preflight.py --fetch    # also download model weights now

Every check is something that has actually broken a live demo: a missing
ffmpeg, a CPU-only torch wheel that silently installs when the CUDA index URL
is forgotten, a numpy 2.x that mediapipe refuses to import against, weights
that were never cached and now need 400 MB over conference wifi.

Exit code 0 = ready.  1 = at least one hard failure.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, CONFIG, ensure_dirs        # noqa: E402

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #

def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    # mediapipe 0.10.21 is the last release exposing mp.solutions.face_mesh and
    # it ships per-interpreter wheels only up to cp312.  3.13 gets you no wheel.
    if (v.major, v.minor) == (3, 12):
        record(OK, "python", detail)
    elif (v.major, v.minor) in {(3, 10), (3, 11)}:
        record(WARN, "python", detail + " -- 3.12 is the tested version")
    else:
        record(FAIL, "python", detail + " -- need 3.12 (mediapipe 0.10.21 has no wheel here)")


def check_versions() -> None:
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major == 1:
        record(OK, "numpy", np.__version__)
    else:
        record(FAIL, "numpy", f"{np.__version__} -- mediapipe 0.10.21 pins numpy<2; "
                              "run: pip install numpy==1.26.4")

    for mod in ("scipy", "soundfile", "cv2", "fastapi", "uvicorn"):
        try:
            m = importlib.import_module(mod)
            record(OK, mod, getattr(m, "__version__", "?"))
        except Exception as exc:
            record(FAIL, mod, f"import failed: {exc}")


def check_torch() -> str:
    try:
        import torch
    except Exception as exc:
        record(FAIL, "torch", f"import failed: {exc}")
        return "cpu"

    if not torch.cuda.is_available():
        # The classic failure: `pip install torch` grabs the CPU wheel from
        # PyPI, everything imports fine, and inference is 30x slower.
        record(WARN, "torch", f"{torch.__version__} -- CUDA NOT available; "
                              "CPU inference will be slow. Reinstall with "
                              "--index-url https://download.pytorch.org/whl/cu128")
        return "cpu"

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 2**30
    record(OK, "torch", f"{torch.__version__} cuda={torch.version.cuda} "
                        f"| {name} ({vram:.1f} GB)")

    # Does this wheel actually carry code for THIS card?
    #
    # `cuda.is_available()` only proves a driver and a runtime exist -- it says
    # nothing about whether the binary was compiled for the installed GPU's
    # compute capability.  A wheel built for newer architectures loads happily,
    # reports CUDA available, and then fails at the first real kernel launch
    # with "no kernel image is available for execution on the device" -- from
    # deep inside SepFormer, minutes into a job, looking like a model bug.
    #
    # Pascal (GTX 10xx, sm_61) is the live case: the cu128 wheel that
    # requirements.txt names for the Ada target machine does not carry sm_61.
    # PTX from a lower arch can be JIT'd forward but never backward, so a
    # too-new wheel has no path to run here at all.
    cap = f"{props.major}.{props.minor}"
    sm = f"sm_{props.major}{props.minor}"
    archs = torch.cuda.get_arch_list()
    if sm in archs:
        record(OK, "gpu arch", f"{sm} (cc {cap}) present in wheel: {' '.join(archs)}")
    else:
        # A PTX blob for an EARLIER arch can be JIT-compiled forward to this one.
        ptx = [a for a in archs if a.startswith("compute_")]
        jit = [p for p in ptx if int(p.split("_")[1]) <= props.major * 10 + props.minor]
        if jit:
            record(WARN, "gpu arch", f"{sm} absent; may JIT from {jit} "
                                     f"(slow first launch)")
        else:
            record(FAIL, "gpu arch",
                   f"{sm} (cc {cap}) NOT in this wheel: {' '.join(archs)} -- "
                   f"kernels will fail at launch. Reinstall torch from an index "
                   f"that still builds {sm}.")
            return "cpu"

    # Report the chunk size this card will ACTUALLY get, using the same
    # resolver the pipeline uses -- not a threshold restated here, which is how
    # a preflight starts reassuring you about a value the job does not use.
    free, total = torch.cuda.mem_get_info()
    chunk_s = CONFIG.audio.resolve_chunk_s("cuda")
    a = CONFIG.audio
    predicted = a.vram_fixed_mb + a.vram_per_chunk_s_mb * chunk_s
    detail = (f"{free / 2**30:.1f} GB free of {vram:.1f} GB -- chunk_s "
              f"{chunk_s:.1f}s, predicted peak {predicted:.0f} MB "
              f"({100 * predicted / (free / 2**20):.0f}% of free)")
    if chunk_s < CONFIG.audio.chunk_s:
        record(WARN, "vram", detail + f" (reduced from {CONFIG.audio.chunk_s:.1f}s; "
                                      f"close other GPU apps to raise it)")
    else:
        record(OK, "vram", detail)

    try:                                        # a real allocation, not a query
        t0 = time.time()
        x = torch.randn(2048, 2048, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        record(OK, "cuda matmul", f"{(time.time() - t0) * 1000:.0f} ms")
        del x
        torch.cuda.empty_cache()
    except Exception as exc:
        record(FAIL, "cuda matmul", str(exc))
        return "cpu"
    return "cuda"


def check_mediapipe() -> None:
    try:
        import mediapipe as mp
    except Exception as exc:
        record(WARN, "mediapipe", f"import failed ({exc}); "
                                  "vision falls back to the OpenCV Haar detector")
        return
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        record(OK, "mediapipe", f"{mp.__version__} (solutions.face_mesh present)")
    else:
        record(WARN, "mediapipe", f"{mp.__version__} -- no mp.solutions.face_mesh; "
                                  "pin mediapipe==0.10.21")


def check_ffmpeg() -> None:
    from app import media
    try:
        exe = media.ffmpeg_exe()
    except RuntimeError as exc:
        record(FAIL, "ffmpeg", str(exc).splitlines()[0])
        return
    p = subprocess.run([exe, "-version"], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    ver = (p.stdout or "").splitlines()[0] if p.returncode == 0 else "unreadable"
    where = "PATH" if shutil.which("ffmpeg") else "imageio-ffmpeg (bundled)"
    record(OK if p.returncode == 0 else FAIL, "ffmpeg", f"{ver}  [{where}]")

    if not media.ffprobe_exe():
        record(WARN, "ffprobe", "not on PATH -- duration probing is skipped (harmless)")


def check_weights(fetch: bool) -> None:
    if CONFIG.runtime.separator == "passthrough":
        record(OK, "weights", "passthrough separator -- none needed")
        return

    hf = Path(os.environ.get("HF_HOME", CACHE_DIR / "hf"))
    _check_sepformer(hf, fetch)
    # The AV path needs two more models, and neither is fetched by the first
    # SepFormer job.  A copied-over .cache is the trap: the HF snapshot dirs
    # survive the copy but their blobs may not, so test for the real files.
    _check_avtse(fetch)
    _check_ecapa(hf, fetch)
    _check_reid(fetch)
    _check_adapter()


def _check_adapter() -> None:
    """The active speaker/room adapter (app/adapt.py), if there is one.

    Checked without the GPU: the file must parse, name the released weights,
    and every tensor must fit a parameter of the model -- the same strict
    policy load_model applies, so a job cannot be the first to find out.
    """
    try:
        path = CONFIG.avtse.resolve_adapter()
    except FileNotFoundError as exc:
        record(FAIL, "adapter", str(exc))
        return
    if path is None:
        record(OK, "adapter", "none active -- released weights "
                              "(python -m app.adapt to make one for your team)")
        return
    try:
        from app.avtse import av_mossformer2, default_args, read_adapter
        raw = read_adapter(path)
        shapes = {n: p.shape for n, p in av_mossformer2(default_args()).named_parameters()}
        bad = [k for k, v in raw["state"].items() if shapes.get(k) != v.shape]
        if bad:
            record(FAIL, "adapter", f"{path.name}: {len(bad)} tensors do not fit the model")
            return
        faces = len(raw.get("faces") or [])
        record(OK if faces else WARN, "adapter",
               f"{path.stem}: {len(raw['state'])} tensors, trained {raw.get('created')}, "
               + (f"{faces} known faces" if faces else
                  "NO face embeddings -- it will be applied to strangers too"))
    except Exception as exc:
        record(FAIL, "adapter", f"{path.name}: {type(exc).__name__}: {exc}")


def _check_reid(fetch: bool) -> None:
    """SFace + YuNet, for re-identifying faces across camera cuts (app/reid.py).

    Without them the tracker still runs, by position only -- which on an
    edited clip loses most of each speaker. Worth knowing before a demo.
    """
    from huggingface_hub import try_to_load_from_cache
    from app.reid import _SFACE, _YUNET

    missing = [f for r, f in (_SFACE, _YUNET)
               if not isinstance(try_to_load_from_cache(r, f), str)]
    if not missing:
        record(OK, "face re-id", "SFace + YuNet cached")
        return
    if not fetch:
        record(WARN, "face re-id", "not cached -- first job downloads ~39 MB. "
                                   "Run with --fetch to do it now.")
        return
    try:
        from app.reid import FaceEmbedder
        ok = FaceEmbedder().available
        record(OK if ok else FAIL, "face re-id",
               "downloaded and loadable" if ok else "could not load SFace/YuNet")
    except Exception as exc:
        record(FAIL, "face re-id", f"{type(exc).__name__}: {exc}")


def _check_sepformer(hf: Path, fetch: bool) -> None:
    cached = list(hf.rglob("*sepformer-whamr16k*")) if hf.exists() else []
    if cached:
        record(OK, "sepformer", f"cached in {hf}")
        return

    if not fetch:
        record(WARN, "sepformer", f"not cached under {hf} -- first job downloads "
                                  "~400 MB. Run with --fetch to do it now.")
        return

    record(WARN, "sepformer", "downloading (~400 MB) ...")
    try:
        from app.separation import build_separator
        sep = build_separator("sepformer", device="cpu",
                              cache_dir=str(CACHE_DIR / "models"))
        sep.load()
        record(OK, "sepformer", "downloaded and loadable")
    except Exception as exc:
        record(FAIL, "sepformer", f"{type(exc).__name__}: {exc}")


def _check_avtse(fetch: bool) -> None:
    from huggingface_hub import try_to_load_from_cache
    from app.avtse.loader import CHECKPOINT_FILE, REPO_ID

    path = try_to_load_from_cache(REPO_ID, CHECKPOINT_FILE)
    if isinstance(path, str) and Path(path).is_file():
        record(OK, "av-tse", f"cached ({Path(path).stat().st_size / 2**20:.0f} MB)")
        return

    if not fetch:
        record(WARN, "av-tse", "not cached -- first AV job downloads ~734 MB. "
                               "Run with --fetch to do it now.")
        return

    record(WARN, "av-tse", "downloading (~734 MB) ...")
    try:
        from app.avtse.loader import clear_cache, load_model
        load_model("cpu")          # strict=True: a partial download fails here
        clear_cache()
        record(OK, "av-tse", "downloaded and loadable")
    except Exception as exc:
        record(FAIL, "av-tse", f"{type(exc).__name__}: {exc}")


def _check_ecapa(hf: Path, fetch: bool) -> None:
    ckpt = hf / "ecapa" / "embedding_model.ckpt"
    if ckpt.is_file():
        record(OK, "ecapa", f"cached in {ckpt.parent}")
        return

    if not fetch:
        record(WARN, "ecapa", "not cached -- first AV job downloads ~80 MB. "
                              "Run with --fetch to do it now.")
        return

    record(WARN, "ecapa", "downloading (~80 MB) ...")
    try:
        from app.identity import load_encoder
        load_encoder("cpu")
        record(OK, "ecapa", "downloaded and loadable")
    except Exception as exc:
        record(FAIL, "ecapa", f"{type(exc).__name__}: {exc}")


def check_paths() -> None:
    ensure_dirs()
    from app.config import PROJECT_ROOT, RUNS_DIR, STATIC_DIR
    for name in ("index.html", "app.js", "style.css"):
        p = STATIC_DIR / name
        record(OK if p.exists() else FAIL, f"static/{name}",
               "" if p.exists() else "missing")
    try:
        probe = RUNS_DIR / ".writetest"
        probe.write_text("x")
        probe.unlink()
        free = shutil.disk_usage(PROJECT_ROOT).free / 2**30
        record(OK, "runs dir", f"writable, {free:.1f} GB free")
    except Exception as exc:
        record(FAIL, "runs dir", str(exc))


def check_roundtrip() -> None:
    """End-to-end DSP sanity: does the gate actually produce exact zeros?"""
    import numpy as np
    from app import dsp

    sr = CONFIG.audio.sample_rate
    t = np.arange(3 * sr) / sr
    loud = (np.sin(2 * np.pi * 200 * t) * (t < 1.0)).astype(np.float32)
    quiet = (loud * 10 ** (-25 / 20))[::-1].copy()          # a -25 dB "ghost"
    gated = dsp.apply_gate(np.stack([loud, quiet]), sample_rate=sr, cfg=CONFIG.gate)

    silent_frac = float((gated[0][int(1.2 * sr):] == 0.0).mean())
    if silent_frac > 0.98:
        record(OK, "gate", f"{silent_frac * 100:.1f}% bit-exact zeros after speech ends")
    else:
        record(FAIL, "gate", f"only {silent_frac * 100:.1f}% zeros -- "
                             "retune GateConfig in app/config.py")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true",
                    help="download model weights now instead of on the first job")
    args = ap.parse_args()

    print("=" * 74)
    print("AV Speaker Isolation -- preflight")
    print("=" * 74)

    check_python()
    check_versions()
    device = check_torch()
    check_mediapipe()
    check_ffmpeg()
    check_paths()
    check_weights(args.fetch)
    check_roundtrip()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]

    print("=" * 74)
    if fails:
        print(f"NOT READY -- {len(fails)} failure(s), {len(warns)} warning(s):")
        for _, name, detail in fails:
            print(f"  * {name}: {detail}")
        return 1
    print(f"READY on {device}" + (f" -- {len(warns)} warning(s)" if warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
