"""Measure SepFormer's real peak VRAM against chunk_s, on THIS card.

    .venv\\Scripts\\python.exe scripts\\bench_vram.py
    .venv\\Scripts\\python.exe scripts\\bench_vram.py --max 12

``AudioConfig`` used to carry a hand-estimated small-VRAM chunk length, derived
from the shape of the model: dual-path attention is quadratic in chunk count, so
peak activation memory "must" grow ~chunk_s^2.  This script measured it and the
estimate was wrong -- the growth is linear below ~16 s (see the fit printed at
the end, and the note in ``AudioConfig``).  An estimate is the wrong basis for
the one number that decides whether a live demo OOMs; the config now stores the
measured line and re-derives chunk_s per card at run time.

For each candidate chunk length it runs a real ``separate_batch`` on synthetic
audio and reports ``torch.cuda.max_memory_allocated``.  It stops at the first
OOM, which is itself the answer: the largest chunk that survived is the largest
this card can be trusted with, and the headroom column says by how much.

Peak memory is reset between runs but the model stays loaded -- weights are a
fixed cost you always pay, so they belong in the measurement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from app.config import CACHE_DIR, CONFIG                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=float, default=12.0,
                    help="largest chunk length to try, seconds")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device -- nothing to measure")
        return 1

    props = torch.cuda.get_device_properties(0)
    free0, total = torch.cuda.mem_get_info()
    print("=" * 78)
    print(f"VRAM benchmark -- {props.name}")
    print(f"  total {total / 2**30:.2f} GB | free before load {free0 / 2**30:.2f} GB "
          f"| {(total - free0) / 2**30:.2f} GB held by other processes")
    print("=" * 78)

    from app.separation import SepformerSeparator

    sep = SepformerSeparator(device=args.device,
                             cache_dir=str(CACHE_DIR / "models"))
    t0 = time.time()
    sep.load()
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 2**20
    print(f"\nmodel loaded in {time.time() - t0:.1f}s -- weights resident: "
          f"{weights_mb:.0f} MB")

    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"free after load: {free_after_load / 2**30:.2f} GB\n")

    sr = CONFIG.audio.sample_rate
    rng = np.random.default_rng(0)

    header = (f"{'chunk_s':>8}  {'peak MB':>9}  {'activations':>12}  "
              f"{'free left':>10}  {'wall':>7}   result")
    print(header)
    print("-" * len(header))

    rows: list[tuple[float, float, float]] = []
    largest_ok = 0.0

    for chunk_s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        if chunk_s > args.max:
            break
        # Speech-like: band-limited noise, not white -- amplitude distribution
        # does not change allocation, but keeping it realistic costs nothing.
        n = int(chunk_s * sr)
        x = rng.standard_normal(n).astype(np.float32) * 0.1

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.inference_mode():
                t = torch.from_numpy(x).unsqueeze(0).to(args.device)
                est = sep._model.separate_batch(t)
                est = est.squeeze(0).transpose(0, 1).cpu().numpy()
                del t, est
            torch.cuda.synchronize()
            wall = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20
            free_now, _ = torch.cuda.mem_get_info()
            act = peak - weights_mb
            rows.append((chunk_s, peak, act))
            largest_ok = chunk_s
            rtf = wall / chunk_s
            print(f"{chunk_s:>8.1f}  {peak:>9.0f}  {act:>12.0f}  "
                  f"{free_now / 2**20:>9.0f}M  {wall:>6.2f}s   ok  "
                  f"(RTF {rtf:.2f}x)")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{chunk_s:>8.1f}  {'--':>9}  {'--':>12}  {'--':>10}  "
                  f"{'--':>7}   OOM")
            break
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"{chunk_s:>8.1f}  {'--':>9}  {'--':>12}  {'--':>10}  "
                  f"{'--':>7}   {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:40]}")
            break

    print("-" * len(header))

    if not rows:
        print("\nNothing fit. This card cannot run the model on GPU.")
        return 1

    # How does activation memory actually scale?  The docstring claims ~n^2;
    # check it rather than repeating it.
    if len(rows) >= 3:
        import math
        (c1, _, a1), (c2, _, a2) = rows[0], rows[-1]
        if a1 > 0 and a2 > 0 and c2 > c1:
            expo = math.log(a2 / a1) / math.log(c2 / c1)
            print(f"\nactivation memory scales as chunk_s^{expo:.2f} "
                  f"(measured {c1:.0f}s -> {c2:.0f}s)")

    a = CONFIG.audio
    cfg_chunk = a.chunk_s
    resolved = a.resolve_chunk_s(args.device)
    print(f"\nlargest chunk that ran : {largest_ok:.1f}s")
    print(f"config chunk_s         : {cfg_chunk:.1f}s "
          f"({'FITS' if largest_ok >= cfg_chunk else 'DOES NOT FIT'})")
    print(f"resolve_chunk_s() here : {resolved:.1f}s "
          f"({'fits' if largest_ok >= resolved else 'DOES NOT FIT'})")

    # Check the stored linear model against what was just measured, rather than
    # recommending a new number from a rule of thumb.  If these two columns
    # diverge on your card, re-fit vram_fixed_mb / vram_per_chunk_s_mb -- that is
    # what resolve_chunk_s extrapolates from on machines nobody benchmarked.
    print(f"\nstored model: peak_MB ~= {a.vram_fixed_mb:.0f} + "
          f"{a.vram_per_chunk_s_mb:.0f} * chunk_s")
    print(f"{'chunk_s':>8}  {'measured':>9}  {'predicted':>10}  {'error':>7}")
    worst = 0.0
    for chunk_s, peak, _act in rows:
        pred = a.vram_fixed_mb + a.vram_per_chunk_s_mb * chunk_s
        err = 100.0 * (pred - peak) / peak
        worst = max(worst, abs(err))
        print(f"{chunk_s:>8.1f}  {peak:>9.0f}  {pred:>10.0f}  {err:>+6.1f}%")
    verdict = "model holds" if worst <= 10.0 else "RE-FIT the model"
    print(f"\nworst error {worst:.1f}% -- {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
