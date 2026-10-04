# Research: Local Backend

Status: **DONE** — 2026-08-11. Every finding below is verified by execution on
this Windows box or against upstream sources (PyPI, pytorch.org, Starlette).

Covers: dependency resolution, the serving architecture (upload, SSE, model
lifecycle, Range), and the VRAM/chunking operating point.

---

## 1. Dependencies — verified, pinned, with the traps documented

`requirements.txt` was rewritten and every pin verified against the PyPI JSON
API and download.pytorch.org (2026-08-11). The three traps that would have
bitten at `pip install` time:

1. **torch/torchaudio mismatch.** torch reaches 2.13.0, but torchaudio stops at
   2.11.0. The newest matched pair is **2.11.0** — pin it and stop.
2. **The jax trap.** mediapipe 0.10.21 declares `numpy<2` but depends on
   *unpinned* jax; jax 0.7.2+ requires `numpy>=2.0` and 0.11.0 requires
   `numpy>=2.1`. **jax 0.7.1 is the last release compatible with numpy 1.26.4**
   (numpy 1.26.4 is itself the last 1.x). jaxlib must be paired exactly.
3. **The opencv collision.** mediapipe depends on `opencv-contrib-python`;
   `opencv-python` and `opencv-contrib-python` both ship `cv2` and clobber each
   other. We use only the core cv2 API, so the contrib wheel (a superset) is the
   safe single provider.

Also pinned: `starlette>=0.49.1` (see §4), `protobuf` resolves to 4.25.9 via
abi3, `imageio-ffmpeg` as the ffmpeg fallback.

**Verify with** `scripts/preflight.py` before demo day.

---

## 2. Upload path — the demo-breaker, found and fixed

**The bug:** `create_job` did `tempfile.mkstemp(suffix=suffix)[1]`. `mkstemp`
returns `(fd, path)`; keeping only the path **leaks an open OS handle**. On
Linux the file can still be moved, so the code worked everywhere except the one
place that matters. On Windows, `shutil.move` fails with
`PermissionError: [WinError 32]` — reproduced 100%, then fixed.

**The fix (two layers):**
- `JobStore.allocate()` / `discard()`: no temp file at all. The upload streams
  straight into the job directory, which also kills a second Windows issue: a
  cross-volume `shutil.move` (`%TEMP%` on C: → runs dir) was a full copy +
  delete, writing a 500 MB upload twice.
- `create_job` discards the job directory on *any* failure, including
  `ClientDisconnect` mid-upload (`BaseException`, not `Exception`), so
  `adopt_existing()` can never resurrect a half-written file after a restart.

Verified: happy path, abort path, and hostile filenames (`../../etc/passwd`,
`con.mp4`, 300-char) — only the suffix ever survives, always inside `runs/`.

---

## 3. SSE — subscribe race and the terminal-event guarantee

**Race:** `job_events` yielded `job.snapshot()` *then* subscribed. The worker
thread runs independently, so an event could fire between the snapshot and the
queue existing — and be lost. Reordered: **subscribe first**, snapshot second.
Worst case is now a duplicate event, which the client applies idempotently.

**Stuck at 0%:** a client connecting to an *already-finished* job got its
terminal event from the snapshot, then sat on the empty queue forever. Added a
`TERMINAL` early-return.

**Queue overflow:** `call_soon_threadsafe(q.put_nowait, …)` raising
`QueueFull` inside a loop callback is un-catchable from our code — asyncio logs
and drops the event. `_offer()` now drops the *oldest* event instead, so the
**terminal event can never be evicted**. Tested both directly and via the real
threaded hop: 256 events survive, last is always `done`, no loop errors.

---

## 4. Range requests for video.mp4 — guaranteed, not hoped

**The trap:** `artifact()` hand-wrote `Accept-Ranges: bytes`. But that header is
a promise; the *implementation* is Starlette's. `FileResponse` only learned
`Range` in **0.39.0**, and fastapi 0.115's own metadata allowed
`starlette<0.42`. The header was a lie on older installs.

**The fix:** `starlette>=0.49.1` pinned in requirements (0.49.1 also carries the
fix for the Range-parser advisory GHSA-7f5h-v6xp-fcq8 — we parse Range on an
endpoint that serves user-uploaded files) and the hand-written header removed —
`FileResponse` sets `accept-ranges` itself. Verified against Starlette master
(`_parse_range_header`, 206/416 paths) and the release notes.

---

## 5. Model lifecycle — warm-up serialized, failures audible

- **`warm()`** submits to the *same* single-slot pool as inference, so an
  upload during warm-up queues behind it instead of racing it into a second
  model load (the OOM the pool exists to prevent).
- Warm-up failures were silent: the future was dropped on the floor. Now a
  done-callback logs them, so a failed CUDA-init/download is visible before the
  first job dies of it mid-demo.
- `warm()` is fire-and-forget: if the model isn't ready when the first job
  starts, `separate()` calls `load()` itself and pays the 3× on the spot — the
  tradeoff for never blocking startup.
- `shutdown()` cancels futures and releases the model.

---

## 6. Duration cap — was dead config, now enforced twice

`max_duration_s: 300.0` was declared and **never read**. A 20-minute upload
under the 512 MB limit went straight through.

- **Cheap probe first** (before `normalize_video` re-encodes the whole file with
  libx264): `probe_duration()` via ffprobe.
- **Authoritative check after decode**: `probe_duration` returns `0.0` when
  ffprobe is missing — and `imageio-ffmpeg` bundles ffmpeg *without* ffprobe,
  which is exactly the fallback path on a machine with no system ffmpeg. The
  second check is the guarantee, measured off the decoded samples.
- Errors surface as `ClipTooLongError` → job ERROR with a readable message.

---

## 7. Permutation alignment — silent-boundary coin flip, fixed

`_align_permutation` took the raw argmax/Hungarian on the overlap correlation.
**Measured:** during a shared pause (both speakers quiet — ubiquitous in
conversation), identity is preserved **16/40 trials. A coin flip.** And because
`prev_tail` chains from the already-aligned chunk, one bad call propagates for
the rest of the clip. The docstring itself said the result "sounds exactly like
bleed-through" — that's a false alarm on the gate, sent you debugging the wrong
stage.

**The fix — evidence-gated:**
- `PERM_SILENCE_RMS` (1e-3): if either side of the overlap is below −60 dBFS,
  there is no signal to correlate → keep order.
- `PERM_MARGIN` (0.15): the winner must beat the identity assignment by this
  much, or keep order. Noise scores ~0.006 (1/√k over 2 s); real speech clears
  it ~1.8. Held decisions log a warning.

Validated 200/200 across: shared pause (was 16/40), digital silence, both
active kept, both active swapped (still 200/200 — the gate suppresses noise,
not signal), one active, one active swapped.

**Diagnostics** now flow into `meta.json` (`alignment.*`: boundaries, flipped,
held_boundaries, min_margin). When Objective A appears to regress, check
`held_boundaries` *before* touching the gate — a swap mid-clip is audibly
indistinguishable from bleed-through.

---

## 8. VRAM / chunking operating point

Analytic estimate for SepFormer-whamr16k (encoder stride 8, dual-path chunk
250, d_model 256, 8 repeats) — torch isn't installed on this box, so these are
bounds, not measurements. Check against real GPU numbers in Step 6:

| chunk_s | inter-chunk attention share |
|--------:|----------------------------:|
| 10      | 24%                         |
| 30      | 49%                         |
| 60      | 66%                         |
| 300     | 91%                         |

**`chunk_s=10`, `overlap_s=2` stays.** Activation estimate ~1.6 GiB at 10 s on
an 8 GB card; the quadratic-in-chunk-count term only dominates beyond ~30 s.
Chunking exists so duration stops being an OOM variable, not because 8 GB can't
hold a short clip.

**fp16 is not adopted:** SepFormer post-norm produces the *denominator* of a
ratio; a denorm/overflow in one activation is a silent output error, and the
downstream gate needs *bit-exact* zeros that a quantized tail can't promise.
Loads in fp32, and there is no per-chunk `torch.cuda.empty_cache()` — a full
empty between chunks doubles the time spent allocating.

At the 300 s cap: ~38 chunks → ~30 s on a 4050/4060 (at 0.8 s/chunk), ~6 min on
CPU-only. A 60 s demo clip is 8 chunks, ~6 s.

---

## 9. Verified but unchanged

- **GIL / head-of-line:** one worker thread + `max_workers=1` = one job per
  GPU, exactly the concurrency policy needed. Progress is chunk-granular (one
  publish per chunk), so SSE is not flooded; artifact GETs and the SSE streams
  don't share the worker.
- **Device:** CUDA when available, CPU fallback; model has no `pin_memory`,
  `to()` is on the eval path, `dtype` is default fp32.
- **Ports/static:** bound to 127.0.0.1 (no unauthenticated upload endpoint on
  conference wifi); no GZipMiddleware (buffers SSE); static mounted read-only.
