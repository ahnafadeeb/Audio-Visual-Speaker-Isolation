"""FastAPI application.

Process shape (ARCHITECTURE_V2.md §5.1)::

    uvicorn (1 worker, asyncio loop)        <- HTTP, SSE, static. Never touches torch.
       └── ThreadPoolExecutor(max_workers=1)  <- the single inference lane
              └── warm model singleton

Run it::

    python -m app.main                 # or: run.bat

Bound to 127.0.0.1 deliberately.  On conference wifi, 0.0.0.0 would expose an
unauthenticated file-upload-and-transcode endpoint to the room.  ``HOST=0.0.0.0``
opts in, for a phone on a private hotspot to upload a clip it just recorded.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import CONFIG, STATIC_DIR, ensure_dirs
from .jobs import JobState, JobStore
from .serialization import json_safe
from .serialization import dumps as json_dumps

log = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
TERMINAL = {JobState.DONE.value, JobState.ERROR.value}
ARTIFACTS = {
    "video.mp4": "video/mp4",
    "stems_demo.wav": "audio/wav",
    "stems_raw.wav": "audio/wav",
    "tracks.json": "application/json",
    "meta.json": "application/json",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    store = JobStore(CONFIG)
    app.state.store = store
    log.info("device=%s separator=%s",
             CONFIG.runtime.resolve_device(), CONFIG.runtime.separator)
    store.adopt_existing()
    store.warm()
    try:
        yield
    finally:
        store.shutdown()


app = FastAPI(title="AV Speaker Isolation", lifespan=lifespan)

# NOTE: do NOT add GZipMiddleware. It buffers responses to compress them, which
# means an SSE stream never flushes and the progress bar hangs forever.


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "device": CONFIG.runtime.resolve_device(),
        "separator": CONFIG.runtime.separator,
        "sample_rate": CONFIG.audio.sample_rate,
    }


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type {suffix!r}; "
                                 f"expected one of {sorted(ALLOWED_SUFFIXES)}")

    store: JobStore = request.app.state.store
    limit = CONFIG.runtime.max_upload_mb * 1024 * 1024
    written = 0

    # Stream straight into the job directory.  No temp file: see JobStore.allocate
    # for why the mkstemp-then-move version failed on Windows every single time.
    job = store.allocate(file.filename or "input.mp4")
    try:
        with job.input_path.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        413, f"file exceeds {CONFIG.runtime.max_upload_mb} MB")
                fh.write(chunk)
        if written == 0:
            raise HTTPException(400, "empty upload")
    except BaseException:
        # Includes the client vanishing mid-upload, which raises ClientDisconnect
        # rather than a subclass of Exception.  Leaving the directory behind
        # would let adopt_existing() resurrect a half-written file on restart.
        store.discard(job)
        raise

    await store.start(job)
    return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
async def job_status(request: Request, job_id: str) -> JSONResponse:
    job = request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    # json_safe, not a bare return: the snapshot embeds `meta`, and a perfectly
    # muted channel reports a non-finite residual floor.  Starlette's
    # JSONResponse renders with allow_nan=False, so returning the dict directly
    # makes this endpoint 500 on exactly the jobs that worked best -- and this
    # is the endpoint the /?job=<id> deep link boots from.
    return JSONResponse(json_safe(job.snapshot()))


@app.get("/api/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    store: JobStore = request.app.state.store
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    async def stream():
        # subscribe() BEFORE reading the snapshot: the worker thread runs
        # independently of this loop, so doing it the other way round leaves a
        # window in which an event fires after the snapshot is taken but before
        # the queue exists, and it is lost.  In this order the worst case is a
        # duplicate event, which the client applies idempotently.
        q = store.subscribe(job_id)
        try:
            first = job.snapshot()
            yield _sse(first)
            # A client that connects to an ALREADY-finished job gets its
            # terminal event from the snapshot above and nothing further will
            # ever be published -- without this check the generator would sit on
            # the queue emitting pings until the socket dropped.
            if first["state"] in TERMINAL:
                return
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"          # keep intermediaries from reaping
                    continue
                yield _sse(payload)
                if payload["state"] in TERMINAL:
                    break
        finally:
            store.unsubscribe(job_id, q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


def _sse(payload: dict) -> str:
    # json_dumps, not json.dumps: bare dumps defaults to allow_nan=True and
    # emits the non-standard `-Infinity` token, which the browser's
    # JSON.parse(e.data) rejects.  That kills the terminal `done` event and the
    # progress bar hangs forever at the end of a job that actually succeeded.
    return f"data: {json_dumps(payload)}\n\n"


@app.get("/api/jobs/{job_id}/artifact/{name}")
async def artifact(request: Request, job_id: str, name: str) -> FileResponse:
    if name not in ARTIFACTS:
        raise HTTPException(404, "unknown artifact")
    job = request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    path = job.dir / name
    if not path.exists():
        raise HTTPException(404, "artifact not ready")
    # Range handling is Starlette's (FileResponse, >=0.39.0 -- see requirements),
    # including the Accept-Ranges header, so do not hand-write one here: an
    # Accept-Ranges we set ourselves would still be advertised on a Starlette too
    # old to honour it, which is worse than not advertising it.  Seeking the
    # <video> depends on this working.
    return FileResponse(path, media_type=ARTIFACTS[name])


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn
    # Import string + __main__ guard: on Windows there is no fork, so --reload
    # and any multiprocessing re-import this module. Without the guard you get
    # a second model load or a recursive spawn loop. Run without --reload for
    # the demo -- the reloader doubles VRAM during the overlap window.
    # PORT lets a second copy run beside a live one. The host is loopback
    # unless HOST says otherwise: this is an unauthenticated upload endpoint.
    # HOST=0.0.0.0 is for a live demo on a private network -- a phone on the
    # same Wi-Fi opens http://<laptop-ip>:8000, records with its own camera
    # from the file picker, and uploads straight to the laptop's GPU.
    import os
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        import socket
        try:
            ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
        except OSError:
            ips = []
        # print, not log: uvicorn has not configured logging yet.
        print(f"[warn] listening on {host}:{port} -- anyone on this network can upload videos.")
        for ip in ips:
            print(f"       from a phone on the same network: http://{ip}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
