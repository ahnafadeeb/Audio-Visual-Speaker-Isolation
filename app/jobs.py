"""Job store: state, progress events, and the single inference lane."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import RUNS_DIR, Config
from .pipeline import Pipeline

log = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    state: JobState = JobState.QUEUED
    stage: str = "queued"
    pct: float = 0.0
    message: str = ""
    error: str | None = None
    meta: dict | None = None
    input_name: str = "input.mp4"
    created: float = field(default_factory=time.time)

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.job_id

    @property
    def input_path(self) -> Path:
        return self.dir / self.input_name

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "stage": self.stage,
            "pct": round(self.pct, 4),
            "message": self.message,
            "error": self.error,
            "meta": self.meta,
        }


class JobStore:
    """Owns the jobs, the worker pool, and per-job event queues.

    ``max_workers=1`` is load-bearing: it is the GPU admission gate.  Two
    concurrent jobs on an 8 GB card OOM; serialising them is the entire
    concurrency policy a single-user demo app needs.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._pipeline = Pipeline(cfg)

    # -- lifecycle ---------------------------------------------------------- #

    def warm(self) -> None:
        """Pre-load the model so the first job is not 3x slower.

        Submitted to the same single-slot pool as inference, so a job uploaded
        during warm-up queues behind it rather than racing it into a second
        model load -- which on an 8 GB card is the OOM this pool exists to
        prevent.
        """
        if self.cfg.runtime.separator == "passthrough":
            return
        try:
            if self.cfg.runtime.separator == "avtse":
                # AVTSESeparator is built per job (it needs the faces), but the
                # weights are cached process-wide by avtse.load_model, so
                # loading them here is what the first job then reuses.
                from .avtse import load_model
                device = self.cfg.runtime.resolve_device()
                fut = self.pool.submit(load_model, device,
                                       adapter=self.cfg.avtse.resolve_adapter())
                fut.add_done_callback(self._log_warm_result)
                return
            sep = self._pipeline.separator()
            if hasattr(sep, "load"):
                fut = self.pool.submit(sep.load)
                # Without this callback the future is dropped on the floor and a
                # failed download/CUDA-init is silent until the first job dies
                # of it, three minutes into a demo.
                fut.add_done_callback(self._log_warm_result)
        except Exception as exc:                              # pragma: no cover
            log.warning("model warm-up failed (will retry per job): %s", exc)

    @staticmethod
    def _log_warm_result(fut) -> None:
        exc = fut.exception()
        if exc is not None:
            log.warning("model warm-up failed (will retry per job): %s: %s",
                        type(exc).__name__, exc)
        else:
            log.info("model warm-up complete")

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        sep = getattr(self._pipeline, "_separator", None)
        if sep is not None and hasattr(sep, "release"):
            sep.release()

    # -- creation ----------------------------------------------------------- #

    def allocate(self, upload_name: str) -> Job:
        """Reserve a job directory.  The caller streams the upload into
        ``job.input_path`` itself, then calls :meth:`start`.

        Allocate-then-stream, rather than the obvious write-to-tempfile-then-move:

        1. ``tempfile.mkstemp()`` returns ``(fd, path)``.  Keeping only the path
           leaks an *open OS handle*, and Windows then refuses to move or unlink
           the file -- ``PermissionError: [WinError 32]``.  On Linux the same
           code works, so the bug is invisible until demo day.  Not having a
           temp file at all is a stronger fix than remembering to close it.
        2. ``%TEMP%`` is on C:, the runs directory is wherever the project
           lives.  A cross-volume ``shutil.move`` is a full copy + delete, so a
           500 MB upload was written twice and read once for no reason.

        The job id is a server-generated UUID and the ONLY request-scoped input
        to any path we build.  The uploaded filename contributes its suffix and
        nothing else -- a demo box is still a web server on a socket.
        """
        suffix = Path(upload_name).suffix.lower()[:8] or ".mp4"
        job = Job(job_id=uuid.uuid4().hex[:12], input_name=f"input{suffix}")
        job.dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job.job_id] = job
        self._queues[job.job_id] = []
        return job

    def discard(self, job: Job) -> None:
        """Undo :meth:`allocate` for an upload that never completed."""
        self.jobs.pop(job.job_id, None)
        self._queues.pop(job.job_id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def adopt_existing(self) -> int:
        """Register completed jobs already on disk as DONE.

        The store is in-memory, so without this a server restart orphans every
        artefact directory -- restart uvicorn mid-demo and the clip you just
        spent two minutes processing becomes unreachable.  It is also how
        ``scripts/make_demo_job.py`` hands its synthetic job to the UI.
        """
        if not RUNS_DIR.exists():
            return 0
        found = 0
        for d in sorted(RUNS_DIR.iterdir()):
            meta_path = d / "meta.json"
            if not d.is_dir() or d.name in self.jobs or not meta_path.is_file():
                continue
            if not (d / "stems_demo.wav").is_file() or not (d / "video.mp4").is_file():
                continue                      # a crashed run, not a finished one
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job = Job(job_id=d.name, state=JobState.DONE, stage="done",
                      pct=1.0, message="loaded from disk", meta=meta,
                      input_name=next((p.name for p in sorted(d.glob("input.*"))),
                                      "input.mp4"),
                      created=meta_path.stat().st_mtime)
            self.jobs[job.job_id] = job
            self._queues[job.job_id] = []
            found += 1
        if found:
            log.info("adopted %d existing job(s) from %s", found, RUNS_DIR)
        return found

    # -- events ------------------------------------------------------------- #

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._queues.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        try:
            self._queues.get(job_id, []).remove(q)
        except ValueError:
            pass

    def _publish(self, job_id: str, payload: dict, loop) -> None:
        """Called from the worker thread; hops to the event loop thread."""
        for q in list(self._queues.get(job_id, [])):
            try:
                loop.call_soon_threadsafe(self._offer, q, payload)
            except RuntimeError:
                pass                      # loop already closed: shutting down

    @staticmethod
    def _offer(q: asyncio.Queue, payload: dict) -> None:
        """Enqueue on the loop thread, dropping the OLDEST event if full.

        ``put_nowait`` straight from ``call_soon_threadsafe`` would raise
        ``QueueFull`` *inside a loop callback*, where this class cannot catch it
        -- asyncio logs it and drops the event.  Since a full queue means a
        stalled reader, the stale head is the right thing to lose; the terminal
        event, which is the one the browser is actually waiting for, must not
        be.
        """
        while True:
            try:
                q.put_nowait(payload)
                return
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover -- lost the race
                    return

    # -- execution ---------------------------------------------------------- #

    async def start(self, job: Job) -> None:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(self.pool, self._run_blocking, job, loop)

    def _run_blocking(self, job: Job, loop) -> None:
        job.state = JobState.RUNNING

        def progress(stage: str, pct: float, message: str) -> None:
            job.stage, job.pct, job.message = stage, pct, message
            self._publish(job.job_id, job.snapshot(), loop)

        try:
            src = job.input_path
            if not src.is_file():             # tolerate a dir seeded by hand/CLI
                src = next(job.dir.glob("input.*"))
            result = self._pipeline.run(src, job.dir, progress=progress)
            job.meta = result.meta
            job.state, job.pct, job.stage = JobState.DONE, 1.0, "done"
            job.message = "complete"
        except Exception as exc:
            log.exception("job %s failed", job.job_id)
            job.state, job.error = JobState.ERROR, f"{type(exc).__name__}: {exc}"
            job.message = "failed"
        finally:
            self._publish(job.job_id, job.snapshot(), loop)
