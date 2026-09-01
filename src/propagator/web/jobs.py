"""In-memory background job registry for interactive simulation runs.

A local, single-user tool doesn't need Celery/Redis: one job runs at a
time in a single-worker thread pool, and its state lives in a plain dict
guarded by a lock. `GET /api/simulate/{id}` polls `JobState` fields that
the runner thread mutates as it progresses.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import numpy as np
import numpy.typing as npt

from propagator.io.geo import GeographicInfo
from propagator.web.schemas import SimulateRequest


class JobStatus(str, Enum):
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FrameData:
    time_s: int
    fire_probability: npt.NDArray[np.floating]
    stats: dict
    # Lazily filled in by the routers on first request and reused on
    # every later one: `fire_probability` never changes once a frame is
    # recorded, so re-running marching-squares/reprojection or PNG
    # encoding on every slider tick over an already-viewed frame is
    # wasted work.
    png_cache: bytes | None = None
    isochrone_cache: list | None = None


@dataclass
class JobState:
    id: str
    status: JobStatus
    request: SimulateRequest
    created_at: datetime
    current_time_s: int = 0
    time_limit_s: int = 0
    frame_times: list[int] = field(default_factory=list)
    frames: dict[int, FrameData] = field(default_factory=dict)
    geo_info: GeographicInfo | None = None
    warning: str | None = None
    error: str | None = None
    cancel_requested: bool = False


class JobBusyError(Exception):
    """Raised by `submit` when a job is already running."""


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        request: SimulateRequest,
        runner: Callable[[JobState, "JobManager"], None],
    ) -> str:
        with self._lock:
            active = [
                j
                for j in self._jobs.values()
                if j.status
                in (
                    JobStatus.PENDING,
                    JobStatus.PREPARING_DATA,
                    JobStatus.RUNNING,
                )
            ]
            if active:
                raise JobBusyError(
                    "A simulation is already running; wait for it to "
                    "finish or cancel it before starting a new one."
                )
            # Only one simulation is ever active at a time and the UI only
            # ever displays the latest one, so finished jobs (and their
            # per-frame fire_probability grids) would otherwise accumulate
            # in memory, unfreed, for the lifetime of the process across a
            # normal "tweak params, rerun" session.
            for old_id, old_job in list(self._jobs.items()):
                if old_job.status in (
                    JobStatus.DONE,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    del self._jobs[old_id]

            job_id = uuid.uuid4().hex
            job = JobState(
                id=job_id,
                status=JobStatus.PENDING,
                request=request,
                created_at=datetime.now(),
                time_limit_s=request.time_limit_s,
            )
            self._jobs[job_id] = job

        self._executor.submit(runner, job, self)
        return job_id

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.cancel_requested = True
            return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            # If the job's background thread is still running, it holds
            # its own reference to `job` regardless of the pop above, so
            # it would otherwise keep occupying the single worker
            # indefinitely; flagging cancellation here makes it notice
            # and stop at its next loop iteration.
            job.cancel_requested = True
            return True
