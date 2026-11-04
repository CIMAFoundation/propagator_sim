"""Shared FastAPI dependencies. A single module-level JobManager backs the
whole app (one simulation at a time, in-memory state); tests override
`get_job_manager` via `app.dependency_overrides` to inject a fresh one."""

from __future__ import annotations

from propagator.web.jobs import JobManager

_job_manager = JobManager()


def get_job_manager() -> JobManager:
    return _job_manager
