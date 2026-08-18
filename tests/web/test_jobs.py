from __future__ import annotations

import threading
import time

import pytest

from propagator.web.jobs import JobBusyError, JobManager, JobStatus
from propagator.web.schemas import SimulateRequest


def make_request(**overrides) -> SimulateRequest:
    defaults = dict(
        center_lat=42.42,
        center_lon=12.11,
        ignition_lat=42.45,
        ignition_lon=12.25,
        radius_km=5.0,
        realizations=2,
        time_limit_h=1.0,
    )
    defaults.update(overrides)
    return SimulateRequest(**defaults)


def fast_success_runner(job, manager):
    job.status = JobStatus.RUNNING
    for t in (3600,):
        job.frame_times.append(t)
        job.current_time_s = t
    job.status = JobStatus.DONE


def blocking_runner_factory(release_event: threading.Event, started_event: threading.Event):
    def runner(job, manager):
        job.status = JobStatus.RUNNING
        started_event.set()
        release_event.wait(timeout=5)
        job.status = JobStatus.DONE

    return runner


def failing_runner(job, manager):
    job.status = JobStatus.RUNNING
    job.status = JobStatus.FAILED
    job.error = "boom"


def test_submit_and_get_returns_expected_state():
    manager = JobManager()
    job_id = manager.submit(make_request(), fast_success_runner)

    deadline = time.time() + 5
    job = manager.get(job_id)
    while job.status not in (JobStatus.DONE, JobStatus.FAILED) and time.time() < deadline:
        time.sleep(0.05)
        job = manager.get(job_id)

    assert job is not None
    assert job.status == JobStatus.DONE
    assert job.frame_times == [3600]


def test_get_unknown_job_returns_none():
    manager = JobManager()
    assert manager.get("does-not-exist") is None


def test_submit_rejects_second_job_while_one_is_running():
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()
    runner = blocking_runner_factory(release, started)

    manager.submit(make_request(), runner)
    assert started.wait(timeout=5)

    with pytest.raises(JobBusyError):
        manager.submit(make_request(), runner)

    release.set()


def test_cancel_sets_flag_on_job_state():
    manager = JobManager()
    job_id = manager.submit(make_request(), fast_success_runner)
    deadline = time.time() + 5
    while manager.get(job_id).status != JobStatus.DONE and time.time() < deadline:
        time.sleep(0.05)

    # job already finished, but cancel() should still just flip the flag
    assert manager.cancel(job_id) is True
    assert manager.get(job_id).cancel_requested is True
    assert manager.cancel("missing") is False


def test_delete_removes_job():
    manager = JobManager()
    job_id = manager.submit(make_request(), fast_success_runner)
    deadline = time.time() + 5
    while manager.get(job_id).status != JobStatus.DONE and time.time() < deadline:
        time.sleep(0.05)

    assert manager.delete(job_id) is True
    assert manager.get(job_id) is None
    assert manager.delete(job_id) is False


def test_failed_job_reports_error():
    manager = JobManager()
    job_id = manager.submit(make_request(), failing_runner)
    deadline = time.time() + 5
    job = manager.get(job_id)
    while job.status != JobStatus.FAILED and time.time() < deadline:
        time.sleep(0.05)
        job = manager.get(job_id)

    assert job.status == JobStatus.FAILED
    assert job.error == "boom"
