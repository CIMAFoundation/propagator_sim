from __future__ import annotations

import threading
import time

import pytest

from propagator.web.jobs import JobBusyError, JobManager, JobStatus

from .conftest import make_request


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


def test_submit_rejects_a_second_job_while_one_is_fetching_pois():
    """Regression test: the busy guard listed PENDING/PREPARING_DATA/
    RUNNING explicitly, so the later-added FETCHING_POIS left a window
    (tens of seconds on a slow Overpass endpoint) where a second run was
    accepted and then sat in PENDING behind the single worker."""
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()

    def poi_fetching_runner(job, manager):
        job.status = JobStatus.FETCHING_POIS
        started.set()
        release.wait(timeout=5)
        job.status = JobStatus.DONE

    manager.submit(make_request(), poi_fetching_runner)
    assert started.wait(timeout=5)

    with pytest.raises(JobBusyError):
        manager.submit(make_request(), poi_fetching_runner)

    release.set()


def test_every_non_terminal_status_counts_as_busy():
    """The guard is derived by negating TERMINAL_STATUSES precisely so a
    status added later is busy by default; keep that property enforced."""
    from propagator.web.jobs import TERMINAL_STATUSES

    assert TERMINAL_STATUSES == {
        JobStatus.DONE,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    for status in JobStatus:
        if status in TERMINAL_STATUSES:
            continue
        manager = JobManager()
        started = threading.Event()
        release = threading.Event()

        def runner(job, manager, _status=status):
            job.status = _status
            started.set()
            release.wait(timeout=5)
            job.status = JobStatus.DONE

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


def test_delete_cancels_a_still_running_job():
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()
    runner = blocking_runner_factory(release, started)

    job_id = manager.submit(make_request(), runner)
    assert started.wait(timeout=5)

    job = manager.get(job_id)
    assert job is not None
    assert job.cancel_requested is False

    assert manager.delete(job_id) is True
    assert manager.get(job_id) is None
    # the background thread holds its own reference to `job` regardless
    # of the registry pop above, so it must still see cancel_requested
    # flip to True instead of running forever unnoticed.
    assert job.cancel_requested is True

    release.set()


def test_submit_evicts_finished_jobs_from_previous_runs():
    manager = JobManager()
    first_id = manager.submit(make_request(), fast_success_runner)
    deadline = time.time() + 5
    while manager.get(first_id).status != JobStatus.DONE and time.time() < deadline:
        time.sleep(0.05)

    second_id = manager.submit(make_request(), fast_success_runner)

    # the UI only ever displays the latest run, so a finished job should
    # be evicted once a new one starts rather than kept in memory forever
    assert manager.get(first_id) is None
    assert manager.get(second_id) is not None


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
