from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from propagator.io.geo import GeographicInfo
from propagator.web.app import app
from propagator.web.deps import get_job_manager
from propagator.web.jobs import FrameData, JobManager, JobStatus


def make_geo_info() -> GeographicInfo:
    return GeographicInfo.from_bounds(
        west=500000.0,
        south=4699000.0,
        east=500600.0,
        north=4700000.0,
        rows=20,
        cols=20,
        zone=33,
    )


def synchronous_runner(job, manager):
    job.status = JobStatus.RUNNING
    job.geo_info = make_geo_info()
    fp = np.zeros((20, 20), dtype=np.float32)
    fp[8:12, 8:12] = 0.8
    job.frames[3600] = FrameData(
        time_s=3600,
        fire_probability=fp,
        stats={"n_active": 3, "area_mean": 1000.0, "area_50": 500.0, "area_75": 100.0, "area_90": 0.0},
    )
    job.frame_times.append(3600)
    job.current_time_s = 3600
    job.status = JobStatus.DONE


@pytest.fixture()
def client(monkeypatch):
    test_manager = JobManager()
    # the router imports `run_job` by name, so patch its reference there
    monkeypatch.setattr(
        "propagator.web.routers.simulate.run_job", synchronous_runner
    )

    def fake_get_job_manager():
        return test_manager

    app.dependency_overrides[get_job_manager] = fake_get_job_manager
    yield TestClient(app), test_manager
    app.dependency_overrides.clear()


def base_request(**overrides):
    body = dict(
        center_lat=42.42,
        center_lon=12.11,
        ignition_lat=42.45,
        ignition_lon=12.25,
        radius_km=5,
    )
    body.update(overrides)
    return body


def test_start_and_poll_until_done(client):
    http, _ = client
    res = http.post("/api/simulate", json=base_request())
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        r = http.get(f"/api/simulate/{job_id}")
        assert r.status_code == 200
        status = r.json()
        if status["status"] == "done":
            break
        time.sleep(0.05)

    assert status["status"] == "done"


def test_unknown_job_returns_404(client):
    http, _ = client
    res = http.get("/api/simulate/does-not-exist")
    assert res.status_code == 404


def test_invalid_request_returns_422(client):
    http, _ = client
    res = http.post("/api/simulate", json=base_request(wind_dir=999))
    assert res.status_code == 422


def test_frames_and_frame_endpoints_after_completion(client):
    http, manager = client
    res = http.post("/api/simulate", json=base_request())
    job_id = res.json()["job_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        if http.get(f"/api/simulate/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.05)

    frames_res = http.get(f"/api/simulate/{job_id}/frames")
    assert frames_res.status_code == 200
    frames = frames_res.json()
    assert frames["frame_times_s"] == [3600]
    assert frames["bounds_wgs84"] is not None

    frame_res = http.get(f"/api/simulate/{job_id}/frame/3600")
    assert frame_res.status_code == 200
    assert frame_res.json()["stats"]["n_active"] == 3

    png_res = http.get(f"/api/simulate/{job_id}/frame/3600/image.png")
    assert png_res.status_code == 200
    assert png_res.headers["content-type"] == "image/png"
    assert png_res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_manual_page_is_served(client):
    http, _ = client
    res = http.get("/manual.html")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")

    index_res = http.get("/index.html")
    assert 'href="manual.html"' in index_res.text


def test_cancel_and_delete(client):
    http, manager = client
    res = http.post("/api/simulate", json=base_request())
    job_id = res.json()["job_id"]

    cancel_res = http.post(f"/api/simulate/{job_id}/cancel")
    assert cancel_res.status_code == 200

    delete_res = http.delete(f"/api/simulate/{job_id}")
    assert delete_res.status_code == 200
    assert http.get(f"/api/simulate/{job_id}").status_code == 404
