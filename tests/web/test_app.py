from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from propagator.core.models import CellArrivalSample
from propagator.io.geo import GeographicInfo
from propagator.io.osm_poi import POI
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
        stats={
            "n_active": 3,
            "area_mean": 1000.0,
            "area_50": 500.0,
            "area_75": 100.0,
            "area_90": 0.0,
        },
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
    assert frames["pois"] == []

    frame_res = http.get(f"/api/simulate/{job_id}/frame/3600")
    assert frame_res.status_code == 200
    assert frame_res.json()["stats"]["n_active"] == 3
    assert frame_res.json()["poi_arrival"] == []

    png_res = http.get(f"/api/simulate/{job_id}/frame/3600/image.png")
    assert png_res.status_code == 200
    assert png_res.headers["content-type"] == "image/png"
    assert png_res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_frame_png_and_isochrones_are_cached_after_first_request(client):
    http, manager = client
    res = http.post("/api/simulate", json=base_request())
    job_id = res.json()["job_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        if http.get(f"/api/simulate/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.05)

    frame = manager.get(job_id).frames[3600]
    assert frame.png_cache is None
    assert frame.isochrone_cache is None

    http.get(f"/api/simulate/{job_id}/frame/3600/image.png")
    assert frame.png_cache is not None
    cached_png = frame.png_cache
    http.get(f"/api/simulate/{job_id}/frame/3600/image.png")
    assert frame.png_cache is cached_png  # not recomputed on the 2nd request

    http.get(f"/api/simulate/{job_id}/frame/3600")
    assert frame.isochrone_cache is not None
    cached_iso = frame.isochrone_cache
    http.get(f"/api/simulate/{job_id}/frame/3600")
    assert frame.isochrone_cache is cached_iso


def test_line_poi_arrival_is_aggregated_across_samples(client):
    http, manager = client
    res = http.post("/api/simulate", json=base_request())
    job_id = res.json()["job_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        if http.get(f"/api/simulate/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.05)

    job = manager.get(job_id)
    job.pois = [
        POI(
            osm_id=1,
            osm_type="way",
            category="power_line",
            name="Test Line",
            lat=42.0,
            lon=12.0,
            tags={},
            voltage="132000",
            operator="Terna",
            geometry=((42.0, 12.0), (42.01, 12.01)),
        )
    ]
    job.frames[3600].poi_arrival = (
        CellArrivalSample("way/1#0", 1, 1, False, float("nan"), float("nan")),
        CellArrivalSample("way/1#1", 2, 2, True, 100.0, 120.0),
    )

    frame_res = http.get(f"/api/simulate/{job_id}/frame/3600")
    assert frame_res.status_code == 200
    poi_arrival = frame_res.json()["poi_arrival"]
    assert len(poi_arrival) == 1
    entry = poi_arrival[0]
    assert entry["id"] == "way/1"
    assert entry["reached"] is True
    assert entry["arrival_time_h"] == pytest.approx(120.0 / 3600.0)
    assert entry["voltage"] == "132000"
    assert entry["operator"] == "Terna"

    frames_res = http.get(f"/api/simulate/{job_id}/frames")
    poi_out = frames_res.json()["pois"]
    assert poi_out == [
        {
            "id": "way/1",
            "name": "Test Line",
            "category": "power_line",
            "lat": 42.0,
            "lon": 12.0,
            "voltage": "132000",
            "operator": "Terna",
            "geometry": [[42.0, 12.0], [42.01, 12.01]],
        }
    ]


def test_poi_with_no_in_grid_samples_is_still_reported_as_unreached(client):
    """Regression test: a POI whose every sampled cell fell outside the
    grid contributes no CellArrivalSample, and the arrival list was built
    from the samples alone -- so it was fetched, counted against
    max_pois, listed by `/frames`, and then never drawn, since the map
    renders from `poi_arrival`."""
    http, manager = client
    res = http.post("/api/simulate", json=base_request())
    job_id = res.json()["job_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        if http.get(f"/api/simulate/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.05)

    job = manager.get(job_id)
    job.pois = [
        POI(
            osm_id=1,
            osm_type="node",
            category="hospital",
            name="Off-grid",
            lat=42.0,
            lon=12.0,
            tags={},
        )
    ]
    job.frames[3600].poi_arrival = ()  # every sample fell outside the grid
    job.frames[3600].poi_arrival_cache = None

    poi_arrival = http.get(f"/api/simulate/{job_id}/frame/3600").json()[
        "poi_arrival"
    ]

    assert len(poi_arrival) == 1
    assert poi_arrival[0]["id"] == "node/1"
    assert poi_arrival[0]["reached"] is False
    assert poi_arrival[0]["arrival_time_h"] is None


def test_manual_page_is_served(client):
    http, _ = client
    res = http.get("/manual.html")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")

    index_res = http.get("/index.html")
    assert 'href="manual.html"' in index_res.text


def test_italian_manual_and_locales_are_served(client):
    http, _ = client
    res = http.get("/manual.it.html")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")

    for lang in ("en", "it"):
        res = http.get(f"/locales/{lang}.json")
        assert res.status_code == 200
        assert res.json()["run.button"]


def test_cancel_and_delete(client):
    http, manager = client
    res = http.post("/api/simulate", json=base_request())
    job_id = res.json()["job_id"]

    cancel_res = http.post(f"/api/simulate/{job_id}/cancel")
    assert cancel_res.status_code == 200

    delete_res = http.delete(f"/api/simulate/{job_id}")
    assert delete_res.status_code == 200
    assert http.get(f"/api/simulate/{job_id}").status_code == 404
