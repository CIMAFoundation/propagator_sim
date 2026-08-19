from __future__ import annotations

from datetime import datetime

import numpy as np
import rasterio as rio
from pyproj import CRS, Transformer

from propagator.io.data_prep import AreaDataResult
from propagator.io.geo import GeographicInfo
from propagator.web.jobs import JobManager, JobState, JobStatus
from propagator.web.runner import (
    build_simulator,
    run_job,
    run_loop,
    schedule_actions,
)
from propagator.web.schemas import SimulateRequest


def make_request(**overrides) -> SimulateRequest:
    defaults = dict(
        center_lat=42.42,
        center_lon=12.11,
        ignition_lat=42.42,
        ignition_lon=12.11,
        radius_km=1.0,
        cellsize=30.0,
        realizations=2,
        time_limit_h=2.0,
        time_resolution_h=1.0,
    )
    defaults.update(overrides)
    return SimulateRequest(**defaults)


def make_job(request: SimulateRequest) -> JobState:
    return JobState(
        id="test-job",
        status=JobStatus.PENDING,
        request=request,
        created_at=datetime.now(),
        time_limit_s=request.time_limit_s,
    )


def test_run_loop_produces_frames_with_consistent_shape_and_growing_area():
    request = make_request()
    veg = np.full((20, 20), 4, dtype=np.int32)  # all grassland: burns
    dem = np.zeros((20, 20), dtype=np.float32)
    job = make_job(request)

    simulator = build_simulator(veg, dem, request, ign_row=10, ign_col=10)
    run_loop(simulator, job)

    assert job.status == JobStatus.DONE
    assert len(job.frame_times) > 0
    shapes = {frame.fire_probability.shape for frame in job.frames.values()}
    assert shapes == {veg.shape}

    area_history = [job.frames[t].stats["area_mean"] for t in job.frame_times]
    assert area_history == sorted(area_history)


def test_run_loop_does_not_overshoot_time_limit_s():
    # time_limit_h isn't a multiple of time_resolution_h, so the last
    # window must be truncated rather than always advancing by a full
    # time_resolution_h past the requested limit.
    request = make_request(time_limit_h=2.5, time_resolution_h=1.0)
    veg = np.full((20, 20), 4, dtype=np.int32)  # all grassland: burns
    dem = np.zeros((20, 20), dtype=np.float32)
    job = make_job(request)

    simulator = build_simulator(veg, dem, request, ign_row=10, ign_col=10)
    run_loop(simulator, job)

    assert job.status == JobStatus.DONE
    assert job.frame_times[-1] == request.time_limit_s
    assert all(t <= request.time_limit_s for t in job.frame_times)


def test_run_loop_stops_when_cancelled():
    request = make_request(time_limit_h=48.0, time_resolution_h=1.0)
    veg = np.full((40, 40), 4, dtype=np.int32)
    dem = np.zeros((40, 40), dtype=np.float32)
    job = make_job(request)
    job.cancel_requested = True

    simulator = build_simulator(veg, dem, request, ign_row=20, ign_col=20)
    run_loop(simulator, job)

    assert job.status == JobStatus.CANCELLED


def _pixel_center_lonlat(transform, crs, row, col):
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(x, y)
    return lat, lon


def test_schedule_actions_heavy_action_blocks_ignition_cell():
    size = 30
    utm_epsg = "EPSG:32633"
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - size * 30, 500000.0 + size * 30, 4700000.0),
        shape=(size, size),
    )
    ign_row, ign_col = 15, 15
    lat0, lon0 = _pixel_center_lonlat(transform, utm_epsg, ign_row, ign_col)
    lat1, lon1 = _pixel_center_lonlat(transform, utm_epsg, ign_row, ign_col + 1)

    veg = np.full((size, size), 4, dtype=np.int32)  # all grassland: burns
    dem = np.zeros((size, size), dtype=np.float32)

    request = make_request(
        radius_km=1.0,
        cellsize=30.0,
        time_limit_h=3.0,
        time_resolution_h=1.0,
        actions=[
            {
                "action_type": "heavy_action",
                "time_h": 0.0,
                "line": [[lat0, lon0], [lat1, lon1]],
            }
        ],
    )
    job = make_job(request)

    simulator = build_simulator(veg, dem, request, ign_row=ign_row, ign_col=ign_col)
    schedule_actions(simulator, request, geo_info)
    run_loop(simulator, job)

    assert job.status == JobStatus.DONE
    # the ignition cell was neutralized by the heavy_action before the fire
    # ever started spreading: it registers as burned (ignition forces the
    # cell alight regardless of fuel) but never propagates beyond it, so
    # burned area stays pinned at exactly one cell and the front dies
    # immediately (n_active == 0) instead of growing like in the
    # unprotected case (test_run_loop_produces_frames_with_consistent_shape_and_growing_area)
    one_cell_area = request.cellsize**2
    for t in job.frame_times:
        stats = job.frames[t].stats
        assert stats["area_mean"] == one_cell_area
        assert stats["n_active"] == 0


def _write_tiny_geotiff(path, array, transform, crs):
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def test_run_job_full_orchestration_with_stubbed_prepare_area_data(monkeypatch, tmp_path):
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path, np.zeros((size, size), dtype=np.float32), transform, "EPSG:32633"
    )
    _write_tiny_geotiff(
        fuel_path,
        np.full((size, size), 4, dtype=np.int16),
        transform,
        "EPSG:32633",
    )

    with rio.open(dem_path) as f:
        geo_info = GeographicInfo.from_file(f)

    def fake_prepare_area_data(*args, **kwargs):
        return AreaDataResult(
            dem_path=dem_path,
            fuel_path=fuel_path,
            geo_info=geo_info,
            utm_epsg="EPSG:32633",
            ignition_fuel_code=4,
            ignition_warning=None,
        )

    monkeypatch.setattr(
        "propagator.web.runner.prepare_area_data", fake_prepare_area_data
    )
    monkeypatch.setattr(
        "propagator.web.runner.latlon_to_rowcol", lambda *a, **k: (10, 10)
    )

    manager = JobManager()
    request = make_request(time_limit_h=1.0, time_resolution_h=1.0)
    job = make_job(request)

    run_job(job, manager)

    assert job.status == JobStatus.DONE
    assert job.error is None
    assert job.geo_info is not None
    assert len(job.frame_times) > 0


def test_run_job_fails_fast_when_ignition_outside_grid(monkeypatch, tmp_path):
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path, np.zeros((size, size), dtype=np.float32), transform, "EPSG:32633"
    )
    _write_tiny_geotiff(
        fuel_path,
        np.full((size, size), 4, dtype=np.int16),
        transform,
        "EPSG:32633",
    )
    with rio.open(dem_path) as f:
        geo_info = GeographicInfo.from_file(f)

    def fake_prepare_area_data(*args, **kwargs):
        return AreaDataResult(
            dem_path=dem_path,
            fuel_path=fuel_path,
            geo_info=geo_info,
            utm_epsg="EPSG:32633",
            ignition_fuel_code=None,
            ignition_warning="Ignition point (0, 0) falls outside the generated grid.",
        )

    monkeypatch.setattr(
        "propagator.web.runner.prepare_area_data", fake_prepare_area_data
    )

    manager = JobManager()
    request = make_request(time_limit_h=1.0, time_resolution_h=1.0)
    job = make_job(request)

    run_job(job, manager)

    assert job.status == JobStatus.FAILED
    assert "outside the generated grid" in job.error
    assert job.frame_times == []
