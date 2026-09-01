from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import rasterio as rio
from pyproj import CRS, Transformer

from propagator.io.data_prep import AreaDataResult
from propagator.io.geo import GeographicInfo
from propagator.io.osm_poi import POI, OverpassError
from propagator.web.jobs import JobManager, JobState, JobStatus
from propagator.web.runner import (
    build_sample_cells,
    build_simulator,
    run_job,
    run_loop,
    schedule_actions,
)
from propagator.web.schemas import SimulateRequest

from .conftest import make_request


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
        bounds=(
            500000.0,
            4700000.0 - size * 30,
            500000.0 + size * 30,
            4700000.0,
        ),
        shape=(size, size),
    )
    ign_row, ign_col = 15, 15
    lat0, lon0 = _pixel_center_lonlat(transform, utm_epsg, ign_row, ign_col)
    lat1, lon1 = _pixel_center_lonlat(
        transform, utm_epsg, ign_row, ign_col + 1
    )

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

    simulator = build_simulator(
        veg, dem, request, ign_row=ign_row, ign_col=ign_col
    )
    schedule_actions(simulator, request, geo_info)
    # seed the global RNG (numba draws from it) so the stochastic spread
    # attempt from the neutralized cell is reproducible, and restore the
    # previous state afterwards
    rng_state = np.random.get_state()
    np.random.seed(20240601)
    try:
        run_loop(simulator, job)
    finally:
        np.random.set_state(rng_state)

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


def test_run_job_full_orchestration_with_stubbed_prepare_area_data(
    monkeypatch, tmp_path
):
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path,
        np.zeros((size, size), dtype=np.float32),
        transform,
        "EPSG:32633",
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
    monkeypatch.setattr(
        "propagator.web.runner.fetch_area_pois", lambda *a, **k: []
    )

    manager = JobManager()
    request = make_request(time_limit_h=1.0, time_resolution_h=1.0)
    job = make_job(request)

    run_job(job, manager)

    assert job.status == JobStatus.DONE
    assert job.error is None
    assert job.geo_info is not None
    assert len(job.frame_times) > 0


def test_run_job_honours_a_cancel_issued_during_the_poi_fetch(
    monkeypatch, tmp_path
):
    """Regression test: the fetch can take ~100 s against an unresponsive
    endpoint, and there was no cancel check between it and run_loop -- so
    a cancel was only honoured after the simulator had been built, while
    `submit` kept rejecting new runs for that whole window."""
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path,
        np.zeros((size, size), dtype=np.float32),
        transform,
        "EPSG:32633",
    )
    _write_tiny_geotiff(
        fuel_path,
        np.full((size, size), 4, dtype=np.int16),
        transform,
        "EPSG:32633",
    )
    with rio.open(dem_path) as f:
        geo_info = GeographicInfo.from_file(f)

    monkeypatch.setattr(
        "propagator.web.runner.prepare_area_data",
        lambda *a, **k: AreaDataResult(
            dem_path=dem_path,
            fuel_path=fuel_path,
            geo_info=geo_info,
            utm_epsg="EPSG:32633",
            ignition_fuel_code=4,
            ignition_warning=None,
        ),
    )
    monkeypatch.setattr(
        "propagator.web.runner.latlon_to_rowcol", lambda *a, **k: (10, 10)
    )

    manager = JobManager()
    request = make_request(
        time_limit_h=1.0, time_resolution_h=1.0, include_pois=True
    )
    job = make_job(request)

    def cancel_during_fetch(*args, **kwargs):
        # the user hits Cancel while Overpass is still being waited on
        job.cancel_requested = True
        return []

    monkeypatch.setattr(
        "propagator.web.runner.fetch_area_pois", cancel_during_fetch
    )

    built = []
    real_build = build_simulator
    monkeypatch.setattr(
        "propagator.web.runner.build_simulator",
        lambda *a, **k: built.append(1) or real_build(*a, **k),
    )

    run_job(job, manager)

    assert job.status == JobStatus.CANCELLED
    assert built == [], "must not build the simulator after a cancel"


def test_run_job_reports_pois_and_arrival(monkeypatch, tmp_path):
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path,
        np.zeros((size, size), dtype=np.float32),
        transform,
        "EPSG:32633",
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

    poi = POI(
        osm_id=1,
        osm_type="node",
        category="hospital",
        name="Test Hospital",
        lat=42.42,
        lon=12.11,
        tags={"amenity": "hospital"},
    )

    monkeypatch.setattr(
        "propagator.web.runner.prepare_area_data", fake_prepare_area_data
    )
    monkeypatch.setattr(
        "propagator.web.runner.latlon_to_rowcol", lambda *a, **k: (10, 10)
    )
    fetch_calls = []

    def fake_fetch_area_pois(*args, **kwargs):
        fetch_calls.append(kwargs)
        return [poi]

    monkeypatch.setattr(
        "propagator.web.runner.fetch_area_pois", fake_fetch_area_pois
    )

    manager = JobManager()
    request = make_request(
        time_limit_h=1.0,
        time_resolution_h=1.0,
        include_pois=True,
        max_pois=250,
        poi_categories=["hospital", "power"],
    )
    job = make_job(request)

    run_job(job, manager)

    assert job.status == JobStatus.DONE
    assert job.pois == [poi]
    assert job.poi_cells == [("node/1#0", 10, 10)]
    assert any(job.frames[t].poi_arrival for t in job.frame_times)
    assert len(fetch_calls) == 1
    assert fetch_calls[0]["max_pois"] == 250
    assert fetch_calls[0]["categories"] == ["hospital", "power"]


@pytest.mark.parametrize(
    "poi_error",
    [
        OverpassError("boom"),
        # The POI overlay is best-effort, so *any* failure below it must
        # degrade to a warning rather than failing the whole run: a
        # malformed Overpass body (JSONDecodeError), a requests error
        # other than the three fetch_overpass retries on (SSL,
        # TooManyRedirects), a corrupted cache entry, or an OSError
        # writing the cache.
        ValueError("boom"),
        OSError("boom"),
    ],
    ids=["overpass-error", "unexpected-value-error", "os-error"],
)
def test_run_job_continues_with_warning_when_overpass_fails(
    monkeypatch, tmp_path, poi_error
):
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path,
        np.zeros((size, size), dtype=np.float32),
        transform,
        "EPSG:32633",
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

    def fake_fetch_area_pois(*args, **kwargs):
        raise poi_error

    monkeypatch.setattr(
        "propagator.web.runner.prepare_area_data", fake_prepare_area_data
    )
    monkeypatch.setattr(
        "propagator.web.runner.latlon_to_rowcol", lambda *a, **k: (10, 10)
    )
    monkeypatch.setattr(
        "propagator.web.runner.fetch_area_pois", fake_fetch_area_pois
    )

    manager = JobManager()
    request = make_request(time_limit_h=1.0, time_resolution_h=1.0)
    job = make_job(request)

    run_job(job, manager)

    assert job.status == JobStatus.DONE
    assert job.pois == []
    assert job.poi_warning is not None and "boom" in job.poi_warning


def test_build_sample_cells_drops_pois_outside_grid():
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - 20 * 30, 500000.0 + 20 * 30, 4700000.0),
        shape=(20, 20),
    )
    inside_lat, inside_lon = _pixel_center_lonlat(
        transform, "EPSG:32633", 5, 5
    )

    pois = [
        POI(1, "node", "hospital", "Inside", inside_lat, inside_lon, {}),
        POI(2, "node", "hospital", "Outside", 0.0, 0.0, {}),
    ]

    cells = build_sample_cells(pois, geo_info, "EPSG:32633")
    assert cells == [("node/1#0", 5, 5)]


def test_build_sample_cells_samples_along_line_geometry():
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - 20 * 30, 500000.0 + 20 * 30, 4700000.0),
        shape=(20, 20),
    )
    p0 = _pixel_center_lonlat(transform, "EPSG:32633", 2, 2)
    p1 = _pixel_center_lonlat(transform, "EPSG:32633", 5, 5)
    p2 = _pixel_center_lonlat(transform, "EPSG:32633", 8, 8)

    line_poi = POI(
        1,
        "way",
        "power_line",
        None,
        p1[0],
        p1[1],
        {},
        geometry=(p0, p1, p2),
    )

    cells = build_sample_cells([line_poi], geo_info, "EPSG:32633")
    assert cells == [
        ("way/1#0", 2, 2),
        ("way/1#1", 5, 5),
        ("way/1#2", 8, 8),
    ]


def test_build_sample_cells_caps_vertices_per_poi_keeping_both_ends():
    """A long OSM way (power lines carry hundreds of vertices) must not
    expand into hundreds of sample cells: each is re-sampled every frame
    and retained for the life of the job. The kept vertices stay spread
    over the whole extent, ends included, so coverage is preserved."""
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - 60 * 30, 500000.0 + 60 * 30, 4700000.0),
        shape=(60, 60),
    )
    # a 50-vertex diagonal, one vertex per cell
    geometry = tuple(
        _pixel_center_lonlat(transform, "EPSG:32633", i, i) for i in range(50)
    )
    poi = POI(
        1,
        "way",
        "power_line",
        None,
        geometry[0][0],
        geometry[0][1],
        {},
        geometry=geometry,
    )

    cells = build_sample_cells(
        [poi], geo_info, "EPSG:32633", max_vertices_per_poi=8
    )

    assert len(cells) == 8
    rows = [row for _key, row, _col in cells]
    assert rows[0] == 0 and rows[-1] == 49  # both ends kept
    assert rows == sorted(rows)
    # keys stay contiguous from #0, as the uniform scheme promises
    assert [k for k, _r, _c in cells] == [f"way/1#{i}" for i in range(8)]


def test_build_sample_cells_bounds_total_cells_not_just_per_poi():
    """Regression test: the per-POI cap alone still multiplies by the POI
    count (5000 POIs x 64 vertices = 320k cells, each re-sampled every
    frame and retained for the job's life), so a total budget has to
    shrink the per-POI allowance as the POI count grows."""
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - 60 * 30, 500000.0 + 60 * 30, 4700000.0),
        shape=(60, 60),
    )
    geometry = tuple(
        _pixel_center_lonlat(transform, "EPSG:32633", i, i) for i in range(50)
    )
    pois = [
        POI(
            i,
            "way",
            "power_line",
            None,
            geometry[0][0],
            geometry[0][1],
            {},
            geometry=geometry,
        )
        for i in range(10)
    ]

    cells = build_sample_cells(
        pois,
        geo_info,
        "EPSG:32633",
        max_vertices_per_poi=64,
        max_sample_cells=20,
    )

    # 20 cells over 10 geometries -> 2 vertices each, not 64
    assert len(cells) == 20


def test_sampled_vertices_accepts_a_single_vertex_budget():
    """`max_vertices_per_poi=1` is the natural way to ask for "the POI as
    a point"; it used to divide by zero."""
    from propagator.web.runner import _sampled_vertices

    geometry = ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    assert _sampled_vertices(geometry, 1) == [(1.0, 1.0)]


def test_build_sample_cells_leaves_short_geometries_untouched():
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - 20 * 30, 500000.0 + 20 * 30, 4700000.0),
        shape=(20, 20),
    )
    geometry = tuple(
        _pixel_center_lonlat(transform, "EPSG:32633", i, i) for i in range(3)
    )
    poi = POI(
        1,
        "way",
        "power_line",
        None,
        geometry[0][0],
        geometry[0][1],
        {},
        geometry=geometry,
    )

    cells = build_sample_cells(
        [poi], geo_info, "EPSG:32633", max_vertices_per_poi=8
    )
    assert len(cells) == 3


def test_build_sample_cells_dedupes_repeated_cells_along_geometry():
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    geo_info = GeographicInfo(
        crs=CRS.from_epsg(32633),
        trans=transform,
        bounds=(500000.0, 4700000.0 - 20 * 30, 500000.0 + 20 * 30, 4700000.0),
        shape=(20, 20),
    )
    p0 = _pixel_center_lonlat(transform, "EPSG:32633", 3, 3)
    p1 = _pixel_center_lonlat(transform, "EPSG:32633", 3, 3)
    p2 = _pixel_center_lonlat(transform, "EPSG:32633", 7, 7)

    line_poi = POI(
        1,
        "way",
        "power_line",
        None,
        p0[0],
        p0[1],
        {},
        geometry=(p0, p1, p2),
    )

    cells = build_sample_cells([line_poi], geo_info, "EPSG:32633")
    assert cells == [
        ("way/1#0", 3, 3),
        ("way/1#1", 7, 7),
    ]


def test_run_job_fails_fast_when_ignition_outside_grid(monkeypatch, tmp_path):
    size = 20
    transform = rio.Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4700000.0)
    dem_path = tmp_path / "dem.tif"
    fuel_path = tmp_path / "fuel.tif"
    _write_tiny_geotiff(
        dem_path,
        np.zeros((size, size), dtype=np.float32),
        transform,
        "EPSG:32633",
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
