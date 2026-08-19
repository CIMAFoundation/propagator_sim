"""Orchestrates a simulation job: prepare area data, build a `Propagator`,
and step through it, capturing one frame per `time_resolution_h`.

Bounds stay stable across the whole run because `job.geo_info` is set
once, from the loaded DEM/fuel grid, before the loop starts (that grid is
already sized to `radius_km` around the requested center by
`prepare_area_data`, and fully populated — no per-frame reprojection or
cropping is needed for the map overlay to line up frame to frame).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from shapely import LineString

from propagator.core import FUEL_SYSTEM_LEGACY, BoundaryConditions, Propagator
from propagator.io.actions import (
    CanadairAction,
    HeavyAction,
    HelicopterAction,
    WaterlineAction,
)
from propagator.io.boundary_conditions import TimedInput
from propagator.io.data_prep import (
    AreaDataError,
    latlon_to_rowcol,
    prepare_area_data,
)
from propagator.io.geo import GeographicInfo
from propagator.io.loader.geotiff import load_data_from_files
from propagator.web.jobs import FrameData, JobManager, JobState, JobStatus
from propagator.web.schemas import SimulateRequest

ACTION_CLASSES = {
    "waterline_action": WaterlineAction,
    "canadair": CanadairAction,
    "helicopter": HelicopterAction,
    "heavy_action": HeavyAction,
}


def cache_dir() -> Path:
    override = os.environ.get("PROPAGATOR_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".propagator" / "cache"


def build_simulator(
    veg, dem, request, ign_row: int, ign_col: int
) -> Propagator:
    """Construct and initialize a `Propagator` ready to step, from a
    loaded veg/dem grid and a validated `SimulateRequest`."""
    simulator = Propagator(
        veg=veg,
        dem=dem,
        realizations=request.realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        cellsize=request.cellsize,
        do_spotting=request.do_spotting,
        out_of_bounds_mode="ignore",
    )
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            ignitions=[(ign_row, ign_col)],
            wind_speed=request.wind_speed,
            wind_dir=request.wind_dir,
            moisture=request.moisture,
        )
    )
    return simulator


def schedule_actions(
    simulator: Propagator,
    request: SimulateRequest,
    geo_info: GeographicInfo,
) -> None:
    """Schedule firefighting actions (canadair/helicopter/waterline/heavy)
    as future `BoundaryConditions`, reusing `TimedInput.get_boundary_conditions`
    (the same rasterization the CLI uses) rather than reimplementing it.
    Must be called before the simulation loop starts stepping."""
    if not request.actions:
        return
    non_vegetated = simulator.fuels.get_non_vegetated()
    for action_req in request.actions:
        action_cls = ACTION_CLASSES[action_req.action_type]
        line = LineString([(lon, lat) for lat, lon in action_req.line])
        timed_input = TimedInput(
            time=int(round(action_req.time_h * 3600)),
            actions=[action_cls(geometries=[line])],
        )
        bc = timed_input.get_boundary_conditions(geo_info, non_vegetated)
        simulator.set_boundary_conditions(bc)


def run_loop(simulator: Propagator, job: JobState) -> None:
    """Step `simulator` to completion (or until cancelled/time_limit_s),
    recording one `FrameData` per `time_resolution_s` into `job`."""
    request = job.request
    ref_date = datetime.now(timezone.utc)
    job.status = JobStatus.RUNNING
    while True:
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            return
        next_time = simulator.next_time()
        if next_time is None:
            break
        simulator.step(seconds=request.time_resolution_s)
        output = simulator.get_output()
        stats = output.stats.to_dict(output.time, ref_date)
        job.frames[output.time] = FrameData(
            time_s=output.time,
            fire_probability=output.fire_probability,
            stats=stats,
        )
        job.frame_times.append(output.time)
        job.current_time_s = output.time
        if simulator.time > request.time_limit_s:
            break
    job.status = JobStatus.DONE


def run_job(job: JobState, manager: JobManager) -> None:
    """Full job orchestration: download/build area data, run the
    simulation loop, and report status/errors on `job` as it progresses."""
    request = job.request
    work_dir = Path(tempfile.mkdtemp(prefix="propagator_web_"))
    try:
        job.status = JobStatus.PREPARING_DATA
        area = prepare_area_data(
            request.center_lat,
            request.center_lon,
            request.radius_km,
            cellsize=request.cellsize,
            ignition_lat=request.ignition_lat,
            ignition_lon=request.ignition_lon,
            output_dir=work_dir,
            cache_dir=cache_dir(),
        )
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            return
        job.warning = area.ignition_warning
        if area.ignition_fuel_code is None:
            # out of bounds: row/col indices would be unusable (and could
            # silently wrap via numpy negative indexing), so fail fast
            # instead of igniting the wrong cell.
            raise AreaDataError(area.ignition_warning)

        dem, veg, geo_info = load_data_from_files(
            str(area.fuel_path), str(area.dem_path)
        )
        job.geo_info = geo_info

        ign_row, ign_col = latlon_to_rowcol(
            geo_info.trans,
            area.utm_epsg,
            request.ignition_lat,
            request.ignition_lon,
        )

        simulator = build_simulator(veg, dem, request, ign_row, ign_col)
        schedule_actions(simulator, request, geo_info)
        run_loop(simulator, job)
    except AreaDataError as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
