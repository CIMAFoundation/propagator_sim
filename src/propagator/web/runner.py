"""Orchestrates a simulation job: prepare area data, build a `Propagator`,
and step through it, capturing one frame per `time_resolution_h`.

Bounds stay stable across the whole run because `job.geo_info` is set
once, from the loaded DEM/fuel grid, before the loop starts (that grid is
already sized to `radius_km` around the requested center by
`prepare_area_data`, and fully populated — no per-frame reprojection or
cropping is needed for the map overlay to line up frame to frame).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pyproj import Transformer
from shapely import LineString

from propagator.core import FUEL_SYSTEM_LEGACY, BoundaryConditions, Propagator
from propagator.io.actions import ACTION_CLASSES
from propagator.io.boundary_conditions import TimedInput
from propagator.io.data_prep import (
    AreaDataError,
    latlon_to_rowcol,
    prepare_area_data,
)
from propagator.io.geo import GeographicInfo
from propagator.io.loader.geotiff import load_data_from_files
from propagator.io.osm_poi import POI, OverpassError, fetch_area_pois
from propagator.web.jobs import FrameData, JobManager, JobState, JobStatus
from propagator.web.schemas import SimulateRequest

logger = logging.getLogger(__name__)


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


def build_sample_cells(
    pois: list[POI], geo_info: GeographicInfo, utm_epsg: str
) -> list[tuple[str, int, int]]:
    """Convert each POI to one or more (row, col) grid cells via the same
    transform used for the ignition point, dropping any that fall outside
    the grid.

    A POI with a `geometry` (a line or polygon way, e.g. a power line or
    a substation footprint) is sampled at every vertex rather than a
    single representative point, so fire arrival is detected anywhere
    along its extent, not just at its centroid. Each sample gets a
    `"{poi.key}#{i}"` key (grid cells revisited by the geometry are
    deduplicated); a plain point POI still gets a single `"...#0"` key,
    for a uniform key scheme regardless of geometry.
    """
    cells = []
    height, width = geo_info.shape
    # One transformer for every vertex of every POI: building one costs
    # ~0.2 ms, and a run with many geometry-bearing POIs (power lines,
    # substation footprints) converts thousands of vertices here.
    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    for poi in pois:
        points = (
            poi.geometry
            if poi.geometry and len(poi.geometry) > 1
            else [(poi.lat, poi.lon)]
        )
        seen: set[tuple[int, int]] = set()
        idx = 0
        for lat, lon in points:
            row, col = latlon_to_rowcol(
                geo_info.trans, utm_epsg, lat, lon, to_utm=to_utm
            )
            if not (0 <= row < height and 0 <= col < width):
                continue
            if (row, col) in seen:
                continue
            seen.add((row, col))
            cells.append((f"{poi.key}#{idx}", row, col))
            idx += 1
    return cells


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
    while simulator.time < request.time_limit_s:
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
            return
        if simulator.next_time() is None:
            break
        # `_step_window` always advances `simulator.time` by exactly the
        # requested window, so the last step must be clamped to the
        # remaining budget or the run would always overshoot
        # time_limit_s by up to one time_resolution_h.
        step_s = min(
            request.time_resolution_s, request.time_limit_s - simulator.time
        )
        simulator.step(seconds=step_s)
        output = simulator.get_output(sample_cells=job.poi_cells)
        stats = output.stats.to_dict(output.time, ref_date)
        job.frames[output.time] = FrameData(
            time_s=output.time,
            fire_probability=output.fire_probability,
            stats=stats,
            poi_arrival=output.poi_arrival,
        )
        job.frame_times.append(output.time)
        job.current_time_s = output.time
    job.status = JobStatus.DONE


def run_job(job: JobState, manager: JobManager) -> None:
    """Full job orchestration: download/build area data, run the
    simulation loop, and report status/errors on `job` as it progresses."""
    request = job.request
    work_dir: Path | None = None
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="propagator_web_"))
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

        if request.include_pois:
            job.status = JobStatus.FETCHING_POIS
            # The POI overlay is best-effort: the simulation itself is
            # already fully specified by the DEM/fuel data fetched above,
            # so *nothing* here may fail the run. Catch broadly rather
            # than only OverpassError -- a malformed Overpass body
            # (JSONDecodeError), any other requests exception (SSL,
            # redirects), a corrupted cache entry, or an OSError writing
            # the cache would otherwise reach run_job's generic handler
            # and mark an otherwise valid simulation FAILED.
            try:
                job.pois = fetch_area_pois(
                    request.center_lat,
                    request.center_lon,
                    request.radius_km,
                    cache_dir=cache_dir(),
                    max_pois=request.max_pois,
                    categories=request.poi_categories,
                )
                job.poi_cells = build_sample_cells(
                    job.pois, geo_info, area.utm_epsg
                )
            except OverpassError as e:
                job.poi_warning = f"Could not fetch OpenStreetMap POIs: {e}"
                job.pois = []
                job.poi_cells = []
            except Exception as e:
                job.poi_warning = (
                    f"Could not fetch OpenStreetMap POIs: "
                    f"{type(e).__name__}: {e}"
                )
                job.pois = []
                job.poi_cells = []
                logger.exception(
                    "Unexpected error fetching POIs for job %s; continuing "
                    "without the POI overlay",
                    job.id,
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
        logger.exception("Unhandled error while running job %s", job.id)
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
