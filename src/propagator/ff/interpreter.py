from __future__ import annotations

import heapq
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import geopandas as gpd
import pytz
from shapely.geometry import Point

from propagator.cli.console import (
    get_console,
    info_msg,
    status_propagator_msg,
    warn_msg,
)
from propagator.core import Propagator
from propagator.core.constants import (
    CELLSIZE,
    FUEL_SYSTEM_LEGACY_DICT,
    MOISTURE_MODEL_DEFAULT,
    REALIZATIONS,
    ROS_DEFAULT,
)
from propagator.core.models import PropagatorOutput
from propagator.core.numba import (
    FUEL_SYSTEM_LEGACY,
    get_p_moisture_fn,
    get_p_time_fn,
)
from propagator.core.runner import SimulationRunner
from propagator.ff.parser import Command, FFParseError, Schedule, parse_script
from propagator.io.boundary_conditions import TimedInput
from propagator.io.geo import GeographicInfo
from propagator.io.loader.cog import utm_zone_of_lon
from propagator.io.loader.netcdf import PropagatorDataFromNetCDF
from propagator.io.writer.isochrones_geojson import extract_isochrone
from propagator.io.writer.metadata_json import MetadataJSONWriter
from propagator.io.writer.raster_geotiff import GeoTiffWriter, write_geotiff

# Commands that only make sense for forefire's Lagrangian front-tracker
# (explicit perimeter geometry, per-node state) or its DataBroker/HTTP
# layer -- propagator has no equivalent. `systemExec` is intentionally left
# out: shelling out to arbitrary commands read from a script file is a
# command-injection risk, not a missing feature.
_UNSUPPORTED_COMMANDS = frozenset(
    {
        "FireFront",
        "FireNode",
        "addLayer",
        "plot",
        "listenHTTP",
        "systemExec",
        "parallelInit",
    }
)

# forefire propagationModel name (case-insensitive) -> propagator ros_model.
# Iso/BalbiNov2011/Balbi2015 have no propagator equivalent.
_ROS_MODEL_MAP = {"rothermel": "rothermel"}

# Front-discretization/numerical-scheme tuning knobs that only mean
# something to forefire's Lagrangian front-tracker (node spacing, curvature
# scheme, CFL...). Real .ff scripts set these routinely (see e.g.
# tests/runff/params.ff upstream) even though they're meaningless for a
# raster/ensemble core; accepted and stored, never applied, rather than
# failing scripts that don't otherwise ask propagator to do anything it
# can't.
_IGNORED_PARAMETERS = frozenset(
    {
        "perimeterResolution",
        "spatialIncrement",
        "spatialCFLmax",
        "normalScheme",
        "smoothing",
        "relax",
        "curvatureComputation",
        "curvatureScheme",
        "frontDepthComputation",
        "frontDepthScheme",
        "burningTresholdFlux",
        "minimalPropagativeFrontDepth",
        "maxFrontDepth",
        "initialFrontDepth",
        "propagationSpeedAdjustmentFactor",
        "windReductionFactor",
        "noInitialScan",
        "minSpeed",
        "ForeFireDataDirectory",
        "NetCDFfile",
        "parallelInit",
        "InitTime",
    }
)


class UnsupportedCommandError(Exception):
    def __init__(self, detail: str, line_no: int = 0):
        self.detail = detail
        self.line_no = line_no
        suffix = f" (line {line_no})" if line_no else ""
        super().__init__(f"unsupported in propagator: {detail}{suffix}")


class ForeFireScriptRunner:
    """Executes a subset of forefire's `.ff` command language against
    propagator's own raster/ensemble core.

    Life cycle: `loadData` builds the landscape loader and, immediately
    after, the `Propagator`/`RustPropagator` instance (propagator loads the
    whole domain up front, unlike forefire's incremental DataBroker).
    `startFire`/`trigger` accumulate `TimedInput` boundary conditions keyed
    by simulation time; they are submitted to the simulator lazily, just
    before the simulator clock needs to reach that time. `@t=`/`@nowplus=`
    scheduling defers a command's execution until the simulator clock
    reaches that time, interleaved with `step`/`goTo`.
    """

    def __init__(
        self,
        *,
        core: Literal["numba", "rust"] = "numba",
        seed: Optional[int] = None,
        freeze_dir: Optional[str] = None,
        verbose: bool = False,
        case_directory: Optional[str] = None,
    ) -> None:
        self.core = core
        self.seed = seed
        self.freeze_dir = freeze_dir
        self.verbose = verbose
        self.case_directory = Path(case_directory or ".")

        self.params: dict[str, Any] = {
            "ros_model": ROS_DEFAULT,
            "moisture_model": MOISTURE_MODEL_DEFAULT,
            "dumpMode": "geojson",
            "fireOutputDirectory": ".",
            "experiment": "ForeFire",
            "isoThreshold": 0.5,
        }
        self._fuel_system = FUEL_SYSTEM_LEGACY

        self.loader: Optional[PropagatorDataFromNetCDF] = None
        self.geo_info: Optional[GeographicInfo] = None
        self.simulator: Any = None
        self.runner: Optional[SimulationRunner] = None
        self.init_date: datetime = datetime.now(tz=pytz.UTC)

        self._domain_bbox: Optional[tuple[float, float, float, float]] = None
        self._bc: dict[int, TimedInput] = {}
        self._submitted_times: set[int] = set()
        self._schedule: list[tuple[int, int, Command]] = []
        self._seq = 0

        self._commands = {
            "FireDomain": self._cmd_fire_domain,
            "loadData": self._cmd_load_data,
            "startFire": self._cmd_start_fire,
            "trigger": self._cmd_trigger,
            "setParameter": self._cmd_set_parameter,
            "setParameters": self._cmd_set_parameter,
            "getParameter": self._cmd_get_parameter,
            "step": self._cmd_step,
            "goTo": self._cmd_goto,
            "print": self._cmd_print,
            "save": self._cmd_save,
            "include": self._cmd_include,
            "clear": self._cmd_clear,
            "quit": self._cmd_quit,
        }

    # ------------------------------------------------------------------
    # script driving
    # ------------------------------------------------------------------
    def run_file(self, path: str | Path) -> None:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        prev_dir = self.case_directory
        self.case_directory = path.parent
        try:
            for cmd in parse_script(text):
                self.execute(cmd)
        finally:
            self.case_directory = prev_dir

    def execute(self, cmd: Command) -> None:
        if cmd.schedule is not None:
            due_time = self._resolve_schedule_time(cmd.schedule)
            heapq.heappush(self._schedule, (due_time, self._seq, cmd))
            self._seq += 1
            return
        self._dispatch(cmd)

    def _dispatch(self, cmd: Command) -> None:
        if cmd.name in _UNSUPPORTED_COMMANDS:
            raise UnsupportedCommandError(f"{cmd.name}[]", cmd.line_no)
        handler = self._commands.get(cmd.name)
        if handler is None:
            raise UnsupportedCommandError(f"{cmd.name}[]", cmd.line_no)
        handler(cmd)

    def _resolve_schedule_time(self, schedule: Schedule) -> int:
        if schedule.kind == "t":
            return int(schedule.value)
        return self._current_time() + int(schedule.value)

    def _current_time(self) -> int:
        return int(self.simulator.time) if self.simulator is not None else 0

    # ------------------------------------------------------------------
    # time control: step[]/goTo[] share this, interleaving due scheduled
    # commands and boundary-condition submission with the actual stepping
    # ------------------------------------------------------------------
    def _advance_to(self, target_time: int) -> PropagatorOutput:
        self._require_simulator()
        assert self.runner is not None
        while True:
            self._flush_pending_bc(target_time)
            if not self._schedule or self._schedule[0][0] > target_time:
                break
            due_time, _, cmd = heapq.heappop(self._schedule)
            if due_time > self.simulator.time:
                self.runner.advance_to(due_time)
            self._dispatch(cmd)
        return self.runner.advance_to(target_time)

    def _flush_pending_bc(self, upto_time: int) -> None:
        assert self.geo_info is not None
        non_vegetated = self._fuel_system.get_non_vegetated()
        due = sorted(
            t
            for t in self._bc
            if t not in self._submitted_times and t <= upto_time
        )
        for t in due:
            bc = self._bc[t].get_boundary_conditions(
                self.geo_info, non_vegetated
            )
            self.simulator.set_boundary_conditions(bc)
            self._submitted_times.add(t)

    def _get_bc(self, t: int) -> TimedInput:
        if t not in self._bc:
            self._bc[t] = TimedInput(time=t)
        return self._bc[t]

    def _require_simulator(self) -> None:
        if self.simulator is None:
            raise FFParseError(
                "no landscape loaded yet: a `loadData[...]` command must "
                "precede this one"
            )

    # ------------------------------------------------------------------
    # command handlers
    # ------------------------------------------------------------------
    def _cmd_fire_domain(self, cmd: Command) -> None:
        if "BBoxWSEN" in cmd.args:
            self._domain_bbox = tuple(float(v) for v in cmd.args["BBoxWSEN"])  # type: ignore[assignment]

    def _cmd_load_data(self, cmd: Command) -> None:
        if len(cmd.positional) < 1:
            raise FFParseError("loadData[]: requires a landscape file path")
        filename = str(cmd.positional[0])
        path = Path(filename)
        if not path.is_absolute():
            path = self.case_directory / path
        if len(cmd.positional) >= 2:
            self.init_date = _parse_iso_date(str(cmd.positional[1]))

        self.loader = PropagatorDataFromNetCDF(
            nc_file=str(path), bbox_wsen=self._domain_bbox
        )
        self.geo_info = self.loader.get_geo_info()
        self._build_simulator()

    def _build_simulator(self) -> None:
        assert self.loader is not None and self.geo_info is not None
        dem = self.loader.get_dem()
        veg = self.loader.get_veg()
        step_x, step_y = self.geo_info.get_stepx_stepy()
        cellsize = (abs(step_x) + abs(step_y)) / 2.0 or CELLSIZE
        ros_model = self.params["ros_model"]
        moisture_model = self.params["moisture_model"]

        if self.core == "rust":
            from propagator.rust_core import Propagator as RustPropagator

            self.simulator = RustPropagator(
                dem=dem,
                veg=veg,
                realizations=REALIZATIONS,
                fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
                do_spotting=False,
                origin=(0, 0),
                seed=self.seed,
                freeze_dir=self.freeze_dir,
                out_of_bounds_mode="raise",
                ros_model=ros_model,
                moisture_model=moisture_model,
                cellsize=cellsize,
            )
        else:
            self.simulator = Propagator(
                dem=dem,
                veg=veg,
                realizations=REALIZATIONS,
                fuels=self._fuel_system,
                do_spotting=False,
                origin=(0, 0),
                seed=self.seed,
                freeze_dir=self.freeze_dir,
                out_of_bounds_mode="raise",
                cellsize=cellsize,
                p_time_fn=get_p_time_fn(ros_model),
                p_moist_fn=get_p_moisture_fn(moisture_model),
            )

        self.runner = SimulationRunner(
            simulator=self.simulator,
            freeze_dir=self.freeze_dir,
            verbose=self.verbose,
        )

        # unlike forefire (which falls back to per-fuel moisture defaults
        # from the fuel table), propagator's core requires the very first
        # boundary condition to carry moisture/wind explicitly, or it never
        # initializes those state arrays at all (see Propagator._get_moisture)
        # -- seed 0.0 defaults here; a `trigger[wind;...;t=0]` or future
        # moisture parameter overwrites the same TimedInput in place.
        bc0 = self._get_bc(0)
        if bc0.moisture is None:
            bc0.moisture = 0.0
        if bc0.w_speed is None:
            bc0.w_speed = 0.0
        if bc0.w_dir is None:
            bc0.w_dir = 0.0

    def _cmd_start_fire(self, cmd: Command) -> None:
        self._require_simulator()
        t = int(cmd.args.get("t", self._current_time()))
        if "lonlat" in cmd.args:
            lon, lat = cmd.args["lonlat"][0], cmd.args["lonlat"][1]
        elif "loc" in cmd.args:
            lon, lat = self._loc_to_lonlat(cmd.args["loc"])
        else:
            raise FFParseError(
                "startFire[]: requires loc=(x,y,z) or lonlat=(lon,lat)"
            )
        bc = self._get_bc(t)
        bc.ignitions = [*(bc.ignitions or []), Point(lon, lat)]

    def _cmd_trigger(self, cmd: Command) -> None:
        self._require_simulator()
        # forefire scripts write either `fuelType=wind` or a bare `wind`
        # positional token (see e.g. tests/runff/real_case.ff upstream).
        fuel_type = cmd.args.get("fuelType") or (
            "wind" if "wind" in cmd.positional else None
        )
        if fuel_type != "wind":
            raise UnsupportedCommandError(
                f"trigger[fuelType={fuel_type!r}] "
                "(only fuelType=wind has a propagator equivalent)",
                cmd.line_no,
            )
        if "vel" not in cmd.args:
            raise FFParseError(
                "trigger[fuelType=wind]: requires vel=(vx,vy,vz)"
            )
        vx, vy = cmd.args["vel"][0], cmd.args["vel"][1]
        t = int(cmd.args.get("t", self._current_time()))
        # vel is the wind vector in m/s; propagator wants speed in km/h and
        # direction clockwise from north, blowing towards (vector heading).
        speed_kmh = math.hypot(vx, vy) * 3.6
        bearing = (90.0 - math.degrees(math.atan2(vy, vx))) % 360.0
        bc = self._get_bc(t)
        bc.w_speed = speed_kmh
        bc.w_dir = bearing

    def _loc_to_lonlat(self, loc: tuple[float, ...]) -> tuple[float, float]:
        if self._domain_bbox is None:
            raise FFParseError(
                "loc=(x,y,z) needs a preceding "
                "FireDomain[...;BBoxWSEN=...] to georeference Cartesian "
                "coordinates"
            )
        from pyproj import Proj

        west, south, east, north = self._domain_bbox
        zone = utm_zone_of_lon((west + east) / 2.0)
        proj = Proj(proj="utm", zone=zone, datum="WGS84")
        sw_x, sw_y = proj(west, south)
        world_x, world_y = sw_x + loc[0], sw_y + loc[1]
        lon, lat = proj(world_x, world_y, inverse=True)
        return lon, lat

    def _cmd_set_parameter(self, cmd: Command) -> None:
        for key, value in cmd.args.items():
            self._set_parameter(key, value)

    def _set_parameter(self, key: str, value: Any) -> None:
        if key == "propagationModel":
            model = str(value).lower()
            if model not in _ROS_MODEL_MAP:
                raise UnsupportedCommandError(
                    f"propagationModel={value!r} "
                    "(only Rothermel has a propagator equivalent)"
                )
            self.params["ros_model"] = _ROS_MODEL_MAP[model]
        elif key == "dumpMode":
            if value not in ("geojson", "gpkg"):
                raise UnsupportedCommandError(
                    f"dumpMode={value!r} (only geojson/gpkg are supported)"
                )
            self.params["dumpMode"] = value
        elif key in ("fireOutputDirectory", "experiment", "caseDirectory"):
            if key == "caseDirectory":
                self.case_directory = Path(str(value))
            else:
                self.params[key] = str(value)
        elif key == "fuelsTableFile":
            # propagator-native YAML fuel table, NOT forefire's CSV format
            from propagator.cli.main import fuels_from_yaml

            try:
                self._fuel_system = fuels_from_yaml(value)
            except Exception as e:
                raise UnsupportedCommandError(
                    f"fuelsTableFile={value!r}: propagator needs its own "
                    "YAML fuel table, not forefire's CSV format "
                    f"({e})"
                ) from e
        elif key == "isoThreshold":
            self.params["isoThreshold"] = float(value)
        elif key in _IGNORED_PARAMETERS:
            self.params[key] = value
            if self.verbose:
                info_msg(
                    f"setParameter[{key}={value}]: no propagator "
                    "equivalent (front-tracker tuning knob), ignored"
                )
        else:
            raise UnsupportedCommandError(f"parameter {key!r}")

    def _cmd_get_parameter(self, cmd: Command) -> None:
        if not cmd.positional:
            raise FFParseError("getParameter[]: requires a parameter key")
        key = str(cmd.positional[0])
        info_msg(f"{key} = {self.params.get(key, '<unset>')}")

    def _cmd_step(self, cmd: Command) -> None:
        (dt,) = cmd.require("dt")
        self._advance_and_report(self._current_time() + int(dt))

    def _cmd_goto(self, cmd: Command) -> None:
        (t,) = cmd.require("t")
        self._advance_and_report(int(t))

    def _advance_and_report(self, target_time: int) -> None:
        output = self._advance_to(target_time)
        status_propagator_msg(
            self.init_date, output.time, output.stats, verbose=self.verbose
        )

    def _cmd_print(self, cmd: Command) -> None:
        self._require_simulator()
        assert self.geo_info is not None
        output = self.simulator.get_output()
        threshold = float(self.params["isoThreshold"])
        geoms = extract_isochrone(
            output.fire_probability,
            self.geo_info.trans,
            thresholds=[threshold],
        )
        geom = geoms.get(threshold)
        if geom is None:
            warn_msg(
                f"print[]: no isochrone at threshold {threshold} "
                f"(t={output.time})"
            )
            return
        gdf = gpd.GeoDataFrame(
            {"time": [output.time]}, geometry=[geom], crs=self.geo_info.crs
        )
        if not cmd.positional:
            get_console().print(gdf.to_json())
            return
        filename = str(cmd.positional[0])
        path = self.case_directory / filename
        driver = "GPKG" if self.params["dumpMode"] == "gpkg" else "GeoJSON"
        gdf.to_file(path, driver=driver)

    def _cmd_save(self, cmd: Command) -> None:
        self._require_simulator()
        assert self.geo_info is not None and self.loader is not None
        if "filename" in cmd.args:
            filename = str(cmd.args["filename"])
            fields = cmd.args.get("fields", ())
            if isinstance(fields, str):
                fields = (fields,)
            stem = Path(filename).stem
            for name in fields:
                layer = self.loader.get_layer(str(name))
                out_path = self.case_directory / f"{stem}_{name}.tif"
                write_geotiff(
                    str(out_path),
                    layer,
                    self.geo_info.trans,
                    self.geo_info.crs,
                    dtype=layer.dtype,
                )
            return

        output = self.simulator.get_output()
        out_dir = self.case_directory / str(self.params["fireOutputDirectory"])
        out_dir.mkdir(parents=True, exist_ok=True)
        raster_mapping = {
            "fire_probability": lambda o: o.fire_probability,
            "mean_arrival_time": lambda o: o.mean_arrival_time,
            "min_arrival_time": lambda o: o.min_arrival_time,
        }
        GeoTiffWriter(
            start_date=self.init_date,
            raster_variables_mapping=raster_mapping,
            output_folder=out_dir,
            geo_info=self.geo_info,
            dst_crs=self.geo_info.crs,
        ).write_rasters(output)
        MetadataJSONWriter(
            start_date=self.init_date,
            output_folder=out_dir,
            prefix=str(self.params["experiment"]),
        ).write_metadata(output)

    def _cmd_include(self, cmd: Command) -> None:
        if not cmd.positional:
            raise FFParseError("include[]: requires a script file path")
        self.run_file(self.case_directory / str(cmd.positional[0]))

    def _cmd_clear(self, cmd: Command) -> None:
        """Drop the current landscape/simulator so a following `loadData[]`
        starts a fresh case, mirroring forefire's `clear[]`."""
        self.loader = None
        self.geo_info = None
        self.simulator = None
        self.runner = None
        self._bc.clear()
        self._submitted_times.clear()

    def _cmd_quit(self, cmd: Command) -> None:
        # no-op: forefire's `quit[]` tears down its interactive session
        # (relevant to listenHTTP[]); a script simply ends after this line.
        pass


def _parse_iso_date(text: str) -> datetime:
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt
