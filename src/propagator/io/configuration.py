from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Mapping, Optional, Tuple
from warnings import warn

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from shapely import Geometry

from propagator.core.constants import CELLSIZE, REALIZATIONS
from propagator.core.models import BoundaryConditions

# ---- project utils ----------------------------------------------------------
from propagator.core.numba import (
    FUEL_SYSTEM_LEGACY,
    MoistureModel,
    RateOfSpreadModel,
    fuelsystem_from_dict,
    get_p_moisture_fn,
    get_p_time_fn,
)
from propagator.core.numba.models import FuelSystem
from propagator.io.boundary_conditions import TimedInput
from propagator.io.geo import GeographicInfo
from propagator.io.geometry import (
    DEFAULT_EPSG_GEOMETRY,
    GeometryKind,
    parse_geometry_list,
)


# ---- configuration ----------------------------------------------------------
class PropagatorConfigurationLegacy(BaseModel):
    """Propagator configuration"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = Field(
        None, description="Name of the simulation (optional)"
    )

    fuel_config: Optional[Path] = Field(
        None, description="Path to fuel configuration file (YAML)"
    )
    mode: Literal["tiles", "geotiff"] = Field(
        "geotiff",
        description="Mode of static data load: 'tiles' for automatic, "
        "'geotiff' for giving DEM and FUEL in input."
        "[default: geotiff]",
    )
    dem: Optional[Path] = Field(
        None,
        description="Path to DEM file (GeoTIFF), required in 'geotiff' mode",
    )
    fuel: Optional[Path] = Field(
        None,
        description="Path to FUEL file (GeoTIFF), required in 'geotiff' mode",
    )
    tilespath: Optional[Path] = Field(
        None,
        description="Path to TILES folder (GeoTIFF), required in 'tiles' mode",
    )
    output: Path = Field(
        ...,
        description="Path to output folder where results will be saved",
    )
    record: bool = Field(
        False,
        description="Export run logs",
    )
    realizations: int = Field(
        REALIZATIONS, ge=1, description="Number of realizations"
    )
    init_date: datetime = Field(
        default_factory=datetime.now,
        description="Datetime of the simulated event",
    )
    time_resolution: int = Field(
        60, gt=0, description="Simulation resolution [minutes]"
    )
    time_limit: int = Field(
        1440, gt=0, description="Simulation limit [minutes]"
    )
    epsg: int = Field(
        DEFAULT_EPSG_GEOMETRY,  # default to WGS84
        description="EPSG of geometries",
    )
    ignitions: Optional[List[Geometry]] = Field(
        None, description="List of ignitions at simulation start (time=0)."
    )
    boundary_conditions: List[TimedInput] = Field(
        default_factory=list, description="List of boundary conditions"
    )
    do_spotting: bool = Field(False, description="Spotting option")
    ros_model: RateOfSpreadModel = Field("wang", description="ROS model name")
    prob_moist_model: MoistureModel = Field(
        "trucchia", description="Moisture model name"
    )
    cellsize: float = Field(
        CELLSIZE, gt=0.0, description="Cell size in meters"
    )
    p_time_fn: Optional[object] = Field(default=None, exclude=True)
    p_moist_fn: Optional[object] = Field(default=None, exclude=True)
    fuel_system: FuelSystem = Field(default=FUEL_SYSTEM_LEGACY, exclude=True)

    # ---------- checks ----------
    @field_validator("fuel_config", mode="before")
    @classmethod
    def _check_fuel_config_file(cls, v: str | Path | None) -> Optional[Path]:
        if v is None:
            return None
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if not v.is_file():
            raise ValueError("Fuel configuration file not found.")
        return v

    @field_validator("dem", "fuel", mode="before")
    @classmethod
    def _check_dem_fuel_files(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if v is not None and not v.is_file():
            raise ValueError("file not found.")
        return v

    @field_validator("output", mode="before")
    @classmethod
    def _check_output_folder(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the folder exists
        if not v.is_dir():
            os.makedirs(v, exist_ok=True)
        return v

    @field_validator("init_date", mode="before")
    @classmethod
    def _parse_init_date(cls, v: str | datetime) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            fmt_ok = ("%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")
            for fmt in fmt_ok:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            # if no format matched, raise error
            raise ValueError(
                f"init_date string not recognized: {v!r}. Expected formats: {fmt_ok}"
            )
        # if v is neither str nor datetime, raise error
        raise TypeError(
            f"init_date must be a datetime or string, \
            got {type(v).__name__}"
        )

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_parse_geoms(cls, data: dict):
        if not isinstance(data, dict):
            return data
        # get epsg
        epsg = data.get("epsg", DEFAULT_EPSG_GEOMETRY)

        # 2) top-level ignitions (strings -> Geometry w/ epsg)
        if "ignitions" in data:
            data["ignitions"] = parse_geometry_list(
                data["ignitions"],
                allowed={
                    GeometryKind.POINT,
                    GeometryKind.LINE,
                    GeometryKind.POLYGON,
                },
                epsg=epsg,
            )

        # 3) nested ignitions inside boundary_conditions[*]
        bcs = data.get("boundary_conditions")
        if isinstance(bcs, list):
            data["boundary_conditions"] = [
                TimedInput.model_validate(bc, context={"epsg": epsg})
                for bc in bcs
            ]
        return data

    # ---------- cross-field checks & friendly console messages ----------
    @model_validator(mode="after")
    def _post_setup(self):
        # geotiff mode: DEM/FUEL required and must exist
        if self.mode == "geotiff":
            if not self.dem:
                raise ValueError("DEM path must be set in 'geotiff' mode")
            if not self.fuel:
                raise ValueError("FUEL path must be set in 'geotiff' mode")
        elif self.mode == "tiles" and (self.dem or self.fuel):
            warn(
                "DEM and FUEL paths are ignored in 'tiles' mode. "
                "Please remove them from the configuration."
            )

        # set the functions
        self.p_time_fn = get_p_time_fn(self.ros_model)
        self.p_moist_fn = get_p_moisture_fn(self.prob_moist_model)
        if self.p_time_fn is None:
            raise ValueError(f"Unknown ROS model: {self.ros_model}")
        if self.p_moist_fn is None:
            raise ValueError(
                f"Unknown moisture model: \
                {self.prob_moist_model}"
            )

        # set fuel system
        if self.fuel_config is not None:
            self.fuel_system = fuels_from_yaml(self.fuel_config)

        # check if boundary condition is empty
        if len(self.boundary_conditions) == 0:
            raise ValueError("boundary_conditions must not be empty.")

        # check if time == 0 is present
        t0_bc = next(
            (bc for bc in self.boundary_conditions if bc.time == 0), None
        )
        if t0_bc is None:
            raise ValueError(
                "boundary_conditions must include an entry with time = 0."
            )

        # add initial ignitions (if present) to the firt boundary condition
        if self.ignitions:
            if t0_bc.ignitions is None:
                t0_bc.ignitions = []
            t0_bc.ignitions.extend(self.ignitions)
            # # single source of truth: clear at top-level
            # self.ignitions = None

        # now, check if t0 has an ignition > must have, otherwise error
        if not t0_bc.ignitions or len(t0_bc.ignitions) == 0:
            raise ValueError(
                "Initial ignitions must be provided either at top-level or in "
                "the first boundary condition (time=0)."
            )

        # check if there are repetitions in boundary conditions
        times = [bc.time for bc in self.boundary_conditions]
        if len(times) != len(set(times)):
            raise ValueError("boundary_conditions have duplicate times.")

        return self

    def get_boundary_conditions(
        self, geo_info: GeographicInfo, non_vegetated: int
    ) -> List[BoundaryConditions]:
        # NOTE: boundary conditions should be sorted by time already

        return [
            bc.get_boundary_conditions(geo_info, non_vegetated)
            for bc in self.boundary_conditions
        ]

    def get_ignitions_middle_point(self) -> Optional[Tuple[float, float]]:
        middle_points = [
            bc.extract_ignitions_middle_point()
            for bc in self.boundary_conditions
        ]
        middle_points = [mp for mp in middle_points if mp is not None]
        if not middle_points:
            return None
        # Return the average of the middle points
        avg_x = float(np.mean([pt[0] for pt in middle_points]))  # type: ignore
        avg_y = float(np.mean([pt[1] for pt in middle_points]))  # type: ignore
        return avg_x, avg_y


def fuels_from_yaml(path: str | Path) -> FuelSystem:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fuels_node = data.get("fuels")
    if not isinstance(fuels_node, Mapping):
        raise ValueError("YAML must contain 'fuels' (mapping)")
    # coerce IDs to int and build Fuel objects
    fs = fuelsystem_from_dict(fuels_node)  # type: ignore
    return fs
