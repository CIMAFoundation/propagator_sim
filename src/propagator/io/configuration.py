from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from shapely import Geometry

from propagator.core.constants import (
    CELLSIZE,
    REALIZATIONS,
    ROS_DEFAULT,
    MOISTURE_MODEL_DEFAULT,
)
from propagator.core.models import BoundaryConditions

# ---- project utils ----------------------------------------------------------
from propagator.core.numba import (
    MoistureModel,
    RateOfSpreadModel,
    get_p_moisture_fn,
    get_p_time_fn,
)
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

    # --- basic info ---
    name: Optional[str] = Field(
        None, description="Name of the simulation (optional)"
    )
    init_date: datetime = Field(
        default_factory=datetime.now,
        description="Datetime of the simulated event",
    )
    time_limit: int = Field(
        1440, gt=0, description="Simulation limit [minutes]"
    )
    ignitions: Optional[List[Geometry]] = Field(
        None, description="List of ignitions at simulation start (time=0)."
    )
    epsg: int = Field(
        DEFAULT_EPSG_GEOMETRY,  # default to WGS84
        description="EPSG of geometries",
    )

    # --- settings ---
    time_resolution: int = Field(
        60, gt=0, description="Simulation resolution [minutes]"
    )
    cellsize: float = Field(
        CELLSIZE, gt=0.0, description="Cell size in meters"
    )
    do_spotting: bool = Field(False, description="Spotting option")
    ros_model: RateOfSpreadModel = Field(
        ROS_DEFAULT, description="ROS model name")
    prob_moist_model: MoistureModel = Field(
        MOISTURE_MODEL_DEFAULT, description="Moisture model name"
    )
    realizations: int = Field(
        REALIZATIONS, ge=1, description="Number of realizations"
    )

    # --- models ---
    p_time_fn: Optional[object] = Field(default=None, exclude=True)
    p_moist_fn: Optional[object] = Field(default=None, exclude=True)

    # --- boundary conditions ---
    boundary_conditions: List[TimedInput] = Field(
        default_factory=list, description="List of boundary conditions"
    )

    # ---------- checks ----------
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

    @model_validator(mode="after")
    def _post_setup(self):
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

        # checks on boundary conditions
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
