from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from propagator import (
    BoundaryConditions,
)

# ---- project utils ----------------------------------------------------------
from propagator.io.actions import Action, parse_actions
from propagator.io.geo import GeographicInfo
from propagator.io.geometry import (
    DEFAULT_EPSG_GEOMETRY,
    Geometry,
    GeometryKind,
    get_middle_point,
    parse_geometry_list,
    rasterize_geometries,
)


# ---- simulation inputs ------------------------------------------------------
class TimedInput(BaseModel):
    """Single time-step boundary conditions."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    time: int = Field(0, description="seconds from simulation start")

    # Weather conditions
    w_dir: Optional[float] = Field(
        None,
        description="wind direction clockwise in degrees from north (north=0)",
    )
    w_speed: Optional[float] = Field(None, description="wind speed in km/h")
    moisture: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="fuel moisture in percentage (0-100)",
    )

    actions: Optional[list[Action]] = None

    # Optional per-step ignitions (POINT/LINE/POLYGON)
    ignitions: Optional[List[Geometry]] = None

    @field_validator("w_speed", mode="before")
    @classmethod
    def _coerce_speed(cls, v):
        if v is None:
            return v
        return float(v)

    @field_validator("w_dir", mode="before")
    @classmethod
    def _coerce_wdir(cls, v):
        if v is None:
            return v
        x = float(v)
        if x < 0:
            x = 360 + (x % 360)
        elif x >= 360:
            x = x % 360
        return x

    @field_validator("time")
    @classmethod
    def _time_nonnegative(cls, v):
        if v < 0:
            raise ValueError("time must be >= 0")
        return v

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_actions(cls, data: dict, info):
        if not isinstance(data, dict):
            return data
        epsg = (info.context or {}).get("epsg", DEFAULT_EPSG_GEOMETRY)
        #  legacy ignitions parsing
        if "ignitions" in data:
            v = data["ignitions"]
            if isinstance(v, list) and (not v or isinstance(v[0], str)):
                data["ignitions"] = parse_geometry_list(
                    v,
                    allowed={
                        GeometryKind.POINT,
                        GeometryKind.LINE,
                        GeometryKind.POLYGON,
                    },
                    epsg=epsg,
                )
        # let actions.py parse and normalize legacy fields
        legacy_actions = parse_actions(data, epsg=epsg)
        # append to any already-provided "actions"
        existing_actions = data.get("actions")
        if existing_actions is None:
            data["actions"] = legacy_actions
        elif isinstance(existing_actions, list):
            data["actions"] = [*existing_actions, *legacy_actions]
        else:
            raise ValueError("actions must be provided as a list")
        return data

    def get_boundary_conditions(
        self,
        geo_info: GeographicInfo,
        non_vegetated: int,
    ) -> BoundaryConditions:
        # rasterize weather conditions > so far given as scalars
        w_speed_arr = None
        w_dir_arr = None
        moisture_arr = None
        ignition_mask = None
        additional_moisture = None
        vegetation_changes = None

        if self.w_speed is not None:
            w_speed_arr = np.ones(geo_info.shape) * self.w_speed
        if self.w_dir is not None:
            w_dir_arr = np.ones(geo_info.shape) * self.w_dir
        if self.moisture is not None:
            moisture_arr = np.ones(geo_info.shape) * self.moisture

        if self.ignitions is not None:
            ignition_mask = rasterize_geometries(
                geometries=self.ignitions,
                geo_info=geo_info,
                default_value=1,  # set 1 for ignited pixels
                dtype="uint8",
                merge_alg="replace",
            )

        if self.actions is not None:
            for action in self.actions:
                # moisture actions
                moist_action = action.rasterize_action_moisture(geo_info)
                if moist_action is not None:
                    if additional_moisture is None:
                        additional_moisture = np.zeros(geo_info.shape)
                    if moisture_arr is None:
                        moisture_arr = np.zeros(geo_info.shape)
                    # in case of multiple actions, take the one
                    # that have maximum effect e.g. max moisture
                    moisture_arr_tmp = moisture_arr + additional_moisture
                    moist_action = np.where(
                        np.isnan(moist_action),
                        0.0,
                        moist_action,
                    )
                    moist_final = np.maximum(
                        moisture_arr_tmp,
                        moist_action,
                    )
                    additional_moisture = moist_final - moisture_arr
                # fuel actions
                fuel_action = action.rasterize_action_fuel(
                    geo_info, non_vegetated
                )
                if fuel_action is not None:
                    if vegetation_changes is None:
                        vegetation_changes = np.full(geo_info.shape, np.nan)

                    vegetation_changes = np.where(  # type: ignore
                        np.isnan(fuel_action),
                        vegetation_changes,
                        fuel_action,
                    )

        # convert info in Propagator BoundaryConditions
        return BoundaryConditions(
            time=self.time,
            wind_speed=w_speed_arr,
            wind_dir=w_dir_arr,
            moisture=moisture_arr,
            ignition_mask=ignition_mask,
            additional_moisture=additional_moisture,
            vegetation_changes=vegetation_changes,
        )

    def extract_ignitions_middle_point(self) -> Optional[Tuple[float, float]]:
        """
        Extracts the middle coordinates from ignitions
        """
        if self.ignitions is None or len(self.ignitions) == 0:
            return None
        ignitions_middle_points = [
            get_middle_point(ignition) for ignition in self.ignitions
        ]
        ignitions_middle_points = [
            mp for mp in ignitions_middle_points if mp is not None
        ]

        if not ignitions_middle_points:
            return None

        # Return the average of the middle points
        avg_x = float(np.mean([pt[0] for pt in ignitions_middle_points]))  # type: ignore
        avg_y = float(np.mean([pt[1] for pt in ignitions_middle_points]))  # type: ignore
        return avg_x, avg_y
