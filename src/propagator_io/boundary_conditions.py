from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from propagator.propagator import (
    BoundaryConditions,
)

# ---- project utils ----------------------------------------------------------
from propagator.utils import normalize
from propagator_io.actions import Action, parse_actions
from propagator_io.geo import GeographicInfo
from propagator_io.geometry import (
    DEFAULT_EPSG_GEOMETRY,
    Geometry,
    GeometryKind,
    rasterize_geometries,
    parse_geometry_list,
    get_middle_point,
)


# ---- wind helpers -----------------------------------------------------------
def meteo_deg_to_model_rad(deg: float) -> float:
    return normalize((180.0 - float(deg) + 90.0) * np.pi / 180.0)


# ---- simulation inputs ------------------------------------------------------
class TimedInput(BaseModel):
    """Single time-step boundary conditions."""

    model_config = ConfigDict(extra="allow")

    time: int = Field(0, description="minutes from simulation start")

    # Weather conditions
    w_dir: float = Field(
        default_factory=lambda: meteo_deg_to_model_rad(0.0),
        description="wind direction in radians",
    )  # ????
    w_speed: float = Field(0.0, description="wind speed in km/h")
    moisture: float = Field(
        0.0, ge=0.0, le=100.0, description="fuel moisture in percent (0-100)"
    )

    actions: Optional[list[Action]] = None

    # Optional per-step ignitions (POINT/LINE/POLYGON)
    ignitions: Optional[List[Geometry]] = None

    @field_validator("w_speed", mode="before")
    @classmethod
    def _coerce_speed(cls, v):
        if v is None:
            return 0.0
        return float(v)

    @field_validator("w_dir", mode="before")
    @classmethod
    def _coerce_wdir(cls, v):
        if v is None:
            return meteo_deg_to_model_rad(0.0)
        x = float(v)
        if x > 2 * np.pi or x < -2 * np.pi:
            return meteo_deg_to_model_rad(x)
        return normalize(x)

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
                    v, allowed={GeometryKind.POINT, GeometryKind.LINE, GeometryKind.POLYGON}, epsg=epsg
                )
        # let actions.py parse and normalize legacy fields
        new_actions, consumed = parse_actions(data, epsg=epsg)
        if len(new_actions) != 0:
            # append to any already-provided "actions"
            data["actions"] = list(data.get("actions", [])) + new_actions
            # remove consumed legacy keys so they don't error as "extra"
            for k in consumed:
                data.pop(k, None)
        return data

    def get_boundary_conditions(
        self,
        geo_info: GeographicInfo,
        non_vegetated: int,
    ) -> BoundaryConditions:
        # rasterize weather conditions > so far given as scalars
        w_speed_arr = np.ones(geo_info.shape) * self.w_speed
        w_dir_arr = np.ones(geo_info.shape) * self.w_dir
        moisture_arr = np.ones(geo_info.shape) * self.moisture
        ignition_mask = None
        additional_moisture = None
        vegetation_changes = None

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
                fuel_action = action.rasterize_action_fuel(geo_info, non_vegetated)
                if fuel_action is not None:
                    if vegetation_changes is None:
                        vegetation_changes = np.zeros(geo_info.shape, dtype=int)
                    vegetation_changes = np.where(
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
        avg_x = float(np.mean([pt[0] for pt in ignitions_middle_points]))
        avg_y = float(np.mean([pt[1] for pt in ignitions_middle_points]))
        return avg_x, avg_y
