from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pyparsing import abstractmethod
from scipy import ndimage

from propagator.io.geo import GeographicInfo
from propagator.io.geometry import (
    Geometry,
    GeometryKind,
    is_allowed,
    parse_geometry_list,
    rasterize_geometries,
)


def build_mask(
    geometries: List[Geometry], geo_info: GeographicInfo
) -> np.ndarray:
    """Boolean mask of the action geometries."""
    m = rasterize_geometries(
        geometries=geometries,
        geo_info=geo_info,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype="uint8",
    )
    return m.astype(bool)


class ActionType(str, Enum):
    WATERLINE_ACTION = "waterline_action"
    CANADAIR = "canadair"
    HELICOPTER = "helicopter"
    HEAVY_ACTION = "heavy_action"


# constants > moisture values (%)
WATERLINE_ACTION_MOIST_VALUE = 27
CANADAIR_MOIST_VALUE = 25
CANADAIR_BUFFER_MOIST_VALUE = 22
HELICOPTER_MOIST_VALUE = 22
HELICOPTER_BUFFER_MOIST_VALUE = 20


# ---------- Base class ----------
class Action(BaseModel):
    geometries: List[Geometry] = Field(default_factory=list, exclude=True)
    model_config = ConfigDict(
        arbitrary_types_allowed=True  # use shapely geometry for geometries
    )

    @classmethod
    @abstractmethod
    def allowed_kinds(cls) -> set[GeometryKind]: ...

    @field_validator("geometries")
    @classmethod
    def _check_allowed(cls, geoms: List[Geometry]) -> List[Geometry]:
        allowed = cls.allowed_kinds()
        for g in geoms:
            if not is_allowed(g, allowed):
                raise ValueError(
                    f"{cls.__name__} supports {allowed},\
                    got {g}"
                )
        return geoms

    def rasterize_action_moisture(
        self, geo_info: GeographicInfo
    ) -> Optional[npt.NDArray[np.floating]]:
        return None

    def rasterize_action_fuel(
        self, geo_info: GeographicInfo, fuel: int
    ) -> Optional[npt.NDArray[np.floating]]:
        return None


# ---------- Concrete actions ----------


class WaterlineAction(Action):
    action_type: Literal[ActionType.WATERLINE_ACTION] = Field(
        default=ActionType.WATERLINE_ACTION, frozen=True
    )

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def rasterize_action_moisture(
        self, geo_info: GeographicInfo
    ) -> Optional[npt.NDArray[np.floating]]:
        mask_action = build_mask(self.geometries, geo_info)
        mask_buffer = ndimage.binary_dilation(mask_action)
        moisture_action = np.where(
            mask_buffer, WATERLINE_ACTION_MOIST_VALUE, np.nan
        )
        return moisture_action


class CanadairAction(Action):
    action_type: Literal[ActionType.CANADAIR] = Field(
        default=ActionType.CANADAIR, frozen=True
    )

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def rasterize_action_moisture(
        self, geo_info: GeographicInfo
    ) -> npt.NDArray[np.floating]:
        mask_action = build_mask(self.geometries, geo_info)
        mask_buffer = ndimage.binary_dilation(mask_action)
        moisture_action = np.where(
            mask_buffer, CANADAIR_BUFFER_MOIST_VALUE, np.nan
        )
        moisture_action = np.where(
            mask_action, CANADAIR_MOIST_VALUE, moisture_action
        )
        return moisture_action


class HelicopterAction(Action):
    action_type: Literal[ActionType.HELICOPTER] = Field(
        default=ActionType.HELICOPTER, frozen=True
    )

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def rasterize_action(
        self, geo_info: GeographicInfo
    ) -> npt.NDArray[np.floating]:
        mask_action = build_mask(self.geometries, geo_info)
        # create "jittered" seed points near the line pixels
        iy, ix = np.nonzero(mask_action)
        seed_mask = np.zeros(geo_info.shape, dtype=bool)
        if iy.size:
            jit = np.random.randint(-1, 2, size=(iy.size, 2))  # [-1, 0, 1]
            jy = np.clip(iy + jit[:, 0], 0, geo_info.shape[0] - 1)
            jx = np.clip(ix + jit[:, 1], 0, geo_info.shape[1] - 1)
            seed_mask[jy, jx] = True
        # one-pixel buffer around seed points
        buffer_mask = ndimage.binary_dilation(seed_mask)
        # create moisture action
        moisture_action = np.where(
            buffer_mask, HELICOPTER_BUFFER_MOIST_VALUE, np.nan
        )
        moisture_action[seed_mask] = HELICOPTER_MOIST_VALUE
        return moisture_action


class HeavyAction(Action):
    action_type: Literal[ActionType.HEAVY_ACTION] = Field(
        default=ActionType.HEAVY_ACTION, frozen=True
    )

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def rasterize_action_fuel(
        self, geo_info: GeographicInfo, fuel: int
    ) -> npt.NDArray[np.floating]:
        mask_action = build_mask(self.geometries, geo_info)
        mask_buffer = ndimage.binary_dilation(mask_action)
        fuel_action = np.where(mask_buffer, fuel, np.nan)
        return fuel_action


def parse_actions(
    data: dict[str, Any],
    epsg: int,
) -> list[Action]:
    """
    Parse legacy action fields from a dictionary and convert them to Action objects.
    Consumes legacy fields from the input dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Input dictionary potentially containing legacy action fields.
    epsg : int
        EPSG code for geometry parsing.
    Returns
    -------
    list[Action]
        List of parsed Action objects.
    """
    actions: list[Action] = []
    if ActionType.WATERLINE_ACTION.value in data:
        raw = data.pop(ActionType.WATERLINE_ACTION.value)
        geometries = parse_geometry_list(
            raw,
            allowed=WaterlineAction.allowed_kinds(),
            epsg=epsg,
        )
        if geometries:
            actions.append(WaterlineAction(geometries=geometries))

    if ActionType.CANADAIR.value in data:
        raw = data.pop(ActionType.CANADAIR.value)
        geometries = parse_geometry_list(
            raw,
            allowed=CanadairAction.allowed_kinds(),
            epsg=epsg,
        )
        if geometries:
            actions.append(CanadairAction(geometries=geometries))

    if ActionType.HELICOPTER.value in data:
        raw = data.pop(ActionType.HELICOPTER.value)
        geometries = parse_geometry_list(
            raw,
            allowed=HelicopterAction.allowed_kinds(),
            epsg=epsg,
        )
        if geometries:
            actions.append(HelicopterAction(geometries=geometries))

    if ActionType.HEAVY_ACTION.value in data:
        raw = data.pop(ActionType.HEAVY_ACTION.value)
        geometries = parse_geometry_list(
            raw,
            allowed=HeavyAction.allowed_kinds(),
            epsg=epsg,
        )
        if geometries:
            actions.append(HeavyAction(geometries=geometries))

    return actions
