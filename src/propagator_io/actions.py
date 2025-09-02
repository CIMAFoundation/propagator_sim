from __future__ import annotations

from typing import Any, Iterable, Type, cast, List, Literal, Optional
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from functools import lru_cache
from pyparsing import abstractmethod
from scipy import ndimage
from collections import defaultdict

from propagator_io.geometry import Geometry, parse_geometry_list, is_allowed
from propagator_io.geo import GeographicInfo
from propagator_io.geometry import GeometryKind, rasterize_geometries


class ActionType(str, Enum):
    WATERLINE_ACTION = "waterline_action"
    CANADAIR = "canadair"
    HELICOPTER = "helicopter"
    HEAVY_ACTION = "heavy_action"


# constants
# NOTE: fixed moisture values in [0, 1]
WATERLINE_ACTION_MOIST_VALUE = 0.27
CANADAIR_MOIST_VALUE = 0.25
CANADAIR_BUFFER_MOIST_VALUE = 0.22
HELICOPTER_MOIST_VALUE = 0.22
HELICOPTER_BUFFER_MOIST_VALUE = 0.2


# ---------- Base class ----------
class Action(BaseModel):
    geometries: List[Geometry] = Field(default_factory=list, exclude=True)
    model_config = ConfigDict(
        arbitrary_types_allowed=True    # use shapely geometry for geometries
    )

    @classmethod
    @abstractmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        ...

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

    def _mask(self, geo_info: GeographicInfo) -> np.ndarray:
        """Boolean mask of the action geometries."""
        m = rasterize_geometries(
            geometries=self.geometries,
            geo_info=geo_info,
            fill=0,
            default_value=1,
            all_touched=True,
            dtype="uint8",
        )
        return m.astype(bool)

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
        mask_action = self._mask(geo_info)
        mask_buffer = ndimage.binary_dilation(mask_action)
        moisture_action = np.where(mask_buffer, WATERLINE_ACTION_MOIST_VALUE, np.nan)
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
        mask_action = self._mask(geo_info)
        mask_buffer = ndimage.binary_dilation(mask_action)
        moisture_action = np.where(mask_buffer, CANADAIR_BUFFER_MOIST_VALUE, np.nan)
        moisture_action = np.where(mask_action, CANADAIR_MOIST_VALUE, moisture_action)
        return moisture_action


class HelicopterAction(Action):
    action_type: Literal[ActionType.HELICOPTER] = Field(
        default=ActionType.HELICOPTER, frozen=True
    )

    @classmethod
    def allowed_kinds(cls) -> set[GeometryKind]:
        return {GeometryKind.LINE}

    def rasterize_action(self, geo_info: GeographicInfo) -> npt.NDArray[np.floating]:
        mask_action = self._mask(geo_info)
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
        moisture_action = np.where(buffer_mask, HELICOPTER_BUFFER_MOIST_VALUE, np.nan)
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
        mask_action = self._mask(geo_info)
        mask_buffer = ndimage.binary_dilation(mask_action)
        fuel_action = np.where(mask_buffer, fuel, np.nan)
        return fuel_action


# ---------- parsing for boundary conditions definition ----------


def _iter_subclasses(cls: Type[Action]) -> Iterable[Type[Action]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


@lru_cache(maxsize=1)
def get_action_registry() -> dict[ActionType, Type[Action]]:
    """
    Build once by introspecting subclasses (no manual lists).
    Pydantic v2 stores field metadata on `model_fields`.
    """
    reg: dict[ActionType, Type[Action]] = {}
    for sub in _iter_subclasses(Action):
        # Be defensive: model_fields may not exist on unrelated classes
        fields: dict[str, Any] = cast(dict[str, Any], getattr(sub, "model_fields", {}))
        info = fields.get("action_type")
        default = getattr(info, "default", None)
        if isinstance(default, ActionType):
            reg[default] = sub
    return reg


@lru_cache(maxsize=1)
def _action_name_set() -> frozenset[str]:
    return frozenset(a.value for a in ActionType)


def load_action(obj: dict[str, Any]) -> Action:
    """
    Instantiate the right Action subclass from a dict containing 'action_type'.
    Accepts str or ActionType.
    """
    atype_raw = obj.get("action_type")
    atype = (
        ActionType(atype_raw)
        if isinstance(atype_raw, str)
        else cast(ActionType, atype_raw)
    )
    cls = get_action_registry().get(atype)
    if cls is None:
        raise ValueError(f"Unknown action_type: {atype_raw!r}")
    return cls.model_validate(obj)


def parse_actions(
    data: dict[str, Any],
    epsg: int,
) -> tuple[list[Action], set[str]]:
    reg = get_action_registry()
    valid_names = _action_name_set()
    # 1) gather geometries per action type
    acc: dict[ActionType, list[Geometry]] = defaultdict(list)
    consumed: set[str] = set()
    for key, raw in list(data.items()):
        # skip non-actions or empty payloads
        if key not in valid_names or not raw:
            continue
        atype = ActionType(key)
        cls = reg.get(atype)
        if cls is None:
            # unknown/unregistered action -> ignore
            continue

        geoms = parse_geometry_list(raw, allowed=cls.allowed_kinds(), epsg=epsg)
        if geoms:
            acc[atype].extend(geoms)
            consumed.add(key)
    # build a single Action per type with merged geometries
    actions: list[Action] = []
    for atype, geoms in acc.items():
        cls = reg[atype]
        actions.append(cls(geometries=geoms))
    return actions, consumed
