"""Rust-core adapter presenting the numba ``Propagator`` interface.

Wraps the ``propagator_rust`` native extension (built from
``rust/propagator-py``) so the CLI can drive either core through the same
calls. Inputs/outputs are marshalled to/from the exact dataclasses the rest
of the package expects (:class:`BoundaryConditions`,
:class:`PropagatorOutput`, :class:`PropagatorStats`), and out-of-bounds
errors are re-raised as the core's :class:`PropagatorOutOfBoundsError`.

Build the extension with::

    maturin develop -m rust/propagator-py/Cargo.toml --release
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from propagator.core.models import (
    BoundaryConditions,
    PropagatorOutput,
    PropagatorStats,
)
from propagator.core.propagator import PropagatorOutOfBoundsError

try:
    import propagator_rust as _rust
except ImportError as exc:  # pragma: no cover - only hit without the wheel
    raise ImportError(
        "The Rust core is not installed. Build it with "
        "`maturin develop -m rust/propagator-py/Cargo.toml --release` "
        "(or `make rust-core`) before using --core rust."
    ) from exc


def _normalize_fuels(fuels: dict) -> dict:
    """Coerce fuel ids (and spread-probability target ids) to ``int`` so the
    Rust binding, which expects integer keys, accepts YAML-loaded dicts."""
    out: dict[int, dict] = {}
    for fid, fuel in fuels.items():
        fuel = dict(fuel)
        sp = fuel.get("spread_probability")
        if isinstance(sp, dict):
            fuel["spread_probability"] = {
                int(k): float(v) for k, v in sp.items()
            }
        out[int(fid)] = fuel
    return out


def _as_field(value: Any) -> Any:
    """Scalars pass through as floats; rasters as contiguous float32."""
    if value is None:
        return None
    if np.isscalar(value):
        return float(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return np.ascontiguousarray(arr, dtype=np.float32)


class Propagator:
    """Drop-in replacement for the numba ``Propagator`` backed by Rust."""

    def __init__(
        self,
        *,
        dem: np.ndarray,
        veg: np.ndarray,
        realizations: int,
        fuels_dict: dict,
        do_spotting: bool = False,
        origin: tuple[int, int] = (0, 0),
        seed: Optional[int] = None,
        freeze_dir: Optional[Path] = None,
        out_of_bounds_mode: str = "raise",
        ros_model: str = "wang",
        moisture_model: str = "trucchia",
        cellsize: float = 20.0,
        n_threads: Optional[int] = None,
    ) -> None:
        dem = np.ascontiguousarray(dem, dtype=np.float64)
        veg = np.ascontiguousarray(veg, dtype=np.int32)
        self._sim = _rust.Propagator(
            dem,
            veg,
            realizations=realizations,
            fuels=_normalize_fuels(fuels_dict),
            do_spotting=do_spotting,
            origin=(int(origin[0]), int(origin[1])),
            seed=seed,
            freeze_dir=str(freeze_dir) if freeze_dir is not None else None,
            out_of_bounds_mode=out_of_bounds_mode,
            ros_model=ros_model,
            moisture_model=moisture_model,
            cellsize=float(cellsize),
            n_threads=n_threads,
        )

    # ---- properties mirrored from the numba Propagator -------------------
    @property
    def time(self) -> int:
        return self._sim.time

    @property
    def origin(self) -> tuple[int, int]:
        return self._sim.origin

    @property
    def veg(self) -> np.ndarray:
        return self._sim.veg

    def realizations(self) -> int:
        return self._sim.realizations

    # ---- driving the simulation -----------------------------------------
    def next_time(self) -> Optional[int]:
        return self._sim.next_time()

    def set_boundary_conditions(self, bc: BoundaryConditions) -> None:
        ignitions = bc.ignitions
        if ignitions is not None and not isinstance(ignitions, np.ndarray):
            # list of (row, col[, realization]) tuples -> plain python ints
            ignitions = [tuple(int(v) for v in ign) for ign in ignitions]
        elif isinstance(ignitions, np.ndarray):
            ignitions = np.ascontiguousarray(ignitions, dtype=bool)

        additional = bc.additional_moisture
        if additional is not None:
            additional = np.ascontiguousarray(additional, dtype=np.float32)
        veg_changes = bc.vegetation_changes
        if veg_changes is not None:
            veg_changes = np.ascontiguousarray(veg_changes, dtype=np.float64)

        self._sim.set_boundary_conditions(
            int(bc.time),
            moisture=_as_field(bc.moisture),
            wind_dir=_as_field(bc.wind_dir),
            wind_speed=_as_field(bc.wind_speed),
            ignitions=ignitions,
            additional_moisture=additional,
            vegetation_changes=veg_changes,
        )

    def step(self, seconds: int) -> None:
        try:
            self._sim.step(int(seconds))
        except _rust.PropagatorOutOfBoundsError as exc:
            # surface as the core's exception the CLI already catches
            raise PropagatorOutOfBoundsError(str(exc)) from None

    def expand(
        self,
        new_veg: np.ndarray,
        new_dem: np.ndarray,
        new_origin: tuple[int, int],
    ) -> None:
        new_veg = np.ascontiguousarray(new_veg, dtype=np.int32)
        new_dem = np.ascontiguousarray(new_dem, dtype=np.float64)
        self._sim.expand(
            new_veg, new_dem, (int(new_origin[0]), int(new_origin[1]))
        )

    def freeze_inactive_tiles(self) -> int:
        return self._sim.freeze_inactive_tiles()

    def get_output(self) -> PropagatorOutput:
        out = self._sim.get_output()
        s = out["stats"]
        stats = PropagatorStats(
            n_active=int(s["n_active"]),
            area_mean=float(s["area_mean"]),
            area_50=float(s["area_50"]),
            area_75=float(s["area_75"]),
            area_90=float(s["area_90"]),
        )
        return PropagatorOutput(
            time=int(out["time"]),
            fire_probability=out["fire_probability"],
            spotting_generation_probability=out[
                "spotting_generation_probability"
            ],
            spotting_receiving_probability=out[
                "spotting_receiving_probability"
            ],
            mean_arrival_time=out["mean_arrival_time"],
            min_arrival_time=out["min_arrival_time"],
            ros_mean=out["ros_mean"],
            ros_max=out["ros_max"],
            fli_mean=out["fli_mean"],
            fli_max=out["fli_max"],
            stats=stats,
        )
