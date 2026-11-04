from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from propagator.core.constants import TILE_SIZE
from propagator.core.models import PropagatorOutput
from propagator.io.geo import GeographicInfo
from propagator.io.loader.cog import PropagatorDataFromCogs


class _SteppableSimulator(Protocol):
    time: int
    origin: tuple[int, int]

    def step(self, seconds: int) -> None: ...
    def get_output(self) -> PropagatorOutput: ...
    def boundary_pressure(self) -> tuple[bool, bool, bool, bool]: ...
    def expand(self, veg, dem, new_origin: tuple[int, int]) -> None: ...
    def freeze_inactive_tiles(self) -> int: ...


@dataclass
class SimulationRunner:
    """Drives a `Propagator`/`RustPropagator` instance forward in time.

    Shared by the plain JSON-config CLI (`cli/main.py`) and the `.ff` script
    interpreter (`ff/interpreter.py`) so out-of-bounds/domain-growth/freeze
    handling lives in exactly one place.
    """

    simulator: _SteppableSimulator
    cog_loader: Optional[PropagatorDataFromCogs] = None
    grow_margin: int = 512
    #: Ceiling on the domain, in cells. A growth that would pass it is
    #: refused, so the run carries on clipped at its current edge instead of
    #: growing until the process is killed. None means no ceiling.
    max_domain_cells: Optional[int] = None
    freeze_dir: object | None = None
    on_domain_grow: Optional[Callable[[GeographicInfo], None]] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        # The Rust core's `expand()` requires north/west growth to be a
        # multiple of TILE_SIZE; validate here rather than relying on the
        # CLI (`cli/main.py`) so any caller of this library gets a clear
        # error instead of a confusing core-level failure mid-run.
        if self.grow_margin % TILE_SIZE:
            raise ValueError(
                f"grow_margin must be a multiple of TILE_SIZE ({TILE_SIZE}) "
                f"cells; got {self.grow_margin}"
            )

    def _expand_by(
        self,
        grow_n: int,
        grow_s: int,
        grow_w: int,
        grow_e: int,
        reason: str,
    ) -> Optional[GeographicInfo]:
        """Load and adopt a window wider by the given per-side cell counts.

        The one place the domain actually changes size, whether the fire asked
        for the ground (`grow_domain`) or a boundary condition did
        (`grow_to_cover`). Returns the new geo info, or None when nothing had
        to change or `max_domain_cells` would be passed — growth is
        best-effort, and a refusal leaves the run going on the domain it has.
        """
        assert self.cog_loader is not None
        if not (grow_n or grow_s or grow_w or grow_e):
            return None

        # North/west growth shifts the origin, which the core's expand() only
        # accepts in whole tiles.
        grow_n = -(-grow_n // TILE_SIZE) * TILE_SIZE
        grow_w = -(-grow_w // TILE_SIZE) * TILE_SIZE

        row0, col0 = self.simulator.origin
        rows, cols = self.simulator.veg.shape  # type: ignore[attr-defined]
        new_origin = (row0 - grow_n, col0 - grow_w)
        new_shape = (rows + grow_n + grow_s, cols + grow_w + grow_e)

        if (
            self.max_domain_cells is not None
            and new_shape[0] * new_shape[1] > self.max_domain_cells
        ):
            if self.verbose:
                from propagator.cli.console import info_msg

                info_msg(
                    f"{reason}: {new_shape[0]}x{new_shape[1]} cells would pass "
                    f"the {self.max_domain_cells}-cell cap; not growing"
                )
            return None

        new_dem, new_veg, new_geo_info = self.cog_loader.load_window(
            new_origin, new_shape
        )
        self.simulator.expand(new_veg, new_dem, new_origin)

        if self.on_domain_grow is not None:
            self.on_domain_grow(new_geo_info)

        if self.verbose:
            from propagator.cli.console import info_msg

            info_msg(
                f"{reason}: domain grown to {new_shape[0]}x{new_shape[1]} "
                f"cells (origin {new_origin})"
            )
        return new_geo_info

    def grow_domain(self) -> Optional[GeographicInfo]:
        """Enlarge the domain by grow_margin cells, but only on the side(s)
        the fire actually reached (reported by the core), loading the wider
        window from the COGs (in-place, nothing is lost)."""

        assert self.cog_loader is not None
        margin = self.grow_margin
        north, south, west, east = self.simulator.boundary_pressure()
        if not (north or south or west or east):
            # no edge reported (shouldn't happen right after a halt): grow
            # every side so the run can still make progress
            north = south = west = east = True

        grew = [
            name
            for name, flag in (
                ("N", north),
                ("S", south),
                ("W", west),
                ("E", east),
            )
            if flag
        ]
        return self._expand_by(
            margin if north else 0,
            margin if south else 0,
            margin if west else 0,
            margin if east else 0,
            f"Fire reached the boundary ({'+'.join(grew)})",
        )

    def grow_to_cover(
        self,
        bounds: tuple[float, float, float, float],
        bounds_epsg: Optional[int] = None,
    ) -> Optional[GeographicInfo]:
        """Grow the domain until it covers `bounds`, plus `grow_margin`.

        The deliberate counterpart to `grow_domain`: there the fire reached the
        edge, here something scheduled ahead of it — an ignition or a
        suppression action placed away from the front — sits on ground the
        domain does not hold yet. Rasterizing such a geometry against the
        current domain would silently drop the part outside it, so the caller
        grows first and schedules after.

        `bounds` is `(minx, miny, maxx, maxy)`, in the grid's own CRS unless
        `bounds_epsg` says otherwise. Only the sides that fall short grow, so
        nothing here assumes a direction of spread.
        """
        assert self.cog_loader is not None
        geo_info = self.cog_loader.get_geo_info()

        if bounds_epsg is not None:
            from pyproj import CRS
            from rasterio.warp import transform_bounds

            src = CRS.from_user_input(bounds_epsg)
            dst = CRS.from_user_input(geo_info.crs)
            if src != dst:
                bounds = transform_bounds(src, dst, *bounds)

        west, south, east, north = geo_info.bounds
        rows, cols = geo_info.shape
        cell_x = (east - west) / cols
        cell_y = (north - south) / rows
        minx, miny, maxx, maxy = bounds
        margin = self.grow_margin

        def shortfall(distance: float, size: float) -> int:
            return math.ceil(distance / size) + margin if distance > 0 else 0

        return self._expand_by(
            shortfall(maxy - north, cell_y),
            shortfall(south - miny, cell_y),
            shortfall(west - minx, cell_x),
            shortfall(maxx - east, cell_x),
            "Boundary condition outside the domain",
        )

    def advance_to(self, target_time: int) -> PropagatorOutput:
        """Advance the simulator until `target_time`, growing the domain on
        out-of-bounds if a `cog_loader` is configured (otherwise the
        `PropagatorOutOfBoundsError` propagates to the caller). Returns the
        output snapshot once `target_time` is reached."""
        from propagator.core import PropagatorOutOfBoundsError

        while True:
            try:
                self.simulator.step(seconds=target_time - self.simulator.time)
                break
            except PropagatorOutOfBoundsError:
                if self.cog_loader is None:
                    raise
                self.grow_domain()

        output = self.simulator.get_output()

        if self.freeze_dir is not None:
            self.simulator.freeze_inactive_tiles()

        return output
