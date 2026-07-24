from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

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
    freeze_dir: object | None = None
    on_domain_grow: Optional[Callable[[GeographicInfo], None]] = None
    verbose: bool = False

    def grow_domain(self) -> None:
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

        grow_n = margin if north else 0
        grow_s = margin if south else 0
        grow_w = margin if west else 0
        grow_e = margin if east else 0

        row0, col0 = self.simulator.origin
        rows, cols = self.simulator.veg.shape  # type: ignore[attr-defined]
        # north/west growth shifts the origin; the shift stays a multiple of
        # TILE_SIZE because grow_margin is, which the core's expand requires
        new_origin = (row0 - grow_n, col0 - grow_w)
        new_shape = (rows + grow_n + grow_s, cols + grow_w + grow_e)

        new_dem, new_veg, new_geo_info = self.cog_loader.load_window(
            new_origin, new_shape
        )
        self.simulator.expand(new_veg, new_dem, new_origin)

        if self.on_domain_grow is not None:
            self.on_domain_grow(new_geo_info)

        if self.verbose:
            from propagator.cli.console import info_msg

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
            info_msg(
                f"Fire reached the boundary ({'+'.join(grew)}): domain grown "
                f"to {new_shape[0]}x{new_shape[1]} cells (origin {new_origin})"
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
