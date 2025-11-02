"""Models and data structures for the NumPy wildfire propagation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propagator.core.constants import FUEL_SYSTEM_LEGACY_DICT
from propagator.core.models import PropagatorError


@dataclass(frozen=True)
class Fuel:
    """Fuel descriptor mirroring the Numba jitclass version."""

    name: str
    v0: float
    d0: float
    hhv: float
    d1: float = 0.0
    humidity: float = -9999.0
    spotting: bool = False
    prob_ign_by_embers: float = 0.0
    burn: bool = True


class FuelSystem:
    """Container for fuel parameters and transition probabilities."""

    def __init__(self, n_fuels: int):
        self.fuels_id: dict[int, int] = {}
        self.v0 = np.zeros(n_fuels, dtype=np.float64)
        self.d0 = np.zeros(n_fuels, dtype=np.float64)
        self.d1 = np.zeros(n_fuels, dtype=np.float64)
        self.hhv = np.zeros(n_fuels, dtype=np.float64)
        self.humidity = np.zeros(n_fuels, dtype=np.float64)
        self.spread_probability = np.zeros(
            (n_fuels, n_fuels), dtype=np.float64
        )
        self.spotting = np.zeros(n_fuels, dtype=bool)
        self.prob_ign_by_embers = np.zeros(n_fuels, dtype=np.float64)
        self.burn = np.ones(n_fuels, dtype=bool)
        self.name: dict[int, str] = {}
        self._non_vegetated = -1

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    def get_transition_probability(self, from_id: int, to_id: int) -> float:
        if from_id not in self.fuels_id or to_id not in self.fuels_id:
            raise PropagatorError(
                f"Fuel IDs {from_id} or {to_id} do not exist."
            )
        i = self.fuels_id[from_id]
        j = self.fuels_id[to_id]
        return float(self.spread_probability[i, j])

    def add_fuel(
        self,
        fuel_id: int,
        name: str,
        v0: float,
        d0: float,
        hhv: float,
        d1: float = 0.0,
        humidity: float = -9999.0,
        spotting: bool = False,
        prob_ign_by_embers: float = 0.0,
        burn: bool = True,
    ) -> None:
        if fuel_id in self.fuels_id:
            raise PropagatorError(f"Fuel ID {fuel_id} already exists.")
        index = len(self.fuels_id)
        if index >= len(self.v0):
            raise PropagatorError(
                "FuelSystem capacity exceeded when adding new fuel."
            )
        self.fuels_id[fuel_id] = index
        self.v0[index] = v0
        self.d0[index] = d0
        self.d1[index] = d1
        self.hhv[index] = hhv
        self.humidity[index] = humidity
        self.spotting[index] = spotting
        self.prob_ign_by_embers[index] = prob_ign_by_embers
        self.burn[index] = burn
        self.name[index] = name
        if not burn:
            self._non_vegetated = fuel_id

    def add_transition_probability(
        self, from_id: int, to_id: int, prob: float
    ) -> None:
        if from_id not in self.fuels_id or to_id not in self.fuels_id:
            raise PropagatorError(
                f"Fuel IDs {from_id} or {to_id} do not exist."
            )
        i = self.fuels_id[from_id]
        j = self.fuels_id[to_id]
        self.spread_probability[i, j] = prob

    def get_fuel(self, fuel_id: int) -> Fuel:
        if fuel_id not in self.fuels_id:
            raise PropagatorError(f"Fuel ID {fuel_id} does not exist.")
        index = self.fuels_id[fuel_id]
        return Fuel(
            name=self.name[index],
            v0=float(self.v0[index]),
            d0=float(self.d0[index]),
            hhv=float(self.hhv[index]),
            d1=float(self.d1[index]),
            humidity=float(self.humidity[index]),
            spotting=bool(self.spotting[index]),
            prob_ign_by_embers=float(self.prob_ign_by_embers[index]),
            burn=bool(self.burn[index]),
        )

    def disable_spotting(self) -> None:
        self.spotting[:] = False
        self.prob_ign_by_embers[:] = 0.0


def fuelsystem_from_dict(fuels: dict[int, dict]) -> FuelSystem:
    """Create a ``FuelSystem`` from a configuration dictionary."""

    n_fuels = len(fuels)
    fuelsystem = FuelSystem(n_fuels)
    for fuel_id, fuel in fuels.items():
        humid = fuel.get("humidity", -9999.0)
        d1 = fuel.get("d1", 0.0)
        humidity = humid / 100 if humid != -9999.0 else humid
        if humidity == -9999.0 and d1 != 0.0:
            raise PropagatorError(
                f"Inconsistent fuel data for fuel ID {fuel_id}: "
                "humidity is -9999.0 but d1 is not 0.0."
            )
        fuelsystem.add_fuel(
            fuel_id,
            fuel["name"],
            fuel["v0"] / 60,  # converts from m/h to m/min
            fuel["d0"],
            fuel["hhv"],
            d1,
            humidity,
            fuel.get("spotting", False),
            fuel.get("prob_ign_by_embers", 0.0),
            fuel.get("burn", True),
        )
    for from_id, fuel in fuels.items():
        for to_id, prob in fuel["spread_probability"].items():
            fuelsystem.add_transition_probability(from_id, to_id, prob)
    return fuelsystem


FUEL_SYSTEM_LEGACY = fuelsystem_from_dict(FUEL_SYSTEM_LEGACY_DICT)
