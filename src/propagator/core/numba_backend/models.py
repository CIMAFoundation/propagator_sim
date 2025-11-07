"""Models and data structures for the Numba JIT-compiled wildfire propagation engine.

This file defines the data structures used in the Numba JIT-compiled wildfire propagation engine.
"""

import numpy as np
from numba import types
from numba.experimental import jitclass
from numba.typed import Dict

from propagator.core.constants import FUEL_SYSTEM_LEGACY_DICT
from propagator.core.models import PropagatorError

spec = [
    ("v0", types.float64),
    ("d0", types.float64),
    ("d1", types.float64),
    ("hhv", types.float64),
    ("humidity", types.float64),
    ("spotting", types.boolean),
    ("prob_ign_by_embers", types.float64),
    ("burn", types.boolean),
    ("name", types.string),
]


@jitclass(spec)  # type: ignore
class Fuel:
    def __init__(
        self,
        name: str,
        v0: float,
        d0: float,
        hhv: float,
        d1: float = 0.0,
        humidity: float = -9999.0,
        spotting: bool = False,
        prob_ign_by_embers: float = 0.0,
        burn: bool = True,
    ):
        """
        Initialize a Fuel object.

        Parameters
        ----------
        name : str
            The name of the fuel type
        v0 : float
            The initial spread rate (m/min)
        d0 : float
            The dead fuel density (kg/m^2)
        hhv : float
            The higher heating value (KJ/kg)
        d1 : float, optional
            The live fuel density (kg/m^2)
        humidity : float, optional
            The fuel moisture content (fraction)
        spotting : bool, optional
            Whether the fuel type is prone to spotting (default is False)
        prob_ign_by_embers : float, optional
            The probability of ignition by embers (default is 0.0)
        burn : bool, optional
            Whether the fuel type is combustible (default is True)
        """
        self.v0 = v0
        self.d0 = d0
        self.d1 = d1
        self.hhv = hhv
        self.humidity = humidity
        self.spotting = spotting
        self.prob_ign_by_embers = prob_ign_by_embers
        self.burn = burn
        self.name = name


spec = [
    ("fuels_id", types.DictType(types.int64, types.int64)),
    ("v0", types.float64[:]),
    ("d0", types.float64[:]),
    ("d1", types.float64[:]),
    ("hhv", types.float64[:]),
    ("humidity", types.float64[:]),
    ("spread_probability", types.float64[:, :]),
    ("spotting", types.boolean[:]),
    ("prob_ign_by_embers", types.float64[:]),
    ("burn", types.boolean[:]),
    ("name", types.DictType(types.int64, types.string)),
    ("_non_vegetated", types.int64),
]


@jitclass(spec)  # type: ignore
class FuelSystem:
    def __init__(self, n_fuels: int):
        self.fuels_id = Dict.empty(
            key_type=types.int64, value_type=types.int64
        )
        self.v0 = np.zeros(n_fuels, dtype=np.float64)
        self.d0 = np.zeros(n_fuels, dtype=np.float64)
        self.d1 = np.zeros(n_fuels, dtype=np.float64)
        self.hhv = np.zeros(n_fuels, dtype=np.float64)
        self.humidity = np.zeros(n_fuels, dtype=np.float64)
        self.spread_probability = np.zeros(
            (n_fuels, n_fuels), dtype=np.float64
        )
        self.spotting = np.zeros(n_fuels, dtype=np.bool_)
        self.prob_ign_by_embers = np.zeros(n_fuels, dtype=np.float64)
        self.burn = np.ones(n_fuels, dtype=np.bool_)
        self.name = Dict.empty(key_type=types.int64, value_type=types.string)
        self._non_vegetated = -1

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    # def which_spotting(self) -> set[int]:
    #     return set(fid for fid, f in self.spotting.items() if f)

    # ---------- public getters ----------
    def get_transition_probability(self, from_id: int, to_id: int) -> float:
        if from_id not in self.fuels_id or to_id not in self.fuels_id:
            raise PropagatorError(
                f"Fuel IDs {from_id} or {to_id} do not exist."
            )
        i = self.fuels_id[from_id]
        j = self.fuels_id[to_id]
        return self.spread_probability[i, j]  # type: ignore

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
        """
        Adds a Fuel object to the FuelSystem.

        Parameters
        ----------
        fuel_id : int
            The unique identifier for the fuel type
        name : str
            The name of the fuel type
        v0 : float
            The initial spread rate (m/min)
        d0 : float
            The dead fuel density (kg/m^2)
        hhv : float
            The higher heating value (KJ/kg)
        d1 : float, optional
            The live fuel density (kg/m^2)
        humidity : float, optional
            The fuel moisture content (fraction)
        spotting : bool, optional
            Whether the fuel type is prone to spotting (default is False)
        prob_ign_by_embers : float, optional
            The probability of ignition by embers (default is 0.0)
        burn : bool, optional
            Whether the fuel type is combustible (default is True)
        """
        n = len(self.fuels_id.keys())
        if fuel_id in self.fuels_id:
            raise PropagatorError(f"Fuel ID {fuel_id} already exists.")
        self.fuels_id[fuel_id] = n
        self.v0[n] = v0
        self.d0[n] = d0
        self.d1[n] = d1
        self.hhv[n] = hhv
        self.humidity[n] = humidity
        self.spotting[n] = spotting
        self.prob_ign_by_embers[n] = prob_ign_by_embers
        self.burn[n] = burn
        self.name[n] = name
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
        i = self.fuels_id[fuel_id]
        return Fuel(
            self.name[i],  # type: ignore
            self.v0[i],  # type: ignore
            self.d0[i],  # type: ignore
            self.hhv[i],  # type: ignore
            self.d1[i],  # type: ignore
            self.humidity[i],  # type: ignore
            self.spotting[i],  # type: ignore
            self.prob_ign_by_embers[i],  # type: ignore
            self.burn[i],  # type: ignore
        )

    def disable_spotting(self):
        for i in range(len(self.spotting)):
            self.spotting[i] = False
            self.prob_ign_by_embers[i] = 0.0


def fuelsystem_from_dict(fuels: dict[int, dict]) -> FuelSystem:
    n_fuels = len(fuels)
    fuelsystem = FuelSystem(n_fuels)
    for k, fuel in fuels.items():
        humid = fuel.get("humidity", -9999.0)
        d1 = fuel.get("d1", 0.0)
        # converts from percentage to fraction
        humidity = humid / 100 if humid != -9999.0 else humid
        # check if humidity and d1 are consistent:
        # if humidity is -9999.0, d1 must be 0.0
        if humidity == -9999.0 and d1 != 0.0:
            raise PropagatorError(
                f"Inconsistent fuel data for fuel ID {k}: "
                "humidity is -9999.0 but d1 is not 0.0."
            )
        fuelsystem.add_fuel(
            k,
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
