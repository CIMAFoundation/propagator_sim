from typing import Literal

TICK_PRECISION = 10
CELLSIZE = 20  # [m]
REALIZATIONS = 100

# --- DEFAULT MODELS ---
ROS_DEFAULT: Literal["wang", "rothermel"] = "wang"
MOISTURE_MODEL_DEFAULT: Literal["trucchia", "baghino"] = "trucchia"


# --- FUEL SYSTEM LEGACY ---
NO_FUEL = 0


FUEL_SYSTEM_LEGACY_DICT = {
    # key_fuel: dict(
    #     name="name_fuel",
    #     v0=140,  # nominal rate of spread - in m/h
    #     d0=1.5,  # dead fuel load - in kg/m2
    #     hhv=20000,  # higher heating value - in kJ/kg
    #     d1=3,  # live fuel load - in kg/m2 (optional)
    #     humidity=60, # live fuel moisture - in percentage (optional)
    #     spread_probability={  # spread probability to other fuel types
    #         1: 0.3,
    #         2: 0.375,
    #         3: 0.005,
    #         4: 0.45,
    #         5: 0.225,
    #         6: 0.25,
    #         7: 0.075,
    #     },
    #     spotting=False,  # if the fuel type is prone to spotting (optional)
    #     prob_ign_by_embers=0.0,  # prob. of ignition by embers (optional)
    #     burn=True,  # if the fuel type is combustible (optional)
    # ),
    1: dict(
        name="broadleaves",
        v0=140,
        d0=1.5,
        d1=3,
        hhv=20000,
        humidity=60,
        spread_probability={
            1: 0.3,
            2: 0.375,
            3: 0.005,
            4: 0.45,
            5: 0.225,
            6: 0.25,
            7: 0.075,
        },
    ),
    2: dict(
        name="shrubs",
        v0=140,
        d0=1,
        d1=3,
        hhv=21000,
        humidity=45,
        spread_probability={
            1: 0.375,
            2: 0.375,
            3: 0.005,
            4: 0.475,
            5: 0.325,
            6: 0.25,
            7: 0.1,
        },
    ),
    3: dict(
        name="non-vegetated",
        v0=20,
        d0=0.1,
        hhv=100,
        spread_probability={
            1: 0.005,
            2: 0.005,
            3: 0.005,
            4: 0.005,
            5: 0.005,
            6: 0.005,
            7: 0.005,
        },
        burn=False,
    ),
    4: dict(
        name="grassland",
        v0=120,
        d0=0.5,
        hhv=17000,
        spread_probability={
            1: 0.25,
            2: 0.35,
            3: 0.005,
            4: 0.475,
            5: 0.1,
            6: 0.3,
            7: 0.075,
        },
    ),
    5: dict(
        name="conifers",
        v0=200,
        d0=1,
        d1=4,
        hhv=21000,
        humidity=55,
        spread_probability={
            1: 0.275,
            2: 0.4,
            3: 0.005,
            4: 0.475,
            5: 0.35,
            6: 0.475,
            7: 0.275,
        },
        spotting=True,
        prob_ign_by_embers=0.4,
    ),
    6: dict(
        name="agro-forestry areas",
        v0=120,
        d0=0.5,
        d1=2,
        hhv=19000,
        humidity=60,
        spread_probability={
            1: 0.25,
            2: 0.3,
            3: 0.005,
            4: 0.375,
            5: 0.2,
            6: 0.35,
            7: 0.075,
        },
    ),
    7: dict(
        name="non-fire prone forests",
        v0=60,
        d0=1,
        d1=2,
        hhv=18000,
        humidity=65,
        spread_probability={
            1: 0.25,
            2: 0.375,
            3: 0.005,
            4: 0.475,
            5: 0.35,
            6: 0.25,
            7: 0.075,
        },
    ),
}
