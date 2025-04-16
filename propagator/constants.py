import numpy as np
from numpy import pi, array

CELLSIZE = 20

D1 = 0.5
D2 = 1.4
D3 = 8.2
D4 = 2.0
D5 = 50.0
A = 1 - ((D1 * (D2 * np.tanh((0 / D3) - D4))) + (0 / D5))

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBOURS_ARRAY = array(NEIGHBOURS)
NEIGHBOURS_DISTANCE = array([[1.414, 1, 1.414], [1, 0, 1], [1.414, 1, 1.414]])
NEIGHBOURS_ANGLE = array(
    [[pi * 3 / 4, pi / 2, pi / 4], [pi, np.nan, 0], [-pi * 3 / 4, -pi / 2, -pi / 4]]
)


# aggiunte per spotting
# insieme punti lontani 2 celle
NEIGHBOURS_AT2 = [
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, 2),
    (0, -2),
    (0, 2),
    (1, -2),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
]
NEIGHBOURS_AT2_ARRAY = array(NEIGHBOURS_AT2)
NEIGHBOURS_AT2_DISTANCE = array(
    [
        [2.828, 2.236, 2, 2.236, 2.828],
        [2.236, 1.414, 1, 1.414, 2.236],
        [2, 1, 0, 1, 2],
        [2.236, 1.414, 1, 1.414, 2.236],
        [2.828, 2.236, 2, 2.236, 2.828],
    ]
)
NEIGHBOURS_AT2_ANGLE = array(
    [
        [pi * 3 / 4, pi * 13 / 20, pi / 2, pi * 7 / 20, pi / 4],
        [pi * 17 / 20, pi * 3 / 4, pi / 2, pi / 4, pi * 3 / 20],
        [pi, pi, np.nan, 0, 0],
        [pi * 17 / 20, pi * 3 / 4, pi / 2, pi / 4, pi * 3 / 20],
        [-pi * 3 / 4, -pi * 13 / 20, -pi / 2, -pi * 7 / 20, -pi / 4],
    ]
)
# insieme punti lontani 3 celle
NEIGHBOURS_AT3 = [
    (-3, -3),
    (-3, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-3, 2),
    (-3, 3),
    (-2, -3),
    (-2, 3),
    (-1, -3),
    (-1, 3),
    (0, -3),
    (0, 3),
    (1, -3),
    (1, 3),
    (2, -3),
    (2, 3),
    (3, -3),
    (3, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
]
NEIGHBOURS_AT3_ARRAY = array(NEIGHBOURS_AT3)

NEIGHBOURS_AT3_DISTANCE = array(
    [
        [4.243, 3.606, 3.162, 3, 3.162, 3.606, 4.243],
        [3.606, 2.828, 2.236, 2, 2.236, 2.828, 3.606],
        [3.162, 2.236, 1.414, 1, 1.414, 2.236, 3.162],
        [3, 2, 1, 0, 1, 2, 3],
        [3.162, 2.236, 1.414, 1, 1.414, 2.236, 3.162],
        [3.606, 2.828, 2.236, 2, 2.236, 2.828, 3.606],
        [4.243, 3.606, 3.162, 3, 3.162, 3.606, 4.243],
    ]
)
NEIGHBOURS_AT3_ANGLE = array(
    [
        [pi * 3 / 4, pi * 7 / 10, pi * 3 / 5, pi / 2, pi * 2 / 5, pi * 3 / 10, pi / 4],
        [pi * 4 / 5, pi * 3 / 4, pi * 13 / 20, pi / 2, pi * 7 / 20, pi / 4, pi / 5],
        [pi * 9 / 10, pi * 17 / 20, pi * 3 / 4, pi / 2, pi / 4, pi * 3 / 20, pi / 10],
        [pi, pi, pi, np.nan, 0, 0, 0],
        [
            -pi * 9 / 10,
            -pi * 17 / 20,
            -pi * 3 / 4,
            -pi / 2,
            -pi / 4,
            -pi * 3 / 20,
            -pi / 10,
        ],
        [
            -pi * 4 / 5,
            -pi * 3 / 4,
            -pi * 13 / 20,
            -pi / 2,
            -pi * 7 / 20,
            -pi / 4,
            -pi / 5,
        ],
        [
            -pi * 3 / 4,
            -pi * 7 / 10,
            -pi * 3 / 5,
            -pi / 2,
            -pi * 2 / 5,
            -pi * 3 / 10,
            -pi / 4,
        ],
    ]
)
# costante per calcolare distanza in fire-spotting
FIRE_SPOTTING_DISTANCE_COEFFICIENT = 0.191

# parametri Rothermel
ROTHERMEL_ALPHA1 = 0.0693
ROTHERMEL_ALPHA2 = 0.0576
# parametri Wang
WANG_BETA1 = 0.1783
WANG_BETA2 = 3.533
WANG_BETA3 = 1.2

####costanti per moisture
# probabilità
M1 = -3.5995
M2 = 5.2389
M3 = -2.6355
M4 = 1.019
# RoS
C_MOIST = -0.014

# The following constants are used in the Fire-Spotting model. Alexandridis et al. (2009,2011)

LAMBDA_SPOTTING = 2.0
SPOTTING_RN_MEAN = 100
SPOTTING_RN_STD = 25
# P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition by spotting and P_cd is a correction factor that
# depends on vegetation type and density...
P_CD_CONIFER = 0.4
P_C0 = 0.6

# variable for fireline intensity
Q = 2442.0
