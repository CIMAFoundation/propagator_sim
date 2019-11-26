import numpy as np
from numpy import pi, array

'''
D1 = 0.5
D2 = 2
D3 = 10
D4 = 2
D5 = 50
'''

D1 = 0.5
D2 = 1.4
D3 = 8.2
D4 = 2.0
D5 = 50.0


A = 1 - ((D1 * (D2 * np.tanh((0 / D3) - D4))) + (0 / D5))

neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
n_arr = array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])

dist = array([[1.414, 1, 1.414], [1, 0, 1], [1.414, 1, 1.414]])
angle = array([[pi*3/4, pi/2, pi/4], [pi, np.nan, 0], [-pi*3/4, -pi/2, -pi/4]])
cellsize = 20

#parametri Rothermel
alpha1 = 0.0693
alpha2 = 0.0576
#parametri Wang
beta1 = 0.1783
beta2 = 3.533
beta3 = 1.2