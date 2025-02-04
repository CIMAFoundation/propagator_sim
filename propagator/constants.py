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

#aggiunte per spotting
#insieme punti lontani 2 celle
neighbours2 = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2), (-1, 2), (0, -2), (0, 2), (1, -2), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
n2_arr = array([(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2), (-1, 2), (0, -2), (0, 2), (1, -2), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)])
dist2 = array([[2.828, 2.236, 2, 2.236, 2.828], [2.236, 1.414, 1, 1.414, 2.236], [2, 1, 0, 1, 2], [2.236, 1.414, 1, 1.414, 2.236], [2.828, 2.236, 2, 2.236, 2.828]])
angle2 = array([[pi*3/4, pi*13/20, pi/2, pi*7/20, pi/4], [pi*17/20, pi*3/4, pi/2, pi/4, pi*3/20], [pi, pi, np.nan, 0, 0], [pi*17/20, pi*3/4, pi/2, pi/4, pi*3/20], [-pi*3/4, -pi*13/20, -pi/2, -pi*7/20, -pi/4]])
#insieme punti lontani 3 celle
neighbours3 = [(-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-2, -3), (-2, 3), (-1, -3), (-1, 3), (0, -3), (0, 3), (1, -3), (1, 3), (2, -3), (2, 3), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3)]
n3_arr = array([(-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-2, -3), (-2, 3), (-1, -3), (-1, 3), (0, -3), (0, 3), (1, -3), (1, 3), (2, -3), (2, 3), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3)])
dist3 = array([[4.243, 3.606, 3.162, 3, 3.162, 3.606, 4.243], [3.606, 2.828, 2.236, 2, 2.236, 2.828, 3.606], [3.162, 2.236, 1.414, 1, 1.414, 2.236, 3.162], [3, 2, 1, 0, 1, 2, 3], [3.162, 2.236, 1.414, 1, 1.414, 2.236, 3.162], [3.606, 2.828, 2.236, 2, 2.236, 2.828, 3.606], [4.243, 3.606, 3.162, 3, 3.162, 3.606, 4.243]])
angle3 = array([[pi*3/4, pi*7/10, pi*3/5, pi/2, pi*2/5, pi*3/10, pi/4], [pi*4/5, pi*3/4, pi*13/20, pi/2, pi*7/20, pi/4, pi/5], [pi*9/10, pi*17/20, pi*3/4, pi/2, pi/4, pi*3/20, pi/10], [pi, pi, pi, np.nan, 0, 0, 0], [-pi*9/10, -pi*17/20, -pi*3/4, -pi/2, -pi/4, -pi*3/20, -pi/10], [-pi*4/5, -pi*3/4, -pi*13/20, -pi/2, -pi*7/20, -pi/4, -pi/5], [-pi*3/4, -pi*7/10, -pi*3/5, -pi/2, -pi*2/5, -pi*3/10, -pi/4]])
#costante per calcolare distanza in fire-spotting
c_2 = 0.191

#parametri Rothermel
alpha1 = 0.0693
alpha2 = 0.0576
#parametri Wang
beta1 = 0.1783
beta2 = 3.533
beta3 = 1.2

####costanti per moisture
# probabilità
M1 = -3.5995
M2 = 5.2389
M3 = -2.6355
M4 = 1.019
# RoS
c_moist = -0.014

# The following constants are used in the Fire-Spotting model. Alexandridis et al. (2009,2011)

lambda_spotting   = 2.0
spotting_rn_mean  = 100
spotting_rn_std   = 25
# P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition by spotting and P_cd is a correction factor that 
#depends on vegetation type and density...
P_cd_conifer = 0.4
P_c0 = 0.6

# variable for fireline intensity
Q = 2442.0

FUEL_SPOTTING = [5]

#####   TAG   ####
WATERLINE_ACTION_TAG = 'waterline_action'
HEAVY_ACTION_TAG = 'heavy_action'
HEAVY_ACTION_RASTER_TAG = 'heavy_action_raster'
HELICOPTER_TAG = 'helicopter'
CANADAIR_TAG = 'canadair'
MOISTURE_TAG = 'moisture'
MOIST_RASTER_TAG = 'moist_raster'
N_THREADS_TAG = 'n_threads'
BOUNDARY_CONDITIONS_TAG = 'boundary_conditions'
INIT_DATE_TAG = 'init_date'
TILESET_TAG = 'tileset'
GRID_DIM_TAG = 'grid_dim'
TIME_RESOLUTION_TAG = 'time_resolution'
OUTPUT_FOLDER_TAG = 'output_folder'
TIME_LIMIT_TAG = 'time_limit'
ROS_MODEL_CODE_TAG = 'ros_model_code'
TIME_TAG = 'time'
W_DIR_TAG = 'w_dir'
W_SPEED_TAG = 'w_speed'

PROB_FILE_TAG = 'prob_file'
V0_TABLE_TAG = 'v0_file'
P_VEGETATION_TAG = 'p_vegetation'
IGNITIONS_TAG = 'ignitions'
GRID_DIM_KM_TAG = 'grid_dim_km'
IGNITIONS_RASTER_TAG = 'ignitions_raster'

ROS_MODEL_TAG = 'ros_model'
DEFAULT_TAG = 'default'
WANG_TAG = 'wang'
ROTHERMEL_TAG = 'rothermel'

PROB_MOIST_CODE_TAG = 'prob_moist_model'
NEW_FORMULATION_TAG = 'new_formula'
STD_FORMULATION_TAG = 'rothermel'

SPOT_FLAG_TAG = 'do_spotting'


WATERLINE_ACTION_VALUE = 0.27

CANADAIR_VALUE = 0.25
CANADAIR_BUFFER_VALUE = 0.22

HELICOPTER_VALUE = 0.22
HELICOPTER_BUFFER_VALUE = 0.2