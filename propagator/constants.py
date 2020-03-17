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

####costanti per moisture
# probabilità
M1 = -3.5995
M2 = 5.2389
M3 = -2.6355
M4 = 1.019
# RoS
c_moist = -0.014


#####   TAG   ####
WATERLINE_ACTION_TAG = 'waterline_action'
HEAVY_ACTION_TAG = 'heavy_action'
HEAVY_ACTION_RASTER_TAG = 'heavy_action_raster'
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
IGNITIONS_TAG = 'ignitions'
GRID_DIM_KM_TAG = 'grid_dim_km'
IGNITIONS_RASTER_TAG = 'ignitions_raster'

ROS_MODEL_TAG = 'ros_model'
DEFAULT_TAG = 'default'
WANG_TAG = 'wang'
ROTHERMEL_TAG = 'rothermel'