#!/usr/bin/env python
__author__ = 'mirko'

import json
import logging
import os
import sys
import traceback
import enum

from datetime import datetime

import numpy as np

from scipy import ndimage


import propagator.logging_config
from propagator.args_parser import parse_params
from propagator.propagator import NoTilesError, Propagator, PropagatorSettings
from propagator.utils import normalize

from propagator.constants import *


class ErrorCodes(enum.Enum):
    OK = 0
    GENERIC_ERROR = 1
    DOMAIN_ERROR = 2
    IGNITIONS_ERROR = 3
    BC_ERROR = 4


def main():  
    args = parse_params()

    if args.param_file is None:
        raise Exception('Error. Missing parameter file')

    try:
        d = json.load(args.param_file)
    except Exception as exp:
        traceback.print_exc(file=open("errlog.txt", "a"))
        raise exp

    n_threads = int(d.get(N_THREADS_TAG, 10))
    grid_dim_km = float(d.get(GRID_DIM_KM_TAG, 10))
    grid_dim = np.floor(grid_dim_km / 20 * 1000)
    grid_dim = int(np.clip(np.floor(grid_dim), 300, 1500))
    tile_set = d.get(TILESET_TAG, DEFAULT_TAG)
    ros_model_code = d.get(ROS_MODEL_TAG, WANG_TAG) #switch per scegliere se usare il modello di Rothermel (rothermel), Wang (wang) oppure il classico Propagator (default)
    
    #controllo che sia stato richiesto il modello di RoS in maniera corretta
    if ros_model_code not in [DEFAULT_TAG , WANG_TAG , ROTHERMEL_TAG]:
        logging.info('WARNING: RoS function is not well defined, the model will use "wang" configuration')

    w_dir_deg = float(d.get(W_DIR_TAG, 0))
    w_dir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
    w_speed = float(d.get(W_SPEED_TAG, 0))
    moisture_100 = int(d.get(MOISTURE_TAG, 0))
    fighting_actions = d.get(FIGHTING_ACTION_TAG, None)
            
    time_resolution = float(d.get(TIME_RESOLUTION_TAG, 60))

    boundary_conditions = d.get(BOUNDARY_CONDITIONS_TAG, [{
        "w_dir": w_dir,
        "w_speed": w_speed,
        "moisture": moisture_100,
        "fighting_action": fighting_actions,  #fighting_action_string = '\n'.join(fighting_actions)
        "time": 0
    }])

    boundary_conditions = sorted(boundary_conditions, key=lambda k: k[TIME_TAG])
    if boundary_conditions[0][TIME_TAG] > 0:
        boundary_conditions.insert(
            0,
            {
                "w_dir": 0.0,
                "w_speed": 0.0,
                "moisture":0,
                "fighting_actions": None,
                "time": 0
            }
        )


    if IGNITIONS_TAG not in d:
        logging.critical('Error. Missing ignitions in parameter file')
        raise Exception('Error. Missing ignitions in parameter file')
    
    ignitions = d[IGNITIONS_TAG]
    ignition_string = '\n'.join(ignitions)

    date_str = d.get(INIT_DATE_TAG)
    if date_str is None:
        init_date = datetime.now()
    else:
        init_date = datetime.strptime(date_str, '%Y%m%d%H%M')

    if args.output_folder and not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    time_limit_min = d.get(TIME_LIMIT_TAG, None)
    if time_limit_min is None and args.time_limit:
        time_limit_min = args.time_limit*60

    if PROB_FILE_TAG in d or V0_TABLE_TAG in d:
        prob_file = d.get(PROB_FILE_TAG, None)
        v0_file = d.get(V0_TABLE_TAG, None)
        propagator.load_parameters(prob_file, v0_file)

    settings = PropagatorSettings(
        n_threads=n_threads,
        boundary_conditions=boundary_conditions,
        init_date=init_date,
        tileset=tile_set,
        grid_dim=grid_dim,
        time_resolution=time_resolution,
        output_folder=args.output_folder,
        time_limit=time_limit_min,
        simp_fact=args.simp_fact,
        debug_mode=args.debug_mode,
        write_vegetation=args.write_vegetation,
        save_realizations=args.save_realizations,
        ros_model_code=ros_model_code
    )

    sim = Propagator(settings)
    easting, northing, zone_number, zone_letter, polys, lines, points = sim.load_ignitions_from_string(ignition_string)
    if args.veg_file is None and args.dem_file is None:
        sim.load_data_from_tiles(easting, northing, zone_number)
    else:
        assert args.veg_file is not None, 'No veg_file parameter defined' 
        assert args.dem_file is not None, 'No dem_file parameter defined'
       
        sim.load_data_from_files(args.veg_file, args.dem_file)

    sim.init_ignitions(polys, lines, points, zone_number)
    sim.run()
    logging.info('completed')


if __name__ == '__main__':
    ERROR_CODE = ErrorCodes.OK
    try:        
        main()
    except NoTilesError as no_tiles:
        ERROR_CODE = ErrorCodes.DOMAIN_ERROR
    
    except Exception as exp:
        traceback.print_exc()
        ERROR_CODE = ErrorCodes.GENERIC_ERROR
        raise
    finally:
        sys.exit(ERROR_CODE.value)
