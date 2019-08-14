#!/usr/bin/env python
__author__ = 'mirko'

import json
import logging
import os
import traceback
from datetime import datetime

import numpy as np

import propagator.logging_config
from propagator import propagator
from propagator.utils import normalize
from propagator.args_parser import parse_params


def main():
    args = parse_params()

    if args.param_file is None:
        raise Exception('Error. Missing parameter file')

    try:
        d = json.load(args.param_file)
    except Exception as exp:
        traceback.print_exc(file=open("errlog.txt", "a"))
        raise exp

    n_threads = int(d.get('n_threads', 10))
    grid_dim_km = float(d.get('grid_dim_km', 10))
    grid_dim = np.floor(grid_dim_km / 20 * 1000)
    grid_dim = int(np.clip(np.floor(grid_dim), 300, 1500))
    tile_set = d.get('tileset', 'default')

    w_dir_deg = float(d.get('w_dir', 0))
    w_dir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
    w_speed = float(d.get('w_speed', 0))

    time_resolution = float(d.get('time_resolution', 60))

    boundary_conditions = d.get('boundary_conditions', [{
        "w_dir": w_dir,
        "w_speed": w_speed,
        "time": 0
    }])

    boundary_conditions = sorted(boundary_conditions, key=lambda k: k['time'])

    if 'ignitions' not in d:
        logging.critical('Error. Missing ignitions in parameter file')
        raise Exception('Error. Missing ignitions in parameter file')
    else:
        ignitions = d['ignitions']
        ignition_string = '\n'.join(ignitions)

    date_str = d.get('init_date')
    if date_str is None:
        init_date = datetime.now()
    else:
        init_date = datetime.strptime(date_str, '%Y%m%d%H%M')

    if args.output_folder and not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    time_limit_min = d.get('time_limit', None)
    if time_limit_min is None and args.time_limit:
        time_limit_min = args.time_limit*60

    if 'prob_file' in d or 'v0_file' in d:
        prob_file = d.get('prob_file', None)
        v0_file = d.get('v0_file', None)
        propagator.load_parameters(prob_file, v0_file)

    try:
        propagator.run(
            args.run_id, n_threads, boundary_conditions,
            init_date, ignition_string, tile_set, grid_dim,
            time_resolution=time_resolution,
            time_limit=time_limit_min,
            output_folder=args.output_folder,
            simp_fact=args.simp_fact,
            debug_mode=args.debug_mode,
            write_vegetation=args.write_vegetation,
            save_realizations=args.save_realizations
        )
    except Exception as exp:
        traceback.print_exc()
        logging.info('error')
        raise exp
    logging.info('completed')


if __name__ == '__main__':
    main()
