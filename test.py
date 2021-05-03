from propagator.propagator import run
from datetime import datetime
import numpy as np
import propagator.logging_config

if __name__ == '__main__':
    run_id = 'test'
    n_threads = 10
    grid_dim_km = 10
    wdir = np.pi/2
    wspeed = 90
    grid_dim = np.floor(grid_dim_km / 20 * 1000)
    grid_dim = int(max(min(np.floor(grid_dim), 1500), 300))
    tileset = 'default'
    init_date = datetime(2016, 1, 1, 0, 0)
    s1 = [
        #"LINE:[44.3204247306364 44.320317268240956 ];[8.44812858849764 8.449995405972006 ]",
        #"POLYGON:[44.32214410219511 44.320869929892176 44.32083922660368 44.32214410219511 ];[8.454050906002523 8.453171141445639 8.45463026314974 8.454050906002523 ]",
        "POINT:44.32372526549074;8.45040310174227"]
    ignition_string = '\n'.join(s1)

    run(run_id, n_threads, wdir, wspeed, init_date, ignition_string, tileset, grid_dim)
