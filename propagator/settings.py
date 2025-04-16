#####   TAG   ####
from dataclasses import dataclass
from propagator.functions import get_p_moist_fn, get_p_time_fn

N_THREADS_TAG = "n_threads"

BOUNDARY_CONDITIONS_TAG = "boundary_conditions"
INIT_DATE_TAG = "init_date"
TILESET_TAG = "tileset"
GRID_DIM_TAG = "grid_dim"
TIME_RESOLUTION_TAG = "time_resolution"
OUTPUT_FOLDER_TAG = "output_folder"
TIME_LIMIT_TAG = "time_limit"

ROS_MODEL_CODE_TAG = "ros_model_code"
W_DIR_TAG = "w_dir"
W_SPEED_TAG = "w_speed"

PROB_FILE_TAG = "prob_file"
V0_TABLE_TAG = "v0_file"
P_VEGETATION_TAG = "p_vegetation"
IGNITIONS_TAG = "ignitions"
GRID_DIM_KM_TAG = "grid_dim_km"

ROS_MODEL_TAG = "ros_model"
DEFAULT_TILES_TAG = "default"
WANG_TAG = "wang"
ROTHERMEL_TAG = "rothermel"

PROB_MOIST_CODE_TAG = "prob_moist_model"
NEW_FORMULATION_TAG = "new_formula"
STD_FORMULATION_TAG = "rothermel"

SPOT_FLAG_TAG = "do_spotting"


@dataclass(frozen=True)
class PropagatorSettings:
    n_threads: int
    boundary_conditions: list
    init_date: str
    tileset: str
    grid_dim: tuple
    time_resolution: int
    output_folder: str
    time_limit: int
    p_time_fn: callable
    p_moist_fn: callable
    do_spotting: bool

    @staticmethod
    def from_dict(**settings_dict) -> "PropagatorSettings":
        n_threads = settings_dict[N_THREADS_TAG]
        boundary_conditions = settings_dict[BOUNDARY_CONDITIONS_TAG]
        init_date = settings_dict[INIT_DATE_TAG]
        tileset = settings_dict[TILESET_TAG]
        grid_dim = settings_dict[GRID_DIM_TAG]
        time_resolution = settings_dict[TIME_RESOLUTION_TAG]
        output_folder = settings_dict[OUTPUT_FOLDER_TAG]
        time_limit = settings_dict[TIME_LIMIT_TAG]
        p_time_fn = get_p_time_fn(settings_dict[ROS_MODEL_CODE_TAG])
        p_moist_fn = get_p_moist_fn(settings_dict[PROB_MOIST_CODE_TAG])
        do_spotting = settings_dict[SPOT_FLAG_TAG]

        return PropagatorSettings(
            n_threads=n_threads,
            boundary_conditions=boundary_conditions,
            init_date=init_date,
            tileset=tileset,
            grid_dim=grid_dim,
            time_resolution=time_resolution,
            output_folder=output_folder,
            time_limit=time_limit,
            p_time_fn=p_time_fn,
            p_moist_fn=p_moist_fn,
            do_spotting=do_spotting,
        )
