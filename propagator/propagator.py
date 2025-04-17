from dataclasses import dataclass, field

import numpy as np
from numpy import tile

from propagator.constants import (
    CELLSIZE,
    LAMBDA_SPOTTING,
    NEIGHBOURS_ANGLE,
    NEIGHBOURS_ARRAY,
    NEIGHBOURS_DISTANCE,
    P_C0,
    P_CD_CONIFER,
)

from propagator.functions import (
    fire_spotting,
    fireline_intensity,
    lhv_canopy,
    lhv_dead_fuel,
    w_h_effect_on_probability,
)
from propagator.scheduler import Scheduler

RNG = np.random.default_rng(12345)


class PropagatorError(Exception):
    pass


@dataclass(frozen=True)
class PropagatorBoundaryConditions:
    time: int
    ignitions: np.ndarray | None
    moisture: np.ndarray | None
    wind_dir: np.ndarray | None
    wind_speed: np.ndarray | None


@dataclass(frozen=True)
class PropagatorActions:
    time: int
    additional_moisture: np.ndarray | None
    vegetation_changes: np.ndarray | None


@dataclass(frozen=True)
class PropagatorStats:
    n_active: int
    area_mean: float
    area_50: float
    area_75: float
    area_90: float


@dataclass(frozen=True)
class PropagatoOutput:
    time: int
    fire_probability: np.ndarray | None
    ros_mean: np.ndarray | None
    ros_max: np.ndarray | None
    fireline_int_mean: np.ndarray | None
    fireline_int_max: np.ndarray | None
    stats: PropagatorStats | None = field(default=None)


@dataclass
class Propagator:
    # domain parameters for the simulation

    # input
    veg: np.ndarray
    dem: np.ndarray
    realizations: int

    # simulation parameters
    ros_0: np.ndarray
    probability_table: np.ndarray
    veg_parameters: np.ndarray
    do_spotting: bool

    # selected simulation functions
    p_time_fn: callable
    p_moist_fn: callable

    # scheduler object
    scheduler: Scheduler = field(init=False, default_factory=Scheduler)

    # simulation state
    time: int = field(init=False, default=0)
    fire: np.ndarray = field(init=False, default=None)
    ros: np.ndarray = field(init=False, default=None)
    fireline_int: np.ndarray = field(init=False, default=None)
    moisture: np.ndarray = field(init=False, default=None)
    wind_dir: np.ndarray = field(init=False, default=None)
    wind_speed: np.ndarray = field(init=False, default=None)
    actions_moisture: np.ndarray = field(
        init=False, default=None
    )  # additional moisture due to fighting actions (ideally it should decay over time)

    def __post_init__(self):
        shape = self.veg.shape
        self.fire = np.zeros(shape + (self.realizations,), dtype=np.int8)
        self.ros = np.zeros(shape + (self.realizations,), dtype=np.float16)
        self.fireline_int = np.zeros(shape + (self.realizations,), dtype=np.float16)
        self.actions_moisture = np.zeros(shape, dtype=np.float16)

    def set_ignitions(self, ignitions: np.ndarray, time: int) -> None:
        """
        Apply ignitions to the state of the simulation.
        """
        points = np.argwhere(ignitions)
        for t in range(self.realizations):
            for p in points:
                self.scheduler.push(np.array([p[0], p[1], t]), time=time)
                self.fire[p[0], p[1], t] = 0

    def compute_fire_probability(self) -> np.ndarray:
        values = np.nanmean(self.fire, 2)
        return values

    def compute_ros_max(self) -> np.ndarray:
        RoS_max = np.nanmax(self.ros, 2)
        return RoS_max

    def compute_ros_mean(self) -> np.ndarray:
        RoS_m = np.where(self.ros > 0, self.ros, np.nan)
        RoS_mean = np.nanmean(RoS_m, 2)
        RoS_mean = np.where(RoS_mean > 0, RoS_mean, 0)
        return RoS_mean

    def compute_fireline_int_max(self) -> np.ndarray:
        fl_I_max = np.nanmax(self.fireline_int, 2)
        return fl_I_max

    def compute_fireline_int_mean(self) -> np.ndarray:
        fl_I_m = np.where(self.fireline_int > 0, self.fireline_int, np.nan)
        fl_I_mean = np.nanmean(fl_I_m, 2)
        fl_I_mean = np.where(fl_I_mean > 0, fl_I_mean, 0)
        return fl_I_mean

    def compute_stats(self, values) -> PropagatorStats:
        n_active = len(self.scheduler.active().tolist())
        # cell_area = float() * float(self.step_y) / 10000.0
        cell_area = 1
        area_mean = float(np.sum(values) * cell_area)
        area_50 = float(np.sum(values >= 0.5) * cell_area)
        area_75 = float(np.sum(values >= 0.75) * cell_area)
        area_90 = float(np.sum(values >= 0.90) * cell_area)

        return PropagatorStats(
            n_active=n_active,
            area_mean=area_mean,
            area_50=area_50,
            area_75=area_75,
            area_90=area_90,
        )

    def propagation_probability(
        self,
        dem_from,
        dem_to,
        veg_from,
        veg_to,
        angle_to,
        dist_to,
        moist,
        w_dir,
        w_speed,
    ):
        dh = dem_to - dem_from
        alpha_wh = w_h_effect_on_probability(angle_to, w_speed, w_dir, dh, dist_to)

        p_moist = self.p_moist_fn(moist)
        p_moist = np.clip(p_moist, 0, 1.0)
        p_veg = self.probability_table[veg_to - 1, veg_from - 1]
        probability = 1 - (1 - p_veg) ** alpha_wh
        probability = np.clip(probability * p_moist, 0, 1.0)

        return probability

    def set_boundary_conditions(
        self, boundary_condition: PropagatorBoundaryConditions
    ) -> None:
        if self.time > boundary_condition.time:
            raise ValueError(
                "Boundary conditions cannot be applied in the past. Please check the time of the boundary conditions."
            )

        if boundary_condition.moisture is not None:
            self.moisture = boundary_condition.moisture
        if boundary_condition.wind_dir is not None:
            self.wind_dir = boundary_condition.wind_dir
        if boundary_condition.wind_speed is not None:
            self.wind_speed = boundary_condition.wind_speed

        if boundary_condition.ignitions is not None:
            self.set_ignitions(boundary_condition.ignitions, boundary_condition.time)

    def apply_actions(self, actions: PropagatorActions) -> None:
        """
        Set the actions to be applied at the current time step.
        """
        if self.time > actions.time:
            raise ValueError(
                "Actions cannot be applied in the past. Please check the time of the actions."
            )

        if actions.additional_moisture is not None:
            self.actions_moisture += actions.additional_moisture
        if actions.vegetation_changes is not None:
            # mutate vegetation where needed
            mask = ~np.isnan(actions.vegetation_changes)
            self.veg[mask] = actions.vegetation_changes[mask]

    def compute_spotting(self, veg_type, update):
        moisture = self.get_moisture()
        # only cells that have veg = fire-prone conifers are selected
        conifer_mask = veg_type == 5
        conifer_r, conifer_c, conifer_t = (
            update[conifer_mask, 0],
            update[conifer_mask, 1],
            update[conifer_mask, 2],
        )

        # calculate number of embers per emitter

        num_embers = RNG.poisson(LAMBDA_SPOTTING, size=conifer_r.shape)

        # create list of source points for each ember
        conifer_arr_r = conifer_r.repeat(repeats=num_embers)
        conifer_arr_c = conifer_c.repeat(repeats=num_embers)
        conifer_arr_t = conifer_t.repeat(repeats=num_embers)
        # calculate angle and distance
        ember_angle = RNG.uniform(0, 2.0 * np.pi, size=conifer_arr_r.shape)
        ember_distance = fire_spotting(ember_angle, self.w_dir, self.w_speed)

        # filter out short embers
        idx_long_embers = ember_distance > 2 * CELLSIZE
        conifer_arr_r = conifer_arr_r[idx_long_embers]
        conifer_arr_c = conifer_arr_c[idx_long_embers]
        conifer_arr_t = conifer_arr_t[idx_long_embers]
        ember_angle = ember_angle[idx_long_embers]
        ember_distance = ember_distance[idx_long_embers]

        # calculate landing locations
        # vertical delta [meters]
        delta_r = ember_distance * np.cos(ember_angle)
        # horizontal delta [meters]
        delta_c = ember_distance * np.sin(ember_angle)
        nb_spot_r = delta_r / CELLSIZE  # number of vertical cells
        nb_spot_r = nb_spot_r.astype(int)
        nb_spot_c = delta_c / CELLSIZE  # number of horizontal cells
        nb_spot_c = nb_spot_c.astype(int)

        # vertical location of the cell to be ignited by the ember
        nr_spot = conifer_arr_r + nb_spot_r
        # horizontal location of the cell to be ignited by the ember
        nc_spot = conifer_arr_c + nb_spot_c
        nt_spot = conifer_arr_t

        shape = self.fire.shape
        # if I surpass the bounds, I stick to them. This way I don't have to reshape anything.
        nr_spot[nr_spot > shape[0] - 1] = shape[0] - 1
        nc_spot[nc_spot > shape[1] - 1] = shape[1] - 1
        nr_spot[nr_spot < 0] = 0
        nc_spot[nc_spot < 0] = 0

        # we want to put another probabilistic filter in order to assess the success of ember ignition.
        #
        # Formula (10) of Alexandridis et al IJWLF 2011
        # P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition by spotting and P_cd is a correction factor that
        # depends on vegetation type and density...
        # In this case, we put P_cd = 0.3 for conifers and 0 for the rest. but it can be generalized..

        # + 0.4 * bushes_mask.... etc etc
        P_c = P_C0 * (1 + P_CD_CONIFER * (self.veg[nr_spot, nc_spot] == 5))

        success_spot_mask = RNG.uniform(size=P_c.shape) < P_c
        nr_spot = nr_spot[success_spot_mask]
        nc_spot = nc_spot[success_spot_mask]
        nt_spot = nt_spot[success_spot_mask]
        # A little more debug on the previous part is advised

        # the following function evalates the time that the embers  will need to burn the entire cell  they land into
        # transition_time_spot = self.p_time(self.dem[nr_spot, nc_spot], self.dem[nr_spot, nc_spot], #evaluation of the propagation time of the "spotted cells"
        transition_time_spot, _ros_spot = self.p_time_fn(
            self.ros_0,
            self.dem[nr_spot, nc_spot],
            self.dem[
                nr_spot, nc_spot
            ],  # evaluation of the propagation time of the "spotted cells"
            # dh=0 (no slope) and veg_from=veg_to to simplify the phenomenon
            self.veg[nr_spot, nc_spot],
            self.veg[nr_spot, nc_spot],
            # ember_angle, ember_distance,
            np.zeros(nr_spot.shape),
            CELLSIZE * np.ones(nr_spot.shape),
            moisture[nr_spot, nc_spot],
            self.w_dir,
            self.w_speed,
        )

        return nr_spot, nc_spot, nt_spot, transition_time_spot

    def apply_updates(self, updates):
        moisture = self.get_moisture()

        # coordinates of the next updates
        update = np.vstack(updates)
        veg_type = self.veg[update[:, 0], update[:, 1]]
        mask = np.logical_and(
            veg_type != 0, self.fire[update[:, 0], update[:, 1], update[:, 2]] == 0
        )

        rows_from, cols_from, realization = (
            update[mask, 0],
            update[mask, 1],
            update[mask, 2],
        )
        self.fire[rows_from, cols_from, realization] = 1

        # # veg type modified due to the heavy fighting actions
        # heavy_acts = bc.get(HEAVY_ACTION_RASTER_TAG, None)
        # if heavy_acts:
        #     for heavyy in heavy_acts:
        #         # da scegliere se mettere a 0 (impossibile che propaghi) 3 (non veg, quindi prova a propagare ma non riesce) o 7(faggete, quindi propaga con bassissima probabilità)
        #         self.veg[heavyy[0], heavyy[1]] = 0

        neighbours_num = NEIGHBOURS_ARRAY.shape[0]
        from_num = rows_from.shape[0]

        neighbours_row_offsets = tile(NEIGHBOURS_ARRAY[:, 0], from_num)
        neighbours_col_offsets = tile(NEIGHBOURS_ARRAY[:, 1], from_num)

        rows_to = rows_from.repeat(neighbours_num) + neighbours_row_offsets
        cols_to = cols_from.repeat(neighbours_num) + neighbours_col_offsets
        nt = realization.repeat(neighbours_num)

        # let's apply a random noise to wind direction and speed for all the cells
        # [TODO] wind dir and speed are not scalar anymore! rewrite
        # w_dir_r = (self.wind_dir + (pi / 16) * (0.5 - RNG.random(from_num))).repeat(
        #     nb_num
        # )
        # w_speed_r = (self.wind_speed * (1.2 - 0.4 * RNG.random(from_num))).repeat(
        #     nb_num
        # )
        w_dir_r = self.wind_dir[rows_from, cols_from].repeat(neighbours_num)
        w_speed_r = self.wind_speed[rows_from, cols_from].repeat(neighbours_num)
        moisture_r = moisture[rows_to, cols_to]

        dem_from = self.dem[rows_from, cols_from].repeat(neighbours_num)
        veg_from = self.veg[rows_from, cols_from].repeat(neighbours_num)
        veg_to = self.veg[rows_to, cols_to]
        dem_to = self.dem[rows_to, cols_to]

        angle_to = NEIGHBOURS_ANGLE[
            neighbours_row_offsets + 1, neighbours_col_offsets + 1
        ]
        dist_to = NEIGHBOURS_DISTANCE[
            neighbours_row_offsets + 1, neighbours_col_offsets + 1
        ]

        # keep only pixels where fire can spread
        valid_fire_mask = np.logical_and(
            self.fire[rows_to, cols_to, nt] == 0, veg_to != 0
        )
        dem_from = dem_from[valid_fire_mask]
        veg_from = veg_from[valid_fire_mask]
        dem_to = dem_to[valid_fire_mask]
        veg_to = veg_to[valid_fire_mask]
        angle_to = angle_to[valid_fire_mask]
        dist_to = dist_to[valid_fire_mask]
        w_speed_r = w_speed_r[valid_fire_mask]
        w_dir_r = w_dir_r[valid_fire_mask]
        moisture_r = moisture_r[valid_fire_mask]
        rows_to = rows_to[valid_fire_mask]
        cols_to = cols_to[valid_fire_mask]
        nt = nt[valid_fire_mask]

        # get the probability for all the pixels
        p_prob = self.propagation_probability(
            dem_from,
            dem_to,
            veg_from,
            veg_to,
            angle_to,
            dist_to,
            moisture_r,
            w_dir_r,
            w_speed_r,
        )

        # try the propagation
        occurrences = p_prob > RNG.random(p_prob.shape[0])

        dem_from = dem_from[occurrences]
        veg_from = veg_from[occurrences]
        dem_to = dem_to[occurrences]
        veg_to = veg_to[occurrences]
        angle_to = angle_to[occurrences]
        dist_to = dist_to[occurrences]
        w_speed_r = w_speed_r[occurrences]
        w_dir_r = w_dir_r[occurrences]
        moisture_r = moisture_r[occurrences]
        rows_to = rows_to[occurrences]
        cols_to = cols_to[occurrences]
        nt = nt[occurrences]

        # get the propagation time for the propagating pixels
        # transition_time = self.p_time(dem_from[p], dem_to[p],
        transition_time, ros = self.p_time_fn(
            self.ros_0,
            dem_from,
            dem_to,
            veg_from,
            veg_to,
            angle_to,
            dist_to,
            moisture_r,
            w_dir_r,
            w_speed_r,
        )

        d0 = self.veg_parameters[veg_to - 1, 0]
        d1 = self.veg_parameters[veg_to - 1, 1]
        hhv = self.veg_parameters[veg_to - 1, 2]
        humidity = self.veg_parameters[veg_to - 1, 3]

        # evaluate LHV of dead fuel
        lhv_dead_fuel_value = lhv_dead_fuel(hhv, moisture_r)
        # evaluate LHV of the canopy
        lhv_canopy_value = lhv_canopy(hhv, humidity)
        # evaluate fireline intensity
        fireline_intensity_value = fireline_intensity(
            d0, d1, ros, lhv_dead_fuel_value, lhv_canopy_value
        )

        self.fireline_int[rows_to, cols_to, nt] = fireline_intensity_value
        self.ros[rows_to, cols_to, nt] = ros

        if self.do_spotting:
            nr_spot, nc_spot, nt_spot, transition_time_spot = self.compute_spotting(
                veg_type, update
            )

            # row-coordinates of the "spotted cells" added to the other ones
            rows_to = np.append(rows_to, nr_spot)
            # column-coordinates of the "spotted cells" added to the other ones
            cols_to = np.append(cols_to, nc_spot)
            # time propagation of "spotted cells" added to the other ones
            nt = np.append(nt, nt_spot)
            transition_time = np.append(transition_time, transition_time_spot)

        prop_time = np.around(self.time + transition_time, decimals=1)

        def extract_updates(t):
            idx = np.nonzero(prop_time == t)
            stacked = np.stack((rows_to[idx], cols_to[idx], nt[idx]), axis=1)
            return stacked

        # schedule the new updates
        unique_time = sorted(np.unique(prop_time))
        new_updates = list(map(lambda t: (t, extract_updates(t)), unique_time))

        return new_updates

    def decay_actions_moisture(
        self, time_delta: int, decay_factor: float = 0.01
    ) -> None:
        """
        Decay the actions moisture over time.
        """
        if self.actions_moisture is None:
            return
        self.actions_moisture = np.clip(
            self.actions_moisture - self.actions_moisture * decay_factor, 0, None
        )

    def get_moisture(self) -> np.ndarray:
        """
        Get the fuel moisture at the current time step.
        """
        if self.actions_moisture is None:
            return self.moisture

        moisture = self.moisture + self.actions_moisture
        moisture = np.clip(moisture, 0, 100)

        return moisture

    def step(
        self,
    ) -> None:
        time, updates = self.scheduler.pop()
        time_delta = time - self.time
        self.time = time
        self.decay_actions_moisture(time_delta)
        new_updates = self.apply_updates(updates)
        self.scheduler.push_all(new_updates)

    def get_output(self) -> PropagatoOutput:
        fire_probability = self.compute_fire_probability()
        ros_max = self.compute_ros_max()
        ros_mean = self.compute_ros_mean()
        fireline_intensity_max = self.compute_fireline_int_max()
        fireline_intensity_mean = self.compute_fireline_int_mean()
        stats = self.compute_stats(fire_probability)

        return PropagatoOutput(
            time=self.time,
            fire_probability=fire_probability,
            ros_mean=ros_mean,
            ros_max=ros_max,
            fireline_int_mean=fireline_intensity_mean,
            fireline_int_max=fireline_intensity_max,
            stats=stats,
        )

    def next_time(self) -> int | None:
        """
        Get the next time step.
        """
        if self.time == 0:
            return 0

        if len(self.scheduler) == 0:
            return None

        return self.scheduler.next_time()
