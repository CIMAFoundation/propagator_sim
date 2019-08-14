import json
import logging
import os
from datetime import timedelta

# import utm
from numpy import array, pi, sign, tanh, tile
from numpy.random import rand
from pyproj import Proj
from rasterio import crs, enums, transform, warp

from .constants import *
from .utils import *

# [latifoglie cespugli aree_nude erba conifere coltivi faggete]
try:
    propagator_path = os.environ.get('PROPAGATOR_PATH', './')
    v0 = np.loadtxt(os.path.join(propagator_path, 'v0_table.txt'))
    prob_table = np.loadtxt(os.path.join(propagator_path, 'prob_table.txt'))
except Exception:
    v0, prob_table = None, None


def load_parameters(probability_file=None, v0_file=None):
    """
    Override the default values for vegetation speed and probabilities by loading them from file
    :param probability_file:
    :param time_file:
    :return:
    """
    global v0, prob_table
    if v0_file:
        v0 = np.loadtxt(v0_file)
    if probability_file:
        prob_table = np.loadtxt(probability_file)


def p_time(dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed):
    # velocità di base modulata con la densità(tempo di attraversamento)
    dh = (dem_to - dem_from)
    # dir_to = np.stack((np.cos(angle_to), np.sin(angle_to)), axis=1)
    wh = w_h_effect(angle_to, w_speed, w_dir, dh, dist)

    # tempo in minuti di attraversamento di una cella 
    v = v0[veg_from-1] / 60
    v_wh = np.clip(v * wh, 0.01, 100)
    real_dist = np.sqrt((cellsize*dist)**2 + dh**2)
    t = real_dist / v_wh
    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t


def w_h_effect(angle_to, w_speed, w_dir, dh, dist):
    w_effect_module = (A + (D1 * (D2 * np.tanh((w_speed / D3) - D4))) + (w_speed / D5))
    a = (w_effect_module - 1) / 4
    w_effect_on_direction = (a + 1) * (1 - a ** 2) / (1 - a * np.cos(normalize(w_dir - angle_to)))
    #h_effect = 1 + (tanh((dh / 7) ** 2. * sign(dh)))
    slope = dh/(cellsize*dist)
    h_effect = 2**((tanh((slope * 3) ** 2. * sign(slope))))

    w_h = h_effect * w_effect_on_direction
    #w_h = np.clip(w_h, 0.1, np.Inf)
    return w_h


def w_h_effect_on_p(angle_to, w_speed, w_dir, dh, dist_to):
    """
    scales the wh factor for using it on the probability modulation
    """
    w_speed_norm = np.clip(w_speed, 0, 60)
    wh_orig = w_h_effect(angle_to, w_speed_norm, w_dir, dh, dist_to)
    wh = wh_orig - 1.0
    wh[wh > 0] = wh[wh > 0]/2.13
    wh[wh < 0] = wh[wh < 0]/1.12
    wh += 1.0
    return wh


def p_probability(dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moist, w_dir, w_speed):
    dh = (dem_to - dem_from)
    alpha_wh = w_h_effect_on_p(angle_to, w_speed, w_dir, dh, dist_to)
    
    p_moist = 1
    p_veg = prob_table[veg_to - 1, veg_from - 1]
    p = 1-(1-p_veg)**alpha_wh
    p_clip = np.clip(p, 0, 1.0)

    return p_clip


class PropagatorSettings:
    def __init__(self, settings_dict):
        self.n_threads = settings_dict['n_threads']
        self.boundary_conditions = settings_dict['boundary_conditions']
        self.init_date = settings_dict['init_date'] 
        self.ignition_string = settings_dict['ignition_string']
        self.tileset = settings_dict['tileset']
        self.grid_dim = settings_dict['grid_dim']
        self.time_resolution = settings_dict['time_resolution']
        self.output_folder = settings_dict['output_folder']
        self.time_limit = settings_dict['time_limit']
        self.simp_fact = settings_dict['simp_fact']
        self.debug_mode = settings_dict['debug_mode']
        self.write_vegetation = settings_dict['write_vegetation']
        self.save_realizations = settings_dict['save_realizations']


class PropagatorRun:
    def __init__(self, settings: PropagatorSettings):
        self.settings = settings
        self.ps = Scheduler()
        self.c_time = 0,
        self.f_global = None
        self.veg = None
        self.dem = None
        self.boundary_conditions = None


        pass

    def __init_crs_from_bounds(west:float, south:float, east:float, north:float, cols:int, rows:int, 
                                zone:int, proj:str='utm', datum:str='WGS84'):
        self.__prj = Proj(proj=proj, zone=zone_number, datum=datum)
        self.__trans = transform.from_bounds(west, south, east, north, cols, rows)


    def load_data_from_files(self):
        pass
    
    def __init_ignitions(self):
        img, active_ignitions = \
        rasterize_ignitions((grid_dim, grid_dim),
                            points, lines, polys, west, north, step_x, step_y, zone_number)


    def load_data_from_tiles(self, easting, northing, zone_number):
        logging.info('Loading VEGETATION from "' + self.settings.tileset + '" tileset')
        self.veg, west, north, step_x, step_y = \
            load_tiles(zone_number, easting, northing, grid_dim, 'prop', tileset)
        veg = veg.astype('int8')
        veg[:, (0, -1)] = 0
        veg[(0, -1), :] = 0

        logging.info('Loading DEM "default" tileset')
        self.dem, west, north, step_x, step_y = \
            load_tiles(zone_number, easting, northing, grid_dim, 'quo', 'default')
        dem = dem.astype('float')

        self.moist = np.zeros_like(veg, dtype='float')

    def __compute_values(self):
        values = np.nanmean(self.f_global, 2)
        return values

    def __compute_stats(self, values):
        n_active = len(self.ps.active().tolist())
        area_mean = float(np.sum(values) * step_x * step_y) / 10000
        area_50 = float(np.sum(values >= 0.5) * step_x * step_y) / 10000
        area_75 = float(np.sum(values >= 0.75) * step_x * step_y) / 10000
        area_90 = float(np.sum(values >= 0.90) * step_x * step_y) / 10000

        return n_active, area_mean, area_50, area_75, area_90


    def log(self, n_active, area_mean):
        days = round(self.c_time // (24*60))
        hours = round((self.c_time % (24*60)) // 60)
        minutes = round(self.c_time % 60)        
        logging.info(
            '{0:1.0f}d {1:2.0f}h {2:2.0f}m - {3} active - {4:.1f} [ha]'.format(
                days, hours, minutes, n_active, area_mean
            )
        )


    def __wite_output(self, values, **kwargs):
        filename = os.path.join(self.settings.output_folder, str(self.c_time))
        tiff_file = filename + '.tiff'
        json_file = filename + '.json'

        
        ref_date = str(self.settings.init_date + timedelta(minutes=c_time))


        with open(json_file, 'w') as fp:
            json.dump(
                dict(
                    time=c_time,
                    timeref=ref_date,
                ).update(kwargs),
                fp
            )

            write_geotiff(tiff_file, values*255, self.__trans, self.__prj)

    def write_isochrone(self):
        pass

    def __find_bc(self):
        last_bc = None
        for bc in self.boundary_conditions:
            if self.c_time >= bc['time']:
                last_bc = bc
        return last_bc

    def __init_simulation(self, n_threads, initial_ignitions, active_ignitions):
        self.f_global = np.zeros(initial_ignitions.shape + [n_threads])
        for t in range(n_threads):
            self.f_global[:, :, t] = initial_ignitions.copy()
            for p in active_ignitions:
                self.ps.push(array([p[0], p[1], t]), 0)
                self.f_global[p[0], p[1], t] = 0

    def __update_isochrones(self, isochrones, values):
        isochrones[self.c_time] = extract_isochrone(
                values, dst_trans,
                thresholds=[0, 0.5, 0.75, 0.9],
                simp_fact=self.settings.simp_fact
        )

    def __write_isochrones(self, isochrones):
        isochrone_file = 'isochrones_' + str(c_time) + '.geojson'
        isochrone_path = os.path.join(self.settings.output_folder, isochrone_file)
        save_isochrones(isochrones, isochrone_path, format='geojson')


    def __apply_updates(self, updates, w_speed, w_dir):
        # coordinates of the next updates
        u = np.vstack(updates)
        veg_type = self.veg[u[:, 0], u[:, 1]]
        mask = np.logical_and(
            veg_type != 0,
            f_global[u[:, 0], u[:, 1], u[:, 2]] == 0
        )

        r, c, t = u[mask, 0], u[mask, 1], u[mask, 2]
        f_global[r, c, t] = 1

        nb_num = n_arr.shape[0]
        from_num = r.shape[0]

        nb_arr_r = tile(n_arr[:, 0], from_num)
        nb_arr_c = tile(n_arr[:, 1], from_num)

        nr = r.repeat(nb_num) + nb_arr_r
        nc = c.repeat(nb_num) + nb_arr_c
        nt = t.repeat(nb_num)

        #let's apply a random noise to wind direction and speed for all the cells
        w_dir_r = (w_dir + (pi/16)*(0.5 - rand(from_num))).repeat(nb_num)
        w_speed_r = (w_speed * (1.2 - 0.4 * rand(from_num))).repeat(nb_num)

        dem_from = self.dem[r, c].repeat(nb_num)
        veg_from = self.veg[r, c].repeat(nb_num)
        veg_to = self.veg[nr, nc]
        dem_to = self.dem[nr, nc]
        angle_to = angle[nb_arr_r+1, nb_arr_c+1]
        dist_to = dist[nb_arr_r+1, nb_arr_c+1]

        # exclude all ignited and not valid pixels
        n_mask = np.logical_and(self.f_global[nr, nc, nt] == 0, veg_to != 0)
        dem_from = dem_from[n_mask]
        veg_from = veg_from[n_mask]
        dem_to = dem_to[n_mask]
        veg_to = veg_to[n_mask]
        angle_to = angle_to[n_mask]
        dist_to = dist_to[n_mask]
        w_speed_r = w_speed_r[n_mask]
        w_dir_r = w_dir_r[n_mask]

        nr, nc, nt = nr[n_mask], nc[n_mask], nt[n_mask]

        # get the probability for all the pixels
        p_prob = p_probability(dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moist, w_dir_r, w_speed_r)

        # try the propagation
        p = p_prob > rand(p_prob.shape[0])

        # filter out all not propagated pixels
        p_nr = nr[p]
        p_nc = nc[p]
        p_nt = nt[p]

        # get the propagation time for the propagating pixels
        transition_time = p_time(dem_from[p], dem_to[p],
                                    veg_from[p], veg_to[p],
                                    angle_to[p], dist_to[p],
                                    moist,
                                    w_dir_r[p], w_speed_r[p])

        prop_time = np.around(c_time + transition_time, decimals=1)

        def extract_updates(t):
            idx = np.where(prop_time == t)
            stacked = np.stack((p_nr[idx], p_nc[idx], p_nt[idx]), axis=1)
            return stacked
        
        # schedule the new updates
        unique_time = sorted(np.unique(prop_time))
        new_updates = list(map(
            lambda t: (t, extract_updates(t)), 
        unique_time))
        
        return new_updates


    def run(self):
        isochrones = {}
        self.c_time = 0

        while len(self.ps):
            if self.settings.time_limit and self.c_time > self.settings.time_limit:
                break

            self.c_time, updates = self.ps.pop()

            bc = self.__get_bc()
            
            w_dir_deg = float(bc.get('w_dir', 0))
            wdir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
            wspeed = float(bc.get('w_speed', 0))

            updates = self.__apply_updates(updates, wspeed, wdir)
            self.ps.push_all(updates)

            if c_time % self.settings.time_resolution == 0:
                values = self.__compute_values()
                stats = self.__compute_stats(values)
                n_active, area_mean, area_50, area_75, area_90 = stats
                self.__wite_output(values, 
                    active=n_active, 
                    area_mean=area_mean,
                    area_50=area_50,
                    area_75=area_75,
                    area_90=area_90
                )
                
                self.__update_isochrones(isochrones, values)
                self.__write_isochrones(isochrones, values)
