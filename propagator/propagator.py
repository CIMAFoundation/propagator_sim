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

def get_p_time_fn(ros_model_code):
    ros_models = {
        'default' : p_time_standard,
        'wang' : p_time_wang,
        'rothermel' : p_time_rothermel,
    }
    p_time_function = ros_models.get(ros_model_code, p_time_standard)
    return p_time_function

def p_time_rothermel(dem_from, dem_to, veg_from, veg_to, angle_to, dist, w_dir, w_speed, moist):
    # velocità di base modulata con la densità(tempo di attraversamento)
    dh = (dem_to - dem_from)
    
    v = v0[veg_from-1] / 60 # tempo in minuti di attraversamento di una cella
    
    real_dist = np.sqrt((cellsize*dist)**2 + dh**2)
    
    w_proj = np.cos(w_dir - angle_to) #wind component in propagation direction
    w_spd = (w_speed * w_proj) / 3.6 #wind speed in the direction of propagation

    teta_s_rad = np.arctan(dh / cellsize * dist) #slope angle [rad]
    teta_s = np.degrees(teta_s_rad) #slope angle [°]

    teta_f_rad = np.arctan(0.4226 * w_spd) #flame angle measured from the vertical in the direction of fire spread [rad]
    teta_f = np.degrees(teta_f_rad) #flame angle [°]
    
    sf = np.exp(alpha1 * teta_s) #slope factor
    sf_clip = np.clip(sf , 0.01 , 10) #slope factor clipped at 10
    wf = np.exp(alpha2 * teta_f) #wind factor
    wf_rescaled = wf / 13 #wind factor rescaled to have 10 as max value 
    wf_clip = np.clip(wf_rescaled , 1 , 20) #max value is 20, min is 1

    v_wh_pre = v * sf_clip * wf_clip #Rate of Spread evaluate with Rothermel's model
    v_wh = np.clip(v_wh_pre, 0.01, 100) #adoptable RoS

    t = real_dist / v_wh
    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t

def p_time_wang(dem_from, dem_to, veg_from, veg_to, angle_to, dist, w_dir, w_speed, moist):
  # velocità di base modulata con la densità(tempo di attraversamento)
    dh = (dem_to - dem_from)

    v = v0[veg_from-1] / 60 # tempo in minuti di attraversamento di una cella 
    
    real_dist = np.sqrt((cellsize*dist)**2 + dh**2)
    
    w_proj = np.cos(w_dir - angle_to) #wind component in propagation direction
    w_spd = (w_speed * w_proj)/3.6 #wind speed in the direction of propagation
	
    teta_s_rad = np.arctan(dh / cellsize * dist) #slope angle [rad]
    teta_s_pos = np.absolute(teta_s_rad) #absolute values of slope angle
    p_reverse = np.sign(dh) # +1 if fire spreads upslope, -1 if fire spreads downslope
	
    wf = np.exp(beta1 * w_spd) #wind factor
    wf_clip = np.clip(wf , 0.01 , 10) #clipped at 10
    sf = np.exp(p_reverse * beta2 * np.tan(teta_s_pos)**beta3) #slope factor
    sf_clip = np.clip(sf , 0.01 , 10)

    v_wh_pre = v * wf_clip * sf_clip #Rate of Spread evaluate with Wang Zhengfei's model
    v_wh = np.clip(v_wh_pre, 0.01, 100) #adoptable RoS

    t = real_dist / v_wh
    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t

def p_time_standard(dem_from, dem_to, veg_from, veg_to, angle_to, dist, w_dir, w_speed, moist):
    dh = (dem_to - dem_from)
    v = v0[veg_from-1] / 60
    wh = w_h_effect(angle_to, w_speed, w_dir, dh, dist)
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


class PropagatorError(Exception):
    pass

class NoTilesError(PropagatorError):
    def __init__(self):
        self.message = '''Can't initialize simulation, no data on the selected area'''
        super().__init__(self.message)


class PropagatorConfig:
    pass

class PropagatorSettings:
    def __init__(self, **settings_dict):
        self.n_threads = settings_dict['n_threads']
        self.boundary_conditions = settings_dict['boundary_conditions']
        self.init_date = settings_dict['init_date'] 
        self.tileset = settings_dict['tileset']
        self.grid_dim = settings_dict['grid_dim']
        self.time_resolution = settings_dict['time_resolution']
        self.output_folder = settings_dict['output_folder']
        self.time_limit = settings_dict['time_limit']
        self.p_time_fn = get_p_time_fn(settings_dict['ros_model_code'])
        
        

        #self.simp_fact = settings_dict['simp_fact']
        #self.debug_mode = settings_dict['debug_mode']
        #self.write_vegetation = settings_dict['write_vegetation']
        #self.save_realizations = settings_dict['save_realizations']


class Propagator:
    def __init__(self, settings: PropagatorSettings):
        self.settings = settings
        self.ps = Scheduler()
        self.c_time = 0,
        self.f_global = None
        self.veg = None
        self.dem = None
        self.boundary_conditions = self.settings.boundary_conditions
        self.p_time = settings.p_time_fn
        # make it configurable
        self.dst_crs = crs.CRS({'init': 'EPSG:4326', 'no_defs': True})

    def __init_crs_from_bounds(self, west:float, south:float, east:float, north:float, 
                            cols:int, rows:int, 
                            step_x:float, step_y:float,
                            zone:int, proj:str='utm', datum:str='WGS84'):
        self.__prj = Proj(proj=proj, zone=zone, datum=datum)
        self.__trans = transform.from_bounds(west, south, east, north, cols, rows)
        self.__bounds = (west, south, east, north)
        self.__shape = (rows, cols)
        self.step_x = step_x
        self.step_y = step_y

    def init_ignitions(self, polys, lines, points, zone_number):
        west, south, east, north = self.__bounds
        
        img, active_ignitions = \
            rasterize_ignitions((self.__shape[0], self.__shape[1]),
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number)
        self.__init_simulation(self.settings.n_threads, img, active_ignitions)

    def __compute_values(self):
        values = np.nanmean(self.f_global, 2)
        return values

    def __compute_stats(self, values):
        n_active = len(self.ps.active().tolist())
        cell_area = float(self.step_x) * float(self.step_y) / 10000.0
        area_mean = float(np.sum(values) * cell_area)
        area_50 = float(np.sum(values >= 0.5) * cell_area)
        area_75 = float(np.sum(values >= 0.75) * cell_area)
        area_90 = float(np.sum(values >= 0.90) * cell_area)

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


    def __write_output(self, values, dst_trans, **kwargs):
        filename = os.path.join(self.settings.output_folder, str(self.c_time))
        tiff_file = filename + '.tiff'
        json_file = filename + '.json'

        ref_date = str(self.settings.init_date + timedelta(minutes=self.c_time))
        with open(json_file, 'w') as fp:
            meta = dict(time=self.c_time, timeref=ref_date)
            meta.update(kwargs)
            json.dump(meta, fp)

            write_geotiff(tiff_file, values*255, dst_trans, self.dst_crs)

    def __check_input_files_consistency(self, dem_file, veg_file):
        if dem_file.crs != veg_file.crs:
            raise Exception(f'CRS of input files are inconsistent')

        err_res = abs(dem_file.res[0] - veg_file.res[0])/veg_file.res[0]
        if err_res >  0.01:
            raise Exception(f'Resolution of input files are not consistent')

        bounds_err = np.array([
            dem_file.bounds.left - veg_file.bounds.left,
            dem_file.bounds.right - veg_file.bounds.right,
            dem_file.bounds.top - veg_file.bounds.top,
            dem_file.bounds.bottom - veg_file.bounds.bottom
        ])
        if np.linalg.norm(bounds_err,1) > veg_file.res[0]*2:
            raise Exception(f'Bounding box of input files are not consistent')

    def load_data_from_files(self, veg_filename, dem_filename):
            with rio.open(veg_filename) as veg_file, rio.open(dem_filename) as dem_file:
                self.__check_input_files_consistency(dem_file, veg_file)
                try:
                    self.dem = dem_file.read(1).astype('int16')
                    self.veg = veg_file.read(1).astype('int8')
                    
                    self.veg[:, (0, -1)] = 0
                    self.veg[(0, -1), :] = 0
                    self.veg[(self.veg<0)|(self.veg>6)] = 0                    
                    
                    transform, crs, bounds, res = veg_file.transform, veg_file.crs, veg_file.bounds, veg_file.res

                    self.__prj = Proj(crs.to_wkt())
                    self.__trans = transform
                    self.__bounds = bounds
                    self.__shape = self.veg.shape
                    self.step_x = res[0]
                    self.step_y = res[1]
    
                    self.moist = np.zeros_like(self.veg, dtype='float')

                except IOError:
                    logging.error('Error reading input files')
                    raise

    def load_data_from_tiles(self, easting, northing, zone_number):
        try:
            logging.info('Loading VEGETATION from "' + self.settings.tileset + '" tileset')
            veg, west, north, step_x, step_y = \
                load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'prop', self.settings.tileset)
            veg[:, (0, -1)] = 0
            veg[(0, -1), :] = 0
            self.veg = veg.astype('int8')
            
            logging.info('Loading DEM "default" tileset')
            dem, west, north, step_x, step_y = \
                load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'quo', 'default')
            self.dem = dem.astype('float')

            self.moist = np.zeros_like(veg, dtype='float')
            rows, cols = veg.shape
            south = north - (rows * step_y)
            east = west + (cols * step_x)
            self.__init_crs_from_bounds(west, south, east, north, cols, rows, step_x, step_y, zone_number)
        except FileNotFoundError:
            raise NoTilesError()

    def __find_bc(self):
        last_bc = None
        for bc in self.boundary_conditions:
            if self.c_time >= bc['time']:
                last_bc = bc
        return last_bc

    def __init_simulation(self, n_threads, initial_ignitions, active_ignitions):
        self.f_global = np.zeros(self.__shape + (n_threads,))
        for t in range(n_threads):
            self.f_global[:, :, t] = initial_ignitions.copy()
            for p in active_ignitions:
                self.ps.push(array([p[0], p[1], t]), 0)
                self.f_global[p[0], p[1], t] = 0

    def __update_isochrones(self, isochrones, values, dst_trans):
        isochrones[self.c_time] = extract_isochrone(
                values, dst_trans,
                thresholds=[0, 0.5, 0.75, 0.9],
        )

    def __write_isochrones(self, isochrones):
        isochrone_file = 'isochrones_' + str(self.c_time) + '.geojson'
        isochrone_path = os.path.join(self.settings.output_folder, isochrone_file)
        save_isochrones(isochrones, isochrone_path, format='geojson')


    def __apply_updates(self, updates, w_speed, w_dir):
        # coordinates of the next updates
        u = np.vstack(updates)
        veg_type = self.veg[u[:, 0], u[:, 1]]
        mask = np.logical_and(
            veg_type != 0,
            self.f_global[u[:, 0], u[:, 1], u[:, 2]] == 0
        )

        r, c, t = u[mask, 0], u[mask, 1], u[mask, 2]
        self.f_global[r, c, t] = 1

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
        moist_to = self.moist[nr, nc]

        angle_to = angle[nb_arr_r+1, nb_arr_c+1]
        dist_to = dist[nb_arr_r+1, nb_arr_c+1]

        # exclude all ignited and not valid pixels
        n_mask = np.logical_and(self.f_global[nr, nc, nt] == 0, veg_to != 0)
        dem_from = dem_from[n_mask]
        veg_from = veg_from[n_mask]
        dem_to = dem_to[n_mask]
        veg_to = veg_to[n_mask]
        moist_to = moist_to[n_mask]
        angle_to = angle_to[n_mask]
        dist_to = dist_to[n_mask]
        w_speed_r = w_speed_r[n_mask]
        w_dir_r = w_dir_r[n_mask]
        

        nr, nc, nt = nr[n_mask], nc[n_mask], nt[n_mask]

        # get the probability for all the pixels
        p_prob = p_probability(dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moist_to, w_dir_r, w_speed_r)

        # try the propagation
        p = p_prob > rand(p_prob.shape[0])

        # filter out all not propagated pixels
        p_nr = nr[p]
        p_nc = nc[p]
        p_nt = nt[p]

        # get the propagation time for the propagating pixels
        transition_time = self.p_time(dem_from[p], dem_to[p],
                                    veg_from[p], veg_to[p],
                                    angle_to[p], dist_to[p],
                                    moist_to[p],
                                    w_dir_r[p], w_speed_r[p])

        prop_time = np.around(self.c_time + transition_time, decimals=1)

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

    def load_ignitions_from_string(self, ignition_string):
        mid_lat, mid_lon, polys, lines, points = read_ignition(ignition_string)
        easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)
        return easting, northing, zone_number, zone_letter, polys, lines, points
        

    def run(self):
        isochrones = {}
        self.c_time = 0

        while len(self.ps):
            if self.settings.time_limit and self.c_time > self.settings.time_limit:
                break

            self.c_time, updates = self.ps.pop()
            bc = self.__find_bc()
            w_dir_deg = float(bc.get('w_dir', 0))
            wdir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
            wspeed = float(bc.get('w_speed', 0))

            new_updates = self.__apply_updates(updates, wspeed, wdir)
            self.ps.push_all(new_updates)
            

            if self.c_time % self.settings.time_resolution == 0:
                values = self.__compute_values()
                stats = self.__compute_stats(values)
                n_active, area_mean, area_50, area_75, area_90 = stats
                self.log(n_active, area_mean)

                reprj_values, dst_trans = reproject(
                    values,
                    self.__trans,
                    self.__prj.crs.srs,
                    self.dst_crs
                )

                self.__write_output(
                    reprj_values,
                    dst_trans,
                    active=n_active,
                    area_mean=area_mean,
                    area_50=area_50,
                    area_75=area_75,
                    area_90=area_90
                )
                
                self.__update_isochrones(isochrones, reprj_values, dst_trans)
                self.__write_isochrones(isochrones)
