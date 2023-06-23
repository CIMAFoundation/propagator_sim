import json
import logging
import os
from datetime import timedelta

# import utm
from numpy import array, pi, sign, tanh, tile
from numpy.random import rand

from pyproj import Proj
from rasterio import crs, enums, transform, warp

from scipy import ndimage

from .constants import *
from .utils import *

from . import PROPAGATOR_PATH

# [latifoglie cespugli aree_nude erba conifere coltivi faggete]
try:   
    v0 = np.loadtxt(os.path.join(PROPAGATOR_PATH, 'v0_table.txt'))
    prob_table = np.loadtxt(os.path.join(PROPAGATOR_PATH, 'prob_table.txt'))
    p_veg = np.loadtxt(os.path.join(PROPAGATOR_PATH, 'p_vegetation.txt'))
except Exception:
    raise Exception('Could not load the vegetation speed and probabilities tables')


def load_parameters(probability_file=None, v0_file=None, p_vegetation=None):
    """
    Override the default values for vegetation speed and probabilities by loading them from file
    :param probability_file:
    :param time_file:
    :return:
    """
    global v0, prob_table, p_veg
    if v0_file:
        v0 = np.loadtxt(v0_file)
    if probability_file:
        prob_table = np.loadtxt(probability_file)
    if p_vegetation:
        p_veg = np.loadtxt(p_vegetation)

def get_p_time_fn(ros_model_code):
    ros_models = {
        DEFAULT_TAG : p_time_standard,
        WANG_TAG : p_time_wang,
        ROTHERMEL_TAG : p_time_rothermel,
    }
    p_time_function = ros_models.get(ros_model_code, p_time_wang)
    return p_time_function

def get_p_moist_fn(moist_model_code):
    moist_models = {
        DEFAULT_TAG : moist_proba_correction_1,
        NEW_FORMULATION_TAG : moist_proba_correction_1,
        ROTHERMEL_TAG : moist_proba_correction_2,
    }
    p_moist_function = moist_models.get(moist_model_code, moist_proba_correction_1)
    return p_moist_function

def p_time_rothermel(dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed):
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
    moist_eff = np.exp(c_moist * moist) #moisture effect

    #v_wh = np.clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = np.clip(v_wh_pre * moist_eff, 0.01, 100) #adoptable RoS [m/min]

    t = real_dist / v_wh
    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t , v_wh
    #return t

def p_time_wang(dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed):
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
    moist_eff = np.exp(c_moist * moist) #moisture effect
     
    #v_wh = np.clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = np.clip(v_wh_pre * moist_eff, 0.01, 100) #adoptable RoS [m/min]
    
    t = real_dist / v_wh

    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t , v_wh

def p_time_standard(dem_from, dem_to, veg_from, veg_to, angle_to, dist, moist, w_dir, w_speed):
    dh = (dem_to - dem_from)
    v = v0[veg_from-1] / 60
    wh = w_h_effect(angle_to, w_speed, w_dir, dh, dist)
    moist_eff = np.exp(c_moist * moist) #moisture effect
    #v_wh = np.clip(v * wh, 0.01, 100)
    v_wh = np.clip(v * wh * moist_eff, 0.01, 100)

    real_dist = np.sqrt((cellsize*dist)**2 + dh**2)
    t = real_dist / v_wh
    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t , v_wh

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


def p_probability(self, dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moist, w_dir, w_speed):
    dh = (dem_to - dem_from)
    alpha_wh = w_h_effect_on_p(angle_to, w_speed, w_dir, dh, dist_to)
    
    #p_moist = 1
    p_moist = self.p_moist(moist)  #era self.p_moist
    #p_moist = M1 * moist**3 + M2 * moist**2 + M3 * moist + M4
    p_m = np.clip(p_moist , 0, 1.0)
    p_veg = prob_table[veg_to - 1, veg_from - 1]
    p = 1-(1-p_veg)**alpha_wh
    #p_clip = np.clip(p, 0, 1.0)
    p_clip = np.clip(p * p_m, 0, 1.0)

    return p_clip


def moist_proba_correction_1(moist):
    """ 
    e_m is the moinsture correction to the transition probability p_{i,j}.  
    e_m = f(m), with m the Fine Fuel Moisture Content
    e_m = -11,507x5 + 22,963x4 - 17,331x3 + 6,598x2 - 1,7211x + 1,0003, where x is moisture / moisture of extintion (Mx).  
    Mx = 0.3 
    (reference: Trucchia et al, Fire 2020 )
    """
    Mx = 0.3
    x = np.clip(moist, 0.0, 1.0) / Mx
    p_moist = (-11.507  * x**5) + (22.963 * x**4) + (-17.331 * x**3) + (6.598 * x**2) + (-1.7211 * x) + 1.0003
    p_moist = np.clip(p_moist, 0.0, 1.0)
    return p_moist
    
def moist_proba_correction_2(moist):
    """
    e_m is the moinsture correction to the transition probability p_{i,j}. e_m = f(m), with m the Fine Fuel Moisture Content
    Old formulation by Baghino, adopted in Trucchia et al, Fire 2020.
    Here, the parameters come straight from constants.py.
    """
    p_moist = M1 * moist**3 + M2 * moist**2 + M3 * moist + M4
    return p_moist


def fire_spotting(angle_to, w_dir, w_speed): 
    """this function evaluates the distance that an ember can reach, by the use of the Alexandridis' formulation"""
    r_n = np.random.normal( spotting_rn_mean , spotting_rn_std , size=angle_to.shape ) # main thrust of the ember: sampled from a Gaussian Distribution (Alexandridis et al, 2008 and 2011)
    w_speed_ms = w_speed / 3.6                  #wind speed [m/s]
    d_p = r_n * np.exp( w_speed_ms * c_2 *( np.cos( w_dir - angle_to ) - 1 ) )  #Alexandridis' formulation for spotting distance
    return d_p

# functions useful for evaluating the fire line intensity
def lhv_dead_fuel(hhv, dffm):
    lhv = hhv * (1.0 - (dffm / 100.0)) - Q * (dffm / 100.0)
    return lhv


def lhv_canopy(hhv, hum):
    lhv = hhv * (1.0 - (hum / 100.0)) - Q * (hum / 100.0)
    lhv[np.isnan(lhv)] = 0
    return lhv

  
def fireline_intensity(d0, d1, ros, lhv_dead_fuel, lhv_canopy, rg=None):
    intensity = np.full(ros.shape[0], np.nan, dtype='float32')
    if rg is not None:
        rg_idx = ~np.isnan(rg)
        d1_idx = (d1 != 0.0) & rg_idx
        d0_idx = (d1 == 0.0) & rg_idx
        intensity[d0_idx] = ros[d0_idx] * (( lhv_dead_fuel[d0_idx] * d0[d0_idx] * (1.0 - rg[d0_idx])) / 2 ) / 60.0
        intensity[d1_idx] = ros[d1_idx] * (( lhv_dead_fuel[d1_idx] * d0[d1_idx] + lhv_canopy[d1_idx] * (d1[d1_idx] * (1 - rg[d1_idx]))) / 2 ) / 60.0
        intensity[~rg_idx] = ros[~rg_idx] * (( lhv_dead_fuel[~rg_idx] * d0[~rg_idx] + lhv_canopy[~rg_idx] * d1[~rg_idx]) / 2 ) / 60.0
    else:
        intensity = ros * (( lhv_dead_fuel * d0 + lhv_canopy * d1 ) ) / 60.0 #divided by 60 instead of 3600 because RoS is required in m/s and it is given in m/min (so it has to be divided by 60)
    return intensity


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
        self.n_threads = settings_dict[N_THREADS_TAG]
        self.boundary_conditions = settings_dict[BOUNDARY_CONDITIONS_TAG]
        self.init_date = settings_dict[INIT_DATE_TAG] 
        self.tileset = settings_dict[TILESET_TAG]
        self.grid_dim = settings_dict[GRID_DIM_TAG]
        self.time_resolution = settings_dict[TIME_RESOLUTION_TAG]
        self.output_folder = settings_dict[OUTPUT_FOLDER_TAG]
        self.time_limit = settings_dict[TIME_LIMIT_TAG]
        self.p_time_fn = get_p_time_fn(settings_dict[ROS_MODEL_CODE_TAG])
        self.p_moist_fn = get_p_moist_fn(settings_dict[PROB_MOIST_CODE_TAG])
        self.do_spotting = settings_dict[SPOT_FLAG_TAG]

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
        self.p_moist = settings.p_moist_fn #print("p_moist is ...", self.p_moist)
        self.ros = None
        self.fireline_int = None
        # make it configurable
        self.dst_crs = crs.CRS({'init': 'EPSG:4326', 'no_defs': True})

    def __preprocess_bc(self, boundary_conditions):
        for bc in boundary_conditions:
            self.__rasterize_moisture_fighting_actions(bc)
            self.__rasterize_newignitions(bc)

    
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
            rasterize_actions((self.__shape[0], self.__shape[1]), 
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number)
        self.__preprocess_bc(self.settings.boundary_conditions)
        self.__init_simulation(self.settings.n_threads, img, active_ignitions)
        
    def __compute_values(self):
        values = np.nanmean(self.f_global, 2)
        return values
    
    def __compute_RoS_max(self):
        RoS_max = np.nanmax(self.ros, 2)
        return RoS_max
    
    def __compute_RoS_mean(self):
        RoS_m = np.where(self.ros>0, self.ros, np.NaN)
        RoS_mean = np.nanmean(RoS_m, 2)
        RoS_mean = np.where(RoS_mean > 0, RoS_mean, 0)
        return RoS_mean
    
    def __compute_fireline_int_max(self):
        fl_I_max = np.nanmax(self.fireline_int, 2)
        return fl_I_max
    
    def __compute_fireline_int_mean(self):
        fl_I_m = np.where(self.fireline_int>0, self.fireline_int, np.NaN)
        fl_I_mean = np.nanmean(fl_I_m, 2)
        fl_I_mean = np.where(fl_I_mean > 0, fl_I_mean, 0)
        return fl_I_mean

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
    
    
    def __write_output_RoS_max(self, values, dst_trans, **kwargs):
        filename = os.path.join(self.settings.output_folder, 'RoS_max_' + str(self.c_time))
        tiff_file = filename + '.tiff'
        json_file = filename + '.json'

        ref_date = str(self.settings.init_date + timedelta(minutes=self.c_time))
        with open(json_file, 'w') as fp:
            meta = dict(time=self.c_time, timeref=ref_date)
            meta.update(kwargs)
            json.dump(meta, fp)

            write_geotiff(tiff_file, values * 60 , dst_trans, self.dst_crs, values.dtype) #now it returns the RoS in m/h
   
    def __write_output_RoS_mean(self, values, dst_trans, **kwargs):
        filename = os.path.join(self.settings.output_folder, 'RoS_mean_' + str(self.c_time))
        tiff_file = filename + '.tiff'
        json_file = filename + '.json'

        ref_date = str(self.settings.init_date + timedelta(minutes=self.c_time))
        with open(json_file, 'w') as fp:
            meta = dict(time=self.c_time, timeref=ref_date)
            meta.update(kwargs)
            json.dump(meta, fp)

            write_geotiff(tiff_file, values * 60 , dst_trans, self.dst_crs, values.dtype) #now it returns the RoS in m/h

    def __write_output_I_max(self, values, dst_trans, **kwargs):
        filename = os.path.join(self.settings.output_folder, 'fireline_intensity_max_' + str(self.c_time))
        tiff_file = filename + '.tiff'
        json_file = filename + '.json'

        ref_date = str(self.settings.init_date + timedelta(minutes=self.c_time))
        with open(json_file, 'w') as fp:
            meta = dict(time=self.c_time, timeref=ref_date)
            meta.update(kwargs)
            json.dump(meta, fp)

            write_geotiff(tiff_file, values , dst_trans, self.dst_crs, values.dtype)

    def __write_output_I_mean(self, values, dst_trans, **kwargs):
        filename = os.path.join(self.settings.output_folder, 'fireline_intensity_mean_' + str(self.c_time))
        tiff_file = filename + '.tiff'
        json_file = filename + '.json'

        ref_date = str(self.settings.init_date + timedelta(minutes=self.c_time))
        with open(json_file, 'w') as fp:
            meta = dict(time=self.c_time, timeref=ref_date)
            meta.update(kwargs)
            json.dump(meta, fp)

            write_geotiff(tiff_file, values , dst_trans, self.dst_crs, values.dtype)
    

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
                    
                    self.veg[:, (0, 1, 2, -3, -2, -1)] = 0
                    self.veg[(0, 1, 2, -3, -2, -1), :] = 0
                    self.veg[(self.veg<0)|(self.veg>6)] = 0                    
                    
                    transform, crs, bounds, res = veg_file.transform, veg_file.crs, veg_file.bounds, veg_file.res
                    #self.__prj = Proj(crs.to_wkt()) this was not working anymore
                    self.__prj = Proj(crs.to_proj4()) #it works for pyproj 2.6.1.post1 and rasterio 1.1.7
                    self.__trans = transform
                    self.__bounds = bounds
                    self.__shape = self.veg.shape
                    self.step_x = res[0]
                    self.step_y = res[1]
    
                except IOError:
                    logging.error('Error reading input files')
                    raise

    def load_data_from_tiles(self, easting, northing, zone_number):
        try:
            tileset = self.settings.tileset
            try: 
                logging.info('Loading VEGETATION from "' + tileset + '" tileset')
                veg, west, north, step_x, step_y = \
                    load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'prop', tileset)
            except:
                logging.info('Loading VEGETATION from "' + DEFAULT_TAG + '" tileset')
                veg, west, north, step_x, step_y = \
                    load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'prop', self.settings.tileset)
                
            veg[:, (0, 1, 2, -3, -2, -1)] = 0
            veg[(0, 1, 2, -3, -2, -1), :] = 0
            self.veg = veg.astype('int8')
            
            try: 
                logging.info('Loading DEM from "' + tileset + '" tileset')
                dem, west, north, step_x, step_y = \
                    load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'quo', tileset)
                self.dem = dem.astype('float')
            except:
                logging.info('Loading DEM from "' + DEFAULT_TAG + '" tileset')
                dem, west, north, step_x, step_y = \
                    load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'quo', DEFAULT_TAG)
                self.dem = dem.astype('float')

            rows, cols = veg.shape
            south = north - (rows * step_y)
            east = west + (cols * step_x)
            self.__init_crs_from_bounds(west, south, east, north, cols, rows, step_x, step_y, zone_number)
        except FileNotFoundError:
            raise NoTilesError()

    def __find_bc(self):
        last_bc = None
        for bc in self.boundary_conditions:
            if self.c_time >= bc[TIME_TAG]:
                #n_bc = -1
                last_bc = bc
                #n_bc +=1
        return last_bc #, n_bc

    def __init_simulation(self, n_threads, initial_ignitions, active_ignitions):
        self.f_global = np.zeros(self.__shape + (n_threads,))
        self.ros = np.zeros(self.__shape + (n_threads,))
        self.fireline_int = np.zeros(self.__shape + (n_threads,))
        for t in range(n_threads):
            self.f_global[:, :, t] = initial_ignitions.copy()
            for p in active_ignitions:
                self.ps.push(array([p[0], p[1], t]), 0)
                self.f_global[p[0], p[1], t] = 0
            
            # add ignitions in future boundary conditions
            for conditions in self.boundary_conditions:
                if IGNITIONS_RASTER_TAG in conditions:
                    ignition_bc = conditions[IGNITIONS_RASTER_TAG]
                    time_bc = conditions[TIME_TAG]
                    if ignition_bc is not None: 
                        for ign in ignition_bc:
                            self.ps.push(array([[ ign[0], ign[1], t]]), time_bc)
                        


    def __update_isochrones(self, isochrones, values, dst_trans):
        isochrones[self.c_time] = extract_isochrone(
                values, dst_trans,
                thresholds=[0, 0.5, 0.75, 0.9],
        )

    def __write_isochrones(self, isochrones):
        isochrone_file = 'isochrones_' + str(self.c_time) + '.geojson'
        isochrone_path = os.path.join(self.settings.output_folder, isochrone_file)
        save_isochrones(isochrones, isochrone_path, format='geojson')


    def __apply_updates(self, updates, w_speed, w_dir, moisture):
        # coordinates of the next updates
        bc = self.__find_bc()
        u = np.vstack(updates)
        veg_type = self.veg[u[:, 0], u[:, 1]]
        mask = np.logical_and(
            veg_type != 0,
            self.f_global[u[:, 0], u[:, 1], u[:, 2]] == 0
        )

        r, c, t = u[mask, 0], u[mask, 1], u[mask, 2]
        self.f_global[r, c, t] = 1

        #veg type modified due to the heavy fighting actions
        heavy_acts = bc.get(HEAVY_ACTION_RASTER_TAG , None)
        if heavy_acts:
            for heavyy in heavy_acts:
                self.veg[ heavyy[0] , heavyy[1] ] = 0 #da scegliere se mettere a 0 (impossibile che propaghi) 3 (non veg, quindi prova a propagare ma non riesce) o 7(faggete, quindi propaga con bassissima probabilità)
        
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
        moisture_r = moisture[nr, nc]
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
        moisture_r = moisture_r[n_mask]
        

        nr, nc, nt = nr[n_mask], nc[n_mask], nt[n_mask]

        # get the probability for all the pixels
        p_prob = p_probability(self,dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moisture_r, w_dir_r, w_speed_r)

        # try the propagation
        p = p_prob > rand(p_prob.shape[0])

        # filter out all not propagated pixels
        p_nr = nr[p]
        p_nc = nc[p]
        p_nt = nt[p]

        # get the propagation time for the propagating pixels
        #transition_time = self.p_time(dem_from[p], dem_to[p],
        transition_time , ros = self.p_time(dem_from[p], dem_to[p],
                                    veg_from[p], veg_to[p],
                                    angle_to[p], dist_to[p],
                                    moisture_r[p],
                                    w_dir_r[p], w_speed_r[p])

        d0 = p_veg[veg_to[p]-1,0]
        d1 = p_veg[veg_to[p]-1,1]
        hhv = p_veg[veg_to[p]-1,2]
        humidity = p_veg[veg_to[p]-1,3]

        # evaluate LHV of dead fuel
        lhv_dead_fuel_value = lhv_dead_fuel(hhv, moisture_r[p])
        # evaluate LHV of the canopy
        lhv_canopy_value = lhv_canopy(hhv, humidity)
        # evaluate fireline intensity
        fireline_intensity_value = fireline_intensity(d0, d1, ros, lhv_dead_fuel_value, lhv_canopy_value)

        self.fireline_int[ p_nr , p_nc , p_nt ] = fireline_intensity_value

        self.ros[ p_nr , p_nc , p_nt ] = ros

        ###### fire spotting    ----> FROM HERE
        ##################################################
        if self.settings.do_spotting == True:
            #print("I will do spotting!")
            conifer_mask = (veg_type == 5)      #only cells that have veg = fire-prone conifers are selected
            conifer_r , conifer_c , conifer_t = u[conifer_mask, 0], u[conifer_mask, 1], u[conifer_mask, 2] 
            
            #N_spotters = conifer_r.shape[0]    #number of  fire-prone conifers cells that are  burning
            
            #calculate number of embers per emitter
            N_embers = np.random.poisson( lambda_spotting , size=conifer_r.shape )

            # create list of source points for each ember
            conifer_arr_r = conifer_r.repeat(repeats=N_embers)
            conifer_arr_c = conifer_c.repeat(repeats=N_embers)
            conifer_arr_t = conifer_t.repeat(repeats=N_embers)
            # calculate angle and distance
            ember_angle = np.random.uniform(0 , 2.0*np.pi, size=conifer_arr_r.shape)
            ember_distance  = fire_spotting(ember_angle,  w_dir, w_speed)  

            # filter out short embers
            idx_long_embers = ember_distance > 2*cellsize
            conifer_arr_r = conifer_arr_r[idx_long_embers]
            conifer_arr_c = conifer_arr_c[idx_long_embers]
            conifer_arr_t = conifer_arr_t[idx_long_embers]
            ember_angle = ember_angle[idx_long_embers]
            ember_distance = ember_distance[idx_long_embers]


            # calculate landing locations
            delta_r = ember_distance * np.cos(ember_angle)  #vertical delta [meters]
            delta_c = ember_distance * np.sin(ember_angle)  #horizontal delta [meters]
            nb_spot_r = delta_r / cellsize     #number of vertical cells
            nb_spot_r = nb_spot_r.astype(int)
            nb_spot_c = delta_c / cellsize     #number of horizontal cells
            nb_spot_c = nb_spot_c.astype(int) 

            nr_spot = conifer_arr_r + nb_spot_r         #vertical location of the cell to be ignited by the ember
            nc_spot = conifer_arr_c + nb_spot_c         #horizontal location of the cell to be ignited by the ember
            nt_spot = conifer_arr_t

            #if I surpass the bounds, I stick to them. This way I don't have to reshape anything.
            nr_spot[nr_spot > self.__shape[0] -1 ] = self.__shape[0] -1 
            nc_spot[nc_spot > self.__shape[1] -1 ] = self.__shape[1] -1 
            nr_spot[nr_spot < 0 ] = 0
            nc_spot[nc_spot < 0 ] = 0           



            # we want to put another probabilistic filter in order to assess the success of ember ignition. 
            # 
            #Formula (10) of Alexandridis et al IJWLF 2011
            # P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition by spotting and P_cd is a correction factor that 
            #depends on vegetation type and density...
            # In this case, we put P_cd = 0.3 for conifers and 0 for the rest. but it can be generalized..

            P_c = P_c0 * (1+ P_cd_conifer*(self.veg[nr_spot,nc_spot] == 5)) # + 0.4 * bushes_mask.... etc etc 

            success_spot_mask = np.random.uniform(  size=P_c.shape ) <  P_c 
            nr_spot = nr_spot[success_spot_mask]
            nc_spot = nc_spot[success_spot_mask]
            nt_spot = nt_spot[success_spot_mask]        
            # A little more debug on the previous part is advised

            #the following function evalates the time that the embers  will need to burn the entire cell  they land into
            #transition_time_spot = self.p_time(self.dem[nr_spot, nc_spot], self.dem[nr_spot, nc_spot], #evaluation of the propagation time of the "spotted cells"
            transition_time_spot , ros_spot = self.p_time(self.dem[nr_spot, nc_spot], self.dem[nr_spot, nc_spot], #evaluation of the propagation time of the "spotted cells"
                            self.veg[nr_spot, nc_spot], self.veg[nr_spot, nc_spot],   #dh=0 (no slope) and veg_from=veg_to to simplify the phenomenon
                            np.zeros(nr_spot.shape), cellsize*np.ones(nr_spot.shape), #ember_angle, ember_distance, 
                            moisture[nr_spot, nc_spot],
                            w_dir, w_speed)
            
            p_nr = np.append( p_nr , nr_spot)               #row-coordinates of the "spotted cells" added to the other ones
            p_nc = np.append( p_nc , nc_spot)               #column-coordinates of the "spotted cells" added to the other ones
            p_nt = np.append( p_nt , nt_spot)               #time propagation of "spotted cells" added to the other ones 
            transition_time = np.append( transition_time , np.around( transition_time_spot ) )
        #else:
        #    print("I am not going to spot...")

        ######################################
        ##################### UP TO HERE  <------

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
        mid_lat, mid_lon, polys, lines, points = read_actions(ignition_string)
        easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)
        return easting, northing, zone_number, zone_letter, polys, lines, points
    

    def __rasterize_moisture_fighting_actions(self, bc):
        west, south, east, north = self.__bounds
        waterline_actionss = bc.get(WATERLINE_ACTION_TAG, None)
        moisture_value = bc.get(MOISTURE_TAG, 0)/100
        heavy_actionss = bc.get(HEAVY_ACTION_TAG, None)
        canadairs = bc.get(CANADAIR_TAG, None)          # select canadair actions from boundary conditions
        helicopters = bc.get(HELICOPTER_TAG, None)      # select helicopter actions from boundary conditions

        if waterline_actionss:
            waterline_action_string = '\n'.join(waterline_actionss)
            mid_lat, mid_lon, polys, lines, points = read_actions(waterline_action_string)
            easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

            img, waterline_points = \
                rasterize_actions((self.__shape[0], self.__shape[1]), 
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number, base_value=moisture_value)
            
            mask = (img==1)
            img_mask = ndimage.binary_dilation(mask)
            img[img_mask] = WATERLINE_ACTION_VALUE
        else:
            img = np.ones((self.__shape[0], self.__shape[1])) * moisture_value
        
        ####canadair actions
        if canadairs:
            canadairs_string = '\n'.join(canadairs)
            mid_lat, mid_lon, polys, lines, points = read_actions(canadairs_string)
            easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

            if len(polys)!=0 or len(points)!=0 :
                raise Exception(f'ERROR: Canadair actions must be lines')

            img_can, canadair_points = \
                rasterize_actions((self.__shape[0], self.__shape[1]), 
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number, base_value=moisture_value)
            
            mask_can = (img_can==1)                                 #select points that are directly interested by canadair actions
            img_can_mask = ndimage.binary_dilation(mask_can)        #create a 1 pixel buffer around the selected points
            img[img_can_mask] = CANADAIR_BUFFER_VALUE               #moisture value of the points of the buffer
            img[mask_can] = CANADAIR_VALUE                    #moisture value of the points directly interested by the canadair actions
        
        ####helicopter actions
        if helicopters:
            helicopters_string = '\n'.join(helicopters)
            mid_lat, mid_lon, polys, lines, points = read_actions(helicopters_string)
            easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

            ##if len(polys)!=0 or len(lines)!=0 :
            ##    raise Exception(f'ERROR: Helicopter actions must be points')
            
            img_heli, helicopter_points = \
                rasterize_actions((self.__shape[0], self.__shape[1]), 
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number, base_value=moisture_value)
          
            new_heli_point = []
            for ep in helicopter_points:                #create a randomness in the points where the helicopter acts
                new_x = ep[0] - 1 + round(2*np.random.uniform())
                new_y = ep[1] - 1 + round(2*np.random.uniform())
                new_point = add_point(img_heli, new_y, new_x, 0.6)
                new_heli_point.extend(new_point)

            mask_newheli = (img_heli==0.6)                                  #select points that are directly interested by helicopter actions
            img_new_heli_mask = ndimage.binary_dilation(mask_newheli)       #create a 1 pixel buffer around the selected points
            img[img_new_heli_mask] = HELICOPTER_BUFFER_VALUE # moisture value of the points of the buffer
            img[mask_newheli] = HELICOPTER_VALUE        #moisture value of the points directly interested by the helicopter actions
        
        bc[MOIST_RASTER_TAG] = img 

        if heavy_actionss:
            heavy_action_string = '\n'.join(heavy_actionss)
            mid_lat, mid_lon, polys, lines, points = read_actions(heavy_action_string)
            easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

            image, heavy_action_points = \
                rasterize_actions((self.__shape[0], self.__shape[1]), 
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number)

            new_mask = ( image == 1 )
            new_mask_dilated = ndimage.binary_dilation( new_mask )
            heavy_points = np.where( new_mask_dilated == True )
            heavy_action_points_enlarged = []
            for i in range(len(heavy_points[0])):
                heavies = add_point(new_mask_dilated, heavy_points[1][i], heavy_points[0][i], 1)
                heavy_action_points_enlarged.extend(heavies)

            bc[HEAVY_ACTION_RASTER_TAG] = heavy_action_points_enlarged


    def __rasterize_newignitions(self, bc):
            west, south, east, north = self.__bounds
            new_ignitions = bc.get(IGNITIONS_TAG, None)
            
            if new_ignitions:
                new_ignitions_string = '\n'.join(new_ignitions)
                mid_lat, mid_lon, polys, lines, points = read_actions(new_ignitions_string)
                easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

                img, ignition_pixels = \
                    rasterize_actions((self.__shape[0], self.__shape[1]), 
                                points, lines, polys, west, north, self.step_x, self.step_y, zone_number)

                bc[IGNITIONS_RASTER_TAG] = ignition_pixels
            

    def run(self):
        isochrones = {}
        self.c_time = 0

        while len(self.ps):
            if self.settings.time_limit and self.c_time > self.settings.time_limit:
                break

            bc = self.__find_bc()
            w_dir_deg = float(bc.get(W_DIR_TAG, 0))
            wdir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
            wspeed = float(bc.get(W_SPEED_TAG, 0))
            
            moisture = bc.get(MOIST_RASTER_TAG, None)

            newignitions = bc.get(IGNITIONS_RASTER_TAG, None)

            self.c_time, updates = self.ps.pop()
            
            new_updates = self.__apply_updates(updates, wspeed, wdir, moisture)
            self.ps.push_all(new_updates)
            

            if self.c_time % self.settings.time_resolution == 0:
                values = self.__compute_values()
                RoS_max = self.__compute_RoS_max()
                RoS_mean = self.__compute_RoS_mean()
                fl_I_max = self.__compute_fireline_int_max()
                fl_I_mean = self.__compute_fireline_int_mean()
                stats = self.__compute_stats(values)
                n_active, area_mean, area_50, area_75, area_90 = stats
                self.log(n_active, area_mean)

                reprj_values, dst_trans = reproject(
                    values,
                    self.__trans,
                    self.__prj.srs,  #self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
                    self.dst_crs,
                    trim = True #trim is used to auto-clip the tif where there are null value
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

                reprj_values_ros_max, dst_trans_ros_max = reproject(
                    RoS_max,
                    self.__trans,
                    self.__prj.srs,  #self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
                    self.dst_crs,
                    trim = True #trim is used to auto-clip the tif where there are null value
                )

                self.__write_output_RoS_max(
                    reprj_values_ros_max,
                    dst_trans_ros_max,
                    active=n_active,
                    area_mean=area_mean,
                    area_50=area_50,
                    area_75=area_75,
                    area_90=area_90
                )

                reprj_values_ros_mean, dst_trans_ros_mean = reproject(
                    RoS_mean,
                    self.__trans,
                    self.__prj.srs,  #self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
                    self.dst_crs,
                    trim = True #trim is used to auto-clip the tif where there are null value
                )

                self.__write_output_RoS_mean(
                    reprj_values_ros_mean,
                    dst_trans_ros_mean,
                    active=n_active,
                    area_mean=area_mean,
                    area_50=area_50,
                    area_75=area_75,
                    area_90=area_90
                )

                reprj_values_I_max, dst_trans_I_max = reproject(
                    fl_I_max,
                    self.__trans,
                    self.__prj.srs,  #self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
                    self.dst_crs,
                    trim = True #trim is used to auto-clip the tif where there are null value
                )

                self.__write_output_I_max(
                    reprj_values_I_max,
                    dst_trans_I_max,
                    active=n_active,
                    area_mean=area_mean,
                    area_50=area_50,
                    area_75=area_75,
                    area_90=area_90
                )
                
                reprj_values_I_mean, dst_trans_I_mean = reproject(
                    fl_I_mean,
                    self.__trans,
                    self.__prj.srs,  #self.__prj,  crs.srs #changed due to updates in Pyproj and-or Rasterio...
                    self.dst_crs,
                    trim = True #trim is used to auto-clip the tif where there are null value
                )

                self.__write_output_I_mean(
                    reprj_values_I_mean,
                    dst_trans_I_mean,
                    active=n_active,
                    area_mean=area_mean,
                    area_50=area_50,
                    area_75=area_75,
                    area_90=area_90
                )

                self.__update_isochrones(isochrones, reprj_values, dst_trans)
                self.__write_isochrones(isochrones)
