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
        DEFAULT_TAG : p_time_standard,
        WANG_TAG : p_time_wang,
        ROTHERMEL_TAG : p_time_rothermel,
    }
    p_time_function = ros_models.get(ros_model_code, p_time_wang)
    return p_time_function

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
    v_wh = np.clip(v_wh_pre * moist_eff, 0.01, 100) #adoptable RoS

    t = real_dist / v_wh
    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t

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
    v_wh = np.clip(v_wh_pre * moist_eff, 0.01, 100) #adoptable RoS
    
    t = real_dist / v_wh

    t[t>=1] = np.around(t[t>=1])
    t = np.clip(t, 0.1, np.inf)
    return t

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
    
    #p_moist = 1
    p_moist = M1 * moist**3 + M2 * moist**2 + M3 * moist + M4
    p_m = np.clip(p_moist , 0, 1.0)
    p_veg = prob_table[veg_to - 1, veg_from - 1]
    p = 1-(1-p_veg)**alpha_wh
    #p_clip = np.clip(p, 0, 1.0)
    p_clip = np.clip(p * p_m, 0, 1.0)

    return p_clip


def fire_spotting(angle_to, w_dir, w_speed):            #è la funzione per calcolare la distanza delle celle che possono essere innnescate per spotting al variare dell'angolo delle celle in funzione di velocità e direzione del vento (modello dei greci)
    r_n = np.random.normal( spotting_rn_mean , spotting_rn_std )  # r_n = (w_speed.shape[0]) * 100 
    w_speed_ms = w_speed / 3.6                  #velocità del vento in m/s
    d_p = r_n * np.exp( w_speed_ms * c_2 *( np.cos( w_dir - angle_to ) - 1 ) )          #formula per calcolare la distanza dello spotting in tutti i punti del vicinato a 2 e 3 celle (formulazione dei greci)
    #d_p = r_n * w_speed_ms * c_2 * np.exp(  np.cos( w_dir - angle_to ) - 1 )  #formula per prova brutale, pompando lo spotting
    return d_p

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
                    
                    self.veg[:, (0, 1, 2, -3, -2, -1)] = 0
                    self.veg[(0, 1, 2, -3, -2, -1), :] = 0
                    self.veg[(self.veg<0)|(self.veg>6)] = 0                    
                    
                    transform, crs, bounds, res = veg_file.transform, veg_file.crs, veg_file.bounds, veg_file.res

                    self.__prj = Proj(crs.to_wkt())
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
            logging.info('Loading VEGETATION from "' + self.settings.tileset + '" tileset')
            veg, west, north, step_x, step_y = \
                load_tiles(zone_number, easting, northing, self.settings.grid_dim, 'prop', self.settings.tileset)
            veg[:, (0, 1, 2, -3, -2, -1)] = 0
            veg[(0, 1, 2, -3, -2, -1), :] = 0
            self.veg = veg.astype('int8')
            
            logging.info('Loading DEM "default" tileset')
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
        p_prob = p_probability(dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moisture_r, w_dir_r, w_speed_r)

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
                                    moisture_r[p],
                                    w_dir_r[p], w_speed_r[p])

        ###### materiale per fire spotting    ----> DA QUI
        ##################################################
        conifer_mask = (veg_type == 5)                                                                      #seleziono i punti che hanno veg = fire-prone conifers
        conifer_r , conifer_c , conifer_t = u[conifer_mask, 0], u[conifer_mask, 1], u[conifer_mask, 2] 
        
        N_spotters = conifer_r.shape[0]    #number of  fire-prone conifers cells that are  burning
        '''N_embers = np.random.poisson(lambda_spotting, N_spotters)
        N_embers =  np.around(N_embers)   # Number of embers spotted by the selected cell

        ember_angle = np.random.uniform(0 , 2.0*np.pi, N_spotters)
        w_dir_spot = (w_dir + (pi/16)*(0.5 - rand(N_spotters))).repeat(N_spotters)           #vettore direzione del vento con le dimensioni delle celle distanti 2 e lieve randomizzazione 
        w_speed_spot = (w_speed * (1.2 - 0.4 * rand(N_spotters))).repeat(N_spotters)

        ember_distance  = fire_spotting(ember_angle,  w_dir_spot, w_speed_spot)  #sbagliato!!!

        ember_distance = (ember_distance > cellsize)  #manca da risolvere il bordo, ma magari possiamo usare le coordinate dopo

        delta_r = ember_distance * np.cos(ember_angle)  #vertical delta [meters]
        delta_c = ember_distance * np.sin(ember_angle)  #horizontal delta [meters]

        nb_spot_r = int( delta_r / cellsize )   #number of vertical cells 
        nb_spot_c = int( delta_c / cellsize )   #number of horizontal cells  
        #queste cose sono tutte in modalita' "liste ed operazioni collettive"
        #vanno rimesse in modalita' "singolo ember lanciato dal singolo spot"
        nr_spot = conifer_r[spotting_source] + nb_spot_r         #vertical location of the cell to be ignited by the ember
        nc_spot = conifer_c[spotting_source] + nb_spot_c         #horizontal location of the cell to be ignited by the ember
        nt_spot = conifer_t[spotting_source]

        transition_time_spot = self.p_time(self.dem[nr_spot, nc_spot], self.dem[nr_spot, nc_spot],              #calcolo quanto ci mettono a bruciare le celle innescate dallo spotting
                        self.veg[nr_spot, nc_spot], self.veg[nr_spot, nc_spot],                     #considero dh=0 (pianura) e veg_from=veg_to
                        ember_angle, ember_distance, 
                        moisture[nr_spot, nc_spot],
                        w_dir, w_speed)
        
        p_nr = np.append( p_nr , nr_spot)               #aggiungo le coordinate-riga dei punti innescati per spotting a quelli innescati "normalmente"
        p_nc = np.append( p_nc , nc_spot)               #aggiungo le coordinate-colonna dei punti innescati per spotting a quelli innescati "normalmente"
        p_nt = np.append( p_nt , nt_spot)               #aggiungo il tempo della simulazione dei punti innescati per spotting a quelli innescati "normalmente"
        transition_time = np.append( transition_time , np.around( transition_time_spot ) )'''
        for spotting_source in range(N_spotters):
            #print("fare cose relative alla fonte di spotting selezionata")
            # definire lambda = 3.0 da qualche parte su constants.py. 
            N_embers = np.random.poisson(lambda_spotting) #volendo posso aggiungere ,size=...)
            N_embers =  int(N_embers)
            for ember in range(N_embers):
                #print("Sono un ember, viaggio  e brucio.")            
                #angle of spotting for the considered ember. In radians (?) 
                ember_angle = np.random.uniform(0 , 2.0*np.pi)
                # we need to  compute the angle between the wind direction and the ember_angle...
                ember_distance  = fire_spotting(ember_angle,  w_dir, w_speed)
                    
                if  ember_distance < cellsize:
                    pass
                #elif ember_distance > cellsize * 3:  #provvisorio, da rivedere con molta attenzione!!!!!
                #    pass
                else:
                    print('belin se me la viaggio')
                    #here we  need to  compute the number of horizontal and vertical cells that the ember is  actually travelling.
                    #i_ember = 0  
                    #j_ember = 0
                    
                    delta_r = ember_distance * np.cos(ember_angle)  #vertical delta [meters]
                    delta_c = ember_distance * np.sin(ember_angle)  #horizontal delta [meters]

                    nb_spot_r = int( delta_r / cellsize )   #number of vertical cells 
                    nb_spot_c = int( delta_c / cellsize )   #number of horizontal cells  
                    #queste cose sono tutte in modalita' "liste ed operazioni collettive"
                    #vanno rimesse in modalita' "singolo ember lanciato dal singolo spot"
                    nr_spot = conifer_r[spotting_source] + nb_spot_r         #vertical location of the cell to be ignited by the ember
                    nc_spot = conifer_c[spotting_source] + nb_spot_c         #horizontal location of the cell to be ignited by the ember
                    nt_spot = conifer_t[spotting_source]
                    #vediamo! bisogna cambiare il tempo usando una funzione di tipo Ros...
                    # dobbiamo ora fare due controlli: 1) se  non spariamo fuori  dal campo di gioco     ###se stiamo sotto le 3 celle, non usciamo perchè avevo aumentato il bordo di 0 a 3 celle
                    #2) se  spariamo sopra ad  una vegetazione compatibile. in quel caso si lancia ancora uan moneta con
                    # la formula P_0 ( 1 + P_vegetazionericevente)  
                    ## possiamo adattare questa per il punto 2:
                    #n_mask = np.logical_and(self.f_global[nr_spot, nc_spot, nt_spot] == 0, self.veg[nr_spot, nc_spot] != {0, 3, 7} )
                    transition_time_spot = self.p_time(
                                    np.array([self.dem[nr_spot, nc_spot]]), np.array([self.dem[nr_spot, nc_spot]]),              #calcolo quanto ci mettono a bruciare le celle innescate dallo spotting
                                    np.array([self.veg[nr_spot, nc_spot]]), np.array([self.veg[nr_spot, nc_spot]]),                     #considero dh=0 (pianura) e veg_from=veg_to
                                    np.array([ember_angle]), np.array([ember_distance]), 
                                    np.array([moisture[nr_spot, nc_spot]]),
                                    np.array([w_dir]), np.array([w_speed]))
                    
                    p_nr = np.append( p_nr , nr_spot)               #aggiungo le coordinate-riga dei punti innescati per spotting a quelli innescati "normalmente"
                    p_nc = np.append( p_nc , nc_spot)               #aggiungo le coordinate-colonna dei punti innescati per spotting a quelli innescati "normalmente"
                    p_nt = np.append( p_nt , nt_spot)               #aggiungo il tempo della simulazione dei punti innescati per spotting a quelli innescati "normalmente"
                    transition_time = np.append( transition_time , np.around( transition_time_spot ) )
                # main thrust  of the ember: sampled from a Gaussian Distribution (Alexandridis et al, 2008 and 2011)



        
        
        '''if w_speed >=50: #fire-spotting solo se vento > 50 km/h (soglia modificabile)     NB: se non ci fosse sogli minima, con formula usata da greci miniore è il vento e maggiore è possibilità di avere spotting lontano dalla cella che brucia...
            conifer_mask = (veg_type == 5)                                                                      #seleziono i punti che hanno veg = fire-prone conifers
            conifer_r , conifer_c , conifer_t = u[conifer_mask, 0], u[conifer_mask, 1], u[conifer_mask, 2]      #tra le celle che stanno bruciando, seleziono quelle che sono fire-prone conifers

            nb2_num = n2_arr.shape[0]           #vecchia implem.#numero celle a distanza 2
            nb3_num = n3_arr.shape[0]           #vecchia implem. #numero celle a distanza 3
            from_num_spotting = conifer_r.shape[0]          #numero celle fire-prone conifers che stanno bruciando

                #distanza = 2 celle
            nb2_arr_r = tile(n2_arr[:, 0], from_num_spotting)       #coordinata-riga delle celle distanti 2 (generali)  
            nb2_arr_c = tile(n2_arr[:, 1], from_num_spotting)       #coordinata-colonna delle celle distanti 2 (generali)

            nr2 = conifer_r.repeat(nb2_num) + nb2_arr_r             #coordinata-riga delle celle distanti 2 (riferite al dominio)
            nc2 = conifer_c.repeat(nb2_num) + nb2_arr_c             #coordinata-colonna delle celle distanti 2 (riferite al dominio)
            nt2 = conifer_t.repeat(nb2_num)                         #tempo della simulazione della cella che brucia

            #veg_to2 = self.veg[nr2, nc2]                            #vegetazione delle cella distanti 2
            angle_to2 = angle2[nb2_arr_r+2, nb2_arr_c+2]            #angoli delle celle distanti 2
            dist_to2 = dist2[nb2_arr_r+2, nb2_arr_c+2]              #distanza reale delle celle distanti 2
            w_dir_r2 = (w_dir + (pi/16)*(0.5 - rand(from_num_spotting))).repeat(nb2_num)           #vettore direzione del vento con le dimensioni delle celle distanti 2 e lieve randomizzazione 
            w_speed_r2 = (w_speed * (1.2 - 0.4 * rand(from_num_spotting))).repeat(nb2_num)         #vettore velocità del vento con le dimensioni delle celle distanti 2 e lieve randomizzazione
                    #filtro celle non corrette
            n_mask2 = np.logical_and(self.f_global[nr2, nc2, nt2] == 0, self.veg[nr2, nc2] != {0, 3, 7} )# veg_to2 !=3, veg_to2 !=7 ) #aree nude e faggete sono impostate come impossibili da bruciare per spotting
            #veg_to2 = veg_to2[n_mask2]                  #seleziono veg_to dei punti che si possono innescare per spotting
            angle_to2 = angle_to2[n_mask2]              #seleziono angle_to dei punti che si possono innescare per spotting
            dist_to2 = dist_to2[n_mask2]                #seleziono dist_to dei punti che si possono innescare per spotting
            w_dir_r2 = w_dir_r2[n_mask2]                #seleziono direzione del vento dei punti che si possono innescare per spotting
            w_speed_r2 = w_speed_r2[n_mask2]            #seleziono velocità del vento dei punti che si possono innescare per spotting
            
            nr2, nc2, nt2 = nr2[n_mask2], nc2[n_mask2], nt2[n_mask2]        #seleziono coordiate e tempo della simulazione dei punti che si possono innescare per spotting

            calc_dist2 = fire_spotting(angle_to2, w_dir_r2, w_speed_r2)     #calcolo quali celle a distanza 2 si possono innescare per spotting 

            spotting2 = ( calc_dist2 >= ( dist_to2 * cellsize - 15 )) & ( calc_dist2 <= ( dist_to2 * cellsize + 15 ))       #filtro per selezionare solo distanze attorno alle celle distanti 2

            nr2, nc2, nt2 = nr2[spotting2], nc2[spotting2], nt2[spotting2]      #seleziono solo le celle distanti 2 che sono innescate dallo spotting
            #veg_to2 = veg_to2[spotting2]                                        #seleziono solo le veg delle celle distanti 2 che sono innescate dallo spotting

            #spotting_time2 = ( dist_to2[spotting2] * cellsize ) / ( w_speed_r2[spotting2] / 3.6 )           #calcolo il tempo che impiega il tizzone a raggiungere la cella (molto banale)

            transition_time2 = self.p_time(self.dem[nr2, nc2], self.dem[nr2, nc2],              #calcolo quanto ci mettono a bruciare le celle innescate dallo spotting
                                    self.veg[nr2, nc2], self.veg[nr2, nc2],                     #considero dh=0 (pianura) e veg_from=veg_to
                                    angle_to2[spotting2], dist_to2[spotting2],
                                    moisture[nr2, nc2],
                                    w_dir_r2[spotting2], w_speed_r2[spotting2])

                #distanza = 3 celle
            nb3_arr_r = tile(n3_arr[:, 0], from_num_spotting)       #coordinata-riga delle celle distanti 3 (generali)
            nb3_arr_c = tile(n3_arr[:, 1], from_num_spotting)       #coordinata-colonna delle celle distanti 3 (generali)

            nr3 = conifer_r.repeat(nb3_num) + nb3_arr_r             #coordinata-riga delle celle distanti 3 (riferite al dominio)
            nc3 = conifer_c.repeat(nb3_num) + nb3_arr_c             #coordinata-colonna delle celle distanti 3 (riferite al dominio)
            nt3 = conifer_t.repeat(nb3_num)                         #tempo della simulazione della cella che brucia

            #veg_to3 = self.veg[nr3, nc3]                            #vegetazione delle cella distanti 3
            angle_to3 = angle3[nb3_arr_r+3, nb3_arr_c+3]            #angoli delle celle distanti 3
            dist_to3 = dist3[nb3_arr_r+3, nb3_arr_c+3]              #distanza reale delle celle distanti 3
            w_dir_r3 = (w_dir + (pi/16)*(0.5 - rand(from_num_spotting))).repeat(nb3_num)           #vettore direzione del vento con le dimensioni delle celle distanti 3 e lieve randomizzazione
            w_speed_r3 = (w_speed * (1.2 - 0.4 * rand(from_num_spotting))).repeat(nb3_num)         #vettore velocità del vento con le dimensioni delle celle distanti 3 e lieve randomizzazione
                    #filtro celle non corrette
            n_mask3 = np.logical_and(self.f_global[nr3, nc3, nt3] == 0, self.veg[nr3, nc3] != {0, 3, 7} )# veg_to2 !=3, veg_to2 !=7) #aree nude e faggete sono impostate come impossibili da bruciare per spotting
            #veg_to3 = veg_to3[n_mask3]                  #seleziono veg_to dei punti che si possono innescare per spotting
            angle_to3 = angle_to3[n_mask3]              #seleziono angle_to dei punti che si possono innescare per spotting
            dist_to3 = dist_to3[n_mask3]                #seleziono dist_to dei punti che si possono innescare per spotting
            w_dir_r3 = w_dir_r3[n_mask3]                #seleziono direzione del vento dei punti che si possono innescare per spotting
            w_speed_r3 = w_speed_r3[n_mask3]            #seleziono velocità del vento dei punti che si possono innescare per spotting

            nr3, nc3, nt3 = nr3[n_mask3], nc3[n_mask3], nt3[n_mask3]        #seleziono coordiate e tempo della simulazione dei punti che si possono innescare per spotting

            calc_dist3 = fire_spotting(angle_to3, w_dir_r3, w_speed_r3)     #calcolo quali celle a distanza 3 si possono innescare per spotting

            spotting3 = ( calc_dist3 >= ( dist_to3 * cellsize - 15 ) )      #filtro per selezionare solo distanze attorno alle celle distanti 3

            nr3, nc3, nt3 = nr3[spotting3], nc3[spotting3], nt3[spotting3]      #seleziono solo le celle distanti 3 che sono innescate dallo spotting
            #veg_to3 = veg_to3[spotting3]                                        #seleziono solo le veg delle celle distanti 3 che sono innescate dallo spotting

            #spotting_time3 = ( dist_to3[spotting3] * cellsize ) / ( w_speed_r3[spotting3] / 3.6 )           #calcolo il tempo che impiega il tizzone a raggiungere la cella (molto banale)

            transition_time3 = self.p_time(self.dem[nr3, nc3] , self.dem[nr3, nc3] ,            #calcolo quanto ci mettono a bruciare le celle innescate dallo spotting
                                    self.veg[nr3, nc3], self.veg[nr3, nc3],                     #considero dh=0 (pianura) e veg_from=veg_to
                                    angle_to3[spotting3], dist_to3[spotting3],
                                    moisture[nr3, nc3],
                                    w_dir_r3[spotting3], w_speed_r3[spotting3])

            if len(nr2) & len(nc2) & len(nt2):              #eseguo aggiunta dei punti solo se sono state create correttamente tutte le variabili
                p_nr = np.append( p_nr , nr2)               #aggiungo le coordinate-riga dei punti innescati per spotting a quelli innescati "normalmente"
                p_nc = np.append( p_nc , nc2)               #aggiungo le coordinate-colonna dei punti innescati per spotting a quelli innescati "normalmente"
                p_nt = np.append( p_nt , nt2)               #aggiungo il tempo della simulazione dei punti innescati per spotting a quelli innescati "normalmente"
                transition_time = np.append( transition_time , np.around( transition_time2 ) )        #aggiungo il tempo impiegato per innescare i punti con lo spotting al transition time calcolato per gli altri punti
                #transition_time = np.append( transition_time , np.around( spotting_time2 + transition_time2 ) )

            if len(nr3) & len(nc3) & len(nt3):              #eseguo aggiunta dei punti solo se sono state create correttamente tutte le variabili
                p_nr = np.append( p_nr , nr3)               #aggiungo le coordinate-riga dei punti innescati per spotting a quelli innescati "normalmente"
                p_nc = np.append( p_nc , nc3)               #aggiungo le coordinate-colonna dei punti innescati per spotting a quelli innescati "normalmente"
                p_nt = np.append( p_nt , nt3)               #aggiungo il tempo della simulazione dei punti innescati per spotting a quelli innescati "normalmente"
                transition_time = np.append( transition_time , np.around( transition_time3 ) )        #aggiungo il tempo impiegato per innescare i punti con lo spotting al transition time calcolato per gli altri punti
                #transition_time = np.append( transition_time , np.around( spotting_time3 + transition_time3 ) )'''
        ######################################
        #####################   A QUI  <------

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

        if waterline_actionss:
            waterline_action_string = '\n'.join(waterline_actionss)
            mid_lat, mid_lon, polys, lines, points = read_actions(waterline_action_string)
            easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)

            img, waterline_points = \
                rasterize_actions((self.__shape[0], self.__shape[1]), 
                            points, lines, polys, west, north, self.step_x, self.step_y, zone_number, base_value=moisture_value)
            
            mask = (img==1)
            img_mask = ndimage.binary_dilation(mask)
            img[img_mask] = 0.8  #mettere a 0.9 per le fighting actions con le water line
        else:
            img = np.ones((self.__shape[0], self.__shape[1])) * moisture_value
        
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

            #self.__update_simulation(self.settings.n_threads, newignitions)

            self.c_time, updates = self.ps.pop()
            
            new_updates = self.__apply_updates(updates, wspeed, wdir, moisture)
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
