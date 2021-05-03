# import numpy as np
from datetime import timedelta
# import utm
from numpy import pi, tanh, sign, array, tile
from numpy.random import rand
from .utils import *
from .constants import *

import os
import logging
import json

neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
n_arr = array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])

dist = array([[1.414, 1, 1.414], [1, 0, 1], [1.414, 1, 1.414]])
angle = array([[pi*3/4, pi/2, pi/4], [pi, np.nan, 0], [-pi*3/4, -pi/2, -pi/4]])
cellsize = 20

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


#@jit
def apply_updates(c_time, ps, updates, f_global, w_speed, w_dir, dem, veg, moist):
    # coordinates of the next updates
    u = np.vstack(updates)
    veg_type = veg[u[:, 0], u[:, 1]]
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

    dem_from = dem[r, c].repeat(nb_num)
    veg_from = veg[r, c].repeat(nb_num)
    veg_to = veg[nr, nc]
    dem_to = dem[nr, nc]
    angle_to = angle[nb_arr_r+1, nb_arr_c+1]
    dist_to = dist[nb_arr_r+1, nb_arr_c+1]

    # exclude all ignited and not valid pixels
    n_mask = np.logical_and(f_global[nr, nc, nt] == 0, veg_to != 0)
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
    updates = list(map(
        lambda t: (t, extract_updates(t)), 
    unique_time))
    
    return updates


def find_bc(boundary_conditions, time):
    last_bc = None
    for bc in boundary_conditions:
        if time >= bc['time']:
            last_bc = bc
    return last_bc


def init_simulation(n_threads, img, active_ignitions, ps, f_global):
    for t in range(n_threads):
        f_global[:, :, t] = img.copy()
        for p in active_ignitions:
            ps.push(array([p[0], p[1], t]), 0)
            f_global[p[0], p[1], t] = 0


def run(run_id,
        n_threads, boundary_conditions,
        init_date, ignition_string, tileset, grid_dim,
        time_resolution=60,
        output_folder='work',
        time_limit=2880,
        simp_fact=0.00001,
        debug_mode=False,
        write_vegetation=False,
        save_realizations=False
    ):

    mid_lat, mid_lon, polys, lines, points = read_ignition(ignition_string)
    easting, northing, zone_number, zone_letter = utm.from_latlon(mid_lat, mid_lon)
    veg, west, north, step_x, step_y = load_tiles(zone_number, easting, northing, grid_dim, 'prop', tileset)
    veg = veg.astype('int8')

    veg[:, (0, -1)] = 0
    veg[(0, -1), :] = 0

    dem, west, north, step_x, step_y = load_tiles(zone_number, easting, northing, grid_dim, 'quo', tileset)
    dem = dem.astype('int16')

    moist = np.zeros_like(veg)

    img, active_ignitions = \
        rasterize_ignitions((grid_dim, grid_dim),
                            points, lines, polys, west, north, step_x, step_y, zone_number)

    ps = Scheduler()

    f_global = np.zeros((grid_dim, grid_dim, n_threads), dtype='int8')


    if debug_mode:
        #test configuration
        #'''
        veg = np.ones_like(veg, dtype='int8') * 5
        dem = np.ones_like(veg, dtype='float')
        #for r in range(dem.shape[0]):
        #    dem[r, :] = np.arange(dem.shape[1]) * -5
        #'''


    # reproject and save vegetation to file
    if write_vegetation:
        veg_reprojected, dst_trans, dst_crs = reproject(
            veg.astype('float'),
            (west, north, step_x, step_y),
            zone_number, zone_letter, trim=False
        )

        vegetation_file = os.path.join(output_folder, 'vegetation.tiff')
        write_geotiff(vegetation_file, veg_reprojected, dst_trans, dst_crs)

        dem_reprojected, dst_trans, dst_crs = reproject(
            dem.astype('float'),
            (west, north, step_x, step_y),
            zone_number, zone_letter, trim=False
        )

        dem_file = os.path.join(output_folder, 'dem.tiff')
        write_geotiff(dem_file, dem_reprojected, dst_trans, dst_crs, dtype=np.float64)


    init_simulation(n_threads, img, active_ignitions, ps, f_global)

    isochrones = {}
    c_time = 0

    while len(ps):
        if time_limit and c_time > time_limit:
            break

        c_time, updates = ps.pop()

        bc = find_bc(boundary_conditions, c_time)
        w_dir_deg = float(bc.get('w_dir', 0))
        wdir = normalize((180 - w_dir_deg + 90) * np.pi / 180.0)
        wspeed = float(bc.get('w_speed', 0))

        updates = apply_updates(c_time, ps, updates, f_global, wspeed, wdir, dem, veg, moist)
        ps.push_all(updates)

        if c_time % time_resolution == 0:
            filename = os.path.join(output_folder, str(c_time))
            tiff_file = filename + '.tiff'
            json_file = filename + '.json'

            f = np.sum(f_global, 2) / float(n_threads)

            ref_date = str(init_date + timedelta(minutes=c_time))
            active = ps.active().tolist()
            area_mean = np.sum(f) * step_x * step_y / 10000
            area_50 = float(np.sum(f >= 0.5) * step_x * step_y) / 10000
            area_75 = float(np.sum(f >= 0.75) * step_x * step_y) / 10000
            area_90 = float(np.sum(f >= 0.90) * step_x * step_y) / 10000

            trim = not debug_mode
            values, dst_trans, dst_crs = reproject(
                f,
                (west, north, step_x, step_y),
                zone_number, zone_letter, trim=trim
            )

            n_active = len(active)
            days = round(c_time // (24*60))
            hours = round((c_time % (24*60)) // 60)
            minutes = round(c_time % 60)

            logging.info(
                '{0:1.0f}d {1:2.0f}h {2:2.0f}m - {3} active - {4:.1f} [ha]'.format(
                    days, hours, minutes, n_active, area_mean
                )
            )

            with open(json_file, 'w') as fp:
                json.dump(
                    dict(
                        time=c_time,
                        timeref=ref_date,
                        active=n_active,
                        area_mean=area_mean,
                        area_50=area_50,
                        area_75=area_75,
                        area_90=area_90
                    ),
                    fp
                )

                write_geotiff(tiff_file, values*255, dst_trans, dst_crs)


            if c_time % time_resolution == 0:
                isochrones[c_time] = extract_isochrone(values, dst_trans,
                                                       thresholds=[0.5, 0.75, 0.9],
                                                       simp_fact=simp_fact)
                #save_isochrones(isochrones, 'images/isochrones_' + str(c_time) + '.shp')
                isochrone_file = 'isochrones_' + str(c_time) + '.geojson'
                save_isochrones(isochrones, os.path.join(output_folder, isochrone_file), format='geojson')
                #logging.info('writing isochrone')


            #logging.info(ref_date + ' - ' + str(len(ps.active())) + ' active')

    if save_realizations:
        logging.info('saving single realizations')
        for t in range(f_global.shape[2]):
            values, dst_trans, dst_crs = reproject(
                f_global[:,:,t].astype('float') ,
                (west, north, step_x, step_y),
                zone_number, zone_letter, trim=trim
            )
            
            filename = 'final_' + str(t) + '.tiff'
            tiff_file = os.path.join(output_folder, filename)
            write_geotiff(tiff_file, values*255.0, dst_trans, dst_crs)




