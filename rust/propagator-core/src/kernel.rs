//! The event-driven propagation kernel (spec §5.1, §5.4, §6.1).
//!
//! Runs per realization; realizations are independent given the shared
//! read-only inputs, so the driver parallelizes over them with scoped
//! threads. Unlike the numba kernel there is no suspend-and-regrow
//! protocol: heaps and tile pools are per-realization `Vec`s that grow on
//! demand. The boundary-halt semantics (suspend *before* igniting a
//! boundary-ring cell so the caller can expand the domain losslessly) are
//! preserved.

use crate::front::FrontHeap;
use crate::fuels::{FuelSystem, NO_FUEL};
use crate::grid::Grid2;
use crate::models::{
    fireline_intensity, lhv_fuel, p_time, probability_to_neighbour, MoistureModel, RosModel,
};
use crate::rng::Rng;
use crate::tiles::{local_index, TileGrid, FLAG_FIRE, FLAG_SPOT_GEN, FLAG_SPOT_RECV};

// 8-connected neighbour lattice
pub const N_NEIGHBOURS: usize = 8;
const NEIGHBOURS: [(i64, i64); N_NEIGHBOURS] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

// spotting model constants
/// baseline per-ember ignition probability (before the fuel embers bonus)
const P_C0: f64 = 0.6;
/// Poisson mean number of embers emitted by a burning spotting cell
/// (Alexandridis et al. 2009, 2011)
const LAMBDA_SPOTTING: f64 = 2.0;

// Wind- and intensity-scaled landing-distance model (see docs/spotting.md).
// The median landing distance is
//     d_median = SPOTTING_DISTANCE_REF
//              * (U / SPOTTING_WIND_REF)
//              * (I / SPOTTING_FLI_REF).powf(SPOTTING_FLI_EXPONENT)
// so distance -> 0 as wind -> 0, and grows with the source cell's fireline
// intensity through plume lofting.
/// median landing distance at the reference state [m]
const SPOTTING_DISTANCE_REF: f64 = 100.0;
/// reference wind speed [km/h]
const SPOTTING_WIND_REF: f64 = 20.0;
/// reference fireline intensity [kW/m]
const SPOTTING_FLI_REF: f64 = 10000.0;
/// loft chaining: H ~ I^(2/3), d ~ U*sqrt(H) ~ U*I^(1/3)
const SPOTTING_FLI_EXPONENT: f64 = 1.0 / 3.0;
/// downwind concentration, a wind-INDEPENDENT shape factor (= 1 downwind)
const SPOTTING_ANISOTROPY: f64 = 5.0;
/// lognormal spread of the landing distance about its median (Sardoy et al. 2008)
const SPOTTING_DISTANCE_LOG_SIGMA: f64 = 0.5;
/// floor on the along-trajectory wind fraction cos(w_dir - angle) used for the
/// ember travel time, so near-crosswind embers do not get an unbounded time
const SPOTTING_MIN_ALIGNMENT: f64 = 0.2;

/// The following constants are used in the Fire-Spotting model to assess the delay between ember landing and the development of a fire capable of propagation.
const SPOTTING_TIME_TO_PROPAGATION_MEDIAN: f64 = 600.0;
const SPOTTING_TIME_TO_PROPAGATION_LOG_SIGMA: f64 = 0.4;

/// hard cap on embers per cell, bounding worst-case work per ignition
pub const MAX_SPOTTING_EMBERS: u32 = 32;

/// Distance (in cell units) and angle (meteorological convention: 0 =
/// propagation toward south) per neighbour, computed once.
fn neighbour_geometry() -> [(f64, f64); N_NEIGHBOURS] {
    let mut geometry = [(0.0, 0.0); N_NEIGHBOURS];
    for (i, &(dr, dc)) in NEIGHBOURS.iter().enumerate() {
        let dist = ((dr * dr + dc * dc) as f64).sqrt();
        let angle = ((dc as f64).atan2(-(dr as f64)) + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI);
        geometry[i] = (dist, angle);
    }
    geometry
}

/// Read-only inputs shared by every realization during one kernel run.
pub struct SharedInputs<'a> {
    pub veg: &'a Grid2<i32>,
    pub fuel_idx: &'a Grid2<i32>,
    pub dem: &'a Grid2<f64>,
    /// effective moisture (base + actions, clipped), fraction
    pub moisture: &'a Grid2<f32>,
    /// radians
    pub wind_dir: &'a Grid2<f32>,
    /// km/h
    pub wind_speed: &'a Grid2<f32>,
    pub fuels: &'a FuelSystem,
    pub cellsize: f64,
    pub ros_model: RosModel,
    pub moisture_model: MoistureModel,
    pub do_spotting: bool,
}

/// Mutable state of one realization.
pub struct Realization {
    pub heap: FrontHeap,
    pub tiles: TileGrid,
    pub rng: Rng,
    /// fire reached (or, in halt mode, is about to reach) the boundary
    pub out_of_bounds: bool,
    /// halted before igniting a boundary cell (heap intact)
    pub halted_on_boundary: bool,
}

impl Realization {
    pub fn new(rows: usize, cols: usize, rng: Rng) -> Realization {
        Realization {
            heap: FrontHeap::new(),
            tiles: TileGrid::new(rows, cols),
            rng,
            out_of_bounds: false,
            halted_on_boundary: false,
        }
    }
}

#[inline]
fn on_boundary_ring(row: i64, col: i64, n_rows: usize, n_cols: usize) -> bool {
    row <= 0 || col <= 0 || row >= n_rows as i64 - 1 || col >= n_cols as i64 - 1
}

/// Advance one realization until its next event is past `end_time`.
/// With `halt_on_boundary`, suspends (heap intact, event not popped)
/// before igniting an unburnt boundary-ring cell.
pub fn advance_front_until(
    real: &mut Realization,
    shared: &SharedInputs<'_>,
    end_time: i64,
    halt_on_boundary: bool,
) {
    let geometry = neighbour_geometry();
    let n_rows = shared.veg.rows();
    let n_cols = shared.veg.cols();
    real.halted_on_boundary = false;

    while let Some(next_time) = real.heap.peek_time() {
        if next_time > end_time {
            break;
        }

        if halt_on_boundary {
            let (peek_row, peek_col) = real.heap.peek_cell().unwrap();
            if on_boundary_ring(peek_row as i64, peek_col as i64, n_rows, n_cols)
                && real.tiles.read_flags(peek_row as usize, peek_col as usize) & FLAG_FIRE == 0
            {
                real.out_of_bounds = true;
                real.halted_on_boundary = true;
                break;
            }
        }

        let (time, row_i, col_i, ros_value, fli_value) = real.heap.pop_min().unwrap();
        let row = row_i as usize;
        let col = col_i as usize;

        // lazy deletion: already burnt (or frozen: reads as burnt)
        if real.tiles.read_flags(row, col) & FLAG_FIRE != 0 {
            continue;
        }

        if on_boundary_ring(row_i as i64, col_i as i64, n_rows, n_cols) {
            real.out_of_bounds = true;
        }

        // mark burnt
        let slot = real.tiles.ensure_slot(row, col);
        let local = local_index(row, col);
        {
            let tile = real.tiles.tile_mut(slot);
            tile.flags[local] |= FLAG_FIRE;
            tile.arrival[local] = time as i32;
            tile.ros[local] = ros_value;
            tile.fli[local] = fli_value;
        }

        // ignitable but not combustible: never spreads
        if shared.veg[(row, col)] == NO_FUEL {
            continue;
        }

        let w_dir = shared.wind_dir[(row, col)] as f64;
        let w_speed = shared.wind_speed[(row, col)] as f64;
        let idx_from = shared.fuel_idx[(row, col)] as usize;

        // 8-neighbour spread
        for (neighbour, &(dr, dc)) in NEIGHBOURS.iter().enumerate() {
            let row_to = row_i as i64 + dr;
            let col_to = col_i as i64 + dc;
            if !shared.veg.in_bounds(row_to, col_to) {
                continue;
            }
            let (row_to, col_to) = (row_to as usize, col_to as usize);
            if shared.veg[(row_to, col_to)] == NO_FUEL
                || real.tiles.read_flags(row_to, col_to) & FLAG_FIRE != 0
            {
                continue;
            }

            let (dist_cells, angle) = geometry[neighbour];
            let dist = dist_cells * shared.cellsize;
            let dh = shared.dem[(row_to, col_to)] - shared.dem[(row, col)];
            let moist = shared.moisture[(row_to, col_to)] as f64;
            let idx_to = shared.fuel_idx[(row_to, col_to)] as usize;
            let p0 = shared.fuels.transition_probability(idx_from, idx_to);

            let p = probability_to_neighbour(
                shared.moisture_model,
                angle,
                dist,
                w_dir,
                w_speed,
                moist,
                dh,
                p0,
            );
            if p <= real.rng.uniform() {
                continue;
            }

            let (transition_time, ros, fli) = calculate_fire_behavior(
                shared, idx_from, idx_to, dh, dist, angle, moist, w_dir, w_speed,
            );
            real.heap.push(
                time + transition_time,
                row_to as i32,
                col_to as i32,
                ros as f32,
                fli as f32,
            );
        }

        // ember spotting
        if shared.do_spotting && shared.fuels.spotting[idx_from] {
            compute_spotting(
                real,
                shared,
                time,
                row,
                col,
                slot,
                local,
                w_dir,
                w_speed,
                fli_value as f64,
            );
        }
    }
}

/// (transition time [s, >= 1], ROS [m/min], fireline intensity [kW/m]).
#[inline]
#[allow(clippy::too_many_arguments)]
fn calculate_fire_behavior(
    shared: &SharedInputs<'_>,
    idx_from: usize,
    idx_to: usize,
    dh: f64,
    dist: f64,
    angle: f64,
    moist: f64,
    w_dir: f64,
    w_speed: f64,
) -> (i64, f64, f64) {
    let (time_seconds, ros) = p_time(
        shared.ros_model,
        shared.fuels.v0[idx_from],
        dh,
        angle,
        dist,
        moist,
        w_dir,
        w_speed,
    );
    let transition_time = (time_seconds as i64).max(1);

    let lhv_dead = lhv_fuel(shared.fuels.hhv[idx_to], moist);
    let lhv_canopy = lhv_fuel(shared.fuels.hhv[idx_to], shared.fuels.humidity[idx_to]);
    let fli = fireline_intensity(
        shared.fuels.d0[idx_to],
        shared.fuels.d1[idx_to],
        ros,
        lhv_dead,
        lhv_canopy,
    );
    (transition_time, ros, fli)
}

/// Ember emission from a freshly burnt cell (spec §5.4). Pushes landing
/// events and sets the spotting tracking flags.
#[allow(clippy::too_many_arguments)]
fn compute_spotting(
    real: &mut Realization,
    shared: &SharedInputs<'_>,
    time: i64,
    row: usize,
    col: usize,
    source_slot: i32,
    source_local: usize,
    w_dir: f64,
    w_speed: f64,
    fireline_intensity: f64,
) {
    let num_embers = real.rng.poisson(LAMBDA_SPOTTING).min(MAX_SPOTTING_EMBERS);
    let w_speed_ms = w_speed / 3.6;

    for _ in 0..num_embers {
        let angle = real.rng.uniform_range(0.0, 2.0 * std::f64::consts::PI);
        // Embers are wind-carried and lofted by the fire's own plume: with no
        // wind or no fire intensity there is no transport, hence no spotting.
        let (distance, landing_time_s) = if w_speed_ms <= 0.0 || fireline_intensity <= 0.0 {
            (0.0, 1.0)
        } else {
            // Median landing distance: linear in wind (collapses to zero as
            // the wind dies) and ~I^(1/3) in fireline intensity via lofting.
            let d_median = SPOTTING_DISTANCE_REF
                * (w_speed / SPOTTING_WIND_REF)
                * (fireline_intensity / SPOTTING_FLI_REF).powf(SPOTTING_FLI_EXPONENT);
            // Downwind concentration: a wind-independent shape factor.
            let alignment = (w_dir - angle).cos();
            let directional = (SPOTTING_ANISOTROPY * (alignment - 1.0)).exp();
            // Lognormal landing distance about the (directional) median.
            let median = d_median * directional;
            let distance = real.rng.lognormal(median.ln(), SPOTTING_DISTANCE_LOG_SIGMA);
            // Travel time uses the wind component along the ember trajectory
            // (U * cos(w_dir - angle)), floored to avoid a singularity for
            // near-crosswind embers.
            let transport_speed = w_speed_ms * alignment.max(SPOTTING_MIN_ALIGNMENT);
            (distance, distance / transport_speed)
        };

        // short embers behave like normal contact spread
        if distance < 2.0 * shared.cellsize {
            continue;
        }

        // landing cell (same angle convention as the spread kernel:
        // 0 -> south, pi/2 -> west)
        let delta_r = distance * angle.cos();
        let delta_c = -distance * angle.sin();
        let row_to = row as i64 + (delta_r / shared.cellsize) as i64;
        let col_to = col as i64 + (delta_c / shared.cellsize) as i64;
        if !shared.veg.in_bounds(row_to, col_to) {
            continue;
        }
        let (row_to, col_to) = (row_to as usize, col_to as usize);
        if real.tiles.read_flags(row_to, col_to) & FLAG_FIRE != 0 {
            continue;
        }
        if shared.veg[(row_to, col_to)] == NO_FUEL {
            continue;
        }

        // ignition success (Alexandridis eq. 10)
        let idx_to = shared.fuel_idx[(row_to, col_to)] as usize;
        let p_c = P_C0 * (1.0 + shared.fuels.prob_ign_by_embers[idx_to]);
        if real.rng.uniform() > p_c {
            continue;
        }

        let time_to_propagation = real.rng.lognormal(
            SPOTTING_TIME_TO_PROPAGATION_MEDIAN.ln(),
            SPOTTING_TIME_TO_PROPAGATION_LOG_SIGMA,
        );

        let landing_time = (landing_time_s.ceil() as i64).max(1);
        let time_to_propagation = (time_to_propagation.ceil() as i64).max(1);

        let propagation_time = landing_time + time_to_propagation;

        // tracking flags
        real.tiles.tile_mut(source_slot).flags[source_local] |= FLAG_SPOT_GEN;
        let recv_slot = real.tiles.ensure_slot(row_to, col_to);
        real.tiles.tile_mut(recv_slot).flags[local_index(row_to, col_to)] |= FLAG_SPOT_RECV;

        real.heap.push(
            time + propagation_time,
            row_to as i32,
            col_to as i32,
            0.0,
            0.0,
        );
    }
}

/// Run the kernel for every realization, parallelized over available
/// threads with scoped threads (realizations share only `shared`).
pub fn advance_all(
    realizations: &mut [Realization],
    shared: &SharedInputs<'_>,
    end_time: i64,
    halt_on_boundary: bool,
    n_threads: usize,
) {
    let n_threads = n_threads.max(1).min(realizations.len().max(1));
    if n_threads <= 1 || realizations.len() <= 1 {
        for real in realizations.iter_mut() {
            advance_front_until(real, shared, end_time, halt_on_boundary);
        }
        return;
    }
    let chunk_size = realizations.len().div_ceil(n_threads);
    std::thread::scope(|scope| {
        for chunk in realizations.chunks_mut(chunk_size) {
            scope.spawn(move || {
                for real in chunk.iter_mut() {
                    advance_front_until(real, shared, end_time, halt_on_boundary);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbour_angles_follow_convention() {
        let geometry = neighbour_geometry();
        // neighbour (1, 0) is due south of the source: angle 0
        let south = NEIGHBOURS.iter().position(|&d| d == (1, 0)).unwrap();
        assert!(geometry[south].1.abs() < 1e-12);
        // neighbour (-1, 0) is due north: angle pi
        let north = NEIGHBOURS.iter().position(|&d| d == (-1, 0)).unwrap();
        assert!((geometry[north].1 - std::f64::consts::PI).abs() < 1e-12);
        // neighbour (0, -1) is due west: angle pi/2 (propagation toward west)
        let west = NEIGHBOURS.iter().position(|&d| d == (0, -1)).unwrap();
        assert!((geometry[west].1 - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // diagonals are sqrt(2) away
        let diag = NEIGHBOURS.iter().position(|&d| d == (1, 1)).unwrap();
        assert!((geometry[diag].0 - 2f64.sqrt()).abs() < 1e-12);
    }
}
