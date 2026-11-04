// PROPAGATOR GPU kernel: arrival-time relaxation over (realization x grid).
//
// State per (realization r, cell c): `key = arrival << 4 | code`, where
// `arrival` is the ignition time in seconds and `code` the direction the fire
// came from (0..7 = neighbour index, 8 = forced ignition); 0xFFFFFFFF = not
// burnt. Spread is a relaxation: a cell whose arrival is final pushes
// `arrival + transition` into each spreadable neighbour with atomicMin, once
// per value (`pushed_at` remembers the value it was pushed at, so a cell that
// is lowered later is pushed again). The Bernoulli draw of every edge is a
// hash of (seed, r, cell, direction), fixed for the run. Work is ordered in
// time buckets with a per-(r, tile) earliest-unpushed-arrival so idle tiles
// cost one load per sweep. Weather is a per-cell field (wind direction,
// wind speed, effective moisture) for the current window; the ROS / FLI an
// edge carries are stored as a packed f16 pair when its push wins, so the
// fold needs no weather history.

const INF: u32 = 0xFFFFFFFFu;
const INF_ARRIVAL: u32 = 0x0FFFFFFFu;
const CODE_IGNITION: u32 = 8u;
const CODE_SPOT: u32 = 9u;
const FUEL_COLUMNS: u32 = 7u;

// spotting model (as the CPU kernel)
const P_C0: f32 = 0.6;
const LAMBDA_SPOTTING: f32 = 2.0;
const SPOTTING_DISTANCE_REF: f32 = 100.0;
const SPOTTING_WIND_REF: f32 = 20.0;
const SPOTTING_FLI_REF: f32 = 10000.0;
const SPOTTING_FLI_EXPONENT: f32 = 1.0 / 3.0;
const SPOTTING_ANISOTROPY: f32 = 5.0;
const SPOTTING_DISTANCE_LOG_SIGMA: f32 = 0.5;
const SPOTTING_MIN_ALIGNMENT: f32 = 0.2;
const SPOTTING_TIME_TO_PROPAGATION_MEDIAN: f32 = 600.0;
const SPOTTING_TIME_TO_PROPAGATION_LOG_SIGMA: f32 = 0.4;
const MAX_SPOTTING_EMBERS: u32 = 32u;
// draw indices: 0..7 edges, then the spotting sequence of the cell
const DRAW_SPOTTING_BASE: u32 = 8u;
const NO_FUEL_IDX: u32 = 0xFFFFFFFFu;
const TILE_SHIFT: u32 = 4u;
const TILE_SIZE: u32 = 16u;
const PI: f32 = 3.14159265358979;

const D1: f32 = 0.5;
const D2: f32 = 1.4;
const D3: f32 = 8.2;
const D4: f32 = 2.0;
const D5: f32 = 50.0;
const ROTHERMEL_ALPHA1: f32 = 0.0693;
const ROTHERMEL_ALPHA2: f32 = 0.0576;
const WANG_BETA1: f32 = 0.1783;
const WANG_BETA2: f32 = 3.533;
const WANG_BETA3: f32 = 1.2;
const M1: f32 = -3.5995;
const M2: f32 = 5.2389;
const M3: f32 = -2.6355;
const M4: f32 = 1.019;
const C_MOIST: f32 = -0.014;
const Q: f32 = 2442.0;
const MOISTURE_OF_EXTINCTION: f32 = 0.3;

struct Params {
    rows: u32,
    cols: u32,
    n_reals: u32,
    seed: u32,
    tiles_w: u32,
    tiles_h: u32,
    n_fuels: u32,
    do_spotting: u32,
    bucket_end: u32,
    fold_time: u32,
    ros_model: u32,
    moisture_model: u32,
    cellsize: f32,
    // end of the bucket the next sweep will process: tiles that receive an
    // arrival below it are queued for that sweep
    next_end: u32,
    // 1 when a relaxation sweep over the current list follows `prepare`:
    // those tiles recompute their minimum; 0 at a bucket start (carry all)
    processing: u32,
    // which half of the ping-pong buffers (tile_min, list, member, ctrl) is
    // "current" (0 or 1); the other half is "next"
    phase: u32,
    // world coordinates of local cell (0, 0): draws are hashed by world
    // position so a grown domain reproduces them
    origin_row: i32,
    origin_col: i32,
    _pad0: u32,
    _pad1: u32,
}

// WebGPU allows 10 storage buffers per stage, so everything is packed into
// 8: static inputs in one u32 array (fuel index, dem bits, fuel table bits,
// weather windows), the ping-pong pairs in single buffers with two halves
// selected by `params.phase`, and the worklist counters together with the
// indirect dispatch arguments.
@group(0) @binding(0) var<uniform> params: Params;
// [fuel_idx (N)] [dem bits (N)] [fuel table bits: 7 per fuel (v0, d0, d1,
// hhv, humidity, spotting, prob_ign_by_embers), then n x n spread
// probabilities] [wind_dir bits (N)] [wind_speed bits (N)] [effective
// moisture bits (N)]
@group(0) @binding(1) var<storage, read> inputs: array<u32>;
@group(0) @binding(2) var<storage, read_write> key: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pushed_at: array<atomic<u32>>;
// per (realization, tile) earliest unpushed arrival, two halves
@group(0) @binding(4) var<storage, read_write> tile_min: array<atomic<u32>>;
// fold accumulators per cell: count, arrival_min, arrival_sum (u32 seconds),
// ros_sum, fli_sum (f32), ros_max, fli_max (f32), spot_recv count; 8 x N
@group(0) @binding(5) var<storage, read_write> fold: array<f32>;
// worklists of (realization, tile) indices, two halves
@group(0) @binding(6) var<storage, read_write> list: array<u32>;
// membership flags of the worklists, two halves
@group(0) @binding(7) var<storage, read_write> member: array<atomic<u32>>;
// control words, 4 per half: [count, dispatch x, dispatch y, dispatch z]
@group(0) @binding(8) var<storage, read_write> ctrl: array<atomic<u32>>;
// per (realization, cell): ROS and FLI carried into the cell, pack2x16float
@group(0) @binding(9) var<storage, read_write> payload: array<u32>;

fn n_cells() -> u32 { return params.rows * params.cols; }
fn n_rt() -> u32 { return params.n_reals * params.tiles_w * params.tiles_h; }
fn cur_base() -> u32 { return params.phase * n_rt(); }
fn next_base() -> u32 { return (1u - params.phase) * n_rt(); }
fn ctrl_cur() -> u32 { return params.phase * 4u; }
fn ctrl_next() -> u32 { return (1u - params.phase) * 4u; }

fn fuel_idx_of(c: u32) -> u32 { return inputs[c]; }
fn dem_of(c: u32) -> f32 { return bitcast<f32>(inputs[n_cells() + c]); }
fn fuel_table(i: u32) -> f32 { return bitcast<f32>(inputs[2u * n_cells() + i]); }
fn weather_base() -> u32 {
    return 2u * n_cells() + params.n_fuels * FUEL_COLUMNS + params.n_fuels * params.n_fuels;
}
fn wind_dir_of(c: u32) -> f32 { return bitcast<f32>(inputs[weather_base() + c]); }
fn wind_speed_of(c: u32) -> f32 { return bitcast<f32>(inputs[weather_base() + n_cells() + c]); }
fn moisture_of(c: u32) -> f32 { return bitcast<f32>(inputs[weather_base() + 2u * n_cells() + c]); }

const NEIGHBOUR_DR = array<i32, 8>(-1, -1, -1, 0, 0, 1, 1, 1);
const NEIGHBOUR_DC = array<i32, 8>(-1, 0, 1, -1, 1, -1, 0, 1);

// Distance (cells) and propagation angle per neighbour, as the core's
// `neighbour_geometry` computes them: `atan2(dc, -dr) + pi` in the
// meteorological convention (0 = propagation toward south). Tabulated
// because WGSL's atan2 is implementation-defined for a zero x argument
// (Metal returns the wrong sign for -0.0, swapping east and west).
const SQRT2: f32 = 1.41421356237;
const NEIGHBOUR_DIST = array<f32, 8>(SQRT2, 1.0, SQRT2, 1.0, 1.0, SQRT2, 1.0, SQRT2);
const NEIGHBOUR_ANGLE = array<f32, 8>(
    0.75 * PI, PI, 1.25 * PI, 0.5 * PI, 1.5 * PI, 0.25 * PI, 0.0, 1.75 * PI,
);

fn neighbour_dist_cells(d: u32) -> f32 {
    return NEIGHBOUR_DIST[d];
}

fn neighbour_angle(d: u32) -> f32 {
    return NEIGHBOUR_ANGLE[d];
}

fn fuel_v0(i: u32) -> f32 { return fuel_table(i * FUEL_COLUMNS + 0u); }
fn fuel_d0(i: u32) -> f32 { return fuel_table(i * FUEL_COLUMNS + 1u); }
fn fuel_d1(i: u32) -> f32 { return fuel_table(i * FUEL_COLUMNS + 2u); }
fn fuel_hhv(i: u32) -> f32 { return fuel_table(i * FUEL_COLUMNS + 3u); }
fn fuel_humidity(i: u32) -> f32 { return fuel_table(i * FUEL_COLUMNS + 4u); }
fn fuel_spotting(i: u32) -> bool { return fuel_table(i * FUEL_COLUMNS + 5u) > 0.5; }
fn fuel_prob_ign_by_embers(i: u32) -> f32 { return fuel_table(i * FUEL_COLUMNS + 6u); }
fn fuel_p0(src_idx: u32, to: u32) -> f32 {
    return fuel_table(params.n_fuels * FUEL_COLUMNS + src_idx * params.n_fuels + to);
}

// lowbias32 hash of (seed, realization, world cell, draw index) -> [0, 1)
fn uniform01(r: u32, cell: u32, d: u32) -> f32 {
    let wrow = u32(i32(cell / params.cols) + params.origin_row);
    let wcol = u32(i32(cell % params.cols) + params.origin_col);
    var x = params.seed ^ (r * 0x9E3779B9u) ^ (wrow * 0x85EBCA6Bu) ^ (wcol * 0x27D4EB2Fu) ^ (d * 0xC2B2AE35u);
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x >> 8u) / 16777216.0;
}

fn sgn(x: f32) -> f32 {
    if x > 0.0 { return 1.0; }
    if x < 0.0 { return -1.0; }
    return 0.0;
}

fn w_h_effect(angle: f32, w_speed: f32, w_dir: f32, dh: f32, dist: f32) -> f32 {
    let a_const = 1.0 - D1 * D2 * tanh(-D4);
    let modulus = a_const + D1 * (D2 * tanh(w_speed / D3 - D4)) + w_speed / D5;
    let a = (modulus - 1.0) / 4.0;
    let w_effect = (a + 1.0) * (1.0 - a * a) / (1.0 - a * cos(w_dir - angle));
    let slope = dh / dist;
    let h_effect = pow(2.0, tanh((slope * 3.0) * (slope * 3.0) * sgn(slope)));
    return h_effect * w_effect;
}

fn w_h_effect_on_probability(angle: f32, w_speed: f32, w_dir: f32, dh: f32, dist: f32) -> f32 {
    let w_speed_norm = clamp(w_speed, 0.0, 60.0);
    var wh = w_h_effect(angle, w_speed_norm, w_dir, dh, dist) - 1.0;
    if wh > 0.0 { wh = wh / 2.13; } else if wh < 0.0 { wh = wh / 1.12; }
    return wh + 1.0;
}

fn p_moisture(moist: f32) -> f32 {
    if params.moisture_model == 1u {
        return M1 * moist * moist * moist + M2 * moist * moist + M3 * moist + M4;
    }
    let x = moist / MOISTURE_OF_EXTINCTION;
    let x2 = x * x;
    let p = -11.507 * x2 * x2 * x + 22.963 * x2 * x2 - 17.331 * x2 * x + 6.598 * x2 - 1.7211 * x + 1.0003;
    return clamp(p, 0.0, 1.0);
}

fn probability_to_neighbour(angle: f32, dist: f32, w_dir: f32, w_speed: f32, moist: f32, dh: f32, p0: f32) -> f32 {
    let moisture_effect = p_moisture(moist);
    let alpha_wh = max(w_h_effect_on_probability(angle, w_speed, w_dir, dh, dist), 0.0);
    let p = 1.0 - pow(1.0 - p0, alpha_wh);
    return clamp(p * moisture_effect, 0.0, 1.0);
}

// returns (transition seconds, ros m/min)
fn p_time(v0: f32, dh: f32, angle: f32, dist: f32, moist: f32, w_dir: f32, w_speed: f32) -> vec2<f32> {
    let real_dist = sqrt(dist * dist + dh * dh);
    let moist_eff = exp(C_MOIST * moist);
    var ros: f32;
    if params.ros_model == 1u {
        // Rothermel
        let w_spd = w_speed * cos(w_dir - angle) / 3.6;
        let teta_s = degrees(atan(dh / dist));
        let teta_f = degrees(atan(0.4226 * w_spd));
        let sf = clamp(exp(ROTHERMEL_ALPHA1 * teta_s), 0.01, 10.0);
        let wf = clamp(exp(ROTHERMEL_ALPHA2 * teta_f) / 13.0, 1.0, 20.0);
        ros = clamp(v0 * sf * wf * moist_eff, 0.01, 100.0);
    } else if params.ros_model == 2u {
        // Standard
        let wh = w_h_effect(angle, w_speed, w_dir, dh, dist);
        ros = clamp(v0 * wh * moist_eff, 0.01, 100.0);
    } else {
        // Wang
        let w_spd = w_speed * cos(w_dir - angle) / 3.6;
        let teta_s = atan(dh / dist);
        let wf = clamp(exp(WANG_BETA1 * w_spd), 0.01, 10.0);
        let sf = clamp(exp(sgn(dh) * WANG_BETA2 * pow(tan(abs(teta_s)), WANG_BETA3)), 0.01, 10.0);
        ros = clamp(v0 * wf * sf * moist_eff, 0.01, 100.0);
    }
    return vec2<f32>(real_dist / ros * 60.0, ros);
}

fn lhv_fuel(hhv: f32, moisture: f32) -> f32 {
    return hhv * (1.0 - moisture) - Q * moisture;
}

struct Edge {
    open: bool,       // fire can cross (static) and the draw accepted it
    blocked: bool,    // to_cell off-grid or unburnable
    to_cell: u32,
    transition: u32,
    ros: f32,
    fli: f32,
}

// deterministic edge quantities for source cell `c` toward neighbour `d`
// under the current weather fields; `draw` applies the Bernoulli draw
fn edge(r: u32, c: u32, d: u32, draw: bool) -> Edge {
    var e: Edge;
    e.open = false;
    e.blocked = true;
    let row = i32(c / params.cols);
    let col = i32(c % params.cols);
    let row_to = row + NEIGHBOUR_DR[d];
    let col_to = col + NEIGHBOUR_DC[d];
    if row_to < 0 || col_to < 0 || row_to >= i32(params.rows) || col_to >= i32(params.cols) {
        return e;
    }
    let n = u32(row_to) * params.cols + u32(col_to);
    let idx_from = fuel_idx_of(c);
    let idx_to = fuel_idx_of(n);
    if idx_from == NO_FUEL_IDX || idx_to == NO_FUEL_IDX {
        return e;
    }
    e.blocked = false;
    e.to_cell = n;
    let w_dir = wind_dir_of(c);
    let w_speed = wind_speed_of(c);
    let dist = neighbour_dist_cells(d) * params.cellsize;
    let angle = neighbour_angle(d);
    let dh = dem_of(n) - dem_of(c);
    let moist = moisture_of(n);
    if draw {
        let p = probability_to_neighbour(angle, dist, w_dir, w_speed, moist, dh, fuel_p0(idx_from, idx_to));
        if p <= uniform01(r, c, d) {
            return e;
        }
    }
    e.open = true;
    let tr = p_time(fuel_v0(idx_from), dh, angle, dist, moist, w_dir, w_speed);
    e.transition = max(u32(tr.x), 1u);
    e.ros = tr.y;
    let lhv_dead = lhv_fuel(fuel_hhv(idx_to), moist);
    let lhv_canopy = lhv_fuel(fuel_hhv(idx_to), fuel_humidity(idx_to));
    e.fli = (tr.y / 60.0) * (lhv_dead * fuel_d0(idx_to) + lhv_canopy * fuel_d1(idx_to));
    return e;
}

// standard normal from two uniforms (Box-Muller, no caching)
fn normal01(u1: f32, u2: f32) -> f32 {
    let r = sqrt(-2.0 * log(max(1.0 - u1, 1e-7)));
    return r * cos(2.0 * PI * u2);
}

// Ember emission from a freshly pushed spotting cell (CPU kernel's
// compute_spotting): a Poisson number of embers thrown at random azimuths
// with lognormal, wind- and intensity-scaled landing distances; each landing
// becomes an arrival at `a + landing time + time to propagation`.
fn spot(r: u32, c: u32, a: u32, fli: f32, n_tiles: u32, cells: u32) {
    let w_dir = wind_dir_of(c);
    let w_speed = wind_speed_of(c);
    let w_speed_ms = w_speed / 3.6;
    // Poisson(lambda) by Knuth
    var draw = DRAW_SPOTTING_BASE;
    let limit = exp(-LAMBDA_SPOTTING);
    var n_embers = 0u;
    var product = 1.0;
    loop {
        product = product * uniform01(r, c, draw);
        draw = draw + 1u;
        if product <= limit || n_embers >= MAX_SPOTTING_EMBERS {
            break;
        }
        n_embers = n_embers + 1u;
    }
    if w_speed_ms <= 0.0 || fli <= 0.0 {
        return;
    }
    let row = i32(c / params.cols);
    let col = i32(c % params.cols);
    let d_median = SPOTTING_DISTANCE_REF * (w_speed / SPOTTING_WIND_REF)
        * pow(fli / SPOTTING_FLI_REF, SPOTTING_FLI_EXPONENT);
    for (var k = 0u; k < n_embers; k = k + 1u) {
        let base = DRAW_SPOTTING_BASE + 48u + k * 8u;
        let angle = 2.0 * PI * uniform01(r, c, base);
        let alignment = cos(w_dir - angle);
        let directional = exp(SPOTTING_ANISOTROPY * (alignment - 1.0));
        let median = d_median * directional;
        let z = normal01(uniform01(r, c, base + 1u), uniform01(r, c, base + 2u));
        let distance = exp(log(median) + SPOTTING_DISTANCE_LOG_SIGMA * z);
        if distance < 2.0 * params.cellsize {
            continue;
        }
        let transport_speed = w_speed_ms * max(alignment, SPOTTING_MIN_ALIGNMENT);
        let landing_time_s = distance / transport_speed;
        let delta_r = distance * cos(angle);
        let delta_c = -distance * sin(angle);
        let row_to = row + i32(delta_r / params.cellsize);
        let col_to = col + i32(delta_c / params.cellsize);
        if row_to < 0 || col_to < 0 || row_to >= i32(params.rows) || col_to >= i32(params.cols) {
            continue;
        }
        let n = u32(row_to) * params.cols + u32(col_to);
        let idx_to = fuel_idx_of(n);
        if idx_to == NO_FUEL_IDX {
            continue;
        }
        let p_c = P_C0 * (1.0 + fuel_prob_ign_by_embers(idx_to));
        if uniform01(r, c, base + 3u) > p_c {
            continue;
        }
        let zt = normal01(uniform01(r, c, base + 4u), uniform01(r, c, base + 5u));
        let ttp = exp(log(SPOTTING_TIME_TO_PROPAGATION_MEDIAN) + SPOTTING_TIME_TO_PROPAGATION_LOG_SIGMA * zt);
        let landing = max(u32(ceil(landing_time_s)), 1u);
        let propagation = max(u32(ceil(ttp)), 1u);
        let cand = a + landing + propagation;
        let nk = (cand << 4u) | CODE_SPOT;
        let nidx = r * cells + n;
        let old = atomicMin(&key[nidx], nk);
        if nk < old {
            payload[nidx] = 0u;
            let rt_n = r * n_tiles + tile_of(n);
            atomicMin(&tile_min[next_base() + rt_n], cand);
            if cand < params.next_end {
                queue_next(rt_n);
            }
        }
    }
}

fn tile_of(c: u32) -> u32 {
    let row = c / params.cols;
    let col = c % params.cols;
    return (row >> TILE_SHIFT) * params.tiles_w + (col >> TILE_SHIFT);
}

fn queue_next(rt: u32) {
    if atomicExchange(&member[next_base() + rt], 1u) == 0u {
        let i = atomicAdd(&ctrl[ctrl_next()], 1u);
        list[next_base() + i] = rt;
    }
}

// Per (realization, tile) thread, before a sweep: clear the next list's
// membership and carry the tile minimum forward, except for tiles the sweep
// is about to process, which recompute theirs.
@compute @workgroup_size(256)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rt = gid.x;
    if rt == 0u {
        atomicStore(&ctrl[ctrl_next()], 0u);
    }
    if rt >= n_rt() {
        return;
    }
    atomicStore(&member[next_base() + rt], 0u);
    if params.processing != 0u && atomicLoad(&member[cur_base() + rt]) != 0u {
        atomicStore(&tile_min[next_base() + rt], INF);
    } else {
        atomicStore(&tile_min[next_base() + rt], atomicLoad(&tile_min[cur_base() + rt]));
    }
}

// Per (realization, tile) thread, at a bucket start (after `prepare`): queue
// every tile whose earliest unpushed arrival falls below the bucket end.
@compute @workgroup_size(256)
fn build_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rt = gid.x;
    if rt >= n_rt() {
        return;
    }
    if atomicLoad(&tile_min[next_base() + rt]) < params.next_end {
        queue_next(rt);
    }
}

// After a list has been built: turn its count into 2-D indirect dispatch
// arguments (a dimension is limited to 65535 workgroups).
@compute @workgroup_size(1)
fn finalize_list() {
    let c = atomicLoad(&ctrl[ctrl_next()]);
    atomicStore(&ctrl[ctrl_next() + 1u], min(c, 65535u));
    atomicStore(&ctrl[ctrl_next() + 2u], (c + 65534u) / 65535u);
    atomicStore(&ctrl[ctrl_next() + 3u], 1u);
}

// One workgroup per queued (realization, tile): push every cell of the tile
// whose arrival is below the bucket end and has not been pushed at that
// value; queue the tiles that received arrivals below the next bucket end.
@compute @workgroup_size(256)
fn relax_list(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_tiles = params.tiles_w * params.tiles_h;
    let item = wg.x + 65535u * wg.y;
    if item >= atomicLoad(&ctrl[ctrl_cur()]) {
        return;
    }
    let rt = list[cur_base() + item];
    let r = rt / n_tiles;
    let t = rt % n_tiles;
    let cells = n_cells();
    let tile_row = t / params.tiles_w;
    let tile_col = t % params.tiles_w;
    var local_min = INF;
    let local = lid.x;
    let row = (tile_row << TILE_SHIFT) + local / TILE_SIZE;
    let col = (tile_col << TILE_SHIFT) + local % TILE_SIZE;
    if row < params.rows && col < params.cols {
        let c = row * params.cols + col;
        let idx = r * cells + c;
        let k = atomicLoad(&key[idx]);
        let a = k >> 4u;
        if a != INF_ARRIVAL {
            let pa = atomicLoad(&pushed_at[idx]);
            if a < pa {
                if a >= params.bucket_end {
                    local_min = a;
                } else {
                    let idx_from = fuel_idx_of(c);
                    if idx_from != NO_FUEL_IDX {
                        for (var d = 0u; d < 8u; d = d + 1u) {
                            let e = edge(r, c, d, true);
                            if !e.open {
                                continue;
                            }
                            let cand = a + e.transition;
                            let nk = (cand << 4u) | d;
                            let nidx = r * cells + e.to_cell;
                            let old = atomicMin(&key[nidx], nk);
                            if nk < old {
                                // the carried ROS / FLI of the (so far) winning
                                // edge; two pushes landing in one sweep may leave
                                // the loser's pair, which only affects the mean
                                // ROS / FLI outputs of that cell
                                payload[nidx] = pack2x16float(vec2<f32>(e.ros, e.fli));
                                let rt_n = r * n_tiles + tile_of(e.to_cell);
                                atomicMin(&tile_min[next_base() + rt_n], cand);
                                if cand < params.next_end {
                                    queue_next(rt_n);
                                }
                            }
                        }
                        if params.do_spotting != 0u && fuel_spotting(idx_from) {
                            spot(r, c, a, unpack2x16float(payload[idx]).y, n_tiles, cells);
                        }
                    }
                    atomicMin(&pushed_at[idx], a);
                }
            }
        }
    }
    if local_min != INF {
        atomicMin(&tile_min[next_base() + rt], local_min);
        if local_min < params.next_end {
            queue_next(rt);
        }
    }
}

// After uploading a state (restore / growth): queue nothing, but recompute
// the current half's tile minima from the unpushed cells.
@compute @workgroup_size(256)
fn rebuild_tile_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_state = params.n_reals * n_cells();
    let i = gid.x;
    if i >= n_state {
        return;
    }
    let a = atomicLoad(&key[i]) >> 4u;
    if a == INF_ARRIVAL || a >= atomicLoad(&pushed_at[i]) {
        return;
    }
    let r = i / n_cells();
    let c = i % n_cells();
    atomicMin(&tile_min[cur_base() + r * params.tiles_w * params.tiles_h + tile_of(c)], a);
}

// Fill the state buffers with INF (unburnt / never pushed).
@compute @workgroup_size(256)
fn fill_inf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_state = params.n_reals * n_cells();
    for (var i = gid.x; i < n_state; i = i + 65535u * 256u) {
        atomicStore(&key[i], INF);
        atomicStore(&pushed_at[i], INF);
        payload[i] = 0u;
    }
    if gid.x < 2u * n_rt() {
        atomicStore(&tile_min[gid.x], INF);
    }
}

// Per cell: aggregate over realizations the cells burnt by `fold_time`.
// ROS / FLI carried into a cell are recomputed from the winning edge under
// the window of the source cell's arrival (deterministic, race-free).
@compute @workgroup_size(256)
fn fold_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cells = n_cells();
    let c = gid.x;
    if c >= cells {
        return;
    }
    var count = 0u;
    var arrival_min = INF;
    var arrival_sum = 0.0;
    var ros_sum = 0.0;
    var fli_sum = 0.0;
    var ros_max = 0.0;
    var fli_max = 0.0;
    var spot_recv = 0u;
    for (var r = 0u; r < params.n_reals; r = r + 1u) {
        let k = atomicLoad(&key[r * cells + c]);
        let a = k >> 4u;
        if a == INF_ARRIVAL || a > params.fold_time {
            continue;
        }
        count = count + 1u;
        arrival_min = min(arrival_min, a);
        arrival_sum = arrival_sum + f32(a);
        let code = k & 15u;
        if code < 8u {
            let rf = unpack2x16float(payload[r * cells + c]);
            ros_sum = ros_sum + rf.x;
            fli_sum = fli_sum + rf.y;
            ros_max = max(ros_max, rf.x);
            fli_max = max(fli_max, rf.y);
        } else if code == CODE_SPOT {
            spot_recv = spot_recv + 1u;
        }
    }
    fold[c] = f32(count);
    fold[cells + c] = bitcast<f32>(arrival_min);
    fold[2u * cells + c] = arrival_sum;
    fold[3u * cells + c] = ros_sum;
    fold[4u * cells + c] = fli_sum;
    fold[5u * cells + c] = ros_max;
    fold[6u * cells + c] = fli_max;
    fold[7u * cells + c] = f32(spot_recv);
}

// Debug: edge quantities of cell `fold_time` (reused as the cell index) under
// window 0 for every direction, written as [p, transition, ros, fli, angle,
// uniform, 0, 0] per direction into the fold buffer.
@compute @workgroup_size(1)
fn debug_edges() {
    let c = params.fold_time;
    for (var d = 0u; d < 8u; d = d + 1u) {
        let e = edge(0u, c, d, false);
        let base = d * 8u;
        let row = i32(c / params.cols);
        let col = i32(c % params.cols);
        let row_to = row + NEIGHBOUR_DR[d];
        let col_to = col + NEIGHBOUR_DC[d];
        var p = -1.0;
        if !e.blocked {
            let n = u32(row_to) * params.cols + u32(col_to);
            let dist = neighbour_dist_cells(d) * params.cellsize;
            let angle = neighbour_angle(d);
            let dh = dem_of(n) - dem_of(c);
            p = probability_to_neighbour(angle, dist, wind_dir_of(c), wind_speed_of(c), moisture_of(n), dh, fuel_p0(fuel_idx_of(c), fuel_idx_of(n)));
        }
        fold[base] = p;
        fold[base + 1u] = f32(e.transition);
        fold[base + 2u] = e.ros;
        fold[base + 3u] = e.fli;
        fold[base + 4u] = neighbour_angle(d);
        fold[base + 5u] = uniform01(0u, c, d);
        fold[base + 6u] = wind_dir_of(c);
        fold[base + 7u] = wind_speed_of(c);
    }
}
