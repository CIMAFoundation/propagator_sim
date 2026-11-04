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
// cost one load per sweep.

const INF: u32 = 0xFFFFFFFFu;
const INF_ARRIVAL: u32 = 0x0FFFFFFFu;
const CODE_IGNITION: u32 = 8u;
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
    n_windows: u32,
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
}

struct Window {
    t_start: u32,
    wind_dir: f32,
    wind_speed: f32,
    moisture: f32,
}

// WebGPU allows 10 storage buffers per stage, so everything is packed into
// 8: static inputs in one u32 array (fuel index, dem bits, fuel table bits,
// weather windows), the ping-pong pairs in single buffers with two halves
// selected by `params.phase`, and the worklist counters together with the
// indirect dispatch arguments.
@group(0) @binding(0) var<uniform> params: Params;
// [fuel_idx (N)] [dem bits (N)] [fuel table bits: 5 per fuel, then n x n
// spread probabilities] [windows: t_start, wind_dir, wind_speed, moisture
// bits, 4 words each]
@group(0) @binding(1) var<storage, read> inputs: array<u32>;
@group(0) @binding(2) var<storage, read_write> key: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pushed_at: array<atomic<u32>>;
// per (realization, tile) earliest unpushed arrival, two halves
@group(0) @binding(4) var<storage, read_write> tile_min: array<atomic<u32>>;
// fold accumulators per cell: count, arrival_min, arrival_sum (u32 seconds),
// ros_sum, fli_sum (f32), ros_max, fli_max (f32); 7 x N
@group(0) @binding(5) var<storage, read_write> fold: array<f32>;
// worklists of (realization, tile) indices, two halves
@group(0) @binding(6) var<storage, read_write> list: array<u32>;
// membership flags of the worklists, two halves
@group(0) @binding(7) var<storage, read_write> member: array<atomic<u32>>;
// control words, 4 per half: [count, dispatch x, dispatch y, dispatch z]
@group(0) @binding(8) var<storage, read_write> ctrl: array<atomic<u32>>;

fn n_cells() -> u32 { return params.rows * params.cols; }
fn n_rt() -> u32 { return params.n_reals * params.tiles_w * params.tiles_h; }
fn cur_base() -> u32 { return params.phase * n_rt(); }
fn next_base() -> u32 { return (1u - params.phase) * n_rt(); }
fn ctrl_cur() -> u32 { return params.phase * 4u; }
fn ctrl_next() -> u32 { return (1u - params.phase) * 4u; }

fn fuel_idx_of(c: u32) -> u32 { return inputs[c]; }
fn dem_of(c: u32) -> f32 { return bitcast<f32>(inputs[n_cells() + c]); }
fn fuel_table(i: u32) -> f32 { return bitcast<f32>(inputs[2u * n_cells() + i]); }
fn windows_base() -> u32 {
    return 2u * n_cells() + params.n_fuels * 5u + params.n_fuels * params.n_fuels;
}
fn window_at(w: u32) -> Window {
    let b = windows_base() + 4u * w;
    var win: Window;
    win.t_start = inputs[b];
    win.wind_dir = bitcast<f32>(inputs[b + 1u]);
    win.wind_speed = bitcast<f32>(inputs[b + 2u]);
    win.moisture = bitcast<f32>(inputs[b + 3u]);
    return win;
}

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

fn fuel_v0(i: u32) -> f32 { return fuel_table(i * 5u + 0u); }
fn fuel_d0(i: u32) -> f32 { return fuel_table(i * 5u + 1u); }
fn fuel_d1(i: u32) -> f32 { return fuel_table(i * 5u + 2u); }
fn fuel_hhv(i: u32) -> f32 { return fuel_table(i * 5u + 3u); }
fn fuel_humidity(i: u32) -> f32 { return fuel_table(i * 5u + 4u); }
fn fuel_p0(src_idx: u32, to: u32) -> f32 {
    return fuel_table(params.n_fuels * 5u + src_idx * params.n_fuels + to);
}

// lowbias32 hash -> uniform in [0, 1)
fn uniform01(r: u32, cell: u32, d: u32) -> f32 {
    var x = params.seed ^ (r * 0x9E3779B9u) ^ (cell * 0x85EBCA6Bu) ^ (d * 0xC2B2AE35u);
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

// window index whose start is the latest <= t
fn window_of(t: u32) -> u32 {
    var w = 0u;
    for (var i = 1u; i < params.n_windows; i = i + 1u) {
        if window_at(i).t_start <= t { w = i; }
    }
    return w;
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
// under window `w`; `draw` selects whether to apply the Bernoulli draw
fn edge(r: u32, c: u32, d: u32, w: u32, draw: bool) -> Edge {
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
    let win = window_at(w);
    let dist = neighbour_dist_cells(d) * params.cellsize;
    let angle = neighbour_angle(d);
    let dh = dem_of(n) - dem_of(c);
    let moist = win.moisture;
    if draw {
        let p = probability_to_neighbour(angle, dist, win.wind_dir, win.wind_speed, moist, dh, fuel_p0(idx_from, idx_to));
        if p <= uniform01(r, c, d) {
            return e;
        }
    }
    e.open = true;
    let tr = p_time(fuel_v0(idx_from), dh, angle, dist, moist, win.wind_dir, win.wind_speed);
    e.transition = max(u32(tr.x), 1u);
    e.ros = tr.y;
    let lhv_dead = lhv_fuel(fuel_hhv(idx_to), moist);
    let lhv_canopy = lhv_fuel(fuel_hhv(idx_to), fuel_humidity(idx_to));
    e.fli = (tr.y / 60.0) * (lhv_dead * fuel_d0(idx_to) + lhv_canopy * fuel_d1(idx_to));
    return e;
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
                    if fuel_idx_of(c) != NO_FUEL_IDX {
                        let w = window_of(a);
                        for (var d = 0u; d < 8u; d = d + 1u) {
                            let e = edge(r, c, d, w, true);
                            if !e.open {
                                continue;
                            }
                            let cand = a + e.transition;
                            let nk = (cand << 4u) | d;
                            let nidx = r * cells + e.to_cell;
                            let old = atomicMin(&key[nidx], nk);
                            if nk < old {
                                let rt_n = r * n_tiles + tile_of(e.to_cell);
                                atomicMin(&tile_min[next_base() + rt_n], cand);
                                if cand < params.next_end {
                                    queue_next(rt_n);
                                }
                            }
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

// Fill the state buffers with INF (unburnt / never pushed).
@compute @workgroup_size(256)
fn fill_inf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_state = params.n_reals * n_cells();
    for (var i = gid.x; i < n_state; i = i + 65535u * 256u) {
        atomicStore(&key[i], INF);
        atomicStore(&pushed_at[i], INF);
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
    let row = i32(c / params.cols);
    let col = i32(c % params.cols);
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
            // source is the neighbour opposite to the incoming direction
            let src_row = row - NEIGHBOUR_DR[code];
            let src_col = col - NEIGHBOUR_DC[code];
            let src = u32(src_row) * params.cols + u32(src_col);
            let a_src = atomicLoad(&key[r * cells + src]) >> 4u;
            let e = edge(r, src, code, window_of(a_src), false);
            ros_sum = ros_sum + e.ros;
            fli_sum = fli_sum + e.fli;
            ros_max = max(ros_max, e.ros);
            fli_max = max(fli_max, e.fli);
        }
    }
    fold[c] = f32(count);
    fold[cells + c] = bitcast<f32>(arrival_min);
    fold[2u * cells + c] = arrival_sum;
    fold[3u * cells + c] = ros_sum;
    fold[4u * cells + c] = fli_sum;
    fold[5u * cells + c] = ros_max;
    fold[6u * cells + c] = fli_max;
}

// Debug: edge quantities of cell `fold_time` (reused as the cell index) under
// window 0 for every direction, written as [p, transition, ros, fli, angle,
// uniform, 0, 0] per direction into the fold buffer.
@compute @workgroup_size(1)
fn debug_edges() {
    let c = params.fold_time;
    for (var d = 0u; d < 8u; d = d + 1u) {
        let e = edge(0u, c, d, 0u, false);
        let base = d * 8u;
        let row = i32(c / params.cols);
        let col = i32(c % params.cols);
        let row_to = row + NEIGHBOUR_DR[d];
        let col_to = col + NEIGHBOUR_DC[d];
        var p = -1.0;
        let win = window_at(0u);
        if !e.blocked {
            let n = u32(row_to) * params.cols + u32(col_to);
            let dist = neighbour_dist_cells(d) * params.cellsize;
            let angle = neighbour_angle(d);
            let dh = dem_of(n) - dem_of(c);
            p = probability_to_neighbour(angle, dist, win.wind_dir, win.wind_speed, win.moisture, dh, fuel_p0(fuel_idx_of(c), fuel_idx_of(n)));
        }
        fold[base] = p;
        fold[base + 1u] = f32(e.transition);
        fold[base + 2u] = e.ros;
        fold[base + 3u] = e.fli;
        fold[base + 4u] = neighbour_angle(d);
        fold[base + 5u] = uniform01(0u, c, d);
        fold[base + 6u] = win.wind_dir;
        fold[base + 7u] = win.wind_speed;
    }
}
