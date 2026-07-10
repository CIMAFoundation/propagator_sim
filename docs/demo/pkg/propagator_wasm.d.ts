/* tslint:disable */
/* eslint-disable */

/**
 * Accumulates fuel definitions supplied from JavaScript into a
 * [`FuelSystem`]. The demo builds this from the EU 12-class table defined in
 * `app.html`, keeping the fuel model out of the wasm binary.
 */
export class FuelSystemBuilder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Append one fuel. Units match the config dictionaries (`v0` m/h,
     * `humidity` percent, or `undefined` for a fuel with no live load).
     * `spread_to`/`spread_prob` are parallel arrays giving the base spread
     * probability toward each target fuel id.
     */
    add_fuel(id: number, name: string, v0: number, d0: number, d1: number, hhv: number, humidity: number | null | undefined, spotting: boolean, prob_ign_by_embers: number, burn: boolean, spread_to: Int32Array, spread_prob: Float64Array): void;
    constructor();
}

/**
 * Browser-facing wrapper around the core [`CorePropagator`].
 *
 * Constructed from flat, row-major vegetation (fuel-code) and DEM rasters;
 * weather is supplied as scalars and ignitions as flat `[row, col, …]`
 * pairs. See the module docs for the deliberate limitations.
 */
export class Propagator {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Sides currently pressuring the domain boundary, encoded as a bitmask:
     * north=1, south=2, west=4, east=8. A zero mask means that no specific
     * side was recorded, in which case callers should grow every side.
     */
    boundary_pressure(): number;
    /**
     * Sides with pending front events within `margin` cells, using the same
     * north=1, south=2, west=4, east=8 bitmask as `boundary_pressure`.
     */
    boundary_proximity(margin: number): number;
    /**
     * Mean burned area across realizations, in hectares.
     */
    burned_area_ha(): number;
    /**
     * Grow the domain in place onto the larger `veg`/`dem` anchored at
     * `(origin_row, origin_col)`, preserving all simulation state. The new
     * grid must fully contain the old one and growth to the north/west must
     * be a multiple of [`tile_size`] cells.
     */
    expand(rows: number, cols: number, veg: Int32Array, dem: Float64Array, origin_row: bigint, origin_col: bigint): void;
    /**
     * Fire probability per cell in `[0, 1]`, row-major, `rows * cols` long.
     */
    fire_probability(): Float32Array;
    /**
     * Build a propagator over a `rows x cols` domain.
     *
     * `veg` (fuel codes matching `fuels`) and `dem` (elevation, metres) are
     * row-major and must both hold `rows * cols` entries. `fuels` is the
     * fuel model assembled from JavaScript.
     *
     * By default fires reaching the boundary are ignored, which keeps the
     * demo running when the flame front hits an edge. Pass
     * `halt_on_boundary = true` to have [`Propagator::step`] report boundary
     * hits instead, so the caller can grow the domain with
     * [`Propagator::expand`] and continue.
     */
    constructor(rows: number, cols: number, veg: Int32Array, dem: Float64Array, fuels: FuelSystemBuilder, realizations: number, seed: number, do_spotting: boolean, cellsize: number, ros_model: string, moisture_model: string, halt_on_boundary?: boolean | null);
    /**
     * Next event time (scheduler or front), or `-1` when the run is idle.
     */
    next_time(): bigint;
    /**
     * All per-cell output variables packed into one row-major buffer, so a
     * single [`Propagator::get_output`] call feeds every layer the browser
     * wants to render or animate.
     *
     * Layout is `OUTPUT_VARIABLE_COUNT` grids of `rows * cols` values each,
     * concatenated in this order (also see [`output_variable_count`]):
     *
     * 0. fire probability, `[0, 1]`
     * 1. time of arrival (first), seconds — `0` where never burnt
     * 2. mean rate of spread, m/min — `NaN` where never burnt
     * 3. mean fireline intensity, kW/m — `NaN` where never burnt
     * 4. spotting generation probability, `[0, 1]`
     * 5. spotting receiving probability, `[0, 1]`
     */
    output_snapshot(): Float32Array;
    /**
     * Enqueue suppression-action fields at `time`, merged into any weather /
     * ignition event already scheduled there.
     *
     * Actions are supplied as sparse cell lists so the caller never marshals a
     * full grid:
     *
     * * `moisture_cells` — flat `[row0, col0, row1, col1, …]` cells wetted by
     *   a moisture action (waterline / canadair / helicopter), each with the
     *   matching **percent** increment in `moisture_values`. The core adds
     *   these on top of the base moisture, clamps to `[0, 100]%`, and decays
     *   them over time.
     * * `veg_cells` — flat `[row, col, …]` cells a fuel action (heavy) turns
     *   into the `veg_fuel` non-vegetated fuel id, i.e. a fire break.
     *
     * Pass empty arrays for whichever action kind is absent; a call with no
     * cells at all is a no-op.
     */
    set_action_fields(time: bigint, moisture_cells: Int32Array, moisture_values: Float32Array, veg_cells: Int32Array, veg_fuel: number): void;
    /**
     * Enqueue boundary conditions at `time` (external units: percent
     * moisture, degrees wind direction from north, km/h wind speed).
     *
     * `ignitions` is a flat list of `[row0, col0, row1, col1, …]`; pass an
     * empty array to leave the ignition set unchanged.
     */
    set_boundary_conditions(time: bigint, moisture: number, wind_dir: number, wind_speed: number, ignitions: Int32Array): void;
    /**
     * Propagate for `seconds` of simulation time, applying scheduled events.
     *
     * Returns `true` when the run was constructed with
     * `halt_on_boundary = true` and at least one realization was suspended
     * at the domain boundary: grow the domain with [`Propagator::expand`]
     * and step again to resume it — nothing is lost.
     */
    step(seconds: bigint): boolean;
    /**
     * Number of grid columns.
     */
    readonly cols: number;
    /**
     * See [`Propagator::origin_row`].
     */
    readonly origin_col: bigint;
    /**
     * Grid origin (row, col) in global cell coordinates. Growth to the
     * north/west moves the origin to smaller values.
     */
    readonly origin_row: bigint;
    /**
     * Number of stochastic realizations.
     */
    readonly realizations: number;
    /**
     * Number of grid rows.
     */
    readonly rows: number;
    /**
     * Current simulation time, seconds from start.
     */
    readonly time: bigint;
}

/**
 * Number of per-cell variables packed by [`Propagator::output_snapshot`].
 */
export function output_variable_count(): number;

/**
 * Tile size in cells: north/west domain growth must be a multiple of this.
 */
export function tile_size(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_fuelsystembuilder_free: (a: number, b: number) => void;
    readonly __wbg_propagator_free: (a: number, b: number) => void;
    readonly fuelsystembuilder_add_fuel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
    readonly fuelsystembuilder_new: () => number;
    readonly output_variable_count: () => number;
    readonly propagator_boundary_pressure: (a: number) => number;
    readonly propagator_boundary_proximity: (a: number, b: number) => number;
    readonly propagator_burned_area_ha: (a: number) => [number, number, number];
    readonly propagator_cols: (a: number) => number;
    readonly propagator_expand: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint, i: bigint) => [number, number];
    readonly propagator_fire_probability: (a: number) => [number, number, number, number];
    readonly propagator_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number) => [number, number, number];
    readonly propagator_next_time: (a: number) => bigint;
    readonly propagator_origin_col: (a: number) => bigint;
    readonly propagator_origin_row: (a: number) => bigint;
    readonly propagator_output_snapshot: (a: number) => [number, number, number, number];
    readonly propagator_realizations: (a: number) => number;
    readonly propagator_rows: (a: number) => number;
    readonly propagator_set_action_fields: (a: number, b: bigint, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly propagator_set_boundary_conditions: (a: number, b: bigint, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly propagator_step: (a: number, b: bigint) => [number, number, number];
    readonly propagator_time: (a: number) => bigint;
    readonly tile_size: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
