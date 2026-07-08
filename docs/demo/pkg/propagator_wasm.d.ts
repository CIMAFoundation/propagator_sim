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
     * Mean burned area across realizations, in hectares.
     */
    burned_area_ha(): number;
    /**
     * Fire probability per cell in `[0, 1]`, row-major, `rows * cols` long.
     */
    fire_probability(): Float32Array;
    /**
     * Build a propagator over a `rows x cols` domain.
     *
     * `veg` (fuel codes matching `fuels`) and `dem` (elevation, metres) are
     * row-major and must both hold `rows * cols` entries. `fuels` is the
     * fuel model assembled from JavaScript. Fires reaching the boundary are
     * ignored rather than raised, which keeps the demo running when the
     * flame front hits an edge.
     */
    constructor(rows: number, cols: number, veg: Int32Array, dem: Float64Array, fuels: FuelSystemBuilder, realizations: number, seed: number, do_spotting: boolean, cellsize: number, ros_model: string, moisture_model: string);
    /**
     * Next event time (scheduler or front), or `-1` when the run is idle.
     */
    next_time(): bigint;
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
     */
    step(seconds: bigint): void;
    /**
     * Number of grid columns.
     */
    readonly cols: number;
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

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_fuelsystembuilder_free: (a: number, b: number) => void;
    readonly __wbg_propagator_free: (a: number, b: number) => void;
    readonly fuelsystembuilder_add_fuel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
    readonly fuelsystembuilder_new: () => number;
    readonly propagator_burned_area_ha: (a: number) => [number, number, number];
    readonly propagator_cols: (a: number) => number;
    readonly propagator_fire_probability: (a: number) => [number, number, number, number];
    readonly propagator_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number, number];
    readonly propagator_next_time: (a: number) => bigint;
    readonly propagator_realizations: (a: number) => number;
    readonly propagator_rows: (a: number) => number;
    readonly propagator_set_boundary_conditions: (a: number, b: bigint, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly propagator_step: (a: number, b: bigint) => [number, number];
    readonly propagator_time: (a: number) => bigint;
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
