/* @ts-self-types="./propagator_wasm.d.ts" */

/**
 * Accumulates fuel definitions supplied from JavaScript into a
 * [`FuelSystem`]. The demo builds this from the EU 12-class table defined in
 * `app.html`, keeping the fuel model out of the wasm binary.
 */
export class FuelSystemBuilder {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FuelSystemBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fuelsystembuilder_free(ptr, 0);
    }
    /**
     * Append one fuel. Units match the config dictionaries (`v0` m/h,
     * `humidity` percent, or `undefined` for a fuel with no live load).
     * `spread_to`/`spread_prob` are parallel arrays giving the base spread
     * probability toward each target fuel id.
     * @param {number} id
     * @param {string} name
     * @param {number} v0
     * @param {number} d0
     * @param {number} d1
     * @param {number} hhv
     * @param {number | null | undefined} humidity
     * @param {boolean} spotting
     * @param {number} prob_ign_by_embers
     * @param {boolean} burn
     * @param {Int32Array | number[]} spread_to
     * @param {Float64Array | number[]} spread_prob
     */
    add_fuel(id, name, v0, d0, d1, hhv, humidity, spotting, prob_ign_by_embers, burn, spread_to, spread_prob) {
        const val4 = +id;
        const val5 = +v0;
        const val6 = +d0;
        const val7 = +d1;
        const val8 = +hhv;
        const val9 = +(isLikeNone(humidity) ? 0 : humidity);
        const val10 = +spotting;
        const val11 = +prob_ign_by_embers;
        const val12 = +burn;
        let ptr0, len0, ptr1, len1, ptr2, len2;
        try {
            ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArray32ToWasm0(spread_to, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passArrayF64ToWasm0(spread_prob, wasm.__wbindgen_malloc);
            len2 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 8, 8); }
            if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 4, 4); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 1, 1); }
            throw e;
        }
        const ret = wasm.fuelsystembuilder_add_fuel(this.__wbg_ptr, val4, ptr0, len0, val5, val6, val7, val8, !isLikeNone(humidity), val9, val10, val11, val12, ptr1, len1, ptr2, len2);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    constructor() {
        const ret = wasm.fuelsystembuilder_new();
        this.__wbg_ptr = ret;
        FuelSystemBuilderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) FuelSystemBuilder.prototype[Symbol.dispose] = FuelSystemBuilder.prototype.free;

/**
 * WebGPU execution backend (feature `gpu`): the same scenario inputs as
 * [`Propagator`], run by `propagator-gpu`. Methods that touch the GPU are
 * async (they await buffer mappings). Scope: contact spread with scalar
 * weather and point ignitions; no spotting, actions or growth.
 */
export class GpuPropagator {
    static __wrap(ptr) {
        const obj = Object.create(GpuPropagator.prototype);
        obj.__wbg_ptr = ptr;
        GpuPropagatorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GpuPropagatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_gpupropagator_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.gpupropagator_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Open the GPU and upload the scenario (async; rejects when WebGPU is
     * unavailable).
     * @param {number} rows
     * @param {number} cols
     * @param {Int32Array | number[]} veg
     * @param {Float64Array | number[]} dem
     * @param {FuelSystemBuilder} fuels
     * @param {number} realizations
     * @param {number} seed
     * @param {number} cellsize
     * @param {string} ros_model
     * @param {string} moisture_model
     * @returns {Promise<GpuPropagator>}
     */
    static create(rows, cols, veg, dem, fuels, realizations, seed, cellsize, ros_model, moisture_model) {
        _assertClass(fuels, FuelSystemBuilder);
        const val5 = +rows;
        const val6 = +cols;
        const val7 = +realizations;
        const val8 = +seed;
        const val9 = +cellsize;
        let ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3;
        try {
            ptr0 = passArray32ToWasm0(veg, wasm.__wbindgen_malloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArrayF64ToWasm0(dem, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passStringToWasm0(ros_model, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len2 = WASM_VECTOR_LEN;
            ptr3 = passStringToWasm0(moisture_model, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len3 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr3) { wasm.__wbindgen_free(ptr3, len3 * 1, 1); }
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 1, 1); }
            if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 8, 8); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
            throw e;
        }
        const ret = wasm.gpupropagator_create(val5, val6, ptr0, len0, ptr1, len1, fuels.__wbg_ptr, val7, val8, val9, ptr2, len2, ptr3, len3);
        return ret;
    }
    /**
     * @returns {number}
     */
    get rows() {
        const ret = wasm.gpupropagator_rows(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Weather window starting at `time` plus point ignitions (flat
     * `[row, col, ...]`), as [`Propagator::set_boundary_conditions`].
     * @param {bigint} time
     * @param {number} moisture
     * @param {number} wind_dir
     * @param {number} wind_speed
     * @param {Int32Array | number[]} ignitions
     */
    set_boundary_conditions(time, moisture, wind_dir, wind_speed, ignitions) {
        const val2 = BigInt.asIntN(64, time);
        const val3 = +moisture;
        const val4 = +wind_dir;
        const val5 = +wind_speed;
        const ptr0 = passArray32ToWasm0(ignitions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.gpupropagator_set_boundary_conditions(this.__wbg_ptr, val2, val3, val4, val5, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * `[fire probability (rows*cols) ..., area_mean_ha]` folded on the GPU
     * (async): one readback feeds both the map and the readout.
     * @returns {Promise<Float32Array>}
     */
    snapshot() {
        const ret = wasm.gpupropagator_snapshot(this.__wbg_ptr);
        return ret;
    }
    /**
     * Advance by `seconds` of simulation time (async).
     * @param {bigint} seconds
     * @returns {Promise<void>}
     */
    step(seconds) {
        const ret = wasm.gpupropagator_step(this.__wbg_ptr, seconds);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get time() {
        const ret = wasm.gpupropagator_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) GpuPropagator.prototype[Symbol.dispose] = GpuPropagator.prototype.free;

/**
 * Browser-facing wrapper around the core [`CorePropagator`].
 *
 * Constructed from flat, row-major vegetation (fuel-code) and DEM rasters;
 * weather is supplied as scalars and ignitions as flat `[row, col, …]`
 * pairs. See the module docs for the deliberate limitations.
 */
export class Propagator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PropagatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_propagator_free(ptr, 0);
    }
    /**
     * Sides currently pressuring the domain boundary, encoded as a bitmask:
     * north=1, south=2, west=4, east=8. A zero mask means that no specific
     * side was recorded, in which case callers should grow every side.
     * @returns {number}
     */
    boundary_pressure() {
        const ret = wasm.propagator_boundary_pressure(this.__wbg_ptr);
        return ret;
    }
    /**
     * Sides with pending front events within `margin` cells, using the same
     * north=1, south=2, west=4, east=8 bitmask as `boundary_pressure`.
     * @param {number} margin
     * @returns {number}
     */
    boundary_proximity(margin) {
        const ret = wasm.propagator_boundary_proximity(this.__wbg_ptr, margin);
        return ret;
    }
    /**
     * Mean burned area across realizations, in hectares.
     * @returns {number}
     */
    burned_area_ha() {
        const ret = wasm.propagator_burned_area_ha(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Number of grid columns.
     * @returns {number}
     */
    get cols() {
        const ret = wasm.propagator_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Bytes held by the shared per-cell edge cache (spread quantities
     * reused across realizations).
     * @returns {number}
     */
    edge_cache_bytes() {
        const ret = wasm.propagator_edge_cache_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Grow the domain in place onto the larger `veg`/`dem` anchored at
     * `(origin_row, origin_col)`, preserving all simulation state. The new
     * grid must fully contain the old one and growth to the north/west must
     * be a multiple of [`tile_size`] cells.
     * @param {number} rows
     * @param {number} cols
     * @param {Int32Array | number[]} veg
     * @param {Float64Array | number[]} dem
     * @param {bigint} origin_row
     * @param {bigint} origin_col
     */
    expand(rows, cols, veg, dem, origin_row, origin_col) {
        const val3 = +rows;
        const val4 = +cols;
        const val5 = BigInt.asIntN(64, origin_row);
        const val6 = BigInt.asIntN(64, origin_col);
        let ptr0, len0, ptr1, len1;
        try {
            ptr0 = passArray32ToWasm0(veg, wasm.__wbindgen_malloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArrayF64ToWasm0(dem, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 8, 8); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
            throw e;
        }
        const ret = wasm.propagator_expand(this.__wbg_ptr, val3, val4, ptr0, len0, ptr1, len1, val5, val6);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Fire probability per cell in `[0, 1]`, row-major, `rows * cols` long.
     * @returns {Float32Array}
     */
    fire_probability() {
        const ret = wasm.propagator_fire_probability(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
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
     * @param {number} rows
     * @param {number} cols
     * @param {Int32Array | number[]} veg
     * @param {Float64Array | number[]} dem
     * @param {FuelSystemBuilder} fuels
     * @param {number} realizations
     * @param {number} seed
     * @param {boolean} do_spotting
     * @param {number} cellsize
     * @param {string} ros_model
     * @param {string} moisture_model
     * @param {boolean | null} [halt_on_boundary]
     * @param {number | null} [threads]
     */
    constructor(rows, cols, veg, dem, fuels, realizations, seed, do_spotting, cellsize, ros_model, moisture_model, halt_on_boundary, threads) {
        _assertClass(fuels, FuelSystemBuilder);
        const val5 = +rows;
        const val6 = +cols;
        const val7 = +realizations;
        const val8 = +seed;
        const val9 = +do_spotting;
        const val10 = +cellsize;
        const val11 = +(isLikeNone(threads) ? Number.MAX_SAFE_INTEGER : (threads) >>> 0);
        let ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3;
        try {
            ptr0 = passArray32ToWasm0(veg, wasm.__wbindgen_malloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArrayF64ToWasm0(dem, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passStringToWasm0(ros_model, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len2 = WASM_VECTOR_LEN;
            ptr3 = passStringToWasm0(moisture_model, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len3 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr3) { wasm.__wbindgen_free(ptr3, len3 * 1, 1); }
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 1, 1); }
            if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 8, 8); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
            throw e;
        }
        const ret = wasm.propagator_new(val5, val6, ptr0, len0, ptr1, len1, fuels.__wbg_ptr, val7, val8, val9, val10, ptr2, len2, ptr3, len3, isLikeNone(halt_on_boundary) ? 0xFFFFFF : halt_on_boundary ? 1 : 0, val11);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        PropagatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Next event time (scheduler or front), or `-1` when the run is idle.
     * @returns {bigint}
     */
    next_time() {
        const ret = wasm.propagator_next_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * See [`Propagator::origin_row`].
     * @returns {bigint}
     */
    get origin_col() {
        const ret = wasm.propagator_origin_col(this.__wbg_ptr);
        return ret;
    }
    /**
     * Grid origin (row, col) in global cell coordinates. Growth to the
     * north/west moves the origin to smaller values.
     * @returns {bigint}
     */
    get origin_row() {
        const ret = wasm.propagator_origin_row(this.__wbg_ptr);
        return ret;
    }
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
     * @returns {Float32Array}
     */
    output_snapshot() {
        const ret = wasm.propagator_output_snapshot(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Number of stochastic realizations.
     * @returns {number}
     */
    get realizations() {
        const ret = wasm.propagator_realizations(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of grid rows.
     * @returns {number}
     */
    get rows() {
        const ret = wasm.propagator_rows(this.__wbg_ptr);
        return ret >>> 0;
    }
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
     * @param {bigint} time
     * @param {Int32Array | number[]} moisture_cells
     * @param {Float32Array | number[]} moisture_values
     * @param {Int32Array | number[]} veg_cells
     * @param {number} veg_fuel
     */
    set_action_fields(time, moisture_cells, moisture_values, veg_cells, veg_fuel) {
        const val4 = BigInt.asIntN(64, time);
        const val5 = +veg_fuel;
        let ptr0, len0, ptr1, len1, ptr2, len2;
        try {
            ptr0 = passArray32ToWasm0(moisture_cells, wasm.__wbindgen_malloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArrayF32ToWasm0(moisture_values, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passArray32ToWasm0(veg_cells, wasm.__wbindgen_malloc);
            len2 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 4, 4); }
            if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 4, 4); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
            throw e;
        }
        const ret = wasm.propagator_set_action_fields(this.__wbg_ptr, val4, ptr0, len0, ptr1, len1, ptr2, len2, val5);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Enqueue boundary conditions at `time` (external units: percent
     * moisture, degrees wind direction from north, km/h wind speed).
     *
     * `ignitions` is a flat list of `[row0, col0, row1, col1, …]`; pass an
     * empty array to leave the ignition set unchanged.
     * @param {bigint} time
     * @param {number} moisture
     * @param {number} wind_dir
     * @param {number} wind_speed
     * @param {Int32Array | number[]} ignitions
     */
    set_boundary_conditions(time, moisture, wind_dir, wind_speed, ignitions) {
        const val2 = BigInt.asIntN(64, time);
        const val3 = +moisture;
        const val4 = +wind_dir;
        const val5 = +wind_speed;
        const ptr0 = passArray32ToWasm0(ignitions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.propagator_set_boundary_conditions(this.__wbg_ptr, val2, val3, val4, val5, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Turn boundary halting on or off mid-run.
     *
     * Call with `false` when the domain can no longer grow (a cap reached, a
     * failed fetch): the boundary events the core has been suspending are
     * popped on the next [`Propagator::step`], so the fire clips at the edge
     * and the rest of the front keeps burning. Leaving it on instead freezes
     * the realization, because a suspended boundary event is the earliest
     * event in its heap and nothing behind it can advance.
     * @param {boolean} halt
     */
    set_halt_on_boundary(halt) {
        wasm.propagator_set_halt_on_boundary(this.__wbg_ptr, halt);
    }
    /**
     * Propagate for `seconds` of simulation time, applying scheduled events.
     *
     * Returns `true` when the run was constructed with
     * `halt_on_boundary = true` and at least one realization was suspended
     * at the domain boundary: grow the domain with [`Propagator::expand`]
     * and step again to resume it — nothing is lost.
     * @param {bigint} seconds
     * @returns {boolean}
     */
    step(seconds) {
        const ret = wasm.propagator_step(this.__wbg_ptr, seconds);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Current simulation time, seconds from start.
     * @returns {bigint}
     */
    get time() {
        const ret = wasm.propagator_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Propagator.prototype[Symbol.dispose] = Propagator.prototype.free;

/**
 * Extract isochrones through [`propagator_geo::extract_isochrone`].
 *
 * `values` is a row-major `Float32Array` with exactly `rows * cols` entries.
 * `transform` is a six-value `Float64Array` in GDAL affine order
 * `(a, b, c, d, e, f)`. The returned JavaScript array contains
 * `{ threshold, lines }` objects, where `lines` has GeoJSON MultiLineString
 * coordinate nesting. Omitted optional arguments use the Python defaults.
 * Filtering, topology, smoothing, omitted-threshold, and empty-geometry
 * semantics are defined by `propagator-geo`.
 * @param {Float32Array | number[]} values
 * @param {number} rows
 * @param {number} cols
 * @param {Float64Array | number[]} transform
 * @param {Float64Array | number[] | null} [thresholds]
 * @param {number | null} [med_filt_val]
 * @param {number | null} [min_length]
 * @param {number | null} [smooth_sigma]
 * @param {number | null} [simp_fact]
 * @returns {any}
 */
export function extract_isochrone(values, rows, cols, transform, thresholds, med_filt_val, min_length, smooth_sigma, simp_fact) {
    const val4 = +rows;
    const val5 = +cols;
    const val6 = +(isLikeNone(med_filt_val) ? Number.MAX_SAFE_INTEGER : (med_filt_val) >>> 0);
    const val7 = +(isLikeNone(min_length) ? 0 : min_length);
    const val8 = +(isLikeNone(smooth_sigma) ? 0 : smooth_sigma);
    const val9 = +(isLikeNone(simp_fact) ? 0 : simp_fact);
    let ptr0, len0, ptr1, len1, ptr2, len2;
    try {
        ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        len0 = WASM_VECTOR_LEN;
        ptr1 = passArrayF64ToWasm0(transform, wasm.__wbindgen_malloc);
        len1 = WASM_VECTOR_LEN;
        ptr2 = isLikeNone(thresholds) ? 0 : passArrayF64ToWasm0(thresholds, wasm.__wbindgen_malloc);
        len2 = WASM_VECTOR_LEN;
    } catch (e) {
        if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 8, 8); }
        if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 8, 8); }
        if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
        throw e;
    }
    const ret = wasm.extract_isochrone(ptr0, len0, val4, val5, ptr1, len1, ptr2, len2, val6, !isLikeNone(min_length), val7, !isLikeNone(smooth_sigma), val8, !isLikeNone(simp_fact), val9);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * `true` when this bundle was built with the `threads` feature, i.e. it
 * exports `initThreadPool` and runs realizations on a worker pool.
 * @returns {boolean}
 */
export function has_threads() {
    const ret = wasm.has_threads();
    return ret !== 0;
}

/**
 * Number of per-cell variables packed by [`Propagator::output_snapshot`].
 * @returns {number}
 */
export function output_variable_count() {
    const ret = wasm.output_variable_count();
    return ret >>> 0;
}

/**
 * Tile size in cells: north/west domain growth must be a multiple of this.
 * @returns {number}
 */
export function tile_size() {
    const ret = wasm.tile_size();
    return ret >>> 0;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_208446807d0d1e30: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 1, len1, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, ptr1, true);
        },
        __wbg_Window_2805651c516cb7aa: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_174390815d0a4893: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_f249a2a76caa9a55: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 1, len1, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_741ff2bb612836f5: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_c22311b1cc228e3a: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_string_fc456fee5d3317de: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_be5b34249ff1fc15: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_4c88d8bf264fd984: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_09736879a3f67559: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_b9a325184985e6ff: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_call_a410eb4b59ed5965: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_configure_8e04ab1a867d31e1: function() { return handleError(function (arg0, arg1) {
            arg0.configure(arg1);
        }, arguments); },
        __wbg_copyBufferToBuffer_0c93ff5b22a585e9: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        }, arguments); },
        __wbg_copyBufferToBuffer_d04fe4341d6e0f6a: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4);
        }, arguments); },
        __wbg_createBindGroupLayout_5699617270d84042: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_f539b26ca341308f: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_b8c5b520d2c822b8: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_3352d1ffc36c6fc0: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_224fa2618d9948a0: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_6eab52c327118937: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createShaderModule_cefa51336cb288ae: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_dispatchWorkgroupsIndirect_efe17f2da3a39a3f: function(arg0, arg1, arg2) {
            arg0.dispatchWorkgroupsIndirect(arg1, arg2);
        },
        __wbg_dispatchWorkgroups_56b943172790add0: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_end_82fd0c12ba185009: function(arg0) {
            arg0.end();
        },
        __wbg_finish_09ec094c10f41e7b: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_ec1c191f66a895b1: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_getContext_1938edb9a972f63d: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getMappedRange_28e816a763520e01: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_gpu_514705c6dd1f0dab: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_gpupropagator_new: function(arg0) {
            const ret = GpuPropagator.__wrap(arg0);
            return ret;
        },
        __wbg_label_fceb40a43a358e62: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 1, len1, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, ptr1, true);
        },
        __wbg_length_186163212ffe79c3: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_4e444eb8c2e61c45: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_mapAsync_b0597127f5037286: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroupsPlusVertexBuffers_1face4afb5363937: function(arg0) {
            const ret = arg0.maxBindGroupsPlusVertexBuffers;
            return ret;
        },
        __wbg_maxBindGroups_a0b7e55762f095b9: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_ec9067be38d6ec72: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_a9ce15e541c64cd2: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_44089e129e1a2a1d: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_6bd8156394ba618c: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_7b086f15ef868934: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_8bbe505e0ac8dc18: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_7228ed3935eecc31: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_6092b56a967f6f8d: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_4773e3bd3936df36: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_79222bf8e8a90fbd: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_3ab5762b1b2c7fc2: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_722da31d0de1c256: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderVariables_821096d4b3020715: function(arg0) {
            const ret = arg0.maxInterStageShaderVariables;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_c6aea9e05c0a02ca: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_3aa30f6c002d3a99: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_84a2c2433627fca2: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_25b96748df28dc20: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_53ed734f26e6c1ad: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_9ec487f1ce102992: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_faef91b2e9fa6740: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_3a923eb2df4d9124: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_943321fe5945464e: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_21e89ff761d576db: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_19a0bfb0189f7254: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_49aef6c51d07a280: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_f68c4029e6f047ec: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_ce9f15a623f9aac7: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_minStorageBufferOffsetAlignment_529f1f52e61188fc: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_519e26287e1996a3: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_9641ab32fa612dd1: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_c17f443a6fe5d38d: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_2bfd0f45e60518b1: function() { return handleError(function (arg0, arg1) {
            const ret = new OffscreenCanvas(arg0 >>> 0, arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_new_37eb45aaf5eeb820: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_cd4348290afae586: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_6b60b99917da27d6: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_e09da9dae6ee5935: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen__convert__closures_____invoke__he21c9032761183b0(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_3cd32055a417f095: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_onSubmittedWorkDone_1190213cee1ecf7e: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_prototypesetcall_13162965d1fe7aa3: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_queueMicrotask_1e8503a8a690f0a3: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_4a1a4d8dd332373b: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_e250e2ebc49b39e0: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_a539af006419f2e9: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_5cb8a582e55d08cb: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_dbf78ab4942478b0: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_4aa88516c8ab99c4: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setBindGroup_c83391351ce68826: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_5146f7b8d2b4af5c: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_118fefd528ab0a79: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_2793479888329cb6: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_access_b0a6fbebfbe53242: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_beginning_of_pass_write_index_c49071ff9c074e31: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_383f681565a30d8a: function(arg0, arg1, arg2) {
            arg0.bindGroupLayouts = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_binding_bd7d6b01bb08823a: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_dbed4a234264971a: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_buffer_76841cdbff393f1d: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_91f4e8e05066c37a: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_code_130aebc49c4cadd3: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compute_800b1231f347606b: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_d2d33b52458e984a: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_device_1940fd5f9e2a7fee: function(arg0, arg1) {
            arg0.device = arg1;
        },
        __wbg_set_e4b9a833d288759a: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_end_of_pass_write_index_891ac71cf6a560c5: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_552f52b2aa3b346d: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entries_d9929dbfa2aaa9a7: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entry_point_6c86140cb6e5243a: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_1bef5ba626275d0a: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_format_46b1c6b3acc10b24: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_fd579d96fa4e468c: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_has_dynamic_offset_6e5f3385eb579529: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_label_08745022b1ec7b66: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_11c9eb57693a8459: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_22e3966c8d513ba6: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_5347fc976d15e94c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_644bc1b32c7917b0: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_73c99dab6708c962: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_891876dd0e225368: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_89d2ffaf598df6a4: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_a7095396911c5801: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_cba03ffef7ae90a3: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_200fc3b143da0a9e: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_599d761209e4b03f: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_gpu_auto_layout_mode_eb40dfd1eca5b9d3: function(arg0, arg1) {
            arg0.layout = __wbindgen_enum_GpuAutoLayoutMode[arg1];
        },
        __wbg_set_mapped_at_creation_8af5b87ccb898bbe: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_min_binding_size_f64_c6d68d773b7051d3: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_module_05c14f520e88acaa: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisampled_ae3f9f01a324f57c: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_f64_9374d9e912985773: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_power_preference_6bc1d2697a0e7374: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_query_set_6acda8912691b8d6: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_required_features_5bda1420e19461b2: function(arg0, arg1, arg2) {
            arg0.requiredFeatures = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_required_limits_e78b923d73f9b596: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_0f960a4279e7e400: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_buffer_binding_53a2189a77dd38be: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_texture_view_9153a2356b958667: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_sample_type_563589178fc892fe: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_ccdca76ef8e642ca: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_size_f64_3ff72c0d736c42a1: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_f64_9b65e4916b2eb111: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_storage_texture_b9e95e260c9d49a3: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_texture_84a5f84f0f451291: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_78a9445dbac5c902: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_ac87cc570da17000: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_type_f3d1ad35dc7889f9: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_usage_bac3cabe90090275: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_dimension_e9e9cd07d7ad686d: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_eb197c35c45e572c: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_visibility_65413465e7b24eb6: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_static_accessor_GLOBAL_THIS_003ee833af9699fc: function() {
            const ret = typeof globalThis === 'undefined' ? undefined : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_e4293fb22cbe77da: function() {
            const ret = typeof global === 'undefined' ? undefined : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_53bb20457401453d: function() {
            const ret = typeof self === 'undefined' ? undefined : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_b5e0810bfecb7feb: function() {
            const ret = typeof window === 'undefined' ? undefined : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_077c85cc28e36892: function(arg0, arg1, arg2) {
            arg0.submit(getArrayJsValueViewFromWasm0(arg1, arg2));
        },
        __wbg_then_656f253a5dd7d8ea: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_e3f7f2d365708be8: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unconfigure_835307f58dc68d80: function(arg0) {
            arg0.unconfigure();
        },
        __wbg_unmap_6a96b14c9ef5f7f5: function(arg0) {
            arg0.unmap();
        },
        __wbg_writeBuffer_d7640e64ceb69371: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 89, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__hfab4748bf2a224fe);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUDevice")], shim_idx: 41, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("any")], shim_idx: 41, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246_35);
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("undefined")], shim_idx: 41, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246_36);
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000007: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000008: function(arg0, arg1) {
            var v0 = getArrayF32FromWasm0(arg0, arg1).slice();
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(F32) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            table.set(0, undefined);
            table.set(1024, undefined);
            table.set(1025, null);
            table.set(1026, true);
            table.set(1027, false);
        },
    };
    return {
        __proto__: null,
        "./propagator_wasm_bg.js": import0,
    };
}

function wasm_bindgen__convert__closures_____invoke__hfab4748bf2a224fe(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__hfab4748bf2a224fe(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246_35(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246_35(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246_36(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__h0734b3eec33b7246_36(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__he21c9032761183b0(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen__convert__closures_____invoke__he21c9032761183b0(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuAutoLayoutMode = ["auto"];


const __wbindgen_enum_GpuBufferBindingType = ["uniform", "storage", "read-only-storage"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuSamplerBindingType = ["filtering", "non-filtering", "comparison"];


const __wbindgen_enum_GpuStorageTextureAccess = ["write-only", "read-only", "read-write"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureSampleType = ["float", "unfilterable-float", "depth", "sint", "uint"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];
const FuelSystemBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fuelsystembuilder_free(ptr, 1));
const GpuPropagatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_gpupropagator_free(ptr, 1));
const PropagatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_propagator_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    switch (typeof val) {
        // primitive types
        case 'number':
        case 'boolean':
        case 'undefined':
        return `${val}`;
        case 'string':
        return JSON.stringify(val);
        case 'bigint':
        return `${val}n`;
        case 'symbol':
        return val.description !== undefined
            ? `Symbol(${val.description})`
            : 'Symbol';
        case 'function':
        return typeof val.name === 'string' && val.name !== ''
            ? `Function(${val.name})`
            : 'Function';
        case 'object': {
            if (val === null) {
                return 'null';
            }
            // objects
            if (Array.isArray(val)) {
                return '[' + Array.from(val, debugString).join(', ') + ']';
            }
            // errors
            if (val instanceof Error) {
                return `${val.name}: ${val.message}\n${val.stack}`;
            }

            const toStringOutput = toString.call(val);
            // Test for built-in
            const className = /\[object ([^\]]+)\]/.exec(toStringOutput)?.[1];

            // Failed to match the standard '[object ClassName]'
            if (className === undefined) {
                return toStringOutput;
            }
            if (className === 'Object') {
                // we're a user defined class or Object
                // JSON.stringify avoids problems with cycles, and is generally much
                // easier than looping through ownProperties of `val`.
                try {
                    return 'Object(' + JSON.stringify(val) + ')';
                } catch (_) {
                    return 'Object';
                }
            }
            // TODO we could test for more things here, like `Set`s and `Map`s.
            return className;
        }
        // Per MDN: 'This list of values is exhaustive.'
        // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/typeof#description
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayJsValueViewFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(bytes) {
    numBytesDecoded += bytes.length;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
        cachedTextDecoder.decode();
        numBytesDecoded = bytes.length;
    }
    return cachedTextDecoder.decode(bytes);
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

let __wbg_init_promise;
async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;
    if (__wbg_init_promise === undefined) {
        __wbg_init_promise = __wbg_init_once(module_or_path);
        __wbg_init_promise.catch(() => { __wbg_init_promise = undefined; });
    }
    return __wbg_init_promise;
}

async function __wbg_init_once(module_or_path) {

    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('propagator_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
