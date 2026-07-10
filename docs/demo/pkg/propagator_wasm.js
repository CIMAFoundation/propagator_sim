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
     * @param {Int32Array} spread_to
     * @param {Float64Array} spread_prob
     */
    add_fuel(id, name, v0, d0, d1, hhv, humidity, spotting, prob_ign_by_embers, burn, spread_to, spread_prob) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(spread_to, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(spread_prob, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.fuelsystembuilder_add_fuel(this.__wbg_ptr, id, ptr0, len0, v0, d0, d1, hhv, !isLikeNone(humidity), isLikeNone(humidity) ? 0 : humidity, spotting, prob_ign_by_embers, burn, ptr1, len1, ptr2, len2);
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
     * Grow the domain in place onto the larger `veg`/`dem` anchored at
     * `(origin_row, origin_col)`, preserving all simulation state. The new
     * grid must fully contain the old one and growth to the north/west must
     * be a multiple of [`tile_size`] cells.
     * @param {number} rows
     * @param {number} cols
     * @param {Int32Array} veg
     * @param {Float64Array} dem
     * @param {bigint} origin_row
     * @param {bigint} origin_col
     */
    expand(rows, cols, veg, dem, origin_row, origin_col) {
        const ptr0 = passArray32ToWasm0(veg, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(dem, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.propagator_expand(this.__wbg_ptr, rows, cols, ptr0, len0, ptr1, len1, origin_row, origin_col);
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
     * @param {Int32Array} veg
     * @param {Float64Array} dem
     * @param {FuelSystemBuilder} fuels
     * @param {number} realizations
     * @param {number} seed
     * @param {boolean} do_spotting
     * @param {number} cellsize
     * @param {string} ros_model
     * @param {string} moisture_model
     * @param {boolean | null} [halt_on_boundary]
     */
    constructor(rows, cols, veg, dem, fuels, realizations, seed, do_spotting, cellsize, ros_model, moisture_model, halt_on_boundary) {
        const ptr0 = passArray32ToWasm0(veg, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(dem, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        _assertClass(fuels, FuelSystemBuilder);
        const ptr2 = passStringToWasm0(ros_model, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(moisture_model, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.propagator_new(rows, cols, ptr0, len0, ptr1, len1, fuels.__wbg_ptr, realizations, seed, do_spotting, cellsize, ptr2, len2, ptr3, len3, isLikeNone(halt_on_boundary) ? 0xFFFFFF : halt_on_boundary ? 1 : 0);
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
     * @param {Int32Array} moisture_cells
     * @param {Float32Array} moisture_values
     * @param {Int32Array} veg_cells
     * @param {number} veg_fuel
     */
    set_action_fields(time, moisture_cells, moisture_values, veg_cells, veg_fuel) {
        const ptr0 = passArray32ToWasm0(moisture_cells, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(moisture_values, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(veg_cells, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.propagator_set_action_fields(this.__wbg_ptr, time, ptr0, len0, ptr1, len1, ptr2, len2, veg_fuel);
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
     * @param {Int32Array} ignitions
     */
    set_boundary_conditions(time, moisture, wind_dir, wind_speed, ignitions) {
        const ptr0 = passArray32ToWasm0(ignitions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.propagator_set_boundary_conditions(this.__wbg_ptr, time, moisture, wind_dir, wind_speed, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
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
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./propagator_wasm_bg.js": import0,
    };
}

const FuelSystemBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fuelsystembuilder_free(ptr, 1));
const PropagatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_propagator_free(ptr, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
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
    return decodeText(ptr >>> 0, len);
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

function isLikeNone(x) {
    return x === undefined || x === null;
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

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
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
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

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

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


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
