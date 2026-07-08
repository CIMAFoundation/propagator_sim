/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_fuelsystembuilder_free: (a: number, b: number) => void;
export const __wbg_propagator_free: (a: number, b: number) => void;
export const fuelsystembuilder_add_fuel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
export const fuelsystembuilder_new: () => number;
export const propagator_burned_area_ha: (a: number) => [number, number, number];
export const propagator_cols: (a: number) => number;
export const propagator_fire_probability: (a: number) => [number, number, number, number];
export const propagator_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number, number];
export const propagator_next_time: (a: number) => bigint;
export const propagator_realizations: (a: number) => number;
export const propagator_rows: (a: number) => number;
export const propagator_set_boundary_conditions: (a: number, b: bigint, c: number, d: number, e: number, f: number, g: number) => [number, number];
export const propagator_step: (a: number, b: bigint) => [number, number];
export const propagator_time: (a: number) => bigint;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_start: () => void;
