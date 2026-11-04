# propagator-geo

`propagator-geo` contains geospatial raster algorithms shared by
PROPAGATOR's native Python and WebAssembly bindings. It is separate from
`propagator-core` because raster filtering, affine coordinate conversion,
and vector contour extraction are output/geometry concerns rather than
wildfire simulation concerns.

The crate is dependency-free and does not require GDAL, GEOS, SciPy,
rasterio, or PROJ. Reprojection is intentionally outside its scope: callers
must supply raster values and an affine transform in the desired output CRS.

## `extract_isochrone`

The extractor accepts:

- a flat row-major `f64` probability raster;
- its row and column counts;
- a six-coefficient GDAL affine transform;
- an ordered list of probability thresholds;
- filtering, length, and smoothing options.

For each threshold it optionally median-filters the raster, applies a
cross-shaped binary opening, traces four-connected pixel boundaries,
transforms vertices into world coordinates, removes short rings, and applies
reflected Gaussian smoothing. It returns coordinate arrays with GeoJSON
`MultiLineString` nesting.

The implementation preserves the observable behavior of the former Python
SciPy/rasterio pipeline. Important compatibility details include:

- median filtering starts only when more than 100 raster values are positive;
- the median filter uses zero padding and requires an odd kernel only when it
  actually runs;
- thresholds with no surviving pixels are omitted;
- a surviving threshold may still contain an empty geometry when no closed
  interior contour exists;
- line length is measured after the affine transform and before smoothing;
- Gaussian smoothing uses SciPy-compatible reflected boundaries and always
  recloses the line;
- `simplify_factor` remains accepted but inactive because simplification was
  also disabled in the Python implementation.

The full parameter, error, topology, complexity, and example documentation is
available on `propagator_geo::extract_isochrone` through `cargo doc`.

## Bindings

- `propagator-py` exposes the operation as
  `propagator_rust.extract_isochrone`. It accepts two-dimensional NumPy
  `float32` or `float64` arrays and returns a threshold-keyed Python mapping.
- `propagator-wasm` exposes `extract_isochrone` for `Float32Array` rasters and
  returns JavaScript `{ threshold, lines }` objects. Optional arguments use
  the same defaults as Python.

Both bindings depend directly on this crate. Neither routes geometry work
through `propagator-core`.
