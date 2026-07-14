//! Dependency-free geospatial raster processing for PROPAGATOR.
//!
//! This crate owns algorithms that operate on georeferenced rasters but do
//! not belong to the wildfire simulation engine in `propagator-core`.  It is
//! deliberately independent of that engine so native Python, WebAssembly,
//! command-line, and future server-side bindings can reuse the same geometry
//! implementation without pulling in simulation state.
//!
//! The first operation provided here is [`extract_isochrone`], a compatible
//! Rust implementation of the historical Python isochrone pipeline.  It
//! converts a row-major probability raster into closed, smoothed contour
//! lines in the coordinate reference system described by an affine
//! transform.  The implementation is `std`-only: it does not link GDAL,
//! rasterio, SciPy, GEOS, or PROJ.  CRS reprojection remains the caller's
//! responsibility; this crate only applies the supplied affine transform.
//!
//! # Coordinate and storage conventions
//!
//! - Raster values are a flat row-major slice. Cell `(row, col)` is stored at
//!   `row * cols + col`.
//! - Pixel-boundary vertices use `(column, row)` order internally.
//! - World coordinates use `[x, y]` arrays.
//! - Affines use GDAL's six-coefficient order `[a, b, c, d, e, f]`, where
//!   `x = a*col + b*row + c` and `y = d*col + e*row + f`.
//! - Each output line is explicitly closed: its final coordinate equals its
//!   first coordinate.
//!
//! # Compatibility scope
//!
//! [`extract_isochrone`] intentionally preserves observable behavior from
//! the former Python implementation, including its filtering cutoff, zero
//! padding, four-connected topology, strict minimum-length comparison,
//! omitted empty thresholds, retained empty geometries, Gaussian boundary
//! mode, and currently inactive simplification option.  These details are
//! documented on the function and option fields below.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

/// Isochrone geometry for one probability threshold.
///
/// `lines` has GeoJSON `MultiLineString` coordinate nesting, excluding the
/// GeoJSON object wrapper: each inner vector is one closed line and each
/// coordinate is `[x, y]`.
///
/// A threshold may be present with an empty `lines` vector.  This happens
/// when pixels survive the morphological opening but the final polygonized
/// background feature contains no interior ring, for example when the
/// above-threshold region reaches the raster boundary.  A threshold for
/// which no pixels survive at all is omitted from the returned collection.
#[derive(Clone, Debug, PartialEq)]
pub struct Isochrone {
    /// Probability cutoff used to create this geometry.
    pub threshold: f64,
    /// Closed lines in world `[x, y]` coordinates.
    pub lines: Vec<Vec<[f64; 2]>>,
}

/// Controls filtering, contour selection, and smoothing.
///
/// [`Default`] reproduces the defaults of the public Python API.
#[derive(Clone, Copy, Debug)]
pub struct IsochroneOptions {
    /// Side length of the square median-filter window.
    ///
    /// The filter is only applied when more than 100 raster values are
    /// strictly positive.  When applied, this must be a positive odd number.
    /// Samples beyond the raster boundary are zero, matching
    /// `scipy.signal.medfilt2d`.
    pub median_kernel: usize,
    /// Minimum unsmoothed line length in affine/world units.
    ///
    /// The comparison is strict (`length > min_length`) and is performed
    /// after the affine transform but before Gaussian smoothing.
    pub min_length: f64,
    /// Standard deviation of the one-dimensional Gaussian coordinate filter.
    ///
    /// Coordinates are filtered independently using radius
    /// `floor(4*sigma + 0.5)` and half-sample-symmetric reflection, matching
    /// SciPy's default `gaussian_filter1d(..., mode="reflect", truncate=4)`.
    /// The smoothed line is then explicitly reclosed.
    pub smooth_sigma: f64,
    /// Reserved simplification tolerance, retained for API compatibility.
    ///
    /// The historical Python expression that would use this value was
    /// commented out.  Rust therefore accepts and deliberately ignores it so
    /// changing language backends cannot silently change geometry.
    pub simplify_factor: f64,
}

impl Default for IsochroneOptions {
    fn default() -> Self {
        Self {
            median_kernel: 9,
            min_length: 0.0001,
            smooth_sigma: 0.8,
            simplify_factor: 0.00001,
        }
    }
}

/// Failure returned by [`extract_isochrone`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IsochroneError {
    /// The flat raster length is inconsistent with its declared dimensions.
    ShapeMismatch {
        /// Number of values supplied by the caller.
        actual: usize,
        /// `rows * cols`, calculated with saturating multiplication.
        expected: usize,
    },
    /// Median filtering was required but its square kernel was zero or even.
    InvalidMedianKernel {
        /// Invalid side length supplied in [`IsochroneOptions`].
        kernel: usize,
    },
    /// At least one line needed smoothing but sigma was non-finite or not
    /// strictly positive.
    InvalidSmoothSigma,
}

impl fmt::Display for IsochroneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { actual, expected } => write!(
                f,
                "values length {actual} does not match rows*cols {expected}"
            ),
            Self::InvalidMedianKernel { .. } => {
                f.write_str("Each element of kernel_size should be odd.")
            }
            Self::InvalidSmoothSigma => {
                f.write_str("smooth_sigma must be a positive finite number")
            }
        }
    }
}

impl Error for IsochroneError {}

/// Result type used by this crate's geospatial operations.
pub type Result<T> = std::result::Result<T, IsochroneError>;

type Vertex = (usize, usize); // (column, row)

#[derive(Clone, Copy)]
struct Edge {
    start: Vertex,
    end: Vertex,
}

/// Extract filtered, georeferenced isochrones from a probability raster.
///
/// # Algorithm
///
/// The operation is deterministic and runs these stages in order:
///
/// 1. Count values strictly greater than zero.  If that count exceeds 100,
///    apply a zero-padded square median filter; otherwise retain the original
///    raster unchanged.
/// 2. For each threshold independently, create the binary mask
///    `filtered_value >= threshold`.  `NaN` therefore behaves as false.
/// 3. Apply one binary erosion followed by one binary dilation using the
///    five-cell cross structuring element (center, north, south, west, east).
///    Cells outside the raster are false.
/// 4. Polygonize with four-connected semantics and retain the interior rings
///    of the final background feature.  This reproduces the contours exposed
///    by the former `rasterio.features.shapes` loop, including its behavior
///    for boundary-touching regions and diagonally adjacent components.
/// 5. Apply `transform` to pixel-boundary vertices and discard lines whose
///    unsmoothed world-coordinate length is not strictly greater than
///    [`IsochroneOptions::min_length`].
/// 6. Smooth x and y independently with a reflected Gaussian kernel, then
///    replace the last coordinate with the first to guarantee closure.
///
/// # Arguments
///
/// - `values`: Flat row-major raster. Both binding crates normalize their
///   input numbers to `f64` before calling this function.
/// - `rows`, `cols`: Declared raster dimensions. Zero-sized rasters are
///   accepted when the value slice is also empty.
/// - `transform`: Six affine coefficients `[a, b, c, d, e, f]` in GDAL order.
///   Rotation and shear terms (`b` and `d`) are fully supported.
/// - `thresholds`: Probability cutoffs, processed and returned in caller
///   order. Duplicate thresholds produce duplicate output entries.
/// - `options`: Filtering and geometry controls; see [`IsochroneOptions`].
///
/// # Return value
///
/// The vector preserves threshold order but omits thresholds whose opened
/// mask contains no true cells.  See [`Isochrone`] for the distinction
/// between an omitted threshold and a present threshold with empty geometry.
///
/// # Errors
///
/// Returns [`IsochroneError::ShapeMismatch`] when the flat input length does
/// not equal `rows * cols`.  If median filtering is actually needed, an even
/// or zero kernel returns [`IsochroneError::InvalidMedianKernel`].  A bad
/// smoothing sigma returns [`IsochroneError::InvalidSmoothSigma`] only when a
/// retained line reaches the smoothing stage, matching the lazy failure of
/// the historical implementation.
///
/// # Complexity
///
/// Let `N = rows * cols`, `K = median_kernel²`, `T = thresholds.len()`, and
/// `B` be the number of traced boundary edges.  Memory usage is `O(N + B)`.
/// Without median filtering, runtime is `O(T*N + B)`.  When filtering is
/// enabled, median selection adds expected `O(N*K)` work.
///
/// # Example
///
/// ```
/// use propagator_geo::{extract_isochrone, IsochroneOptions};
///
/// let mut probability = vec![0.0; 7 * 7];
/// for row in 2..5 {
///     for col in 2..5 {
///         probability[row * 7 + col] = 1.0;
///     }
/// }
/// let isochrones = extract_isochrone(
///     &probability,
///     7,
///     7,
///     [20.0, 0.0, 500_000.0, 0.0, -20.0, 4_900_000.0],
///     &[0.5, 0.9],
///     IsochroneOptions::default(),
/// )?;
/// assert_eq!(isochrones.len(), 2);
/// assert!(isochrones.iter().all(|item| item.lines.len() == 1));
/// # Ok::<(), propagator_geo::IsochroneError>(())
/// ```
pub fn extract_isochrone(
    values: &[f64],
    rows: usize,
    cols: usize,
    transform: [f64; 6],
    thresholds: &[f64],
    options: IsochroneOptions,
) -> Result<Vec<Isochrone>> {
    if values.len() != rows.saturating_mul(cols) {
        return Err(IsochroneError::ShapeMismatch {
            actual: values.len(),
            expected: rows.saturating_mul(cols),
        });
    }

    // scipy.signal.medfilt2d is only invoked once more than 100 values are
    // positive.  In particular, an invalid kernel remains harmless for a
    // small burn, which is observable legacy behaviour.
    let positive = values.iter().filter(|value| **value > 0.0).count();
    let filtered = if positive <= 100 {
        values.to_vec()
    } else {
        median_filter(values, rows, cols, options.median_kernel)?
    };

    let mut result = Vec::new();
    for &threshold in thresholds {
        let mask: Vec<bool> = filtered.iter().map(|v| *v >= threshold).collect();
        let opened = dilate_cross(&erode_cross(&mask, rows, cols), rows, cols);
        if !opened.iter().any(|v| *v) {
            continue;
        }

        let mut lines = enclosed_component_rings(&opened, rows, cols)
            .into_iter()
            .map(|ring| {
                ring.into_iter()
                    .map(|(col, row)| apply_transform(transform, col as f64, row as f64))
                    .collect::<Vec<_>>()
            })
            .filter(|ring| line_length(ring) > options.min_length)
            .map(|ring| gaussian_smooth_closed(&ring, options.smooth_sigma))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Make the intentionally unused compatibility option explicit.
        let _ = options.simplify_factor;
        result.push(Isochrone {
            threshold,
            lines: std::mem::take(&mut lines),
        });
    }
    Ok(result)
}

fn median_filter(values: &[f64], rows: usize, cols: usize, kernel: usize) -> Result<Vec<f64>> {
    if kernel == 0 || kernel.is_multiple_of(2) {
        return Err(IsochroneError::InvalidMedianKernel { kernel });
    }
    let radius = kernel / 2;
    let mut window = Vec::with_capacity(kernel.saturating_mul(kernel));
    let mut out = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            window.clear();
            for kr in 0..kernel {
                for kc in 0..kernel {
                    let rr = row as isize + kr as isize - radius as isize;
                    let cc = col as isize + kc as isize - radius as isize;
                    let value = if rr >= 0 && rr < rows as isize && cc >= 0 && cc < cols as isize {
                        values[rr as usize * cols + cc as usize]
                    } else {
                        0.0
                    };
                    window.push(value);
                }
            }
            let middle = window.len() / 2;
            let (_, median, _) = window.select_nth_unstable_by(middle, f64::total_cmp);
            out[row * cols + col] = *median;
        }
    }
    Ok(out)
}

fn erode_cross(mask: &[bool], rows: usize, cols: usize) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    if rows < 3 || cols < 3 {
        return out;
    }
    for row in 1..rows - 1 {
        for col in 1..cols - 1 {
            let idx = row * cols + col;
            out[idx] =
                mask[idx] && mask[idx - cols] && mask[idx + cols] && mask[idx - 1] && mask[idx + 1];
        }
    }
    out
}

fn dilate_cross(mask: &[bool], rows: usize, cols: usize) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            out[idx] = mask[idx]
                || (row > 0 && mask[idx - cols])
                || (row + 1 < rows && mask[idx + cols])
                || (col > 0 && mask[idx - 1])
                || (col + 1 < cols && mask[idx + 1]);
        }
    }
    out
}

/// Return exterior rings of true components which become holes of the
/// border-connected false polygon.  This is the geometry retained by the
/// Python function after it iterates `rasterio.features.shapes`.
fn enclosed_component_rings(mask: &[bool], rows: usize, cols: usize) -> Vec<Vec<Vertex>> {
    let mut labels = vec![usize::MAX; mask.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || labels[start] != usize::MAX {
            continue;
        }
        let label = components.len();
        let mut cells = Vec::new();
        let mut queue = VecDeque::from([start]);
        labels[start] = label;
        while let Some(idx) = queue.pop_front() {
            cells.push(idx);
            let row = idx / cols;
            let col = idx % cols;
            for next in neighbours(row, col, rows, cols) {
                if mask[next] && labels[next] == usize::MAX {
                    labels[next] = label;
                    queue.push_back(next);
                }
            }
        }
        components.push(cells);
    }

    let border_background = border_connected_background(mask, rows, cols);
    let mut rings = Vec::new();
    for (label, cells) in components.iter().enumerate() {
        if cells.iter().any(|idx| {
            let row = *idx / cols;
            let col = *idx % cols;
            row == 0 || row + 1 == rows || col == 0 || col + 1 == cols
        }) {
            continue;
        }
        if !cells.iter().any(|idx| {
            let row = *idx / cols;
            let col = *idx % cols;
            neighbours(row, col, rows, cols)
                .into_iter()
                .any(|next| !mask[next] && border_background[next])
        }) {
            continue;
        }

        let mut edges = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                if labels[idx] != label {
                    continue;
                }
                if row == 0 || labels[idx - cols] != label {
                    edges.push(Edge {
                        start: (col, row),
                        end: (col + 1, row),
                    });
                }
                if col + 1 == cols || labels[idx + 1] != label {
                    edges.push(Edge {
                        start: (col + 1, row),
                        end: (col + 1, row + 1),
                    });
                }
                if row + 1 == rows || labels[idx + cols] != label {
                    edges.push(Edge {
                        start: (col + 1, row + 1),
                        end: (col, row + 1),
                    });
                }
                if col == 0 || labels[idx - 1] != label {
                    edges.push(Edge {
                        start: (col, row + 1),
                        end: (col, row),
                    });
                }
            }
        }
        if let Some(outer) = trace_rings(&edges)
            .into_iter()
            .filter(|ring| signed_area(ring) > 0.0)
            .max_by(|left, right| signed_area(left).total_cmp(&signed_area(right)))
        {
            rings.push(outer);
        }
    }

    let mut idx = 0;
    while idx < rings.len() {
        let mut candidate = idx + 1;
        while candidate < rings.len() {
            if shared_vertex_count(&rings[idx], &rings[candidate]) >= 2 {
                let other = rings.remove(candidate);
                rings[idx] = combined_outer_ring(&rings[idx], &other);
                candidate = idx + 1;
            } else {
                candidate += 1;
            }
        }
        idx += 1;
    }

    rings
        .into_iter()
        .map(|ring| simplify_orthogonal_ring(&ring))
        .collect()
}

fn shared_vertex_count(left: &[Vertex], right: &[Vertex]) -> usize {
    left[..left.len() - 1]
        .iter()
        .filter(|vertex| right[..right.len() - 1].contains(vertex))
        .count()
}

fn combined_outer_ring(left: &[Vertex], right: &[Vertex]) -> Vec<Vertex> {
    let edges: Vec<Edge> = left
        .windows(2)
        .chain(right.windows(2))
        .map(|pair| Edge {
            start: pair[0],
            end: pair[1],
        })
        .collect();
    trace_rings_with_turn(&edges, true)
        .into_iter()
        .filter(|ring| signed_area(ring) > 0.0)
        .max_by(|a, b| signed_area(a).total_cmp(&signed_area(b)))
        .expect("two closed rings produce a closed outer ring")
}

fn neighbours(row: usize, col: usize, rows: usize, cols: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(4);
    if row > 0 {
        out.push((row - 1) * cols + col);
    }
    if col > 0 {
        out.push(row * cols + col - 1);
    }
    if col + 1 < cols {
        out.push(row * cols + col + 1);
    }
    if row + 1 < rows {
        out.push((row + 1) * cols + col);
    }
    out
}

fn border_connected_background(mask: &[bool], rows: usize, cols: usize) -> Vec<bool> {
    let mut reached = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    if rows == 0 || cols == 0 {
        return reached;
    }
    // The opening always leaves the bottom-right corner false. GDAL emits
    // the component that closes last in scan order last, and the Python
    // implementation overwrites its result for every emitted feature. Thus
    // only holes of this final background component survive when a true
    // barrier happens to split the raster boundary.
    let last = rows * cols - 1;
    if !mask[last] {
        reached[last] = true;
        queue.push_back(last);
    }
    while let Some(idx) = queue.pop_front() {
        let row = idx / cols;
        let col = idx % cols;
        for next in neighbours(row, col, rows, cols) {
            if !mask[next] && !reached[next] {
                reached[next] = true;
                queue.push_back(next);
            }
        }
    }
    reached
}

fn trace_rings(edges: &[Edge]) -> Vec<Vec<Vertex>> {
    trace_rings_with_turn(edges, false)
}

fn trace_rings_with_turn(edges: &[Edge], prefer_left: bool) -> Vec<Vec<Vertex>> {
    let mut used = vec![false; edges.len()];
    let mut rings = Vec::new();
    for start_idx in 0..edges.len() {
        if used[start_idx] {
            continue;
        }
        let start = edges[start_idx].start;
        let mut ring = vec![start];
        let mut current_idx = start_idx;
        loop {
            used[current_idx] = true;
            let edge = edges[current_idx];
            ring.push(edge.end);
            if edge.end == start {
                break;
            }
            let incoming = direction(edge.start, edge.end);
            let Some(next_idx) = choose_next_edge(edges, &used, edge.end, incoming, prefer_left)
            else {
                break;
            };
            current_idx = next_idx;
        }
        if ring.len() >= 4 && ring.first() == ring.last() {
            rings.push(ring);
        }
    }
    rings
}

fn direction(start: Vertex, end: Vertex) -> i8 {
    match (
        end.0 as isize - start.0 as isize,
        end.1 as isize - start.1 as isize,
    ) {
        (1, 0) => 0,  // east
        (0, 1) => 1,  // south
        (-1, 0) => 2, // west
        (0, -1) => 3, // north
        _ => unreachable!("pixel boundary edges are unit orthogonal edges"),
    }
}

fn choose_next_edge(
    edges: &[Edge],
    used: &[bool],
    vertex: Vertex,
    incoming: i8,
    prefer_left: bool,
) -> Option<usize> {
    // Stay on the current four-connected true component. Separate
    // corner-touching components are spliced after their rings are traced.
    let turns = if prefer_left {
        [3_i8, 0, 1, 2]
    } else {
        [1_i8, 0, 3, 2]
    };
    for turn in turns {
        let wanted = (incoming + turn) % 4;
        if let Some((idx, _)) = edges.iter().enumerate().find(|(idx, edge)| {
            !used[*idx] && edge.start == vertex && direction(edge.start, edge.end) == wanted
        }) {
            return Some(idx);
        }
    }
    None
}

fn signed_area(ring: &[Vertex]) -> f64 {
    ring.windows(2)
        .map(|pair| pair[0].0 as f64 * pair[1].1 as f64 - pair[1].0 as f64 * pair[0].1 as f64)
        .sum::<f64>()
        / 2.0
}

fn simplify_orthogonal_ring(ring: &[Vertex]) -> Vec<Vertex> {
    let open = &ring[..ring.len() - 1];
    let mut result = Vec::new();
    for idx in 0..open.len() {
        let prev = open[(idx + open.len() - 1) % open.len()];
        let current = open[idx];
        let next = open[(idx + 1) % open.len()];
        let collinear = (prev.0 == current.0 && current.0 == next.0)
            || (prev.1 == current.1 && current.1 == next.1);
        if !collinear {
            result.push(current);
        }
    }
    result.push(result[0]);
    result
}

fn apply_transform(transform: [f64; 6], col: f64, row: f64) -> [f64; 2] {
    [
        transform[0] * col + transform[1] * row + transform[2],
        transform[3] * col + transform[4] * row + transform[5],
    ]
}

fn line_length(line: &[[f64; 2]]) -> f64 {
    line.windows(2)
        .map(|pair| {
            let dx = pair[1][0] - pair[0][0];
            let dy = pair[1][1] - pair[0][1];
            dx.hypot(dy)
        })
        .sum()
}

fn gaussian_smooth_closed(line: &[[f64; 2]], sigma: f64) -> Result<Vec<[f64; 2]>> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(IsochroneError::InvalidSmoothSigma);
    }
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let mut weights = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0;
    for offset in -radius..=radius {
        let weight = (-0.5 * (offset as f64 / sigma).powi(2)).exp();
        weights.push(weight);
        sum += weight;
    }
    for weight in &mut weights {
        *weight /= sum;
    }

    let n = line.len() as isize;
    let mut smoothed = vec![[0.0; 2]; line.len()];
    for idx in 0..n {
        for (weight_idx, offset) in (-radius..=radius).enumerate() {
            let reflected = reflect_index(idx + offset, n) as usize;
            smoothed[idx as usize][0] += line[reflected][0] * weights[weight_idx];
            smoothed[idx as usize][1] += line[reflected][1] * weights[weight_idx];
        }
    }
    let first = smoothed[0];
    *smoothed.last_mut().expect("a ring is non-empty") = first;
    Ok(smoothed)
}

fn reflect_index(mut idx: isize, len: isize) -> isize {
    while idx < 0 || idx >= len {
        idx = if idx < 0 { -idx - 1 } else { 2 * len - idx - 1 };
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_and_smooths_cross_like_scipy_and_rasterio() {
        let mut values = vec![0.0; 7 * 7];
        for row in 2..5 {
            for col in 2..5 {
                values[row * 7 + col] = 1.0;
            }
        }
        let result = extract_isochrone(
            &values,
            7,
            7,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0.5],
            IsochroneOptions::default(),
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lines.len(), 1);
        let line = &result[0].lines[0];
        assert_eq!(line.len(), 13);
        assert!((line[0][0] - 3.2734535749019926).abs() < 1e-12);
        assert!((line[0][1] - 2.02279180090523).abs() < 1e-12);
        assert_eq!(line.first(), line.last());
    }

    #[test]
    fn keeps_threshold_with_empty_geometry_when_opened_pixels_touch_edge() {
        let values = vec![1.0; 7 * 7];
        let result = extract_isochrone(
            &values,
            7,
            7,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0.5],
            IsochroneOptions::default(),
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].lines.is_empty());
    }

    #[test]
    fn single_diagonal_touch_keeps_separate_background_interior_rings() {
        let mut mask = vec![false; 6 * 6];
        mask[2 * 6 + 2] = true;
        mask[3 * 6 + 3] = true;
        let rings = enclosed_component_rings(&mask, 6, 6);
        assert_eq!(rings.len(), 2);
    }

    #[test]
    fn median_filter_uses_zero_padding_only_for_large_burns() {
        let small = vec![1.0; 10 * 10];
        assert!(extract_isochrone(
            &small,
            10,
            10,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0.5],
            IsochroneOptions {
                median_kernel: 2,
                ..Default::default()
            },
        )
        .is_ok());

        let large = vec![1.0; 11 * 10];
        assert!(extract_isochrone(
            &large,
            11,
            10,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0.5],
            IsochroneOptions {
                median_kernel: 2,
                ..Default::default()
            },
        )
        .is_err());
    }
}
