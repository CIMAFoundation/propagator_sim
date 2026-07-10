//! WebAssembly bindings for the Rust PROPAGATOR core.
//!
//! This crate exposes a small, browser-friendly slice of the
//! [`propagator_core`] API through `wasm-bindgen`, mirroring the subset the
//! PyO3 bindings (`propagator-py`) expose but marshalling flat typed arrays
//! instead of numpy rasters. It is meant to drive the interactive demo in the
//! documentation, not to be a full-featured binding:
//!
//! * the engine runs **single-threaded** (`n_threads = 1`) — `wasm32` has no
//!   `std::thread`, and the core's sequential path is bit-for-bit identical;
//! * runs are always **seeded**, so we never touch `SystemTime`/entropy;
//! * on-disk features (tile freezing, checkpoints) are not surfaced.
//!
//! Grids cross the boundary as row-major flat arrays plus a `(rows, cols)`
//! shape, exactly as `Grid2` stores them internally.

use propagator_core::{
    BoundaryConditions, FieldInput, FuelDef, FuelSystem, Grid2, Ignitions, MoistureModel,
    OobMode, Propagator as CorePropagator, PropagatorConfig, PropagatorError, RosModel,
    TILE_SIZE,
};
use wasm_bindgen::prelude::*;

/// Tile size in cells: north/west domain growth must be a multiple of this.
#[wasm_bindgen]
pub fn tile_size() -> usize {
    TILE_SIZE
}

/// Map a core error to a JS `Error`-friendly string value.
fn to_js(err: PropagatorError) -> JsValue {
    JsValue::from_str(&err.to_string())
}

/// Accumulates fuel definitions supplied from JavaScript into a
/// [`FuelSystem`]. The demo builds this from the EU 12-class table defined in
/// `app.html`, keeping the fuel model out of the wasm binary.
#[wasm_bindgen]
#[derive(Default)]
pub struct FuelSystemBuilder {
    defs: Vec<FuelDef>,
}

#[wasm_bindgen]
impl FuelSystemBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FuelSystemBuilder {
        FuelSystemBuilder::default()
    }

    /// Append one fuel. Units match the config dictionaries (`v0` m/h,
    /// `humidity` percent, or `undefined` for a fuel with no live load).
    /// `spread_to`/`spread_prob` are parallel arrays giving the base spread
    /// probability toward each target fuel id.
    #[allow(clippy::too_many_arguments)]
    pub fn add_fuel(
        &mut self,
        id: i32,
        name: String,
        v0: f64,
        d0: f64,
        d1: f64,
        hhv: f64,
        humidity: Option<f64>,
        spotting: bool,
        prob_ign_by_embers: f64,
        burn: bool,
        spread_to: Vec<i32>,
        spread_prob: Vec<f64>,
    ) -> Result<(), JsValue> {
        if spread_to.len() != spread_prob.len() {
            return Err(JsValue::from_str(
                "spread_to and spread_prob must have the same length",
            ));
        }
        self.defs.push(FuelDef {
            id,
            name,
            v0,
            d0,
            d1,
            hhv,
            humidity,
            spotting,
            prob_ign_by_embers,
            burn,
            spread_probability: spread_to.into_iter().zip(spread_prob).collect(),
        });
        Ok(())
    }
}

fn ros_from_str(s: &str) -> Result<RosModel, JsValue> {
    match s.to_ascii_lowercase().as_str() {
        "wang" => Ok(RosModel::Wang),
        "rothermel" => Ok(RosModel::Rothermel),
        "standard" => Ok(RosModel::Standard),
        other => Err(JsValue::from_str(&format!("unknown ROS model: {other}"))),
    }
}

fn moisture_from_str(s: &str) -> Result<MoistureModel, JsValue> {
    match s.to_ascii_lowercase().as_str() {
        "trucchia" => Ok(MoistureModel::Trucchia),
        "baghino" => Ok(MoistureModel::Baghino),
        other => Err(JsValue::from_str(&format!("unknown moisture model: {other}"))),
    }
}

/// Browser-facing wrapper around the core [`CorePropagator`].
///
/// Constructed from flat, row-major vegetation (fuel-code) and DEM rasters;
/// weather is supplied as scalars and ignitions as flat `[row, col, …]`
/// pairs. See the module docs for the deliberate limitations.
#[wasm_bindgen]
pub struct Propagator {
    inner: CorePropagator,
    rows: usize,
    cols: usize,
}

#[wasm_bindgen]
impl Propagator {
    /// Build a propagator over a `rows x cols` domain.
    ///
    /// `veg` (fuel codes matching `fuels`) and `dem` (elevation, metres) are
    /// row-major and must both hold `rows * cols` entries. `fuels` is the
    /// fuel model assembled from JavaScript.
    ///
    /// By default fires reaching the boundary are ignored, which keeps the
    /// demo running when the flame front hits an edge. Pass
    /// `halt_on_boundary = true` to have [`Propagator::step`] report boundary
    /// hits instead, so the caller can grow the domain with
    /// [`Propagator::expand`] and continue.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: usize,
        cols: usize,
        veg: Vec<i32>,
        dem: Vec<f64>,
        fuels: &FuelSystemBuilder,
        realizations: usize,
        seed: u32,
        do_spotting: bool,
        cellsize: f64,
        ros_model: &str,
        moisture_model: &str,
        halt_on_boundary: Option<bool>,
    ) -> Result<Propagator, JsValue> {
        if veg.len() != rows * cols {
            return Err(JsValue::from_str(&format!(
                "veg length {} != rows*cols {}",
                veg.len(),
                rows * cols
            )));
        }
        if dem.len() != rows * cols {
            return Err(JsValue::from_str(&format!(
                "dem length {} != rows*cols {}",
                dem.len(),
                rows * cols
            )));
        }

        let veg = Grid2::from_vec(rows, cols, veg);
        let dem = Grid2::from_vec(rows, cols, dem);
        let mut config = PropagatorConfig::new(veg, dem);
        config.realizations = realizations.max(1);
        config.seed = Some(seed as u64);
        config.do_spotting = do_spotting;
        config.cellsize = cellsize;
        config.ros_model = ros_from_str(ros_model)?;
        config.moisture_model = moisture_from_str(moisture_model)?;
        // Stay off the threaded/entropy/fs paths; boundary handling is the
        // caller's choice (Ignore for the demo, Raise for domain growth).
        config.oob_mode = if halt_on_boundary.unwrap_or(false) {
            OobMode::Raise
        } else {
            OobMode::Ignore
        };
        config.n_threads = Some(1);
        config.fuels = Some(FuelSystem::from_defs(&fuels.defs).map_err(to_js)?);

        let inner = CorePropagator::new(config).map_err(to_js)?;
        Ok(Propagator { inner, rows, cols })
    }

    /// Current simulation time, seconds from start.
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> i64 {
        self.inner.time()
    }

    /// Number of grid rows.
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of grid columns.
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stochastic realizations.
    #[wasm_bindgen(getter)]
    pub fn realizations(&self) -> usize {
        self.inner.realizations()
    }

    /// Enqueue boundary conditions at `time` (external units: percent
    /// moisture, degrees wind direction from north, km/h wind speed).
    ///
    /// `ignitions` is a flat list of `[row0, col0, row1, col1, …]`; pass an
    /// empty array to leave the ignition set unchanged.
    pub fn set_boundary_conditions(
        &mut self,
        time: i64,
        moisture: f64,
        wind_dir: f64,
        wind_speed: f64,
        ignitions: Vec<i32>,
    ) -> Result<(), JsValue> {
        let ignition_points = if ignitions.is_empty() {
            None
        } else {
            if ignitions.len() % 2 != 0 {
                return Err(JsValue::from_str(
                    "ignitions must be a flat list of [row, col] pairs",
                ));
            }
            let pts = ignitions
                .chunks_exact(2)
                .map(|rc| (rc[0] as usize, rc[1] as usize))
                .collect();
            Some(Ignitions::Points(pts))
        };

        let bc = BoundaryConditions {
            time,
            moisture: Some(FieldInput::Scalar(moisture)),
            wind_dir: Some(FieldInput::Scalar(wind_dir)),
            wind_speed: Some(FieldInput::Scalar(wind_speed)),
            ignitions: ignition_points,
            additional_moisture: None,
            vegetation_changes: None,
        };
        self.inner.set_boundary_conditions(bc).map_err(to_js)
    }

    /// Propagate for `seconds` of simulation time, applying scheduled events.
    ///
    /// Returns `true` when the run was constructed with
    /// `halt_on_boundary = true` and at least one realization was suspended
    /// at the domain boundary: grow the domain with [`Propagator::expand`]
    /// and step again to resume it — nothing is lost.
    pub fn step(&mut self, seconds: i64) -> Result<bool, JsValue> {
        match self.inner.step_window(seconds) {
            Ok(()) => Ok(false),
            Err(PropagatorError::OutOfBounds) => Ok(true),
            Err(err) => Err(to_js(err)),
        }
    }

    /// Grid origin (row, col) in global cell coordinates. Growth to the
    /// north/west moves the origin to smaller values.
    #[wasm_bindgen(getter)]
    pub fn origin_row(&self) -> i64 {
        self.inner.origin().0
    }

    /// See [`Propagator::origin_row`].
    #[wasm_bindgen(getter)]
    pub fn origin_col(&self) -> i64 {
        self.inner.origin().1
    }

    /// Grow the domain in place onto the larger `veg`/`dem` anchored at
    /// `(origin_row, origin_col)`, preserving all simulation state. The new
    /// grid must fully contain the old one and growth to the north/west must
    /// be a multiple of [`tile_size`] cells.
    #[allow(clippy::too_many_arguments)]
    pub fn expand(
        &mut self,
        rows: usize,
        cols: usize,
        veg: Vec<i32>,
        dem: Vec<f64>,
        origin_row: i64,
        origin_col: i64,
    ) -> Result<(), JsValue> {
        if veg.len() != rows * cols || dem.len() != rows * cols {
            return Err(JsValue::from_str(&format!(
                "veg/dem length must equal rows*cols ({})",
                rows * cols
            )));
        }
        let veg = Grid2::from_vec(rows, cols, veg);
        let dem = Grid2::from_vec(rows, cols, dem);
        self.inner
            .expand(veg, dem, (origin_row, origin_col))
            .map_err(to_js)?;
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }

    /// Next event time (scheduler or front), or `-1` when the run is idle.
    pub fn next_time(&self) -> i64 {
        self.inner.next_time().unwrap_or(-1)
    }

    /// Fire probability per cell in `[0, 1]`, row-major, `rows * cols` long.
    pub fn fire_probability(&mut self) -> Result<Vec<f32>, JsValue> {
        let out = self.inner.get_output().map_err(to_js)?;
        Ok(out.fire_probability.as_slice().to_vec())
    }

    /// Mean burned area across realizations, in hectares.
    pub fn burned_area_ha(&mut self) -> Result<f64, JsValue> {
        let out = self.inner.get_output().map_err(to_js)?;
        Ok(out.stats.area_mean / 10_000.0)
    }
}
