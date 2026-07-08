//! PyO3 bindings for the Rust PROPAGATOR core.
//!
//! Exposes a `Propagator` class whose surface matches the subset of the
//! Python/numba `Propagator` that the CLI drives, so a thin Python adapter
//! (`propagator.rust_core`) can present the two cores interchangeably.
//! Fuel definitions are passed in *config units* (v0 m/h, humidity percent)
//! exactly as the Python fuel dictionaries store them, so the Rust core's
//! internal `/60` and `/100` conversions reproduce numba's byte for byte.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use propagator_core::{
    BoundaryConditions, FieldInput, FuelDef, FuelSystem, Grid2, Ignitions, MoistureModel, OobMode,
    Propagator as CorePropagator, PropagatorConfig, PropagatorError, RosModel,
};

create_exception!(
    propagator_rust,
    PropagatorOutOfBoundsError,
    PyException,
    "Fire reached the domain boundary while out_of_bounds_mode='raise'."
);

// ---------------------------------------------------------------------------
// conversions
// ---------------------------------------------------------------------------

fn grid_i32(a: PyReadonlyArray2<i32>) -> Grid2<i32> {
    let view = a.as_array();
    let (r, c) = (view.nrows(), view.ncols());
    Grid2::from_vec(r, c, view.iter().copied().collect())
}

fn grid_f64(a: PyReadonlyArray2<f64>) -> Grid2<f64> {
    let view = a.as_array();
    let (r, c) = (view.nrows(), view.ncols());
    Grid2::from_vec(r, c, view.iter().copied().collect())
}

fn grid_f32(a: PyReadonlyArray2<f32>) -> Grid2<f32> {
    let view = a.as_array();
    let (r, c) = (view.nrows(), view.ncols());
    Grid2::from_vec(r, c, view.iter().copied().collect())
}

fn grid_bool(a: PyReadonlyArray2<bool>) -> Grid2<bool> {
    let view = a.as_array();
    let (r, c) = (view.nrows(), view.ncols());
    Grid2::from_vec(r, c, view.iter().copied().collect())
}

fn f32_grid_to_py<'py>(py: Python<'py>, g: &Grid2<f32>) -> Bound<'py, PyArray2<f32>> {
    let (r, c) = g.shape();
    Array2::from_shape_vec((r, c), g.as_slice().to_vec())
        .expect("grid shape matches its data")
        .into_pyarray(py)
}

fn i32_grid_to_py<'py>(py: Python<'py>, g: &Grid2<i32>) -> Bound<'py, PyArray2<i32>> {
    let (r, c) = g.shape();
    Array2::from_shape_vec((r, c), g.as_slice().to_vec())
        .expect("grid shape matches its data")
        .into_pyarray(py)
}

fn err(e: PropagatorError) -> PyErr {
    match e {
        PropagatorError::OutOfBounds => PropagatorOutOfBoundsError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn ros_from_str(s: &str) -> PyResult<RosModel> {
    match s.to_ascii_lowercase().as_str() {
        "wang" => Ok(RosModel::Wang),
        "rothermel" => Ok(RosModel::Rothermel),
        "standard" => Ok(RosModel::Standard),
        _ => Err(PyValueError::new_err(format!("unknown ROS model: {s}"))),
    }
}

fn moisture_from_str(s: &str) -> PyResult<MoistureModel> {
    match s.to_ascii_lowercase().as_str() {
        "trucchia" => Ok(MoistureModel::Trucchia),
        "baghino" => Ok(MoistureModel::Baghino),
        _ => Err(PyValueError::new_err(format!(
            "unknown moisture model: {s}"
        ))),
    }
}

/// Parse a scalar (float) or 2-D float32 raster into a `FieldInput`.
fn parse_field(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<FieldInput>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            if let Ok(v) = o.extract::<f64>() {
                Ok(Some(FieldInput::Scalar(v)))
            } else {
                let arr: PyReadonlyArray2<f32> = o.extract().map_err(|_| {
                    PyValueError::new_err("field must be a float scalar or a 2-D float32 array")
                })?;
                Ok(Some(FieldInput::Grid(grid_f32(arr))))
            }
        }
    }
}

/// Parse ignitions: list of (row, col) / (row, col, realization) tuples, or
/// a 2-D boolean mask.
fn parse_ignitions(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Ignitions>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            if let Ok(pts) = o.extract::<Vec<(usize, usize)>>() {
                Ok(Some(Ignitions::Points(pts)))
            } else if let Ok(pts) = o.extract::<Vec<(usize, usize, usize)>>() {
                Ok(Some(Ignitions::PointsWithRealization(pts)))
            } else {
                let arr: PyReadonlyArray2<bool> = o.extract().map_err(|_| {
                    PyValueError::new_err(
                        "ignitions must be a list of (row, col[, realization]) \
                         tuples or a 2-D boolean array",
                    )
                })?;
                Ok(Some(Ignitions::Mask(grid_bool(arr))))
            }
        }
    }
}

/// Build the Rust fuel system from a Python `{id: {...}}` dict in config
/// units (v0 m/h, humidity percent), mirroring `fuelsystem_from_dict`.
fn parse_fuels(fuels: &Bound<'_, PyDict>) -> PyResult<FuelSystem> {
    let mut defs: Vec<FuelDef> = Vec::with_capacity(fuels.len());
    for (key, value) in fuels.iter() {
        let id: i32 = key.extract()?;
        let d: Bound<'_, PyDict> = value
            .extract()
            .map_err(|_| PyValueError::new_err(format!("fuel {id} is not a mapping")))?;
        let get = |k: &str| d.get_item(k).ok().flatten();

        let name: String = match get("name") {
            Some(v) => v.extract()?,
            None => format!("fuel_{id}"),
        };
        let v0: f64 = get("v0")
            .ok_or_else(|| PyValueError::new_err(format!("fuel {id}: missing v0")))?
            .extract()?;
        let d0: f64 = get("d0")
            .ok_or_else(|| PyValueError::new_err(format!("fuel {id}: missing d0")))?
            .extract()?;
        let hhv: f64 = get("hhv")
            .ok_or_else(|| PyValueError::new_err(format!("fuel {id}: missing hhv")))?
            .extract()?;
        let d1: f64 = match get("d1") {
            Some(v) => v.extract()?,
            None => 0.0,
        };
        let humidity: Option<f64> = match get("humidity") {
            Some(v) => {
                let h: f64 = v.extract()?;
                if h == -9999.0 {
                    None
                } else {
                    Some(h)
                }
            }
            None => None,
        };
        let spotting: bool = match get("spotting") {
            Some(v) => v.extract()?,
            None => false,
        };
        let prob_ign_by_embers: f64 = match get("prob_ign_by_embers") {
            Some(v) => v.extract()?,
            None => 0.0,
        };
        let burn: bool = match get("burn") {
            Some(v) => v.extract()?,
            None => true,
        };
        let spread_probability: Vec<(i32, f64)> = match get("spread_probability") {
            Some(v) => {
                let sp: Bound<'_, PyDict> = v.extract().map_err(|_| {
                    PyValueError::new_err(format!(
                        "fuel {id}: spread_probability must be a mapping"
                    ))
                })?;
                let mut out = Vec::with_capacity(sp.len());
                for (to_key, prob) in sp.iter() {
                    out.push((to_key.extract::<i32>()?, prob.extract::<f64>()?));
                }
                out
            }
            None => Vec::new(),
        };

        defs.push(FuelDef {
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
            spread_probability,
        });
    }
    FuelSystem::from_defs(&defs).map_err(err)
}

// ---------------------------------------------------------------------------
// Propagator
// ---------------------------------------------------------------------------

#[pyclass]
struct Propagator {
    inner: CorePropagator,
}

#[pymethods]
impl Propagator {
    #[new]
    #[pyo3(signature = (
        dem,
        veg,
        realizations = 100,
        fuels = None,
        do_spotting = false,
        origin = (0, 0),
        seed = None,
        freeze_dir = None,
        out_of_bounds_mode = "raise",
        ros_model = "wang",
        moisture_model = "trucchia",
        cellsize = 20.0,
        n_threads = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dem: PyReadonlyArray2<f64>,
        veg: PyReadonlyArray2<i32>,
        realizations: usize,
        fuels: Option<Bound<'_, PyDict>>,
        do_spotting: bool,
        origin: (i64, i64),
        seed: Option<u64>,
        freeze_dir: Option<String>,
        out_of_bounds_mode: &str,
        ros_model: &str,
        moisture_model: &str,
        cellsize: f64,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        let veg = grid_i32(veg);
        let dem = grid_f64(dem);
        let mut config = PropagatorConfig::new(veg, dem);
        config.realizations = realizations;
        config.do_spotting = do_spotting;
        config.origin = origin;
        config.seed = seed;
        config.cellsize = cellsize;
        config.freeze_dir = freeze_dir.map(Into::into);
        config.ros_model = ros_from_str(ros_model)?;
        config.moisture_model = moisture_from_str(moisture_model)?;
        config.oob_mode = match out_of_bounds_mode {
            "ignore" => OobMode::Ignore,
            "raise" => OobMode::Raise,
            other => {
                return Err(PyValueError::new_err(format!(
                    "out_of_bounds_mode must be 'ignore' or 'raise', got {other:?}"
                )))
            }
        };
        config.n_threads = n_threads;
        if let Some(fuels) = fuels {
            config.fuels = Some(parse_fuels(&fuels)?);
        }
        let inner = CorePropagator::new(config).map_err(err)?;
        Ok(Propagator { inner })
    }

    #[getter]
    fn time(&self) -> i64 {
        self.inner.time()
    }

    #[getter]
    fn origin(&self) -> (i64, i64) {
        self.inner.origin()
    }

    #[getter]
    fn realizations(&self) -> usize {
        self.inner.realizations()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn veg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i32>> {
        i32_grid_to_py(py, self.inner.veg())
    }

    fn next_time(&self) -> Option<i64> {
        self.inner.next_time()
    }

    /// `(north, south, west, east)` edges the front is pressing against
    /// after a boundary halt; all-false unless the last `step` hit a
    /// boundary in out_of_bounds_mode='raise'.
    fn boundary_pressure(&self) -> (bool, bool, bool, bool) {
        self.inner.boundary_pressure()
    }

    #[pyo3(signature = (
        time,
        moisture = None,
        wind_dir = None,
        wind_speed = None,
        ignitions = None,
        additional_moisture = None,
        vegetation_changes = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn set_boundary_conditions(
        &mut self,
        time: i64,
        moisture: Option<Bound<'_, PyAny>>,
        wind_dir: Option<Bound<'_, PyAny>>,
        wind_speed: Option<Bound<'_, PyAny>>,
        ignitions: Option<Bound<'_, PyAny>>,
        additional_moisture: Option<PyReadonlyArray2<f32>>,
        vegetation_changes: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let bc = BoundaryConditions {
            time,
            moisture: parse_field(moisture.as_ref())?,
            wind_dir: parse_field(wind_dir.as_ref())?,
            wind_speed: parse_field(wind_speed.as_ref())?,
            ignitions: parse_ignitions(ignitions.as_ref())?,
            additional_moisture: additional_moisture.map(grid_f32),
            vegetation_changes: vegetation_changes.map(grid_f64),
        };
        self.inner.set_boundary_conditions(bc).map_err(err)
    }

    #[pyo3(signature = (seconds))]
    fn step(&mut self, seconds: i64) -> PyResult<()> {
        self.inner.step_window(seconds).map_err(err)
    }

    fn expand(
        &mut self,
        veg: PyReadonlyArray2<i32>,
        dem: PyReadonlyArray2<f64>,
        origin: (i64, i64),
    ) -> PyResult<()> {
        self.inner
            .expand(grid_i32(veg), grid_f64(dem), origin)
            .map_err(err)
    }

    fn freeze_inactive_tiles(&mut self) -> PyResult<usize> {
        self.inner.freeze_inactive_tiles().map_err(err)
    }

    /// Snapshot as a dict of numpy rasters plus a nested `stats` dict; the
    /// Python adapter re-wraps this into the core's dataclasses.
    fn get_output<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = self.inner.get_output().map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("time", out.time)?;
        d.set_item(
            "fire_probability",
            f32_grid_to_py(py, &out.fire_probability),
        )?;
        d.set_item(
            "spotting_generation_probability",
            f32_grid_to_py(py, &out.spotting_generation_probability),
        )?;
        d.set_item(
            "spotting_receiving_probability",
            f32_grid_to_py(py, &out.spotting_receiving_probability),
        )?;
        d.set_item(
            "mean_arrival_time",
            f32_grid_to_py(py, &out.mean_arrival_time),
        )?;
        d.set_item(
            "min_arrival_time",
            f32_grid_to_py(py, &out.min_arrival_time),
        )?;
        d.set_item("ros_mean", f32_grid_to_py(py, &out.ros_mean))?;
        d.set_item("ros_max", f32_grid_to_py(py, &out.ros_max))?;
        d.set_item("fli_mean", f32_grid_to_py(py, &out.fli_mean))?;
        d.set_item("fli_max", f32_grid_to_py(py, &out.fli_max))?;

        let stats = PyDict::new(py);
        stats.set_item("n_active", out.stats.n_active)?;
        stats.set_item("area_mean", out.stats.area_mean)?;
        stats.set_item("area_50", out.stats.area_50)?;
        stats.set_item("area_75", out.stats.area_75)?;
        stats.set_item("area_90", out.stats.area_90)?;
        d.set_item("stats", stats)?;
        Ok(d)
    }
}

#[pymodule]
fn propagator_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Propagator>()?;
    m.add(
        "PropagatorOutOfBoundsError",
        m.py().get_type::<PropagatorOutOfBoundsError>(),
    )?;
    Ok(())
}
