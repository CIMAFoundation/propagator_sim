//! WebGPU execution backend for the PROPAGATOR core, on `wgpu`.
//!
//! Reformulates the event-driven kernel as a relaxation over
//! `realizations x rows x cols` arrival-time fields (see
//! `docs/gpu-core.md`): every edge's Bernoulli draw is a hash of
//! `(seed, realization, cell, direction)`, so each realization is a
//! shortest-path problem on a fixed random graph whose edge weights follow
//! the weather fields in force when the source cell ignites. Work is ordered
//! in time buckets over a GPU-side worklist of active `(realization, tile)`
//! pairs. Outputs are folded on the GPU and read back.
//!
//! Scope: contact spread with the three ROS models and the two moisture
//! models; per-cell or scalar weather per window; firefighting actions
//! (decaying additional moisture, vegetation changes); forced point
//! ignitions; checkpoints and domain growth through
//! [`GpuPropagator::checkpoint`] / [`GpuPropagator::from_checkpoint`];
//! boundary mode "ignore". Not yet: ember spotting.
//!
//! The GPU-facing methods are async (they await buffer mappings) so the same
//! code runs on WebGPU in the browser; native callers get blocking wrappers.

use std::borrow::Cow;

use propagator_core::{FuelSystem, Grid2, MoistureModel, RosModel, NO_FUEL};
use wgpu::util::DeviceExt;

/// GPU tiles are 16x16: one cell per thread of a 256-wide workgroup
const TILE_SHIFT: usize = 4;
const TILE_SIZE: usize = 1 << TILE_SHIFT;
const INF: u32 = u32::MAX;
const CODE_IGNITION: u32 = 8;
const NO_FUEL_IDX: u32 = u32::MAX;
const FOLD_FIELDS: usize = 8;
const WORKGROUP: u32 = 256;
/// control words per ping-pong half: count, dispatch x, y, z
const CTRL_WORDS: usize = 4;
/// per-fuel columns in the fuel table: v0, d0, d1, hhv, humidity, spotting,
/// prob_ign_by_embers
const FUEL_COLUMNS: usize = 7;
/// per-minute fractional decay of action moisture (as the CPU core)
const ACTIONS_MOISTURE_DECAY: f64 = 0.01;

#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    Device(String),
    Config(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter(m) => write!(f, "no GPU adapter: {m}"),
            GpuError::Device(m) => write!(f, "GPU device: {m}"),
            GpuError::Config(m) => write!(f, "invalid configuration: {m}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    cols: u32,
    n_reals: u32,
    seed: u32,
    tiles_w: u32,
    tiles_h: u32,
    n_fuels: u32,
    do_spotting: u32,
    bucket_end: u32,
    fold_time: u32,
    ros_model: u32,
    moisture_model: u32,
    cellsize: f32,
    next_end: u32,
    processing: u32,
    phase: u32,
    origin_row: i32,
    origin_col: i32,
    _pad: [u32; 2],
}

/// A weather input: one value for the whole grid or a per-cell grid, in the
/// core's external units (wind direction degrees from north, wind speed
/// km/h, moisture percent).
#[derive(Clone, Debug)]
pub enum Field {
    Scalar(f64),
    Grid(Grid2<f32>),
}

impl Field {
    fn to_vec(
        &self,
        n_cells: usize,
        shape: (usize, usize),
        transform: impl Fn(f32) -> f32,
    ) -> Result<Vec<f32>> {
        match self {
            Field::Scalar(v) => Ok(vec![transform(*v as f32); n_cells]),
            Field::Grid(g) => {
                if g.shape() != shape {
                    return Err(GpuError::Config(format!(
                        "weather grid shape {:?} does not match the domain {:?}",
                        g.shape(),
                        shape
                    )));
                }
                Ok(g.as_slice().iter().map(|&v| transform(v)).collect())
            }
        }
    }
}

/// Weather and action inputs applied at a time, mirroring the core's
/// `BoundaryConditions`: every field is optional and keeps its previous value
/// when absent.
#[derive(Clone, Debug, Default)]
pub struct GpuBoundaryConditions {
    pub time: i64,
    /// percent
    pub moisture: Option<Field>,
    /// degrees clockwise from north
    pub wind_dir: Option<Field>,
    /// km/h
    pub wind_speed: Option<Field>,
    /// point ignitions `(row, col)` at `time`, every realization
    pub ignitions: Vec<(usize, usize)>,
    /// extra moisture from actions, fraction, added to the decaying action
    /// layer
    pub additional_moisture: Option<Grid2<f32>>,
    /// vegetation overrides, NaN = no change
    pub vegetation_changes: Option<Grid2<f64>>,
}

/// Construction parameters.
pub struct GpuConfig {
    pub veg: Grid2<i32>,
    pub dem: Grid2<f64>,
    pub fuels: FuelSystem,
    pub cellsize: f64,
    pub realizations: usize,
    pub seed: u32,
    pub ros_model: RosModel,
    pub moisture_model: MoistureModel,
    /// ember spotting (as the core's `do_spotting`)
    pub do_spotting: bool,
    /// world (row, col) of local cell (0, 0); the Bernoulli draws are hashed
    /// by world position, so growing the domain keeps them
    pub origin: (i64, i64),
    /// width of a time bucket, seconds. Ordering granularity only: results
    /// do not depend on it (every window ends with sweeps to convergence),
    /// throughput does. 900 s with 6 sweeps measured best on an M4 Pro.
    pub bucket_seconds: u32,
    /// relaxation sweeps enqueued per bucket; sweeps over an empty worklist
    /// cost one small pass each
    pub sweeps_per_bucket: u32,
}

impl GpuConfig {
    pub fn new(veg: Grid2<i32>, dem: Grid2<f64>, fuels: FuelSystem) -> GpuConfig {
        GpuConfig {
            veg,
            dem,
            fuels,
            cellsize: 20.0,
            realizations: 100,
            seed: 12345,
            ros_model: RosModel::default(),
            moisture_model: MoistureModel::default(),
            do_spotting: false,
            origin: (0, 0),
            bucket_seconds: 900,
            sweeps_per_bucket: 6,
        }
    }
}

/// Folded outputs over the realizations, for cells burnt by the fold time.
pub struct GpuOutput {
    pub time: i64,
    pub fire_probability: Grid2<f32>,
    /// seconds; 0 where never burnt
    pub min_arrival_time: Grid2<f32>,
    /// seconds; NaN where never burnt
    pub mean_arrival_time: Grid2<f32>,
    pub ros_mean: Grid2<f32>,
    pub ros_max: Grid2<f32>,
    pub fli_mean: Grid2<f32>,
    pub fli_max: Grid2<f32>,
    /// fraction of realizations in which the cell was ignited by an ember
    pub spotting_receiving_probability: Grid2<f32>,
    /// expected burned area, m^2
    pub area_mean: f64,
    /// area with burn probability >= 0.5, m^2
    pub area_50: f64,
}

/// The dynamic state of a [`GpuPropagator`], read back to the host:
/// re-attach it to a propagator over the same or a larger grid with
/// [`GpuPropagator::restore`] / [`GpuPropagator::from_checkpoint`].
#[derive(Clone)]
pub struct GpuCheckpoint {
    pub time: i64,
    pub rows: usize,
    pub cols: usize,
    pub realizations: usize,
    /// world (row, col) of the checkpoint's cell (0, 0)
    pub origin: (i64, i64),
    /// `arrival << 4 | code` per (realization, cell)
    key: Vec<u32>,
    pushed_at: Vec<u32>,
    payload: Vec<u32>,
    wind_dir: Vec<f32>,
    wind_speed: Vec<f32>,
    moisture: Vec<f32>,
    actions: Option<Vec<f32>>,
    veg: Grid2<i32>,
    weather_set: bool,
}

struct Pipelines {
    relax: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    build_list: wgpu::ComputePipeline,
    finalize_list: wgpu::ComputePipeline,
    fold: wgpu::ComputePipeline,
    fill_inf: wgpu::ComputePipeline,
    rebuild_tile_min: wgpu::ComputePipeline,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    debug_edges: wgpu::ComputePipeline,
}

struct Buffers {
    inputs: wgpu::Buffer,
    key: wgpu::Buffer,
    pushed_at: wgpu::Buffer,
    tile_min: wgpu::Buffer,
    fold: wgpu::Buffer,
    list: wgpu::Buffer,
    member: wgpu::Buffer,
    ctrl: wgpu::Buffer,
    payload: wgpu::Buffer,
    /// indirect dispatch arguments, two halves of `[x, y, z]`, copied out of
    /// `ctrl` (a buffer cannot be bound as storage and used as indirect
    /// arguments in the same dispatch)
    indirect: wgpu::Buffer,
    staging_small: wgpu::Buffer,
    staging_fold: wgpu::Buffer,
    staging_state: wgpu::Buffer,
}

/// GPU-resident simulation state for one scenario.
pub struct GpuPropagator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    bufs: Buffers,
    /// uniform parameter buffers indexed by `[processing][phase]`
    params_bufs: [[wgpu::Buffer; 2]; 2],
    /// bind groups indexed by `[processing][phase]`
    bind_groups: [[wgpu::BindGroup; 2]; 2],
    params: Params,
    rows: usize,
    cols: usize,
    n_tiles: usize,
    realizations: usize,
    cellsize: f64,
    fuels: FuelSystem,
    veg: Grid2<i32>,
    origin: (i64, i64),
    /// word offset of the weather fields in `inputs`
    weather_offset: usize,
    weather_set: bool,
    wind_dir: Vec<f32>,
    wind_speed: Vec<f32>,
    /// base fuel moisture, fraction
    moisture: Vec<f32>,
    /// decaying action moisture, fraction (`None` until an action arrives)
    actions: Option<Vec<f32>>,
    time: i64,
    bucket_seconds: u32,
    sweeps_per_bucket: u32,
    /// which half of the ping-pong buffers is current
    phase: usize,
}

fn ros_code(m: RosModel) -> u32 {
    match m {
        RosModel::Wang => 0,
        RosModel::Rothermel => 1,
        RosModel::Standard => 2,
    }
}

fn moisture_code(m: MoistureModel) -> u32 {
    match m {
        MoistureModel::Trucchia => 0,
        MoistureModel::Baghino => 1,
    }
}

/// Dense fuel index per cell, `NO_FUEL_IDX` where fire can neither enter nor
/// leave (no fuel or a non-burning fuel).
fn fuel_idx_words(fuels: &FuelSystem, veg: &Grid2<i32>) -> Result<Vec<u32>> {
    let grid = fuels
        .build_fuel_index_grid(veg)
        .map_err(|e| GpuError::Config(e.to_string()))?;
    let burn = fuels.burn();
    Ok(veg
        .as_slice()
        .iter()
        .zip(grid.as_slice())
        .map(|(&v, &idx)| {
            if v == NO_FUEL || !burn[idx as usize] {
                NO_FUEL_IDX
            } else {
                idx as u32
            }
        })
        .collect())
}

impl GpuPropagator {
    /// Open the default high-performance adapter and upload the scenario
    /// (blocking; native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(config: GpuConfig) -> Result<GpuPropagator> {
        pollster::block_on(Self::new_async(config))
    }

    pub async fn new_async(config: GpuConfig) -> Result<GpuPropagator> {
        let (rows, cols) = config.veg.shape();
        if config.dem.shape() != (rows, cols) {
            return Err(GpuError::Config("veg and dem shapes differ".into()));
        }
        if config.realizations == 0 {
            return Err(GpuError::Config("realizations must be >= 1".into()));
        }
        let n_cells = rows * cols;
        let tiles_h = rows.div_ceil(TILE_SIZE);
        let tiles_w = cols.div_ceil(TILE_SIZE);
        let n_tiles = tiles_h * tiles_w;
        let n_rt = config.realizations * n_tiles;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("propagator-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let state_bytes = (config.realizations * n_cells * 4) as u64;
        if state_bytes > limits.max_storage_buffer_binding_size as u64 {
            return Err(GpuError::Config(format!(
                "realizations x cells state ({} MiB) exceeds the adapter's storage binding limit ({} MiB)",
                state_bytes >> 20,
                limits.max_storage_buffer_binding_size >> 20
            )));
        }

        // static inputs, packed into one u32 buffer, weather fields last
        let n_fuels = config.fuels.n_fuels();
        let mut inputs: Vec<u32> = fuel_idx_words(&config.fuels, &config.veg)?;
        inputs.extend(config.dem.as_slice().iter().map(|&v| (v as f32).to_bits()));
        for i in 0..n_fuels {
            for table in [
                config.fuels.v0(),
                config.fuels.d0(),
                config.fuels.d1(),
                config.fuels.hhv(),
                config.fuels.humidity(),
            ] {
                inputs.push((table[i] as f32).to_bits());
            }
            let spotting = config.do_spotting && config.fuels.spotting()[i];
            inputs.push((spotting as u32 as f32).to_bits());
            inputs.push((config.fuels.prob_ign_by_embers()[i] as f32).to_bits());
        }
        debug_assert_eq!(FUEL_COLUMNS, 7);
        if config.origin.0.abs() > i32::MAX as i64 / 2
            || config.origin.1.abs() > i32::MAX as i64 / 2
        {
            return Err(GpuError::Config("origin out of range".into()));
        }
        inputs.extend(
            config
                .fuels
                .spread_probabilities()
                .iter()
                .map(|&p| (p as f32).to_bits()),
        );
        let weather_offset = inputs.len();
        inputs.resize(weather_offset + 3 * n_cells, 0);

        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
            n_reals: config.realizations as u32,
            seed: config.seed,
            tiles_w: tiles_w as u32,
            tiles_h: tiles_h as u32,
            n_fuels: n_fuels as u32,
            do_spotting: config.do_spotting as u32,
            bucket_end: 0,
            fold_time: 0,
            ros_model: ros_code(config.ros_model),
            moisture_model: moisture_code(config.moisture_model),
            cellsize: config.cellsize as f32,
            next_end: 0,
            processing: 1,
            phase: 0,
            origin_row: config.origin.0 as i32,
            origin_col: config.origin.1 as i32,
            _pad: [0; 2],
        };

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let empty = |label: &str, len: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let params_for = |processing: u32, phase: u32| Params {
            processing,
            phase,
            ..params
        };
        let params_bufs = [
            [
                init(
                    "params[0][0]",
                    bytemuck::bytes_of(&params_for(0, 0)),
                    uniform,
                ),
                init(
                    "params[0][1]",
                    bytemuck::bytes_of(&params_for(0, 1)),
                    uniform,
                ),
            ],
            [
                init(
                    "params[1][0]",
                    bytemuck::bytes_of(&params_for(1, 0)),
                    uniform,
                ),
                init(
                    "params[1][1]",
                    bytemuck::bytes_of(&params_for(1, 1)),
                    uniform,
                ),
            ],
        ];
        let state_usage = storage | wgpu::BufferUsages::COPY_SRC;
        let bufs = Buffers {
            inputs: init("inputs", bytemuck::cast_slice(&inputs), storage),
            // large state buffers are allocated empty and filled on the GPU
            key: empty("key", config.realizations * n_cells, state_usage),
            pushed_at: empty("pushed_at", config.realizations * n_cells, state_usage),
            payload: empty("payload", config.realizations * n_cells, state_usage),
            tile_min: empty("tile_min", 2 * n_rt, storage),
            list: empty("list", 2 * n_rt, storage),
            member: init(
                "member",
                bytemuck::cast_slice(&vec![0u32; 2 * n_rt]),
                storage,
            ),
            ctrl: init(
                "ctrl",
                bytemuck::cast_slice(&[0u32; 2 * CTRL_WORDS]),
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            indirect: empty(
                "indirect",
                2 * 3,
                wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            ),
            fold: init(
                "fold",
                bytemuck::cast_slice(&vec![0u32; FOLD_FIELDS * n_cells]),
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            staging_small: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_small"),
                size: 256,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            staging_fold: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_fold"),
                size: (FOLD_FIELDS * n_cells * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            staging_state: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_state"),
                size: state_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("propagator"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("propagator"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, rw),
                entry(3, rw),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
                entry(7, rw),
                entry(8, rw),
                entry(9, rw),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("propagator"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            relax: pipeline("relax_list"),
            prepare: pipeline("prepare"),
            build_list: pipeline("build_list"),
            finalize_list: pipeline("finalize_list"),
            fold: pipeline("fold_cells"),
            fill_inf: pipeline("fill_inf"),
            rebuild_tile_min: pipeline("rebuild_tile_min"),
            debug_edges: pipeline("debug_edges"),
        };
        let bind_group = |params_buf: &wgpu::Buffer| {
            fn b(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry {
                    binding,
                    resource: buf.as_entire_binding(),
                }
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("propagator"),
                layout: &layout,
                entries: &[
                    b(0, params_buf),
                    b(1, &bufs.inputs),
                    b(2, &bufs.key),
                    b(3, &bufs.pushed_at),
                    b(4, &bufs.tile_min),
                    b(5, &bufs.fold),
                    b(6, &bufs.list),
                    b(7, &bufs.member),
                    b(8, &bufs.ctrl),
                    b(9, &bufs.payload),
                ],
            })
        };
        let bind_groups = [
            [
                bind_group(&params_bufs[0][0]),
                bind_group(&params_bufs[0][1]),
            ],
            [
                bind_group(&params_bufs[1][0]),
                bind_group(&params_bufs[1][1]),
            ],
        ];

        let mut sim = GpuPropagator {
            device,
            queue,
            pipelines,
            bufs,
            params_bufs,
            bind_groups,
            params,
            rows,
            cols,
            n_tiles,
            realizations: config.realizations,
            cellsize: config.cellsize,
            fuels: config.fuels,
            veg: config.veg,
            origin: config.origin,
            weather_offset,
            weather_set: false,
            wind_dir: vec![0.0; n_cells],
            wind_speed: vec![0.0; n_cells],
            moisture: vec![0.0; n_cells],
            actions: None,
            time: 0,
            bucket_seconds: config.bucket_seconds.max(1),
            sweeps_per_bucket: config.sweeps_per_bucket.max(1),
            phase: 0,
        };
        sim.fill_state();
        Ok(sim)
    }

    pub fn time(&self) -> i64 {
        self.time
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn realizations(&self) -> usize {
        self.realizations
    }

    /// Current vegetation (after any vegetation changes).
    pub fn veg(&self) -> &Grid2<i32> {
        &self.veg
    }

    /// World (row, col) of local cell (0, 0).
    pub fn origin(&self) -> (i64, i64) {
        self.origin
    }

    /// Upload the four parameter variants (`processing` x `phase`).
    fn write_params(&self) {
        for (processing, bufs) in self.params_bufs.iter().enumerate() {
            for (phase, buf) in bufs.iter().enumerate() {
                let p = Params {
                    processing: processing as u32,
                    phase: phase as u32,
                    ..self.params
                };
                self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
            }
        }
    }

    fn bind_group(&self, processing: bool) -> &wgpu::BindGroup {
        &self.bind_groups[processing as usize][self.phase]
    }

    fn encode_state_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, self.bind_group(true), &[]);
        let n = (self.realizations * self.rows * self.cols) as u32;
        pass.dispatch_workgroups(n.div_ceil(WORKGROUP).min(65535), 1, 1);
    }

    /// Set every arrival / pushed-at / tile minimum to INF on the GPU.
    fn fill_state(&mut self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.fill_inf);
            pass.set_bind_group(0, self.bind_group(true), &[]);
            pass.dispatch_workgroups(65535, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn write_weather_field(&self, index: usize, values: &[f32]) {
        let words: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        let n_cells = self.rows * self.cols;
        self.queue.write_buffer(
            &self.bufs.inputs,
            ((self.weather_offset + index * n_cells) * 4) as u64,
            bytemuck::cast_slice(&words),
        );
    }

    /// Effective moisture (base plus decayed actions, clipped) to the GPU.
    fn upload_effective_moisture(&self) {
        match &self.actions {
            None => self.write_weather_field(2, &self.moisture),
            Some(actions) => {
                let eff: Vec<f32> = self
                    .moisture
                    .iter()
                    .zip(actions)
                    .map(|(m, a)| (m + a).clamp(0.0, 1.0))
                    .collect();
                self.write_weather_field(2, &eff);
            }
        }
    }

    /// Decay the action moisture by `seconds` and refresh the GPU field (as
    /// the CPU core does before each propagation segment).
    fn decay_actions(&mut self, seconds: i64) {
        let Some(actions) = &mut self.actions else {
            return;
        };
        let minutes = (seconds as f64 / 60.0).max(0.0);
        if minutes == 0.0 {
            return;
        }
        let factor = (1.0 - ACTIONS_MOISTURE_DECAY).powf(minutes) as f32;
        for v in actions.iter_mut() {
            *v *= factor;
        }
        self.upload_effective_moisture();
    }

    /// Scalar weather for the window starting at `time` plus point
    /// ignitions (convenience over [`set_boundary_conditions`](Self::set_boundary_conditions)).
    pub fn set_scalar_weather(
        &mut self,
        time: i64,
        wind_dir_deg: f64,
        wind_speed_kmh: f64,
        moisture_pct: f64,
        ignitions: &[(usize, usize)],
    ) -> Result<()> {
        self.set_boundary_conditions(GpuBoundaryConditions {
            time,
            moisture: Some(Field::Scalar(moisture_pct)),
            wind_dir: Some(Field::Scalar(wind_dir_deg)),
            wind_speed: Some(Field::Scalar(wind_speed_kmh)),
            ignitions: ignitions.to_vec(),
            ..Default::default()
        })
    }

    /// Apply weather, actions and ignitions at `bc.time`, which must not be in
    /// the past. Weather fields are external units like the core's
    /// `BoundaryConditions`; absent fields keep their values.
    pub fn set_boundary_conditions(&mut self, bc: GpuBoundaryConditions) -> Result<()> {
        if bc.time < self.time {
            return Err(GpuError::Config("boundary conditions in the past".into()));
        }
        let n_cells = self.rows * self.cols;
        let shape = (self.rows, self.cols);
        if let Some(f) = &bc.wind_dir {
            self.wind_dir = f.to_vec(n_cells, shape, |v| v.to_radians())?;
            self.write_weather_field(0, &self.wind_dir);
        }
        if let Some(f) = &bc.wind_speed {
            self.wind_speed = f.to_vec(n_cells, shape, |v| v)?;
            self.write_weather_field(1, &self.wind_speed);
        }
        let mut moisture_changed = false;
        if let Some(f) = &bc.moisture {
            self.moisture = f.to_vec(n_cells, shape, |v| v / 100.0)?;
            moisture_changed = true;
        }
        if let Some(extra) = &bc.additional_moisture {
            if extra.shape() != shape {
                return Err(GpuError::Config(
                    "additional_moisture shape mismatch".into(),
                ));
            }
            let actions = self.actions.get_or_insert_with(|| vec![0.0; n_cells]);
            for (a, b) in actions.iter_mut().zip(extra.as_slice()) {
                *a = (*a + *b).clamp(0.0, 1.0);
            }
            moisture_changed = true;
        }
        if moisture_changed {
            self.upload_effective_moisture();
        }
        if let Some(changes) = &bc.vegetation_changes {
            if changes.shape() != shape {
                return Err(GpuError::Config("vegetation_changes shape mismatch".into()));
            }
            for (cell, change) in self.veg.as_mut_slice().iter_mut().zip(changes.as_slice()) {
                if !change.is_nan() {
                    *cell = *change as i32;
                }
            }
            let words = fuel_idx_words(&self.fuels, &self.veg)?;
            self.queue
                .write_buffer(&self.bufs.inputs, 0, bytemuck::cast_slice(&words));
        }
        if bc.wind_dir.is_some() && bc.wind_speed.is_some() && bc.moisture.is_some() {
            self.weather_set = true;
        }
        self.write_ignitions(bc.time, &bc.ignitions)
    }

    fn write_ignitions(&mut self, time: i64, ignitions: &[(usize, usize)]) -> Result<()> {
        let n_cells = self.rows * self.cols;
        let n_rt = self.realizations * self.n_tiles;
        let k = ((time as u32) << 4) | CODE_IGNITION;
        for &(row, col) in ignitions {
            if row >= self.rows || col >= self.cols {
                return Err(GpuError::Config(format!(
                    "ignition ({row}, {col}) outside the grid"
                )));
            }
            let c = row * self.cols + col;
            let tile = (row >> TILE_SHIFT) * self.cols.div_ceil(TILE_SIZE) + (col >> TILE_SHIFT);
            for r in 0..self.realizations {
                // forced ignitions overwrite: at t = now they are the
                // earliest possible event; they carry no ROS / FLI
                let idx = r * n_cells + c;
                self.queue
                    .write_buffer(&self.bufs.key, (idx * 4) as u64, bytemuck::bytes_of(&k));
                self.queue
                    .write_buffer(&self.bufs.payload, (idx * 4) as u64, &[0u8; 4]);
                let t_idx = self.phase * n_rt + r * self.n_tiles + tile;
                self.queue.write_buffer(
                    &self.bufs.tile_min,
                    (t_idx * 4) as u64,
                    bytemuck::bytes_of(&(time as u32)),
                );
            }
        }
        Ok(())
    }

    fn encode_rt_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        processing: bool,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, self.bind_group(processing), &[]);
        let n = (self.realizations * self.n_tiles) as u32;
        pass.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
    }

    /// Derive the next list's indirect dispatch arguments and copy them into
    /// the indirect buffer's next half.
    fn encode_finalize(&self, encoder: &mut wgpu::CommandEncoder, processing: bool) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.finalize_list);
            pass.set_bind_group(0, self.bind_group(processing), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let next = 1 - self.phase;
        encoder.copy_buffer_to_buffer(
            &self.bufs.ctrl,
            ((next * CTRL_WORDS + 1) * 4) as u64,
            &self.bufs.indirect,
            (next * 12) as u64,
            12,
        );
    }

    /// Queue every tile with unpushed arrivals below `next_end` (bucket
    /// start), then flip the phase.
    fn encode_build_list(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_rt_pass(encoder, &self.pipelines.prepare, false);
        self.encode_rt_pass(encoder, &self.pipelines.build_list, false);
        self.encode_finalize(encoder, false);
        self.phase = 1 - self.phase;
    }

    /// One relaxation sweep over the queued tiles, then flip the phase.
    fn encode_sweep(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_rt_pass(encoder, &self.pipelines.prepare, true);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.relax);
            pass.set_bind_group(0, self.bind_group(true), &[]);
            pass.dispatch_workgroups_indirect(&self.bufs.indirect, (self.phase * 12) as u64);
        }
        self.encode_finalize(encoder, true);
        self.phase = 1 - self.phase;
    }

    /// Map a staging buffer for reading once the queue has caught up. On
    /// native the device is polled to completion; on the web the browser
    /// resolves the mapping while the future is awaited.
    async fn map_read(&self, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let slice = buffer.slice(..bytes);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.await
            .map_err(|_| GpuError::Device("map callback dropped".into()))?
            .map_err(|e| GpuError::Device(format!("map buffer: {e:?}")))?;
        let data = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| GpuError::Device(format!("mapped range: {e:?}")))?;
            mapped.to_vec()
        };
        buffer.unmap();
        Ok(data)
    }

    /// Copy a whole GPU buffer into the state staging buffer and read it.
    async fn read_state_buffer(&self, buffer: &wgpu::Buffer) -> Result<Vec<u32>> {
        let bytes = (self.realizations * self.rows * self.cols * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &self.bufs.staging_state, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let data = self.map_read(&self.bufs.staging_state, bytes).await?;
        Ok(bytemuck::cast_slice(&data).to_vec())
    }

    /// Number of tiles queued for the next sweep (the current worklist).
    async fn read_queued(&self) -> Result<u32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let offset = (self.phase * CTRL_WORDS * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.bufs.ctrl, offset, &self.bufs.staging_small, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        let data = self.map_read(&self.bufs.staging_small, 4).await?;
        Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
    }

    /// Advance to `end_time` (seconds), bucket by bucket, then sweep until no
    /// tile with an arrival below `end_time` is left unpushed (blocking;
    /// native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn advance_to(&mut self, end_time: i64) -> Result<()> {
        pollster::block_on(self.advance_to_async(end_time))
    }

    /// Async form of [`advance_to`](Self::advance_to), usable on the web.
    pub async fn advance_to_async(&mut self, end_time: i64) -> Result<()> {
        if !self.weather_set {
            return Err(GpuError::Config(
                "wind direction, wind speed and moisture must be set first".into(),
            ));
        }
        if end_time <= self.time {
            return Ok(());
        }
        // action moisture decays over the segment before it is simulated
        self.decay_actions(end_time - self.time);
        let mut t = self.time as u32;
        let end = end_time as u32;
        while t < end {
            let bucket_end = (t + self.bucket_seconds).min(end);
            self.params.bucket_end = bucket_end;
            self.params.next_end = bucket_end;
            self.write_params();
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.encode_build_list(&mut encoder);
            for _ in 0..self.sweeps_per_bucket {
                self.encode_sweep(&mut encoder);
            }
            self.queue.submit(Some(encoder.finish()));
            t = bucket_end;
        }
        // converge: everything below end_time must be pushed before the
        // state is observed or the weather changes
        self.params.bucket_end = end;
        self.params.next_end = end;
        self.write_params();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_build_list(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        loop {
            if self.read_queued().await? == 0 {
                break;
            }
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.encode_sweep(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
        }
        self.time = end_time;
        Ok(())
    }

    /// Debug: per-direction `[p, transition, ros, fli, angle, uniform,
    /// wind_dir, wind_speed]` of one cell under the current weather.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn debug_edges(&mut self, row: usize, col: usize) -> Result<Vec<[f32; 8]>> {
        self.params.fold_time = (row * self.cols + col) as u32;
        self.write_params();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.debug_edges);
            pass.set_bind_group(0, self.bind_group(true), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.bufs.fold, 0, &self.bufs.staging_small, 0, 64 * 4);
        self.queue.submit(Some(encoder.finish()));
        let bytes = pollster::block_on(self.map_read(&self.bufs.staging_small, 256))?;
        let data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        Ok((0..8)
            .map(|d| {
                let mut a = [0.0f32; 8];
                a.copy_from_slice(&data[d * 8..d * 8 + 8]);
                a
            })
            .collect())
    }

    /// Fold the realizations on the GPU and read the per-cell outputs back
    /// (blocking; native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_output(&mut self) -> Result<GpuOutput> {
        pollster::block_on(self.get_output_async())
    }

    /// Async form of [`get_output`](Self::get_output), usable on the web.
    pub async fn get_output_async(&mut self) -> Result<GpuOutput> {
        let n_cells = self.rows * self.cols;
        self.params.fold_time = self.time as u32;
        self.write_params();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.fold);
            pass.set_bind_group(0, self.bind_group(true), &[]);
            pass.dispatch_workgroups((n_cells as u32).div_ceil(WORKGROUP), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.bufs.fold,
            0,
            &self.bufs.staging_fold,
            0,
            (FOLD_FIELDS * n_cells * 4) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let bytes = self
            .map_read(&self.bufs.staging_fold, (FOLD_FIELDS * n_cells * 4) as u64)
            .await?;
        let data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

        let n = self.realizations as f32;
        let field = |i: usize| &data[i * n_cells..(i + 1) * n_cells];
        let count = field(0);
        let arrival_min = field(1);
        let arrival_sum = field(2);
        let ros_sum = field(3);
        let fli_sum = field(4);
        let ros_max = field(5);
        let fli_max = field(6);
        let spot_recv = field(7);
        let grid = |v: Vec<f32>| Grid2::from_vec(self.rows, self.cols, v);
        let fire_probability: Vec<f32> = count.iter().map(|c| c / n).collect();
        let min_arrival: Vec<f32> = arrival_min
            .iter()
            .zip(count)
            .map(|(&a, &c)| if c > 0.0 { a.to_bits() as f32 } else { 0.0 })
            .collect();
        let masked_mean = |sum: &[f32]| -> Vec<f32> {
            sum.iter()
                .zip(count)
                .map(|(&s, &c)| if c > 0.0 { s / c } else { f32::NAN })
                .collect()
        };
        let cell_area = self.cellsize * self.cellsize;
        let mut area_mean = 0.0f64;
        let mut area_50 = 0.0f64;
        for &p in &fire_probability {
            area_mean += p as f64;
            if p >= 0.5 {
                area_50 += 1.0;
            }
        }
        Ok(GpuOutput {
            time: self.time,
            mean_arrival_time: grid(masked_mean(arrival_sum)),
            ros_mean: grid(masked_mean(ros_sum)),
            fli_mean: grid(masked_mean(fli_sum)),
            ros_max: grid(ros_max.to_vec()),
            fli_max: grid(fli_max.to_vec()),
            min_arrival_time: grid(min_arrival),
            fire_probability: grid(fire_probability),
            spotting_receiving_probability: grid(spot_recv.iter().map(|c| c / n).collect()),
            area_mean: area_mean * cell_area,
            area_50: area_50 * cell_area,
        })
    }

    // --- checkpoints and domain growth ------------------------------------

    /// Read the dynamic state back into a host checkpoint (blocking; native
    /// only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn checkpoint(&mut self) -> Result<GpuCheckpoint> {
        pollster::block_on(self.checkpoint_async())
    }

    /// Async form of [`checkpoint`](Self::checkpoint).
    pub async fn checkpoint_async(&mut self) -> Result<GpuCheckpoint> {
        Ok(GpuCheckpoint {
            time: self.time,
            rows: self.rows,
            cols: self.cols,
            realizations: self.realizations,
            origin: self.origin,
            key: self.read_state_buffer(&self.bufs.key).await?,
            pushed_at: self.read_state_buffer(&self.bufs.pushed_at).await?,
            payload: self.read_state_buffer(&self.bufs.payload).await?,
            wind_dir: self.wind_dir.clone(),
            wind_speed: self.wind_speed.clone(),
            moisture: self.moisture.clone(),
            actions: self.actions.clone(),
            veg: self.veg.clone(),
            weather_set: self.weather_set,
        })
    }

    /// Replace the dynamic state with a checkpoint's, re-anchored by the two
    /// origins (the checkpoint's world extent must lie inside this grid).
    /// Vegetation in the overlap comes from the checkpoint, weather is
    /// edge-padded, action moisture zero-padded; the draws are unchanged
    /// because they are hashed by world position.
    pub fn restore(&mut self, cp: &GpuCheckpoint) -> Result<()> {
        if cp.realizations != self.realizations {
            return Err(GpuError::Config("realizations mismatch".into()));
        }
        let (dr, dc) = (cp.origin.0 - self.origin.0, cp.origin.1 - self.origin.1);
        if dr < 0 || dc < 0 {
            return Err(GpuError::Config(
                "checkpoint origin lies north/west of this grid".into(),
            ));
        }
        let (row_shift, col_shift) = (dr as usize, dc as usize);
        if cp.rows + row_shift > self.rows || cp.cols + col_shift > self.cols {
            return Err(GpuError::Config(
                "checkpoint does not fit in the grid".into(),
            ));
        }
        let n_cells = self.rows * self.cols;
        let shape = (self.rows, self.cols);
        let place = |src: &[u32], fill: u32| -> Vec<u32> {
            let mut out = vec![fill; self.realizations * n_cells];
            for r in 0..self.realizations {
                for row in 0..cp.rows {
                    let s0 = (r * cp.rows + row) * cp.cols;
                    let d0 = r * n_cells + (row + row_shift) * self.cols + col_shift;
                    out[d0..d0 + cp.cols].copy_from_slice(&src[s0..s0 + cp.cols]);
                }
            }
            out
        };
        let pad_edge = |v: &[f32]| -> Vec<f32> {
            Grid2::from_vec(cp.rows, cp.cols, v.to_vec())
                .pad_edge(shape, row_shift, col_shift)
                .as_slice()
                .to_vec()
        };
        // vegetation: overlap from the checkpoint, fuel index re-uploaded
        let veg = cp.veg.pad_const(shape, row_shift, col_shift, i32::MIN);
        for (dst, &src) in self.veg.as_mut_slice().iter_mut().zip(veg.as_slice()) {
            if src != i32::MIN {
                *dst = src;
            }
        }
        let words = fuel_idx_words(&self.fuels, &self.veg)?;
        self.queue
            .write_buffer(&self.bufs.inputs, 0, bytemuck::cast_slice(&words));

        self.wind_dir = pad_edge(&cp.wind_dir);
        self.wind_speed = pad_edge(&cp.wind_speed);
        self.moisture = pad_edge(&cp.moisture);
        self.actions = cp.actions.as_ref().map(|a| {
            Grid2::from_vec(cp.rows, cp.cols, a.clone())
                .pad_const(shape, row_shift, col_shift, 0.0)
                .as_slice()
                .to_vec()
        });
        self.weather_set = cp.weather_set;
        self.write_weather_field(0, &self.wind_dir);
        self.write_weather_field(1, &self.wind_speed);
        self.upload_effective_moisture();

        self.queue.write_buffer(
            &self.bufs.key,
            0,
            bytemuck::cast_slice(&place(&cp.key, INF)),
        );
        self.queue.write_buffer(
            &self.bufs.pushed_at,
            0,
            bytemuck::cast_slice(&place(&cp.pushed_at, INF)),
        );
        self.queue.write_buffer(
            &self.bufs.payload,
            0,
            bytemuck::cast_slice(&place(&cp.payload, 0)),
        );
        self.time = cp.time;
        // tile minima: INF everywhere, then rebuilt from the unpushed cells
        let n_rt = self.realizations * self.n_tiles;
        self.queue.write_buffer(
            &self.bufs.tile_min,
            0,
            bytemuck::cast_slice(&vec![INF; 2 * n_rt]),
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_state_pass(&mut encoder, &self.pipelines.rebuild_tile_min);
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// A propagator over `config`'s grid and origin (same or larger domain)
    /// resuming from a checkpoint; realizations must match (blocking; native
    /// only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_checkpoint(config: GpuConfig, cp: &GpuCheckpoint) -> Result<GpuPropagator> {
        pollster::block_on(Self::from_checkpoint_async(config, cp))
    }

    pub async fn from_checkpoint_async(
        config: GpuConfig,
        cp: &GpuCheckpoint,
    ) -> Result<GpuPropagator> {
        let mut sim = Self::new_async(config).await?;
        sim.restore(cp)?;
        Ok(sim)
    }
}
