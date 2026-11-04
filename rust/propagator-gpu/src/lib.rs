//! WebGPU execution backend for the PROPAGATOR core, on `wgpu`.
//!
//! Reformulates the event-driven kernel as a relaxation over
//! `realizations x rows x cols` arrival-time fields (see
//! `docs/gpu-core.md`): every edge's Bernoulli draw is a hash of
//! `(seed, realization, cell, direction)`, so each realization is a
//! shortest-path problem on a fixed random graph whose edge weights depend on
//! the weather window the source cell ignites in. Work is ordered in time
//! buckets over a GPU-side worklist of active `(realization, tile)` pairs.
//! Outputs are folded on the GPU and read back.
//!
//! Scope of this backend: contact spread with the three ROS models and the
//! two moisture models, scalar (per-window) weather, forced point ignitions,
//! boundary mode "ignore". Not yet: ember spotting, per-cell weather fields,
//! firefighting actions, domain growth, checkpoints.
//!
//! The GPU-facing methods are async (they await buffer mappings) so the same
//! code runs on WebGPU in the browser; native callers get blocking wrappers.

use std::borrow::Cow;

use propagator_core::{FuelSystem, Grid2, MoistureModel, RosModel, NO_FUEL};
use wgpu::util::DeviceExt;

/// GPU tiles are 16x16: one cell per thread of a 256-wide workgroup
const TILE_SHIFT: usize = 4;
const TILE_SIZE: usize = 1 << TILE_SHIFT;
const CODE_IGNITION: u32 = 8;
const NO_FUEL_IDX: u32 = u32::MAX;
const FOLD_FIELDS: usize = 7;
const WORKGROUP: u32 = 256;
/// words per weather window in the inputs buffer
const WINDOW_WORDS: usize = 4;
const MAX_WINDOWS: usize = 1024;
/// control words per ping-pong half: count, dispatch x, y, z
const CTRL_WORDS: usize = 4;

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
    n_windows: u32,
    bucket_end: u32,
    fold_time: u32,
    ros_model: u32,
    moisture_model: u32,
    cellsize: f32,
    next_end: u32,
    processing: u32,
    phase: u32,
}

#[derive(Clone, Copy, Debug)]
struct Window {
    t_start: u32,
    /// radians
    wind_dir: f32,
    /// km/h
    wind_speed: f32,
    /// fraction
    moisture: f32,
}

impl Window {
    fn words(&self) -> [u32; WINDOW_WORDS] {
        [
            self.t_start,
            self.wind_dir.to_bits(),
            self.wind_speed.to_bits(),
            self.moisture.to_bits(),
        ]
    }
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
    /// expected burned area, m^2
    pub area_mean: f64,
    /// area with burn probability >= 0.5, m^2
    pub area_50: f64,
}

struct Pipelines {
    relax: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    build_list: wgpu::ComputePipeline,
    finalize_list: wgpu::ComputePipeline,
    fold: wgpu::ComputePipeline,
    fill_inf: wgpu::ComputePipeline,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    debug_edges: wgpu::ComputePipeline,
}

/// GPU-resident simulation state for one scenario.
pub struct GpuPropagator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    /// uniform parameter buffers indexed by `[processing][phase]`
    params_bufs: [[wgpu::Buffer; 2]; 2],
    /// bind groups indexed by `[processing][phase]`
    bind_groups: [[wgpu::BindGroup; 2]; 2],
    inputs_buf: wgpu::Buffer,
    key_buf: wgpu::Buffer,
    tile_min_buf: wgpu::Buffer,
    ctrl_buf: wgpu::Buffer,
    /// indirect dispatch arguments, two halves of `[x, y, z]`, copied out of
    /// `ctrl` (a buffer cannot be bound as storage and used as indirect
    /// arguments in the same dispatch)
    indirect_buf: wgpu::Buffer,
    fold_buf: wgpu::Buffer,
    staging_small: wgpu::Buffer,
    staging_fold: wgpu::Buffer,
    params: Params,
    windows: Vec<Window>,
    windows_offset: u64,
    rows: usize,
    cols: usize,
    n_tiles: usize,
    realizations: usize,
    cellsize: f64,
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

        // static inputs, packed into one u32 buffer
        let fuel_idx_grid = config
            .fuels
            .build_fuel_index_grid(&config.veg)
            .map_err(|e| GpuError::Config(e.to_string()))?;
        let burn = config.fuels.burn();
        let n_fuels = config.fuels.n_fuels();
        let mut inputs: Vec<u32> = Vec::with_capacity(
            2 * n_cells + 5 * n_fuels + n_fuels * n_fuels + MAX_WINDOWS * WINDOW_WORDS,
        );
        for (&veg, &idx) in config.veg.as_slice().iter().zip(fuel_idx_grid.as_slice()) {
            inputs.push(if veg == NO_FUEL || !burn[idx as usize] {
                NO_FUEL_IDX
            } else {
                idx as u32
            });
        }
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
        }
        inputs.extend(
            config
                .fuels
                .spread_probabilities()
                .iter()
                .map(|&p| (p as f32).to_bits()),
        );
        let windows_offset = (inputs.len() * 4) as u64;
        inputs.resize(inputs.len() + MAX_WINDOWS * WINDOW_WORDS, 0);

        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
            n_reals: config.realizations as u32,
            seed: config.seed,
            tiles_w: tiles_w as u32,
            tiles_h: tiles_h as u32,
            n_fuels: n_fuels as u32,
            n_windows: 0,
            bucket_end: 0,
            fold_time: 0,
            ros_model: ros_code(config.ros_model),
            moisture_model: moisture_code(config.moisture_model),
            cellsize: config.cellsize as f32,
            next_end: 0,
            processing: 1,
            phase: 0,
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
        let inputs_buf = init("inputs", bytemuck::cast_slice(&inputs), storage);
        // large state buffers are allocated empty and filled on the GPU
        let key_buf = empty(
            "key",
            config.realizations * n_cells,
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let pushed_buf = empty("pushed_at", config.realizations * n_cells, storage);
        let tile_min_buf = empty("tile_min", 2 * n_rt, storage);
        let list_buf = empty("list", 2 * n_rt, storage);
        let member_buf = init(
            "member",
            bytemuck::cast_slice(&vec![0u32; 2 * n_rt]),
            storage,
        );
        let ctrl_buf = init(
            "ctrl",
            bytemuck::cast_slice(&[0u32; 2 * CTRL_WORDS]),
            storage | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let indirect_buf = empty(
            "indirect",
            2 * 3,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        );
        let fold_buf = init(
            "fold",
            bytemuck::cast_slice(&vec![0u32; FOLD_FIELDS * n_cells]),
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let staging_small = staging("staging_small", 256);
        let staging_fold = staging("staging_fold", (FOLD_FIELDS * n_cells * 4) as u64);

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
            debug_edges: pipeline("debug_edges"),
        };
        let bind_group = |params_buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("propagator"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: inputs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: key_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pushed_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tile_min_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: fold_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: list_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: member_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: ctrl_buf.as_entire_binding(),
                    },
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
            params_bufs,
            bind_groups,
            inputs_buf,
            key_buf,
            tile_min_buf,
            ctrl_buf,
            indirect_buf,
            fold_buf,
            staging_small,
            staging_fold,
            params,
            windows: Vec::new(),
            windows_offset,
            rows,
            cols,
            n_tiles,
            realizations: config.realizations,
            cellsize: config.cellsize,
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

    /// Set the weather for the window starting at `time` (seconds; windows
    /// must be added in increasing time order, the first at the current
    /// time) and ignite `ignitions` cells at `time` in every realization.
    pub fn set_boundary_conditions(
        &mut self,
        time: i64,
        wind_dir_deg: f64,
        wind_speed_kmh: f64,
        moisture_pct: f64,
        ignitions: &[(usize, usize)],
    ) -> Result<()> {
        if time < self.time {
            return Err(GpuError::Config("boundary conditions in the past".into()));
        }
        if let Some(last) = self.windows.last() {
            if (last.t_start as i64) > time {
                return Err(GpuError::Config(
                    "windows must be in increasing time".into(),
                ));
            }
        }
        if self.windows.len() >= MAX_WINDOWS {
            return Err(GpuError::Config("too many weather windows".into()));
        }
        let window = Window {
            t_start: time as u32,
            wind_dir: wind_dir_deg.to_radians() as f32,
            wind_speed: wind_speed_kmh as f32,
            moisture: (moisture_pct / 100.0) as f32,
        };
        // a window at the same start time replaces the previous one
        match self.windows.last_mut() {
            Some(last) if last.t_start == window.t_start => *last = window,
            _ => self.windows.push(window),
        }
        let words: Vec<u32> = self.windows.iter().flat_map(|w| w.words()).collect();
        self.queue.write_buffer(
            &self.inputs_buf,
            self.windows_offset,
            bytemuck::cast_slice(&words),
        );
        self.params.n_windows = self.windows.len() as u32;
        self.write_params();

        if !ignitions.is_empty() {
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
                let tile =
                    (row >> TILE_SHIFT) * self.cols.div_ceil(TILE_SIZE) + (col >> TILE_SHIFT);
                for r in 0..self.realizations {
                    // forced ignitions overwrite: at t = now they are the
                    // earliest possible event
                    let idx = r * n_cells + c;
                    self.queue.write_buffer(
                        &self.key_buf,
                        (idx * 4) as u64,
                        bytemuck::bytes_of(&k),
                    );
                    let t_idx = self.phase * n_rt + r * self.n_tiles + tile;
                    self.queue.write_buffer(
                        &self.tile_min_buf,
                        (t_idx * 4) as u64,
                        bytemuck::bytes_of(&(time as u32)),
                    );
                }
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
            &self.ctrl_buf,
            ((next * CTRL_WORDS + 1) * 4) as u64,
            &self.indirect_buf,
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
            pass.dispatch_workgroups_indirect(&self.indirect_buf, (self.phase * 12) as u64);
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

    /// Number of tiles queued for the next sweep (the current worklist).
    async fn read_queued(&self) -> Result<u32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let offset = (self.phase * CTRL_WORDS * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.ctrl_buf, offset, &self.staging_small, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        let data = self.map_read(&self.staging_small, 4).await?;
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
        if self.windows.is_empty() {
            return Err(GpuError::Config("no boundary conditions set".into()));
        }
        if end_time <= self.time {
            return Ok(());
        }
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
    /// wind_dir, wind_speed]` of one cell under the first window.
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
        encoder.copy_buffer_to_buffer(&self.fold_buf, 0, &self.staging_small, 0, 64 * 4);
        self.queue.submit(Some(encoder.finish()));
        let bytes = pollster::block_on(self.map_read(&self.staging_small, 256))?;
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
            &self.fold_buf,
            0,
            &self.staging_fold,
            0,
            (FOLD_FIELDS * n_cells * 4) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let bytes = self
            .map_read(&self.staging_fold, (FOLD_FIELDS * n_cells * 4) as u64)
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
            area_mean: area_mean * cell_area,
            area_50: area_50 * cell_area,
        })
    }
}
