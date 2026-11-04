//! WebGPU execution backend for the PROPAGATOR core, on `wgpu`.
//!
//! Reformulates the event-driven kernel as a relaxation over
//! `realizations x rows x cols` arrival-time fields (see
//! `docs/gpu-core.md`): every edge's Bernoulli draw is a hash of
//! `(seed, realization, cell, direction)`, so each realization is a
//! shortest-path problem on a fixed random graph whose edge weights depend on
//! the weather window the source cell ignites in. Work is ordered in time
//! buckets; per `(realization, tile)` the earliest unpushed arrival lets idle
//! tiles exit after one load. Outputs are folded on the GPU and read back.
//!
//! Scope of this backend: contact spread with the three ROS models and the
//! two moisture models, scalar (per-window) weather, forced point ignitions,
//! boundary mode "ignore". Not yet: ember spotting, per-cell weather fields,
//! firefighting actions, domain growth, checkpoints.

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
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Window {
    t_start: u32,
    /// radians
    wind_dir: f32,
    /// km/h
    wind_speed: f32,
    /// fraction
    moisture: f32,
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

/// GPU-resident simulation state for one scenario.
pub struct GpuPropagator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    relax: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    build_list: wgpu::ComputePipeline,
    fold: wgpu::ComputePipeline,
    debug_edges: wgpu::ComputePipeline,
    fill_inf: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    /// same parameters with `processing = 0`, for bucket-start passes
    params_build_buf: wgpu::Buffer,
    fuel_idx_buf: wgpu::Buffer,
    dem_buf: wgpu::Buffer,
    fuel_table_buf: wgpu::Buffer,
    windows_buf: wgpu::Buffer,
    key_buf: wgpu::Buffer,
    pushed_buf: wgpu::Buffer,
    tile_min: [wgpu::Buffer; 2],
    list: [wgpu::Buffer; 2],
    member: [wgpu::Buffer; 2],
    /// `[count, 1, 1]`: indirect dispatch arguments of the worklist
    indirect: [wgpu::Buffer; 2],
    work_buf: wgpu::Buffer,
    fold_buf: wgpu::Buffer,
    staging_work: wgpu::Buffer,
    staging_fold: wgpu::Buffer,
    params: Params,
    windows: Vec<Window>,
    rows: usize,
    cols: usize,
    n_tiles: usize,
    realizations: usize,
    cellsize: f64,
    time: i64,
    bucket_seconds: u32,
    sweeps_per_bucket: u32,
    /// index of the tile_min buffer holding the current minima
    cur: usize,
    max_windows: usize,
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
    /// Open the default high-performance adapter and upload the scenario.
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
        if n_tiles > 65535 {
            return Err(GpuError::Config(
                "grid too large: more than 65535 tiles per realization".into(),
            ));
        }
        if config.realizations > 65535 {
            return Err(GpuError::Config("more than 65535 realizations".into()));
        }

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

        // static inputs
        let fuel_idx_grid = config
            .fuels
            .build_fuel_index_grid(&config.veg)
            .map_err(|e| GpuError::Config(e.to_string()))?;
        let burn = config.fuels.burn();
        let fuel_idx: Vec<u32> = config
            .veg
            .as_slice()
            .iter()
            .zip(fuel_idx_grid.as_slice().iter())
            .map(|(&veg, &idx)| {
                if veg == NO_FUEL || !burn[idx as usize] {
                    NO_FUEL_IDX
                } else {
                    idx as u32
                }
            })
            .collect();
        let dem: Vec<f32> = config.dem.as_slice().iter().map(|&v| v as f32).collect();
        let n_fuels = config.fuels.n_fuels();
        let mut fuel_table: Vec<f32> = Vec::with_capacity(n_fuels * 5 + n_fuels * n_fuels);
        for i in 0..n_fuels {
            fuel_table.push(config.fuels.v0()[i] as f32);
            fuel_table.push(config.fuels.d0()[i] as f32);
            fuel_table.push(config.fuels.d1()[i] as f32);
            fuel_table.push(config.fuels.hhv()[i] as f32);
            fuel_table.push(config.fuels.humidity()[i] as f32);
        }
        fuel_table.extend(
            config
                .fuels
                .spread_probabilities()
                .iter()
                .map(|&p| p as f32),
        );

        let max_windows = 1024usize;
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
            _pad: 0,
        };

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let params_buf = init(
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let params_build_buf = init(
            "params_build",
            bytemuck::bytes_of(&Params {
                processing: 0,
                ..params
            }),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let fuel_idx_buf = init("fuel_idx", bytemuck::cast_slice(&fuel_idx), storage);
        let dem_buf = init("dem", bytemuck::cast_slice(&dem), storage);
        let fuel_table_buf = init("fuel_table", bytemuck::cast_slice(&fuel_table), storage);
        let windows_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("windows"),
            size: (max_windows * std::mem::size_of::<Window>()) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let filled = |label: &str, len: usize, value: u32| {
            init(
                label,
                bytemuck::cast_slice(&vec![value; len]),
                storage | wgpu::BufferUsages::COPY_SRC,
            )
        };
        // large state buffers are allocated empty and filled on the GPU
        let empty = |label: &str, len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: storage | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let key_buf = empty("key", config.realizations * n_cells);
        let pushed_buf = empty("pushed_at", config.realizations * n_cells);
        let tile_min = [
            empty("tile_min_a", config.realizations * n_tiles),
            empty("tile_min_b", config.realizations * n_tiles),
        ];
        let work_buf = filled("work", 1, 0);
        let list = [
            filled("list_a", config.realizations * n_tiles, 0),
            filled("list_b", config.realizations * n_tiles, 0),
        ];
        let member = [
            filled("member_a", config.realizations * n_tiles, 0),
            filled("member_b", config.realizations * n_tiles, 0),
        ];
        let indirect_usage = storage | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC;
        let indirect = [
            init(
                "indirect_a",
                bytemuck::cast_slice(&[0u32, 1, 1]),
                indirect_usage,
            ),
            init(
                "indirect_b",
                bytemuck::cast_slice(&[0u32, 1, 1]),
                indirect_usage,
            ),
        ];
        let fold_buf = filled("fold", FOLD_FIELDS * n_cells, 0);
        let staging = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let staging_work = staging("staging_work", 4);
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
                entry(2, ro),
                entry(3, ro),
                entry(4, ro),
                entry(5, rw),
                entry(6, rw),
                entry(7, rw),
                entry(8, rw),
                entry(9, rw),
                entry(10, rw),
                entry(11, ro),
                entry(12, rw),
                entry(13, ro),
                entry(14, rw),
                entry(15, rw),
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
        let relax = pipeline("relax_list");
        let prepare = pipeline("prepare");
        let build_list = pipeline("build_list");
        let fold = pipeline("fold_cells");
        let debug_edges = pipeline("debug_edges");
        let fill_inf = pipeline("fill_inf");

        let mut sim = GpuPropagator {
            device,
            queue,
            relax,
            prepare,
            build_list,
            fold,
            debug_edges,
            fill_inf,
            layout,
            params_buf,
            params_build_buf,
            fuel_idx_buf,
            dem_buf,
            fuel_table_buf,
            windows_buf,
            key_buf,
            pushed_buf,
            tile_min,
            list,
            member,
            indirect,
            work_buf,
            fold_buf,
            staging_work,
            staging_fold,
            params,
            windows: Vec::new(),
            rows,
            cols,
            n_tiles,
            realizations: config.realizations,
            cellsize: config.cellsize,
            time: 0,
            bucket_seconds: config.bucket_seconds.max(1),
            sweeps_per_bucket: config.sweeps_per_bucket.max(1),
            cur: 0,
            max_windows,
        };
        sim.fill_state();
        Ok(sim)
    }

    /// Set every arrival / pushed-at / tile minimum to INF on the GPU.
    fn fill_state(&mut self) {
        let bg = self.bind_group(self.cur, false);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.fill_inf);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(65535, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
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

    fn bind_group(&self, cur: usize, build: bool) -> wgpu::BindGroup {
        fn b(buf: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
            buf.as_entire_binding()
        }
        let params = if build {
            &self.params_build_buf
        } else {
            &self.params_buf
        };
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("propagator"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b(params),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b(&self.fuel_idx_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b(&self.dem_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b(&self.fuel_table_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b(&self.windows_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b(&self.key_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b(&self.pushed_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b(&self.tile_min[cur]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b(&self.tile_min[1 - cur]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: b(&self.work_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: b(&self.fold_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: b(&self.list[cur]),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: b(&self.list[1 - cur]),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: b(&self.member[cur]),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: b(&self.member[1 - cur]),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: b(&self.indirect[1 - cur]),
                },
            ],
        })
    }

    fn write_params(&self) {
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        let build = Params {
            processing: 0,
            ..self.params
        };
        self.queue
            .write_buffer(&self.params_build_buf, 0, bytemuck::bytes_of(&build));
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
        if self.windows.len() >= self.max_windows {
            return Err(GpuError::Config("too many weather windows".into()));
        }
        let window = Window {
            t_start: time as u32,
            wind_dir: wind_dir_deg.to_radians() as f32,
            wind_speed: wind_speed_kmh as f32,
            moisture: (moisture_pct / 100.0) as f32,
        };
        // a window at the same start time replaces the previous one
        if let Some(last) = self.windows.last_mut() {
            if last.t_start == window.t_start {
                *last = window;
            } else {
                self.windows.push(window);
            }
        } else {
            self.windows.push(window);
        }
        self.queue
            .write_buffer(&self.windows_buf, 0, bytemuck::cast_slice(&self.windows));
        self.params.n_windows = self.windows.len() as u32;
        self.write_params();

        if !ignitions.is_empty() {
            let n_cells = self.rows * self.cols;
            let k = ((time as u32) << 4) | CODE_IGNITION;
            let tile_cur = &self.tile_min[self.cur];
            for &(row, col) in ignitions {
                if row >= self.rows || col >= self.cols {
                    return Err(GpuError::Config(format!(
                        "ignition ({row}, {col}) outside the grid"
                    )));
                }
                let c = row * self.cols + col;
                let tile =
                    (row >> TILE_SHIFT) * (self.cols.div_ceil(TILE_SIZE)) + (col >> TILE_SHIFT);
                for r in 0..self.realizations {
                    // forced ignitions overwrite (the heap kernel would pop
                    // whichever event comes first; at t = now they are the
                    // earliest possible)
                    let idx = r * n_cells + c;
                    self.queue.write_buffer(
                        &self.key_buf,
                        (idx * 4) as u64,
                        bytemuck::bytes_of(&k),
                    );
                    let t_idx = r * self.n_tiles + tile;
                    self.queue.write_buffer(
                        tile_cur,
                        (t_idx * 4) as u64,
                        bytemuck::bytes_of(&(time as u32)),
                    );
                }
            }
        }
        Ok(())
    }

    fn encode_prepare(&self, encoder: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.prepare);
        pass.set_bind_group(0, bg, &[]);
        let n = (self.realizations * self.n_tiles) as u32;
        pass.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
    }

    /// Queue every tile with unpushed arrivals below `next_end` (bucket
    /// start), then swap the worklists.
    fn encode_build_list(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let bg = self.bind_group(self.cur, true);
        self.encode_prepare(encoder, &bg);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.build_list);
            pass.set_bind_group(0, &bg, &[]);
            let n = (self.realizations * self.n_tiles) as u32;
            pass.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
        }
        self.cur = 1 - self.cur;
    }

    /// One relaxation sweep over the queued tiles, then swap the worklists
    /// and tile-minimum buffers.
    fn encode_sweep(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let bg = self.bind_group(self.cur, false);
        self.encode_prepare(encoder, &bg);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.relax);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups_indirect(&self.indirect[self.cur], 0);
        }
        self.cur = 1 - self.cur;
    }

    /// Number of tiles queued for the next sweep (the current worklist).
    fn read_queued(&self) -> u32 {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.indirect[self.cur], 0, &self.staging_work, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.staging_work.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().expect("map indirect buffer");
        let value = {
            let data = slice.get_mapped_range().expect("mapped indirect buffer");
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        self.staging_work.unmap();
        value
    }

    /// Advance to `end_time` (seconds), bucket by bucket, then sweep until no
    /// tile with an arrival below `end_time` is left unpushed.
    pub fn advance_to(&mut self, end_time: i64) -> Result<()> {
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
            if self.read_queued() == 0 {
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
    pub fn debug_edges(&mut self, row: usize, col: usize) -> Result<Vec<[f32; 8]>> {
        self.params.fold_time = (row * self.cols + col) as u32;
        self.write_params();
        let bg = self.bind_group(self.cur, false);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.debug_edges);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.fold_buf, 0, &self.staging_fold, 0, 64 * 4);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.staging_fold.slice(..256);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv()
            .unwrap()
            .map_err(|e| GpuError::Device(format!("{e:?}")))?;
        let data: Vec<f32> = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| GpuError::Device(format!("{e:?}")))?;
            bytemuck::cast_slice(&mapped).to_vec()
        };
        self.staging_fold.unmap();
        Ok((0..8)
            .map(|d| {
                let mut a = [0.0f32; 8];
                a.copy_from_slice(&data[d * 8..d * 8 + 8]);
                a
            })
            .collect())
    }

    /// Fold the realizations on the GPU and read the per-cell outputs back.
    pub fn get_output(&mut self) -> Result<GpuOutput> {
        let n_cells = self.rows * self.cols;
        self.params.fold_time = self.time as u32;
        self.write_params();
        let bg = self.bind_group(self.cur, false);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.fold);
            pass.set_bind_group(0, &bg, &[]);
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
        let slice = self.staging_fold.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv()
            .unwrap()
            .map_err(|e| GpuError::Device(format!("map fold buffer: {e:?}")))?;
        let data: Vec<f32> = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| GpuError::Device(format!("mapped fold buffer: {e:?}")))?;
            bytemuck::cast_slice(&mapped).to_vec()
        };
        self.staging_fold.unmap();

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
