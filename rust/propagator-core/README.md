# propagator-core

The Rust core of PROPAGATOR: a stochastic cellular-automaton wildfire spread
simulator, std-only and without dependencies. It is the engine behind the
Python adapter (`propagator-py`), the browser demo (`propagator-wasm`) and
applications that embed the simulator directly. The model is specified in
[`docs/rust-core-spec.md`](../../docs/rust-core-spec.md).

## Using it from another Rust project

The crate is not published on crates.io. Depend on it through Git; Cargo finds
the package inside the repository by name. Pin a commit (`rev`) so builds are
reproducible:

```toml
[dependencies]
propagator-core = { git = "https://github.com/CIMAFoundation/propagator_sim", rev = "<commit>" }
```

`branch = "dev"` follows the development branch instead. The crate builds for
`wasm32-unknown-unknown` as well as native targets.

To work on the core and on an application at the same time, point the
application to a local checkout without editing its dependency line:

```toml
# in the application's workspace Cargo.toml
[patch."https://github.com/CIMAFoundation/propagator_sim"]
propagator-core = { path = "../propagator_sim/rust/propagator-core" }
```

The `[profile.release]` section in this crate's `Cargo.toml` only applies when
the crate is built on its own; as a dependency, the application's profile
applies (Cargo ignores profiles of dependencies).

## Minimal example

```rust
use propagator_core::*;

let veg = Grid2::filled(256, 256, 4); // grassland
let dem = Grid2::filled(256, 256, 0.0);
let mut config = PropagatorConfig::new(veg, dem);
config.seed = Some(42);
let mut sim = Propagator::new(config).unwrap();

sim.set_boundary_conditions(BoundaryConditions {
    time: 0,
    moisture: Some(FieldInput::Scalar(8.0)),   // percent
    wind_dir: Some(FieldInput::Scalar(180.0)), // degrees from north
    wind_speed: Some(FieldInput::Scalar(15.0)),// km/h
    ignitions: Some(Ignitions::Points(vec![(128, 128)])),
    ..Default::default()
}).unwrap();

sim.step_window(3600).unwrap(); // one hour
let output = sim.get_output().unwrap();
println!("burned area: {:.1} ha", output.stats.area_mean / 10_000.0);
```

The same example is the crate documentation (`cargo doc --open`).
`tests/integration.rs` covers the rest of the API: actions, checkpoints,
domain growth, tile freezing and `extinguish` (with a per-cell
probability of success).
