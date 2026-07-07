use std::fmt;

/// Errors raised by the propagator core.
#[derive(Debug)]
pub enum PropagatorError {
    /// The fire reached the boundary ring in `OobMode::Raise`. State is
    /// intact: checkpoint, expand the domain and resume.
    OutOfBounds,
    /// Weather (moisture/wind) was never set before propagation started.
    WeatherNotSet,
    /// Invalid boundary conditions (past time, bad ignition format, ...).
    InvalidBoundaryConditions(String),
    /// Checkpoint incompatible with this propagator or unsupported version.
    CheckpointMismatch(String),
    /// Invalid domain growth request.
    InvalidGrowth(String),
    /// Tile freezing requested without a `freeze_dir`.
    FreezeDirNotSet,
    /// Fuel system construction error.
    FuelSystem(String),
    Io(std::io::Error),
}

impl fmt::Display for PropagatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropagatorError::OutOfBounds => write!(
                f,
                "simulation reached the edge of the grid; the state is \
                 intact: checkpoint() and resume on a larger grid, or use \
                 OobMode::Ignore to let the fire stop at the boundary"
            ),
            PropagatorError::WeatherNotSet => write!(
                f,
                "moisture/wind were never set; apply boundary conditions \
                 before stepping"
            ),
            PropagatorError::InvalidBoundaryConditions(msg) => {
                write!(f, "invalid boundary conditions: {msg}")
            }
            PropagatorError::CheckpointMismatch(msg) => {
                write!(f, "checkpoint mismatch: {msg}")
            }
            PropagatorError::InvalidGrowth(msg) => {
                write!(f, "invalid domain growth: {msg}")
            }
            PropagatorError::FreezeDirNotSet => {
                write!(f, "tile freezing requires freeze_dir to be set")
            }
            PropagatorError::FuelSystem(msg) => {
                write!(f, "fuel system error: {msg}")
            }
            PropagatorError::Io(err) => write!(f, "io error: {err}"),
        }
    }
}

impl std::error::Error for PropagatorError {}

impl From<std::io::Error> for PropagatorError {
    fn from(err: std::io::Error) -> Self {
        PropagatorError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, PropagatorError>;
