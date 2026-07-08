//! Time-ordered boundary-condition queue and the external
//! `BoundaryConditions` input type.

use std::collections::BTreeMap;

use crate::error::{PropagatorError, Result};
use crate::fuels::NO_FUEL;
use crate::grid::Grid2;

/// Batch of ignition updates to inject into realizations' front heaps.
///
/// Struct-of-arrays (parallel vectors): index `i` is one ignition of cell
/// `(rows[i], cols[i])` in realization `realizations[i]`, seeding its heap
/// with the given rate of spread and fireline intensity.
#[derive(Clone, Debug, Default)]
pub struct UpdateBatch {
    /// ignition cell rows (local grid coordinates)
    pub rows: Vec<i32>,
    /// ignition cell columns (local grid coordinates)
    pub cols: Vec<i32>,
    /// which realization each ignition targets
    pub realizations: Vec<u32>,
    /// seed rate of spread, m/min (0 for a plain ignition)
    pub ros: Vec<f32>,
    /// seed fireline intensity, kW/m (0 for a plain ignition)
    pub fli: Vec<f32>,
}

impl UpdateBatch {
    /// Number of queued ignitions.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// `true` when no ignitions are queued.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Append one ignition.
    pub fn push(&mut self, row: i32, col: i32, realization: u32, ros: f32, fli: f32) {
        self.rows.push(row);
        self.cols.push(col);
        self.realizations.push(realization);
        self.ros.push(ros);
        self.fli.push(fli);
    }

    /// Append every ignition of `other`.
    pub fn extend(&mut self, other: &UpdateBatch) {
        self.rows.extend_from_slice(&other.rows);
        self.cols.extend_from_slice(&other.cols);
        self.realizations.extend_from_slice(&other.realizations);
        self.ros.extend_from_slice(&other.ros);
        self.fli.extend_from_slice(&other.fli);
    }
}

/// A scheduled event: optional condition grids (already in internal
/// units) plus ignition updates.
#[derive(Clone, Debug, Default)]
pub struct SchedulerEvent {
    pub updates: UpdateBatch,
    pub moisture: Option<Grid2<f32>>,
    pub wind_dir: Option<Grid2<f32>>,
    pub wind_speed: Option<Grid2<f32>>,
    pub additional_moisture: Option<Grid2<f32>>,
    /// vegetation code overrides; NaN = no change
    pub vegetation_changes: Option<Grid2<f64>>,
}

impl SchedulerEvent {
    /// Merge semantics for same-time events: weather overwrites,
    /// additional moisture accumulates, vegetation changes overlay
    /// (later non-NO_FUEL values win), ignitions concatenate.
    pub fn merge(&mut self, other: SchedulerEvent) {
        self.updates.extend(&other.updates);
        if other.moisture.is_some() {
            self.moisture = other.moisture;
        }
        if other.wind_dir.is_some() {
            self.wind_dir = other.wind_dir;
        }
        if other.wind_speed.is_some() {
            self.wind_speed = other.wind_speed;
        }
        match (&mut self.additional_moisture, other.additional_moisture) {
            (Some(mine), Some(theirs)) => {
                for (a, b) in mine
                    .as_mut_slice()
                    .iter_mut()
                    .zip(theirs.as_slice().iter())
                {
                    *a += *b;
                }
            }
            (mine @ None, theirs @ Some(_)) => *mine = theirs,
            _ => {}
        }
        match (&mut self.vegetation_changes, other.vegetation_changes) {
            (Some(mine), Some(theirs)) => {
                for (a, b) in mine
                    .as_mut_slice()
                    .iter_mut()
                    .zip(theirs.as_slice().iter())
                {
                    if *b != NO_FUEL as f64 {
                        *a = *b;
                    }
                }
            }
            (mine @ None, theirs @ Some(_)) => *mine = theirs,
            _ => {}
        }
    }
}

/// Time-ordered event queue, keyed by event time.
///
/// Backed by a `BTreeMap` so the earliest event is always `O(log n)` to find
/// and pop. Events scheduled at the same time are merged (see
/// [`SchedulerEvent::merge`]) rather than kept separate.
#[derive(Clone, Debug, Default)]
pub struct Scheduler {
    queue: BTreeMap<i64, SchedulerEvent>,
}

impl Scheduler {
    /// An empty queue.
    pub fn new() -> Scheduler {
        Scheduler::default()
    }

    /// Enqueue `event` at `time`, merging into any event already there.
    pub fn add_event(&mut self, time: i64, event: SchedulerEvent) {
        match self.queue.get_mut(&time) {
            Some(existing) => existing.merge(event),
            None => {
                self.queue.insert(time, event);
            }
        }
    }

    /// Time of the earliest queued event, or `None` when empty.
    pub fn next_time(&self) -> Option<i64> {
        self.queue.keys().next().copied()
    }

    /// Remove and return the earliest `(time, event)`.
    pub fn pop(&mut self) -> Option<(i64, SchedulerEvent)> {
        let time = self.next_time()?;
        let event = self.queue.remove(&time)?;
        Some((time, event))
    }

    /// Number of distinct event times queued.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// `true` when nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Iterate queued `(time, event)` pairs in time order.
    pub fn events(&self) -> impl Iterator<Item = (&i64, &SchedulerEvent)> {
        self.queue.iter()
    }

    /// Mutably iterate queued events in time order (used to re-anchor them
    /// after domain growth).
    pub fn events_mut(&mut self) -> impl Iterator<Item = (&i64, &mut SchedulerEvent)> {
        self.queue.iter_mut()
    }

    /// Deep-copy the whole queue as `(time, event)` pairs (for checkpoints).
    pub fn snapshot(&self) -> Vec<(i64, SchedulerEvent)> {
        self.queue
            .iter()
            .map(|(time, event)| (*time, event.clone()))
            .collect()
    }
}

/// Scalar-or-grid field input (scalars broadcast to the grid).
#[derive(Clone, Debug)]
pub enum FieldInput {
    Scalar(f64),
    Grid(Grid2<f32>),
}

impl FieldInput {
    pub(crate) fn to_grid(
        &self,
        shape: (usize, usize),
        transform: impl Fn(f32) -> f32,
    ) -> Result<Grid2<f32>> {
        match self {
            FieldInput::Scalar(value) => Ok(Grid2::filled(
                shape.0,
                shape.1,
                transform(*value as f32),
            )),
            FieldInput::Grid(grid) => {
                if grid.shape() != shape {
                    return Err(PropagatorError::InvalidBoundaryConditions(
                        format!(
                            "field shape {:?} does not match grid {:?}",
                            grid.shape(),
                            shape
                        ),
                    ));
                }
                let data = grid.as_slice().iter().map(|v| transform(*v)).collect();
                Ok(Grid2::from_vec(shape.0, shape.1, data))
            }
        }
    }
}

/// Ignition specification.
#[derive(Clone, Debug)]
pub enum Ignitions {
    /// (row, col), applied to every realization
    Points(Vec<(usize, usize)>),
    /// (row, col, realization)
    PointsWithRealization(Vec<(usize, usize, usize)>),
    /// boolean mask, true cells ignite in every realization
    Mask(Grid2<bool>),
}

/// Boundary conditions in *external units* (percent moisture, degrees
/// wind direction, km/h wind speed), applied at `time`.
#[derive(Clone, Debug, Default)]
pub struct BoundaryConditions {
    pub time: i64,
    /// fuel moisture, percent
    pub moisture: Option<FieldInput>,
    /// wind direction, degrees clockwise from north (meteorological)
    pub wind_dir: Option<FieldInput>,
    /// wind speed, km/h
    pub wind_speed: Option<FieldInput>,
    pub ignitions: Option<Ignitions>,
    /// extra moisture from firefighting actions, percent
    pub additional_moisture: Option<Grid2<f32>>,
    /// vegetation overrides, NaN = no change
    pub vegetation_changes: Option<Grid2<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_semantics() {
        let mut scheduler = Scheduler::new();
        let mut first = SchedulerEvent::default();
        first.moisture = Some(Grid2::filled(2, 2, 0.1));
        first.additional_moisture = Some(Grid2::filled(2, 2, 0.05));
        first.updates.push(1, 1, 0, 0.0, 0.0);
        scheduler.add_event(60, first);

        let mut second = SchedulerEvent::default();
        second.moisture = Some(Grid2::filled(2, 2, 0.2));
        second.additional_moisture = Some(Grid2::filled(2, 2, 0.05));
        second.updates.push(0, 0, 1, 0.0, 0.0);
        scheduler.add_event(60, second);

        assert_eq!(scheduler.len(), 1);
        let (time, event) = scheduler.pop().unwrap();
        assert_eq!(time, 60);
        // weather overwritten
        assert_eq!(event.moisture.unwrap()[(0, 0)], 0.2);
        // additional moisture accumulated
        assert!((event.additional_moisture.unwrap()[(0, 0)] - 0.1).abs() < 1e-6);
        // ignitions concatenated
        assert_eq!(event.updates.len(), 2);
    }

    #[test]
    fn pops_in_time_order() {
        let mut scheduler = Scheduler::new();
        scheduler.add_event(300, SchedulerEvent::default());
        scheduler.add_event(60, SchedulerEvent::default());
        assert_eq!(scheduler.next_time(), Some(60));
        assert_eq!(scheduler.pop().unwrap().0, 60);
        assert_eq!(scheduler.pop().unwrap().0, 300);
        assert!(scheduler.pop().is_none());
    }
}
