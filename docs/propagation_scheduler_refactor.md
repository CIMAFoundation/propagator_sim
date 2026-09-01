# Propagation Scheduler Refactor Notes

Goal: redesign propagation scheduling to be numba-friendly and parallel across
realizations, while keeping the current public API and event-driven behavior.

Idea recap
- Keep boundary conditions on the existing Python `Scheduler`.
- Add a dedicated propagation-only scheduler that stores events in arrays and
  supports numba-friendly insertion/pop.
- Add `seconds` (or similar) to `Propagator.step`, so each call advances all
  realizations in parallel for a time window, avoiding per-event step overhead.
- Track time-dependent moisture per realization when running windowed steps.

Context and lessons
- The dominant cost for large grids + many realizations is step granularity,
  not the scheduler itself. Processing a single event time per call leads to
  heavy orchestration overhead.
- Any propagation-only scheduler should be paired with a multi-second step
  window so numba can process a large batch per call.

Proposal to explore next
- Add `seconds: int | None = None` to `Propagator.step`.
  - If `seconds is None`, keep legacy behavior (one event time).
  - If set, process all propagation events in `[time, time + seconds]` in a
    single call, allowing `prange` across realizations.
- Keep boundary conditions event-driven; apply any BC events at or before
  the end of the window.
- Maintain integer seconds only, earliest time per realization, inf allowed.

Potential data structures
- Propagation scheduler:
  - Hash table (time -> head index), heap for next time, event arrays.
  - Per-realization pending counts for `compute_stats`.
  - Optional 2D tiling for memory locality if needed later, but avoid if it
    harms parallelization.
- Status matrices:
  - Can change 3D layout; external API must not rely on shape or ordering.

Time-dependent moisture considerations
- Current behavior (baseline):
  - `moisture` is a 2D field (fractional), set by boundary conditions.
  - `actions_moisture` is a 2D field (fractional), added on top of `moisture`.
  - `actions_moisture` decays by `(1 - k) ** (delta_minutes)` every step.
  - `_get_moisture()` returns `clip(moisture + actions_moisture)`.
- With windowed steps and per-realization time:
  - If realizations advance different amounts in one call, `actions_moisture`
    decay must be per-realization (and possibly `moisture` updates too).
  - A lazy strategy can keep one scalar `last_update_time` plus a scalar
    decay factor:
    - store `actions_moisture` at reference time `t0`
    - current value when working on a cell/realization is `actions_moisture * decay_factor**(t_now - t0)`
    - when adding moisture at `t_now`, add `delta / scale` to the stored array

Session notes and implementation direction (historical: pre-refactor design)
- Previous hot path: `Propagator.step()` popped a single event time, applied BC, and called `next_updates_fn` to generate new updates.
- The existing `Scheduler` is Python-side and groups `UpdateBatchWithTime` by time; this adds overhead at high event counts.
- `next_updates_fn` was numba-jitted but worked on batches; it returned arrays that then got split/queued in Python. It has since been superseded by the front-tracking kernel below (`advance_front_until`) and no longer exists in the codebase.
- For performance, the propagation loop should avoid `UpdateBatch*` in the core loop and run as a numba front-tracking kernel.
- Proposed kernel structure:
  - Per-realization min-heap (arrays of times/rows/cols/ros/fli, plus size).
  - `advance_front_until(end_time)` runs in `prange` across realizations.
  - Each realization pops earliest events, applies fire state, and pushes new events in numba.
- Boundary conditions remain event-driven in Python; propagation runs in windowed slices between BC events.
- In the initial front-tracking implementation, capacity is capped per realization (default `veg.size`); overflow should raise to avoid silent truncation.
- Moisture and wind are currently treated as constant within a window; BC updates happen at window boundaries.
