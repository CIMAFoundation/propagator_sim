//! Per-realization front event heap: a binary min-heap keyed by event
//! time, stored as parallel vectors (struct-of-arrays like the numba
//! implementation, but growing independently per realization — no
//! suspension protocol needed).

/// A binary min-heap of pending ignition events for one realization.
///
/// The heap is stored as five parallel vectors (struct-of-arrays) that are
/// permuted together, so index `i` across all five describes one event.
/// Ordering is by `times` alone; `rows`/`cols`/`ros`/`fli` are payload that
/// rides along with its key. The classic implicit-tree layout is used:
/// element `i`'s children are `2i + 1` and `2i + 2`, its parent `(i - 1) / 2`.
#[derive(Clone, Default)]
pub struct FrontHeap {
    /// event times (seconds from simulation start); the heap key
    pub times: Vec<i64>,
    /// target cell rows (local grid coordinates)
    pub rows: Vec<i32>,
    /// target cell columns (local grid coordinates)
    pub cols: Vec<i32>,
    /// rate of spread carried into the target cell, m/min
    pub ros: Vec<f32>,
    /// fireline intensity carried into the target cell, kW/m
    pub fli: Vec<f32>,
    /// where the event came from: an index into the kernel's 8-neighbour
    /// lattice (the source cell is `target - NEIGHBOURS[src]`), or
    /// [`SRC_EXTERNAL`] for ignitions and ember landings. Lets
    /// [`crate::Propagator::extinguish`] cancel the spread of put-out cells.
    pub src: Vec<u8>,
}

/// Source code of an event not produced by contact spread (an ignition
/// boundary condition or an ember landing).
pub const SRC_EXTERNAL: u8 = u8::MAX;

impl FrontHeap {
    /// An empty heap.
    pub fn new() -> FrontHeap {
        FrontHeap::default()
    }

    /// Number of pending events.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// `true` when no events remain (the realization's front is exhausted).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Time of the earliest pending event without removing it.
    #[inline]
    pub fn peek_time(&self) -> Option<i64> {
        self.times.first().copied()
    }

    /// Target cell of the earliest pending event without removing it.
    #[inline]
    pub fn peek_cell(&self) -> Option<(i32, i32)> {
        if self.is_empty() {
            None
        } else {
            Some((self.rows[0], self.cols[0]))
        }
    }

    /// Swap two events across all parallel arrays (keeps rows aligned).
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.times.swap(i, j);
        self.rows.swap(i, j);
        self.cols.swap(i, j);
        self.ros.swap(i, j);
        self.fli.swap(i, j);
        self.src.swap(i, j);
    }

    /// Schedule an ignition event, keeping the heap invariant.
    ///
    /// The event is appended to the end (the next open leaf) and then
    /// sifted up — repeatedly swapped with its parent while it is earlier —
    /// until every parent precedes its children again. O(log n).
    pub fn push(&mut self, time: i64, row: i32, col: i32, ros: f32, fli: f32) {
        self.push_from(time, row, col, ros, fli, SRC_EXTERNAL);
    }

    /// [`FrontHeap::push`] recording the event's source (see [`FrontHeap::src`]).
    pub fn push_from(&mut self, time: i64, row: i32, col: i32, ros: f32, fli: f32, src: u8) {
        self.src.push(src);
        self.times.push(time);
        self.rows.push(row);
        self.cols.push(col);
        self.ros.push(ros);
        self.fli.push(fli);
        let mut idx = self.times.len() - 1;
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.times[parent] <= self.times[idx] {
                break;
            }
            self.swap(parent, idx);
            idx = parent;
        }
    }

    /// Remove and return the earliest `(time, row, col, ros, fli)` event.
    ///
    /// The root (minimum) is swapped with the last leaf and popped, then the
    /// leaf now at the root is sifted down — repeatedly swapped with its
    /// smaller child while it is later than that child — restoring the heap.
    /// O(log n). Returns `None` on an empty heap.
    pub fn pop_min(&mut self) -> Option<(i64, i32, i32, f32, f32)> {
        if self.is_empty() {
            return None;
        }
        let last = self.times.len() - 1;
        self.swap(0, last);
        let item = (
            self.times.pop().unwrap(),
            self.rows.pop().unwrap(),
            self.cols.pop().unwrap(),
            self.ros.pop().unwrap(),
            self.fli.pop().unwrap(),
        );
        self.src.pop();
        let size = self.times.len();
        let mut idx = 0;
        loop {
            let left = idx * 2 + 1;
            if left >= size {
                break;
            }
            let right = left + 1;
            let mut smallest = left;
            if right < size && self.times[right] < self.times[left] {
                smallest = right;
            }
            if self.times[idx] <= self.times[smallest] {
                break;
            }
            self.swap(idx, smallest);
            idx = smallest;
        }
        Some(item)
    }

    /// Shift every pending event's coordinates (domain growth).
    pub fn shift_coords(&mut self, row_shift: i32, col_shift: i32) {
        if row_shift != 0 {
            for row in &mut self.rows {
                *row += row_shift;
            }
        }
        if col_shift != 0 {
            for col in &mut self.cols {
                *col += col_shift;
            }
        }
    }

    /// Keep only the events for which `keep(row, col, src)` holds, then
    /// restore the heap invariant. Returns how many events were dropped.
    pub fn retain(&mut self, mut keep: impl FnMut(i32, i32, u8) -> bool) -> usize {
        let n = self.times.len();
        let mut w = 0;
        for i in 0..n {
            if keep(self.rows[i], self.cols[i], self.src[i]) {
                self.swap(w, i);
                w += 1;
            }
        }
        self.times.truncate(w);
        self.rows.truncate(w);
        self.cols.truncate(w);
        self.ros.truncate(w);
        self.fli.truncate(w);
        self.src.truncate(w);
        // Floyd heapify: sift down every internal node, last first.
        for start in (0..w / 2).rev() {
            let mut idx = start;
            loop {
                let left = idx * 2 + 1;
                if left >= w {
                    break;
                }
                let right = left + 1;
                let mut smallest = left;
                if right < w && self.times[right] < self.times[left] {
                    smallest = right;
                }
                if self.times[idx] <= self.times[smallest] {
                    break;
                }
                self.swap(idx, smallest);
                idx = smallest;
            }
        }
        n - w
    }

    /// Pending event target cells (unordered).
    pub fn event_cells(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.rows.iter().copied().zip(self.cols.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pops_in_time_order() {
        let mut heap = FrontHeap::new();
        for (i, t) in [5i64, 1, 4, 1, 3, 9, 0].iter().enumerate() {
            heap.push(*t, i as i32, i as i32, 0.0, 0.0);
        }
        let mut last = i64::MIN;
        while let Some((t, _, _, _, _)) = heap.pop_min() {
            assert!(t >= last);
            last = t;
        }
    }

    #[test]
    fn retain_drops_events_and_keeps_order() {
        let mut heap = FrontHeap::new();
        for (i, t) in [5i64, 1, 4, 1, 3, 9, 0, 7, 2].iter().enumerate() {
            heap.push_from(*t, i as i32, 0, 0.0, 0.0, i as u8);
        }
        let dropped = heap.retain(|row, _, _| row % 3 != 0);
        assert_eq!(dropped, 3);
        let mut last = i64::MIN;
        let mut seen = Vec::new();
        while let Some((t, r, _, _, _)) = heap.pop_min() {
            assert!(t >= last);
            last = t;
            seen.push(r);
        }
        seen.sort();
        assert_eq!(seen, vec![1, 2, 4, 5, 7, 8]);
        assert!(heap.src.is_empty());
    }

    #[test]
    fn payload_follows_key() {
        let mut heap = FrontHeap::new();
        heap.push(10, 1, 2, 1.5, 2.5);
        heap.push(5, 3, 4, 3.5, 4.5);
        let (t, r, c, ros, fli) = heap.pop_min().unwrap();
        assert_eq!((t, r, c), (5, 3, 4));
        assert_eq!((ros, fli), (3.5, 4.5));
        let (t, r, c, ros, fli) = heap.pop_min().unwrap();
        assert_eq!((t, r, c), (10, 1, 2));
        assert_eq!((ros, fli), (1.5, 2.5));
    }
}
