//! Per-realization queue of pending ignition events, ordered by time.
//!
//! Event times are integer seconds and the kernel only ever pops the
//! earliest event, so a calendar queue beats a binary heap: a ring of
//! one-second buckets covering a window of [`WINDOW`] seconds from the
//! ring's start, with a bitmap of non-empty buckets to find the next event,
//! and a binary heap for the few events scheduled beyond the window (slow
//! fuels, long spotting delays). Push and pop are O(1) amortized for
//! in-window events instead of O(log n) sifts over five parallel arrays.
//!
//! Invariant: while the queue is non-empty the ring holds at least one
//! event, and every ring event precedes every far event, so the earliest
//! event is always the first non-empty bucket at or after the cursor.
//! Events pushed at or before the cursor's time (a forced ignition at the
//! current time, or a stale time) go into the cursor's bucket and keep
//! their own time, matching a heap that pops them next.
//!
//! Events in the same second pop in LIFO order; a binary heap pops them in
//! an order that depends on its layout, so seeded runs differ from the
//! heap-based kernel in the last bits of the RNG stream but are
//! statistically identical.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Seconds covered by the ring (power of two).
pub const WINDOW: usize = 1 << 11;
const MASK: i64 = WINDOW as i64 - 1;
const WORDS: usize = WINDOW / 64;

/// One pending ignition event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    /// seconds from simulation start
    pub time: i64,
    /// target cell (local grid coordinates)
    pub row: i32,
    pub col: i32,
    /// rate of spread carried into the target, m/min
    pub ros: f32,
    /// fireline intensity carried into the target, kW/m
    pub fli: f32,
}

/// `Event` ordered by time, earliest first, for the far heap.
#[derive(Clone, Copy, Debug)]
struct Far(Event);

impl PartialEq for Far {
    fn eq(&self, other: &Self) -> bool {
        self.0.time == other.0.time
    }
}
impl Eq for Far {}
impl PartialOrd for Far {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Far {
    fn cmp(&self, other: &Self) -> Ordering {
        // reversed: BinaryHeap is a max-heap
        other.0.time.cmp(&self.0.time)
    }
}

/// Calendar queue of pending ignition events for one realization.
#[derive(Clone)]
pub struct FrontQueue {
    /// time of bucket 0 (multiple of `WINDOW`)
    ring_start: i64,
    /// bucket index at or after which the next event lives
    cursor: usize,
    buckets: Vec<Vec<Event>>,
    /// bit `i` set iff `buckets[i]` is non-empty
    occupied: [u64; WORDS],
    in_ring: usize,
    /// events with `time >= ring_start + WINDOW`
    far: BinaryHeap<Far>,
}

impl Default for FrontQueue {
    fn default() -> Self {
        FrontQueue::new()
    }
}

impl FrontQueue {
    /// An empty queue.
    pub fn new() -> FrontQueue {
        FrontQueue {
            ring_start: 0,
            cursor: 0,
            buckets: (0..WINDOW).map(|_| Vec::new()).collect(),
            occupied: [0; WORDS],
            in_ring: 0,
            far: BinaryHeap::new(),
        }
    }

    /// Number of pending events.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.in_ring + self.far.len()
    }

    /// `true` when no events remain (the realization's front is exhausted).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.in_ring == 0 && self.far.is_empty()
    }

    #[inline]
    fn ring_end(&self) -> i64 {
        self.ring_start + WINDOW as i64
    }

    #[inline]
    fn set_occupied(&mut self, idx: usize) {
        self.occupied[idx / 64] |= 1u64 << (idx % 64);
    }

    #[inline]
    fn clear_occupied(&mut self, idx: usize) {
        self.occupied[idx / 64] &= !(1u64 << (idx % 64));
    }

    /// First non-empty bucket at or after `self.cursor`, if any.
    #[inline]
    fn next_occupied(&self) -> Option<usize> {
        let mut word = self.cursor / 64;
        let mut bits = self.occupied[word] & (!0u64 << (self.cursor % 64));
        loop {
            if bits != 0 {
                return Some(word * 64 + bits.trailing_zeros() as usize);
            }
            word += 1;
            if word == WORDS {
                return None;
            }
            bits = self.occupied[word];
        }
    }

    #[inline]
    fn insert_ring(&mut self, idx: usize, event: Event) {
        self.buckets[idx].push(event);
        self.set_occupied(idx);
        self.in_ring += 1;
    }

    /// Move the ring to start at the window containing `time` (ring must
    /// be empty) and pull in every far event that now falls inside it.
    fn rebase(&mut self, time: i64) {
        debug_assert_eq!(self.in_ring, 0);
        self.ring_start = time & !MASK;
        self.cursor = 0;
        let end = self.ring_end();
        while let Some(Far(event)) = self.far.peek() {
            if event.time >= end {
                break;
            }
            let event = self.far.pop().unwrap().0;
            let idx = (event.time - self.ring_start) as usize;
            self.insert_ring(idx, event);
        }
    }

    /// Schedule an ignition event.
    pub fn push(&mut self, time: i64, row: i32, col: i32, ros: f32, fli: f32) {
        let event = Event {
            time,
            row,
            col,
            ros,
            fli,
        };
        if self.in_ring == 0 {
            // empty ring: re-anchor it on this event so it is never far
            self.rebase(time);
            let idx = (time - self.ring_start) as usize;
            self.insert_ring(idx, event);
            return;
        }
        let cursor_time = self.ring_start + self.cursor as i64;
        if time <= cursor_time {
            // at or before the current time: popped next, own time kept
            let idx = self.cursor;
            self.insert_ring(idx, event);
        } else if time < self.ring_end() {
            let idx = (time - self.ring_start) as usize;
            self.insert_ring(idx, event);
        } else {
            self.far.push(Far(event));
        }
    }

    /// Advance the cursor to the next non-empty bucket. Requires a
    /// non-empty ring.
    #[inline]
    fn seek(&mut self) -> usize {
        let idx = self
            .next_occupied()
            .expect("non-empty ring has an occupied bucket");
        self.cursor = idx;
        idx
    }

    /// Time of the earliest pending event without removing it.
    #[inline]
    pub fn peek_time(&self) -> Option<i64> {
        self.peek().map(|event| event.time)
    }

    /// Target cell of the earliest pending event without removing it.
    #[inline]
    pub fn peek_cell(&self) -> Option<(i32, i32)> {
        self.peek().map(|event| (event.row, event.col))
    }

    /// The earliest pending event without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&Event> {
        if self.in_ring == 0 {
            return None;
        }
        let idx = self
            .next_occupied()
            .expect("non-empty ring has an occupied bucket");
        self.buckets[idx].last()
    }

    /// Remove and return the earliest `(time, row, col, ros, fli)` event.
    pub fn pop_min(&mut self) -> Option<(i64, i32, i32, f32, f32)> {
        if self.in_ring == 0 {
            return None;
        }
        let idx = self.seek();
        let event = self.buckets[idx].pop().expect("occupied bucket");
        self.in_ring -= 1;
        if self.buckets[idx].is_empty() {
            self.clear_occupied(idx);
        }
        if self.in_ring == 0 {
            if let Some(Far(next)) = self.far.peek() {
                let time = next.time;
                self.rebase(time);
            }
        }
        Some((event.time, event.row, event.col, event.ros, event.fli))
    }

    /// Shift every pending event's coordinates (domain growth).
    pub fn shift_coords(&mut self, row_shift: i32, col_shift: i32) {
        if row_shift == 0 && col_shift == 0 {
            return;
        }
        for bucket in &mut self.buckets {
            for event in bucket {
                event.row += row_shift;
                event.col += col_shift;
            }
        }
        let mut far = std::mem::take(&mut self.far).into_vec();
        for Far(event) in &mut far {
            event.row += row_shift;
            event.col += col_shift;
        }
        self.far = BinaryHeap::from(far);
    }

    /// Pending event target cells (unordered).
    pub fn event_cells(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.buckets
            .iter()
            .flatten()
            .map(|event| (event.row, event.col))
            .chain(self.far.iter().map(|Far(event)| (event.row, event.col)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(queue: &mut FrontQueue) -> Vec<i64> {
        let mut times = Vec::new();
        while let Some((t, _, _, _, _)) = queue.pop_min() {
            times.push(t);
        }
        times
    }

    #[test]
    fn pops_in_time_order_across_the_window() {
        let mut queue = FrontQueue::new();
        let times = [
            5i64,
            1,
            4,
            1,
            3,
            9,
            0,
            WINDOW as i64 * 3 + 7,
            WINDOW as i64 + 2,
            WINDOW as i64 * 10,
            WINDOW as i64 - 1,
        ];
        for (i, t) in times.iter().enumerate() {
            queue.push(*t, i as i32, i as i32, 0.0, 0.0);
        }
        assert_eq!(queue.len(), times.len());
        let popped = drain(&mut queue);
        let mut sorted = times.to_vec();
        sorted.sort();
        assert_eq!(popped, sorted);
        assert!(queue.is_empty());
    }

    #[test]
    fn payload_follows_key() {
        let mut queue = FrontQueue::new();
        queue.push(10, 1, 2, 1.5, 2.5);
        queue.push(5, 3, 4, 3.5, 4.5);
        assert_eq!(queue.peek_time(), Some(5));
        assert_eq!(queue.peek_cell(), Some((3, 4)));
        let (t, r, c, ros, fli) = queue.pop_min().unwrap();
        assert_eq!((t, r, c), (5, 3, 4));
        assert_eq!((ros, fli), (3.5, 4.5));
        let (t, r, c, ros, fli) = queue.pop_min().unwrap();
        assert_eq!((t, r, c), (10, 1, 2));
        assert_eq!((ros, fli), (1.5, 2.5));
        assert_eq!(queue.pop_min(), None);
    }

    #[test]
    fn interleaved_push_pop_stays_ordered() {
        // simulate a front: pop an event, push a few later ones, repeat
        let mut queue = FrontQueue::new();
        let mut rng = 12345u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        queue.push(0, 0, 0, 0.0, 0.0);
        let mut last = 0i64;
        let mut popped = 0;
        while let Some((t, _, _, _, _)) = queue.pop_min() {
            assert!(
                t >= last,
                "{t} < {last} (popped {popped}, ring_start {}, cursor {}, in_ring {}, far {})",
                queue.ring_start,
                queue.cursor,
                queue.in_ring,
                queue.far.len()
            );
            last = t;
            popped += 1;
            if popped < 20_000 {
                // one or two successors per event, so the front grows
                // slowly and never dies out before the pop budget
                let successors = if next() % 4 == 0 { 2 } else { 1 };
                for _ in 0..successors {
                    // transitions from 1 s to beyond two windows
                    let dt = 1 + (next() % (WINDOW as u64 * 2 + 500)) as i64;
                    queue.push(t + dt, 0, 0, 0.0, 0.0);
                }
            }
        }
        assert!(popped >= 20_000);
    }

    #[test]
    fn events_at_or_before_the_cursor_pop_next_with_their_own_time() {
        let mut queue = FrontQueue::new();
        queue.push(100, 0, 0, 0.0, 0.0);
        queue.push(200, 0, 0, 0.0, 0.0);
        assert_eq!(queue.pop_min().unwrap().0, 100);
        // a forced ignition scheduled "now" and one in the past
        queue.push(100, 1, 1, 0.0, 0.0);
        queue.push(50, 2, 2, 0.0, 0.0);
        assert_eq!(queue.peek_time(), Some(50));
        assert_eq!(drain(&mut queue), vec![50, 100, 200]);
    }

    #[test]
    fn far_events_are_pulled_in_when_the_ring_drains() {
        let mut queue = FrontQueue::new();
        queue.push(3, 0, 0, 0.0, 0.0);
        let far_a = WINDOW as i64 * 5 + 1;
        let far_b = WINDOW as i64 * 5 + 1;
        let far_c = WINDOW as i64 * 7;
        queue.push(far_a, 1, 1, 0.0, 0.0);
        queue.push(far_c, 3, 3, 0.0, 0.0);
        queue.push(far_b, 2, 2, 0.0, 0.0);
        assert_eq!(queue.len(), 4);
        assert_eq!(queue.pop_min().unwrap().0, 3);
        // ring re-anchored on the far events' window
        assert_eq!(queue.peek_time(), Some(far_a));
        assert_eq!(drain(&mut queue), vec![far_a, far_b, far_c]);
    }

    #[test]
    fn shift_and_event_cells_cover_ring_and_far() {
        let mut queue = FrontQueue::new();
        queue.push(1, 1, 2, 0.0, 0.0);
        queue.push(WINDOW as i64 * 4, 3, 4, 0.0, 0.0);
        queue.shift_coords(10, 20);
        let mut cells: Vec<_> = queue.event_cells().collect();
        cells.sort();
        assert_eq!(cells, vec![(11, 22), (13, 24)]);
        let (_, r, c, _, _) = queue.pop_min().unwrap();
        assert_eq!((r, c), (11, 22));
        let (_, r, c, _, _) = queue.pop_min().unwrap();
        assert_eq!((r, c), (13, 24));
    }

    #[test]
    fn clone_is_independent() {
        let mut queue = FrontQueue::new();
        queue.push(7, 0, 0, 0.0, 0.0);
        let mut copy = queue.clone();
        copy.pop_min();
        assert_eq!(queue.len(), 1);
        assert!(copy.is_empty());
    }
}
