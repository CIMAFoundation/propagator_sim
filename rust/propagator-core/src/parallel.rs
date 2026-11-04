//! Parallel iteration over independent items (realizations, fold bands).
//!
//! Scoped std threads by default; rayon's global pool with the `rayon`
//! feature, for targets without std threads (wasm32, where the pool is a set
//! of web workers created from JavaScript).

/// Apply `f` to every item, splitting the slice into at most `n_threads`
/// contiguous chunks that run concurrently. Runs inline for one thread or
/// one item.
pub fn for_each_chunked<T: Send>(items: &mut [T], n_threads: usize, f: impl Fn(&mut T) + Sync) {
    let n_threads = n_threads.max(1).min(items.len().max(1));
    if n_threads <= 1 || items.len() <= 1 {
        for item in items.iter_mut() {
            f(item);
        }
        return;
    }
    let chunk_size = items.len().div_ceil(n_threads);
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        items.par_chunks_mut(chunk_size).for_each(|chunk| {
            for item in chunk.iter_mut() {
                f(item);
            }
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        let f = &f;
        std::thread::scope(|scope| {
            for chunk in items.chunks_mut(chunk_size) {
                scope.spawn(move || {
                    for item in chunk.iter_mut() {
                        f(item);
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visits_every_item_once_for_any_thread_count() {
        for n_threads in [0, 1, 2, 3, 7, 64] {
            let mut items: Vec<u32> = (0..23).collect();
            for_each_chunked(&mut items, n_threads, |x| *x += 100);
            assert_eq!(items, (100..123).collect::<Vec<u32>>(), "{n_threads}");
        }
        let mut empty: Vec<u32> = Vec::new();
        for_each_chunked(&mut empty, 4, |x| *x += 1);
    }
}
