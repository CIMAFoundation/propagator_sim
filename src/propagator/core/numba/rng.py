"""Seeding for the JIT kernels' random number generators.

Numba gives every worker thread its own RNG state, so seeding must happen
inside a parallel region with one iteration per thread. With numba's
static prange scheduling this makes runs reproducible for a fixed machine
and thread count (``NUMBA_NUM_THREADS``); it does not make runs portable
across different thread counts.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange  # type: ignore


@njit(cache=False, parallel=True)
def seed_rngs(seed: int) -> None:
    """Seed every worker thread's RNG deterministically from `seed`."""
    n_threads = get_num_threads()
    for i in prange(n_threads):
        np.random.seed(seed + i)
