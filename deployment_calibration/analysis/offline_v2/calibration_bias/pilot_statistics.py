"""Block-level bootstrap + simple stats for the pilot (EXPLORATORY / PILOT — NOT CONFIRMATORY).

The resampling unit is the BLOCK (never the trial), matching the data-split unit, so per-block dependence is
respected. n_blocks is small (18) — CIs are wide and reported as exploratory.
"""

from __future__ import annotations

import numpy as np


def block_bootstrap_ci(block_values, statistic=np.mean, n_boot=5000, seed=0, alpha=0.05):
    """Bootstrap a statistic over blocks. block_values: 1D array (one value per block). Returns
    {point, lo, hi, n_blocks}."""
    v = np.asarray(block_values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_blocks": 0}
    rng = np.random.default_rng(seed)
    boots = [statistic(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)]
    return {"point": float(statistic(v)), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n_blocks": int(v.size)}


def paired_block_bootstrap_ci(block_a, block_b, n_boot=5000, seed=0, alpha=0.05):
    """Paired difference (a-b) bootstrapped over blocks. Arrays aligned per block."""
    a = np.asarray(block_a, dtype=float)
    b = np.asarray(block_b, dtype=float)
    d = a - b
    return block_bootstrap_ci(d, statistic=np.mean, n_boot=n_boot, seed=seed, alpha=alpha)
