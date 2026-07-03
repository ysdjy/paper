"""Session-level bootstrap confidence intervals (offline_v2).

Candidate groups are NESTED within sessions, so the resampling unit is the SESSION, not the
individual candidate episode. Treating candidate episodes as independent would understate the
CI width. We resample sessions with replacement, rebuild the per-session records, recompute the
statistic, and report the percentile CI.

Deterministic: a numpy Generator seeded from `seed` (Date/random are avoided for reproducibility).
Test mode can drop n_boot to keep unit tests fast.
"""

from __future__ import annotations

import numpy as np


def session_bootstrap(session_ids, stat_fn, n_boot: int = 2000, seed: int = 0,
                      ci: float = 0.95) -> dict:
    """Resample `session_ids` with replacement; recompute stat_fn(resampled_session_ids).

    stat_fn maps a list of (possibly repeated) session ids -> float (or NaN). NaN replicates
    are dropped from the percentile computation and counted.
    """
    session_ids = list(session_ids)
    rng = np.random.default_rng(seed)
    point = stat_fn(session_ids)
    n = len(session_ids)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [session_ids[i] for i in idx]
        v = stat_fn(resampled)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            reps.append(v)
    reps = np.asarray(reps, dtype=float)
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    if reps.size == 0:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": n_boot, "n_valid": 0, "ci_level": ci}
    return {
        "point": float(point) if point is not None and not (isinstance(point, float) and np.isnan(point)) else float("nan"),
        "ci_low": float(np.quantile(reps, lo_q)),
        "ci_high": float(np.quantile(reps, hi_q)),
        "boot_mean": float(reps.mean()),
        "boot_std": float(reps.std()),
        "n_boot": n_boot,
        "n_valid": int(reps.size),
        "ci_level": ci,
    }


def make_session_stat(rows, value_key: str, session_key: str = "session_id"):
    """Build a stat_fn that averages `value_key` over rows, grouped so a resampled session
    contributes ALL its rows (nested-cluster correct).

    rows: list of dicts each having session_key and value_key.
    Returns (stat_fn, unique_session_ids).
    """
    by_session: dict = {}
    for r in rows:
        by_session.setdefault(r[session_key], []).append(float(r[value_key]))
    sids = list(by_session)

    def stat_fn(resampled_sids):
        vals = []
        for sid in resampled_sids:
            vals.extend(by_session.get(sid, []))
        return float(np.mean(vals)) if vals else float("nan")

    return stat_fn, sids
