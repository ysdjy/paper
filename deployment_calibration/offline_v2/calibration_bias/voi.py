"""Value-of-probing for the calibration-bias stage (offline).

Gross VOI(K) = selected true utility using K probes − best no-history (state-agnostic) selector.
Net  VOI(K) = Gross VOI(K) − lambda_time × real cumulative probe time for K probes.

Because calibration bias should move grasp SUCCESS (not just time), we also report a success-only
Gross VOI so probing value is not hidden behind a small time weight.
"""

from __future__ import annotations

import numpy as np

from ..utility import UtilityConfig
from .schema import DEFAULT_FIELDS, FieldMap, ROLE_PROBE


def probe_time_by_k(episodes, fm: FieldMap = DEFAULT_FIELDS, kmax: int = 3) -> dict:
    """Mean-over-session cumulative elapsed time of the first K probes."""
    by_sess = {}
    for e in episodes:
        if e.get(fm.episode_role) != ROLE_PROBE:
            continue
        by_sess.setdefault(e.get(fm.session_id), []).append(e)
    out = {0: 0.0}
    for k in range(1, kmax + 1):
        tot = []
        for ps in by_sess.values():
            ps = sorted(ps, key=lambda z: z.get(fm.order_in_session, 0))[:k]
            tot.append(sum(float(p[fm.y]["skill_elapsed_time"]) for p in ps))
        out[k] = float(np.mean(tot)) if tot else 0.0
    return out


def voi_curve(selected_util_by_k: dict, baseline_util: float, ptime_by_k: dict,
              cfg: UtilityConfig) -> dict:
    """selected_util_by_k: {K: mean selected TRUE utility with K probes}. Returns Gross/Net VOI."""
    out = {}
    for k, u in selected_util_by_k.items():
        gross = u - baseline_util
        cost = cfg.lambda_time * float(ptime_by_k.get(k, 0.0))
        out[k] = {"selected_true_utility": u, "gross_voi": gross,
                  "probe_time_cumulative": float(ptime_by_k.get(k, 0.0)),
                  "probe_time_cost": cost, "net_voi": gross - cost}
    return out
