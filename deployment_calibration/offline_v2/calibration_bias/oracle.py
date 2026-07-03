"""Decision-value analysis for the calibration-bias stage (offline).

Reuses the offline_v2 oracle primitives. The hidden state is `bias_y`; the candidates are grasp
OFFSETS. A matched offset bank keyed by (target[, replicate]) spans bias levels. We report the same
decision-value quantities as the damping stage, PLUS the calibration-specific baselines:
  * best-single OFFSET (train-selected, state-agnostic deployable policy)
  * "robust offset" existence check (is any single offset near-optimal at every bias?) — the gate
    wants this to be FALSE (no robust generalist).
and BOTH success-only and frozen-utility VSI (calibration bias should move success, not just time).
"""

from __future__ import annotations

import numpy as np

from ..oracle import (oracle_candidate_per_group, optimal_candidate_switch_rate,
                      pairwise_rank_reversal_rate, selection_regret, value_of_state_information,
                      _state_agnostic, _state_aware)
from ..utility import UtilityConfig, true_utility
from .schema import DEFAULT_FIELDS, FieldMap, ROLE_CANDIDATE, offset_key
from .validator import matched_offset_bank


def build_bank(episodes, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    """Adapt the matched offset bank to the shape offline_v2.oracle expects (by_damping alias)."""
    mb = matched_offset_bank(episodes, fm)
    groups = []
    for mg in mb["matched_groups"]:
        groups.append({
            "matched_group_id": mg["matched_group_id"],
            "target": mg.get("target_id"),
            "candidate_keys": mg["offset_keys"],
            "dampings": mg["biases"],                 # alias: bias levels play the "state" role
            "by_damping": mg["by_bias"],
        })
    return {"matched": mb["matched"], "matched_groups": groups,
            "reason": None if mb["matched"] else "no matched offset group spans >=2 bias levels",
            "n_matched_groups": len(groups)}


def selection_groups(episodes, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    g = {}
    for e in episodes:
        if e.get(fm.episode_role) != ROLE_CANDIDATE:
            continue
        g.setdefault(e.get(fm.candidate_group), []).append(e)
    return g


def fixed_offset_regret(groups: dict, off_key, cfg: UtilityConfig, fm: FieldMap = DEFAULT_FIELDS,
                        label="fixed_offset") -> dict:
    def choose(eps):
        cand = [e for e in eps if offset_key(e, fm) == off_key]
        return cand[0] if cand else None
    oc = oracle_candidate_per_group(groups, cfg)
    rows = []
    for gid, eps in groups.items():
        pick = choose(eps) or eps[0]
        u = true_utility(pick, cfg)
        rows.append({"group_id": gid, "session_id": eps[0].get(fm.session_id),
                     "regret": oc[gid]["best_true_utility"] - u, "selected_true_utility": u,
                     "selected_success": 1.0 if pick[fm.y]["success"] else 0.0})
    return {"label": label,
            "mean_regret": float(np.mean([r["regret"] for r in rows])),
            "mean_true_utility": float(np.mean([r["selected_true_utility"] for r in rows])),
            "selected_success_rate": float(np.mean([r["selected_success"] for r in rows]))}


def best_single_offset(train_groups: dict, test_groups: dict, cfg: UtilityConfig,
                       fm: FieldMap = DEFAULT_FIELDS) -> dict:
    """Pick ONE offset on train (max mean true utility), evaluate on test."""
    by_off = {}
    for eps in train_groups.values():
        for e in eps:
            by_off.setdefault(offset_key(e, fm), []).append(true_utility(e, cfg))
    if not by_off:
        return {"available": False}
    means = {k: float(np.mean(v)) for k, v in by_off.items()}
    best = max(means, key=means.get)
    res = fixed_offset_regret(test_groups, best, cfg, fm, label=f"best_single_offset(train={best})")
    res["selected_offset"] = best
    res["train_offset_means"] = means
    return res


def robust_offset_exists(bank: dict, cfg: UtilityConfig, near_opt_tol=0.02) -> dict:
    """Gate check: is there a single offset that is within `near_opt_tol` of the per-bias best at
    EVERY bias level (i.e. a robust generalist)? The gate wants this FALSE."""
    if not bank["matched_groups"]:
        return {"available": False}
    # per (matched group, bias) best utility, and per offset utility
    offset_gap = {}  # offset -> max over (group,bias) of (best - this offset) ; small = robust
    for mg in bank["matched_groups"]:
        for bias, cmap in mg["by_damping"].items():
            us = {ck: true_utility(ep, cfg) for ck, ep in cmap.items()}
            best = max(us.values())
            for ck, u in us.items():
                offset_gap[ck] = max(offset_gap.get(ck, 0.0), best - u)
    most_robust = min(offset_gap, key=offset_gap.get)
    worst_case_gap = offset_gap[most_robust]
    return {"available": True, "most_robust_offset": most_robust,
            "worst_case_gap_of_most_robust": worst_case_gap,
            "robust_generalist_exists": worst_case_gap <= near_opt_tol,
            "per_offset_worst_case_gap": offset_gap}


def best_offset_per_bias(bank: dict, cfg: UtilityConfig) -> dict:
    """The argmax offset at each bias, per matched group (for the 'best offset differs' gate)."""
    out = {}
    for mg in bank["matched_groups"]:
        row = {}
        for bias, cmap in mg["by_damping"].items():
            us = {ck: true_utility(ep, cfg) for ck, ep in cmap.items()}
            row[str(bias)] = max(us, key=us.get)
        out[mg["matched_group_id"]] = row
    n_distinct = [len(set(r.values())) for r in out.values()]
    return {"per_group_best_by_bias": out,
            "groups_with_best_offset_differing_across_bias": sum(1 for n in n_distinct if n > 1),
            "n_groups": len(out)}


def decision_value(episodes, cfg: UtilityConfig, fm: FieldMap = DEFAULT_FIELDS,
                   *, success_only: UtilityConfig | None = None) -> dict:
    bank = build_bank(episodes, fm)
    groups = selection_groups(episodes, fm)
    oc = oracle_candidate_per_group(groups, cfg)
    oc_mean = float(np.mean([v["best_true_utility"] for v in oc.values()])) if oc else float("nan")
    vsi = value_of_state_information(bank, cfg)
    sw = optimal_candidate_switch_rate(bank, cfg)
    rr = pairwise_rank_reversal_rate(bank, cfg)
    bso = best_single_offset(groups, groups, cfg, fm)
    robust = robust_offset_exists(bank, cfg)
    bpb = best_offset_per_bias(bank, cfg)
    aware_vals, _ = _state_aware(bank, cfg)
    aware_mean = float(np.mean(aware_vals)) if aware_vals else float("nan")

    out = {
        "frozen_utility": cfg.to_dict(),
        "n_selection_groups": len(groups), "n_matched_groups": bank["n_matched_groups"],
        "oracle_candidate_mean_true_utility": oc_mean,
        "state_agnostic_oracle_mean_true_utility": vsi.get("state_agnostic_mean_true_utility"),
        "state_aware_oracle_mean_true_utility": vsi.get("state_aware_mean_true_utility"),
        "VSI_frozen": vsi.get("vsi"),
        "optimal_candidate_switch_rate": sw.get("switch_rate"),
        "pairwise_rank_reversal_rate": rr.get("reversal_rate"),
        "best_single_offset": {"offset": bso.get("selected_offset"),
                               "mean_regret": bso.get("mean_regret"),
                               "mean_true_utility": bso.get("mean_true_utility"),
                               "selected_success_rate": bso.get("selected_success_rate")},
        "gap_best_single_to_state_aware": aware_mean - bso.get("mean_true_utility", float("nan")),
        "robust_offset": robust,
        "best_offset_per_bias": bpb,
    }
    # success-only VSI + selected-success gain (key co-primary for calibration bias)
    if success_only is not None:
        vsi_s = value_of_state_information(bank, success_only)
        aware_s, _ = _state_aware(bank, success_only)
        agn_s, _ = _state_agnostic(bank, success_only)
        # selected success of state-aware oracle vs best-single-offset
        bso_s = best_single_offset(groups, groups, success_only, fm)
        out["success_only"] = {
            "VSI_success_only": vsi_s.get("vsi"),
            "state_aware_mean": float(np.mean(aware_s)) if aware_s else None,
            "state_agnostic_mean": float(np.mean(agn_s)) if agn_s else None,
            "best_single_offset_success_rate": bso_s.get("selected_success_rate"),
            "state_aware_selected_success_gain": (float(np.mean(aware_s)) - bso_s.get("mean_true_utility"))
            if aware_s else None,
        }
    return out
