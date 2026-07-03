"""Oracle baselines and decision-value analysis (offline_v2).

Distinguishes three oracles and derives the decision-value quantities that tell us whether
hidden-state (damping) knowledge matters for PLAN SELECTION (not just outcome prediction):

  1. Oracle-Candidate     per candidate_group, pick the truly-best candidate (max true U).
                          This is the regret LOWER BOUND. Works on any data.
  2. State-agnostic Oracle knows the candidate bank but NOT damping; must commit to a single
                          candidate per matched group. Requires a MATCHED bank.
  3. State-aware Oracle    knows true damping; may pick a different candidate per damping.
                          Requires a MATCHED bank. It is an UPPER BOUND / diagnostic, not a
                          deployable model.

Derived:
  * Value of State Information  VSI = mean_true_U(state-aware) - mean_true_U(state-agnostic)
  * Optimal Candidate Switch Rate  fraction of matched groups whose argmax-true-U candidate
                                   changes across damping.
  * Pairwise Rank Reversal Rate    fraction of (candidate pair, damping pair) whose true-U
                                   ordering flips.

Switch / reversal / VSI are computed ONLY on a validated matched bank (see pairing.py). On
non-matched data they return None with an explicit reason, never a fabricated number.
"""

from __future__ import annotations

import itertools

import numpy as np

from .utility import UtilityConfig, true_utility


# ---------------------------------------------------------------- Oracle-Candidate
def oracle_candidate_per_group(groups: dict, cfg: UtilityConfig) -> dict:
    """groups: {group_id -> [episodes]}. Returns per-group best true utility & the picked ep."""
    out = {}
    for gid, eps in groups.items():
        us = [(true_utility(e, cfg), e) for e in eps]
        best_u, best_e = max(us, key=lambda t: t[0])
        out[gid] = {
            "best_true_utility": best_u,
            "best_candidate_index": best_e.get("candidate_index"),
            "best_success": bool(best_e["y"]["success"]),
            "best_task_error": float(best_e["y"]["task_outcome_error"]),
            "best_time": float(best_e["y"]["skill_elapsed_time"]),
        }
    return out


def selection_regret(groups: dict, chosen_index_by_group: dict, cfg: UtilityConfig) -> dict:
    """Given a model's chosen candidate_index per group, compute regret vs Oracle-Candidate.

    Returns per-group regret plus realized outcomes of the chosen candidate.
    """
    oc = oracle_candidate_per_group(groups, cfg)
    rows = []
    for gid, eps in groups.items():
        by_idx = {e.get("candidate_index"): e for e in eps}
        pick = by_idx.get(chosen_index_by_group.get(gid))
        if pick is None:  # fall back to first candidate if model gave nothing valid
            pick = eps[0]
        u_pick = true_utility(pick, cfg)
        best = oc[gid]["best_true_utility"]
        rows.append({
            "group_id": gid,
            "session_id": eps[0]["session_id"],
            "regret": best - u_pick,
            "top1": 1.0 if u_pick >= best - 1e-9 else 0.0,
            "selected_success": 1.0 if pick["y"]["success"] else 0.0,
            "selected_task_error": float(pick["y"]["task_outcome_error"]),
            "selected_time": float(pick["y"]["skill_elapsed_time"]),
            "selected_index": pick.get("candidate_index"),
        })
    return {"rows": rows,
            "mean_regret": float(np.mean([r["regret"] for r in rows])) if rows else float("nan"),
            "median_regret": float(np.median([r["regret"] for r in rows])) if rows else float("nan"),
            "top1_acc": float(np.mean([r["top1"] for r in rows])) if rows else float("nan"),
            "selected_success_rate": float(np.mean([r["selected_success"] for r in rows])) if rows else float("nan"),
            "selected_task_error": float(np.mean([r["selected_task_error"] for r in rows])) if rows else float("nan"),
            "selected_time": float(np.mean([r["selected_time"] for r in rows])) if rows else float("nan")}


# ---------------------------------------------------------------- matched-bank oracles
def _state_aware(matched_bank, cfg: UtilityConfig):
    """For each (matched group, damping): best true utility achievable knowing damping."""
    vals, picks = [], {}
    for mg in matched_bank["matched_groups"]:
        picks[mg["matched_group_id"]] = {}
        for damp, cand_map in mg["by_damping"].items():
            us = {ck: true_utility(ep, cfg) for ck, ep in cand_map.items()}
            best_ck = max(us, key=us.get)
            vals.append(us[best_ck])
            picks[mg["matched_group_id"]][damp] = {"candidate_key": best_ck, "true_utility": us[best_ck]}
    return vals, picks


def _state_agnostic(matched_bank, cfg: UtilityConfig, restrict_keys=None):
    """For each matched group, commit to ONE candidate (max mean-over-damping true U), then
    score it at every damping. restrict_keys: optional {mg_id: candidate_key} to force a
    train-estimated choice (protocol mode); else in-sample argmax."""
    vals, picks = [], {}
    for mg in matched_bank["matched_groups"]:
        mgid = mg["matched_group_id"]
        # candidate keys present at every damping (comparable)
        common = set.intersection(*[set(cm) for cm in mg["by_damping"].values()])
        if not common:
            continue
        if restrict_keys is not None and mgid in restrict_keys:
            best_ck = restrict_keys[mgid]
            if best_ck not in common:
                continue
        else:
            mean_u = {ck: np.mean([true_utility(mg["by_damping"][d][ck], cfg) for d in mg["by_damping"]])
                      for ck in common}
            best_ck = max(mean_u, key=mean_u.get)
        picks[mgid] = best_ck
        for damp, cand_map in mg["by_damping"].items():
            vals.append(true_utility(cand_map[best_ck], cfg))
    return vals, picks


def value_of_state_information(matched_bank, cfg: UtilityConfig) -> dict:
    """VSI and the two oracle means. None if bank is not matched/insufficient."""
    if matched_bank is None or not matched_bank.get("matched_groups"):
        return {"available": False, "reason": matched_bank.get("reason") if matched_bank else "no matched bank"}
    aware_vals, aware_picks = _state_aware(matched_bank, cfg)
    agn_vals, agn_picks = _state_agnostic(matched_bank, cfg)
    if not aware_vals or not agn_vals:
        return {"available": False, "reason": "no candidate keys common across damping levels"}
    aware_mean = float(np.mean(aware_vals))
    agn_mean = float(np.mean(agn_vals))
    return {
        "available": True,
        "state_aware_mean_true_utility": aware_mean,
        "state_agnostic_mean_true_utility": agn_mean,
        "vsi": aware_mean - agn_mean,
        "state_aware_picks": aware_picks,
        "state_agnostic_picks": agn_picks,
        "n_matched_groups": len(matched_bank["matched_groups"]),
    }


def optimal_candidate_switch_rate(matched_bank, cfg: UtilityConfig) -> dict:
    """Fraction of matched groups whose truly-best candidate changes across damping."""
    if matched_bank is None or not matched_bank.get("matched_groups"):
        return {"available": False, "reason": matched_bank.get("reason") if matched_bank else "no matched bank"}
    switched, details = 0, []
    n = 0
    for mg in matched_bank["matched_groups"]:
        best_by_damp = {}
        for damp, cand_map in mg["by_damping"].items():
            us = {ck: true_utility(ep, cfg) for ck, ep in cand_map.items()}
            best_by_damp[damp] = max(us, key=us.get)
        n += 1
        uniq = set(best_by_damp.values())
        if len(uniq) > 1:
            switched += 1
        details.append({"matched_group_id": mg["matched_group_id"], "best_by_damping": best_by_damp,
                        "switched": len(uniq) > 1})
    return {"available": True, "switch_rate": switched / n if n else float("nan"),
            "n_switched": switched, "n_groups": n, "details": details}


def pairwise_rank_reversal_rate(matched_bank, cfg: UtilityConfig) -> dict:
    """Fraction of (candidate pair, damping pair) whose true-U ordering flips.

    Only counts strict orderings (ties on either side are skipped, not counted as reversals).
    """
    if matched_bank is None or not matched_bank.get("matched_groups"):
        return {"available": False, "reason": matched_bank.get("reason") if matched_bank else "no matched bank"}
    total, reversed_ = 0, 0
    for mg in matched_bank["matched_groups"]:
        damps = sorted(mg["by_damping"])
        common = sorted(set.intersection(*[set(cm) for cm in mg["by_damping"].values()]))
        u = {d: {ck: true_utility(mg["by_damping"][d][ck], cfg) for ck in common} for d in damps}
        for ci, cj in itertools.combinations(common, 2):
            for di, dj in itertools.combinations(damps, 2):
                a = u[di][ci] - u[di][cj]
                b = u[dj][ci] - u[dj][cj]
                if abs(a) < 1e-9 or abs(b) < 1e-9:
                    continue
                total += 1
                if (a > 0) != (b > 0):
                    reversed_ += 1
    return {"available": True, "reversal_rate": reversed_ / total if total else float("nan"),
            "n_reversed": reversed_, "n_comparisons": total}
