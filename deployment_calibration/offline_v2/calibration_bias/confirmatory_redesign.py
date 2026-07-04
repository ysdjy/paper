"""Offline confirmatory candidate-bank redesign, driven by the 306-episode empirical hard edge.

The 306 band-edge run measured a HARD success threshold: success = 1[|actual_bias + offset| <= tau],
with tau in the empirically-identified interval (max success |eff| 0.0342 < tau < min failure |eff|
0.0343). This module evaluates candidate BANKS geometrically under that hard model (NO fabricated soft
scale) to find a design whose PRIMARY operating contrast (state-aware vs train/val best-single) is
non-degenerate at the future test nominal biases, WITHOUT enlarging the residual.

No Isaac, no new data, no confirmatory generator. Read-only over the 306 residual realizations.
"""

from __future__ import annotations

import numpy as np

# --- frozen empirical + v3 facts ---
TAU_POINT = 0.03425                       # (0.0342 + 0.0343)/2 midpoint of the identified interval
TAU_LOWER = 0.0342                        # max successful |eff|
TAU_UPPER = 0.0343                        # min failed |eff|
TAU_ISOTONIC = 0.0325                     # isotonic 0.5-crossing (sensitivity)
TAU_SENSITIVITY = (TAU_LOWER, TAU_POINT, TAU_UPPER, TAU_ISOTONIC)

RESIDUAL_SIGMA = 0.005                    # frozen v3
RESIDUAL_SUPPORT = (-0.01, 0.01)          # frozen v3
TRAIN_NOMINALS = (-0.04, -0.02, 0.0, 0.02, 0.04)
VAL_NOMINALS = (-0.01, 0.01)
TEST_NOMINALS = (-0.03, 0.03)

# the 18 residual realizations actually drawn in the 306 run (read-only; used as the empirical blocks)
EMPIRICAL_RESIDUALS = (-0.006304, -0.001434, 0.003966, 0.005185, 0.00923, 0.00418, 0.006572, -0.002085,
                       -0.001492, -0.001767, -0.004283, 0.005248, 0.003822, -0.001088, -0.006394,
                       -0.005183, -0.006723, 0.00046)

CANDIDATE_BANKS = {
    "A_three_point": (-0.04, 0.0, 0.04),
    "B_five_point": (-0.06, -0.04, 0.0, 0.04, 0.06),
    "C_current_seven_point": (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06),
}


def success(nominal, residual, offset, tau=TAU_POINT):
    return abs(nominal + residual + offset) <= tau + 1e-12


def state_aware_offset(actual_bias, bank):
    """Oracle-Z / history upper bound: knows the ACTUAL bias, picks the deepest compensating offset."""
    return min(bank, key=lambda o: abs(actual_bias + o))


def best_single_offset(bank, tau=TAU_POINT, residuals=EMPIRICAL_RESIDUALS):
    """State-agnostic best-single: chosen on TRAIN+VAL ONLY (never test). Frozen selection rule:
      1. maximize mean success over (train/val nominal x block residual);
      2. tie -> maximize mean margin (tau - |eff|), a continuous-utility proxy (deeper = better);
      3. tie -> smallest |offset| (neutral action).
    Returns (offset, diagnostics incl. whether a tie occurred and whether the tie is fragile)."""
    tv = list(TRAIN_NOMINALS) + list(VAL_NOMINALS)
    rows = []
    for o in bank:
        succ = np.mean([1.0 if success(nb, r, o, tau) else 0.0 for nb in tv for r in residuals])
        margin = np.mean([tau - abs(nb + r + o) for nb in tv for r in residuals])
        rows.append({"offset": o, "mean_success": float(succ), "mean_margin": float(margin)})
    max_s = max(r["mean_success"] for r in rows)
    tied_s = [r for r in rows if abs(r["mean_success"] - max_s) < 1e-9]
    # step-2 tie-break by margin
    max_m = max(r["mean_margin"] for r in tied_s)
    tied_m = [r for r in tied_s if abs(r["mean_margin"] - max_m) < 1e-9]
    chosen = min(tied_m, key=lambda r: abs(r["offset"]))
    # fragility: is there a success-tie whose margin tie-break decides between different offsets?
    tie_on_success = len(tied_s) > 1
    return chosen["offset"], {"per_offset": rows, "tie_on_success": tie_on_success,
                              "n_success_tied": len(tied_s), "chosen_offset": chosen["offset"],
                              "success_tied_offsets": [r["offset"] for r in tied_s]}


def common_robust_action(bank, tau=TAU_POINT, residuals=EMPIRICAL_RESIDUALS, robust_frac=0.9):
    """Does any single offset succeed at BOTH test nominals for >= robust_frac of the residuals?
    (a robust common action would eliminate history's decision value)."""
    best = {"offset": None, "frac_both": 0.0}
    for o in bank:
        both = np.mean([1.0 if (success(TEST_NOMINALS[0], r, o, tau) and success(TEST_NOMINALS[1], r, o, tau))
                        else 0.0 for r in residuals])
        if both > best["frac_both"]:
            best = {"offset": o, "frac_both": float(both)}
    return {"most_common_offset": best["offset"], "frac_both_test_nominals": best["frac_both"],
            "robust_common_action_exists": best["frac_both"] >= robust_frac}


def all_states_compensable(bank, tau=TAU_POINT):
    """Every train/val/test nominal + every residual EXTREME has some offset within tau."""
    nominals = list(TRAIN_NOMINALS) + list(VAL_NOMINALS) + list(TEST_NOMINALS)
    bad = []
    for nb in nominals:
        for r in RESIDUAL_SUPPORT + (0.0,):
            actual = nb + r
            if min(abs(actual + o) for o in bank) > tau + 1e-12:
                bad.append({"nominal": nb, "residual": r})
    return {"all_compensable": len(bad) == 0, "non_compensable": bad}


def evaluate_design(bank, tau=TAU_POINT, residuals=EMPIRICAL_RESIDUALS):
    bs_off, bs_diag = best_single_offset(bank, tau, residuals)
    per_block = []            # per (test nominal, block) rows
    for nb in TEST_NOMINALS:
        for r in residuals:
            actual = nb + r
            sa_off = state_aware_offset(actual, bank)
            sa = 1 if success(nb, r, sa_off, tau) else 0
            bs = 1 if success(nb, r, bs_off, tau) else 0
            per_block.append({"nominal": nb, "residual": r, "sa_offset": sa_off, "sa_succ": sa,
                              "bs_succ": bs, "gain": sa - bs, "bs_abs_eff": round(abs(nb + r + bs_off), 5)})
    sa_succ = float(np.mean([x["sa_succ"] for x in per_block]))
    bs_succ = float(np.mean([x["bs_succ"] for x in per_block]))
    gains = [x["gain"] for x in per_block]
    # per-block gain aggregated per block (avg over the two test nominals) -> block-bootstrap unit
    by_block = {}
    for x in per_block:
        by_block.setdefault(x["residual"], []).append(x["gain"])
    block_gain = [float(np.mean(v)) for v in by_block.values()]
    bs_labels = set(x["bs_succ"] for x in per_block)
    sa_labels = set(x["sa_succ"] for x in per_block)
    common = common_robust_action(bank, tau, residuals)
    comp = all_states_compensable(bank, tau)
    return {
        "bank": list(bank), "tau": tau, "best_single_offset": bs_off,
        "best_single_tie_on_success": bs_diag["tie_on_success"],
        "best_single_success_tied_offsets": bs_diag["success_tied_offsets"],
        "state_aware_test_success": round(sa_succ, 4),
        "best_single_test_success": round(bs_succ, 4),
        "expected_success_gain": round(sa_succ - bs_succ, 4),
        "per_block_gain_mean": round(float(np.mean(block_gain)), 4),
        "per_block_gain_variance": round(float(np.var(block_gain)), 6),
        "per_block_gain_nondegenerate": bool(np.var(block_gain) > 1e-9),
        "mixed_best_single_labels": len(bs_labels) > 1,
        "mixed_state_aware_labels": len(sa_labels) > 1,
        "common_robust_action_exists": common["robust_common_action_exists"],
        "common_action_frac_both": common["frac_both_test_nominals"],
        "all_states_compensable": comp["all_compensable"],
        "non_compensable": comp["non_compensable"],
        "per_block": per_block,
    }


def sensitivity_over_tau(bank, residuals=EMPIRICAL_RESIDUALS):
    """Is the design's non-degeneracy + gain robust across the empirical threshold interval + isotonic?"""
    out = {}
    for tau in TAU_SENSITIVITY:
        d = evaluate_design(bank, tau, residuals)
        out[f"tau={tau}"] = {"best_single_offset": d["best_single_offset"],
                             "state_aware_success": d["state_aware_test_success"],
                             "best_single_success": d["best_single_test_success"],
                             "gain": d["expected_success_gain"],
                             "nondegenerate": d["per_block_gain_nondegenerate"],
                             "best_single_tie": d["best_single_tie_on_success"]}
    gains = [v["gain"] for v in out.values()]
    nd = [v["nondegenerate"] for v in out.values()]
    ties = [v["best_single_tie"] for v in out.values()]
    return {"per_tau": out, "gain_min": min(gains), "gain_max": max(gains),
            "nondegenerate_all_tau": all(nd), "best_single_tie_any_tau": any(ties)}


# --- probe information / VOI (probe = repeated-task deployment diagnostic FULL-TASK trial) ---
# observed full-task times from the 306 run (proxy for probe cost; success ~fast, failure ~timeout)
PROBE_TIME_SUCCESS_S = 9.42
PROBE_TIME_FAILURE_S = 23.75
LAMBDA_TIME = 0.02                          # frozen utility time weight


def probe_analysis(bank, probe_offsets, tau=TAU_POINT, residuals=EMPIRICAL_RESIDUALS):
    """K=0/1/2 fixed-probe identifiability + selected success on the TEST nominals.
    K=1 uses ONLY the first probe; K=2 both. Probes are fixed-offset full-task diagnostic trials."""
    def probe_succ(nb, r, po):
        return 1 if success(nb, r, po, tau) else 0

    def selected_success(k):
        # classify each (test nominal, residual) by the pattern of the first k probe outcomes, then use
        # the state-aware action for that class (oracle within class). K=0 -> no info -> best-single.
        used = probe_offsets[:k]
        if k == 0:
            bs, _ = best_single_offset(bank, tau, residuals)
            return float(np.mean([1 if success(nb, r, bs, tau) else 0
                                  for nb in TEST_NOMINALS for r in residuals]))
        # within each probe-pattern class, pick the offset maximizing success over that class's members
        cells = [(nb, r) for nb in TEST_NOMINALS for r in residuals]
        by_pat = {}
        for nb, r in cells:
            pat = tuple(probe_succ(nb, r, po) for po in used)
            by_pat.setdefault(pat, []).append((nb, r))
        succ = []
        for pat, members in by_pat.items():
            best_o = max(bank, key=lambda o: np.mean([1 if success(nb, r, o, tau) else 0 for nb, r in members]))
            for nb, r in members:
                succ.append(1 if success(nb, r, best_o, tau) else 0)
        return float(np.mean(succ))

    kcurve = {k: round(selected_success(k), 4) for k in (0, 1, 2)}
    # probe cumulative TIME (proxy): each probe is a full-task trial; time depends on its own outcome
    def probe_time(k):
        used = probe_offsets[:k]
        t = 0.0
        for po in used:
            outs = [probe_succ(nb, r, po) for nb in TEST_NOMINALS for r in residuals]
            frac_s = np.mean(outs)
            t += frac_s * PROBE_TIME_SUCCESS_S + (1 - frac_s) * PROBE_TIME_FAILURE_S
        return t
    base = kcurve[0]
    voi = {}
    for k in (0, 1, 2):
        gross = kcurve[k] - base
        cost = LAMBDA_TIME * probe_time(k)
        voi[k] = {"gross_voi_success": round(gross, 4), "probe_time_s": round(probe_time(k), 2),
                  "probe_time_cost": round(cost, 4), "net_voi": round(gross - cost, 4)}
    # test-nominal classification with the FIRST probe (1-bit discrimination check)
    first = probe_offsets[0]
    pat = {nb: [probe_succ(nb, r, first) for r in residuals] for nb in TEST_NOMINALS}
    distinguishes = (round(np.mean(pat[TEST_NOMINALS[0]])) != round(np.mean(pat[TEST_NOMINALS[1]])))
    return {"probe_offsets": list(probe_offsets), "K_selected_success": kcurve, "voi": voi,
            "first_probe_distinguishes_test_nominals": bool(distinguishes),
            "note": "probe = repeated-task deployment diagnostic full-task trial (NOT low-cost/non-destructive/online); "
                    "Net VOI uses observed full-task times as a proxy and is preliminary."}


def frozen_distribution_check(bank, tau=TAU_POINT, n=4000, seed=0):
    """Robustness under the FROZEN residual distribution (not just the 18 realizations)."""
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < n:
        x = rng.normal(0.0, RESIDUAL_SIGMA)
        if RESIDUAL_SUPPORT[0] <= x <= RESIDUAL_SUPPORT[1]:
            draws.append(float(x))
    d = evaluate_design(bank, tau, draws)
    return {"state_aware_success": d["state_aware_test_success"],
            "best_single_success": d["best_single_test_success"], "gain": d["expected_success_gain"],
            "nondegenerate": d["per_block_gain_nondegenerate"],
            "best_single_offset": d["best_single_offset"]}
