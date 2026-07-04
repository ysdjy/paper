"""Driver: offline confirmatory candidate-bank redesign comparison + verdict + geometry validation.

Emits the comparison table (md+json), geometry validation json, and redesign verdict json into --out.
No Isaac, no new data, no confirmatory generator. Read-only over the 306 residual realizations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import confirmatory_redesign as R

PROBE_OFFSETS = (-0.04, 0.04)     # first = -0.04 (in every candidate bank considered)
GAIN_MIN = 0.15
BS_TEST_SUCCESS_RANGE = (0.20, 0.80)     # recommended redesign selection band (NOT an original prereg number)
SA_TEST_SUCCESS_MIN = 0.85


def evaluate_all():
    rows = []
    for name, bank in R.CANDIDATE_BANKS.items():
        d = R.evaluate_design(bank)
        sens = R.sensitivity_over_tau(bank)
        fd = R.frozen_distribution_check(bank)
        pr = R.probe_analysis(bank, PROBE_OFFSETS)
        # non-degenerate primary contrast criteria (section 9)
        crit = {
            "A_state_aware_gt_best_single": d["state_aware_test_success"] > d["best_single_test_success"],
            "B_gain_ge_0.15": d["expected_success_gain"] >= GAIN_MIN,
            "C_per_block_gain_nondegenerate": d["per_block_gain_nondegenerate"],
            "D_test_blocks_both_best_single_outcomes": d["mixed_best_single_labels"],
            "E_residual_flips_primary_operating_cell": d["mixed_best_single_labels"],
            "F_state_aware_not_broadly_failing(>=0.85)": d["state_aware_test_success"] >= SA_TEST_SUCCESS_MIN,
            "G_no_common_robust_action": not d["common_robust_action_exists"],
            "best_single_train_val_stable(no_thin_tie)": (not d["best_single_tie_on_success"]) and _stable_margin(d, bank),
            "all_states_compensable": d["all_states_compensable"],
            "nondegenerate_all_tau": sens["nondegenerate_all_tau"],
            "bs_test_success_in_band": BS_TEST_SUCCESS_RANGE[0] <= d["best_single_test_success"] <= BS_TEST_SUCCESS_RANGE[1] + 0.02,
            "sa_test_success_ge_0.85": d["state_aware_test_success"] >= SA_TEST_SUCCESS_MIN,
        }
        rows.append({
            "design_id": name, "candidate_bank": list(bank),
            "residual_support": list(R.RESIDUAL_SUPPORT), "best_single_offset": d["best_single_offset"],
            "best_single_test_success": d["best_single_test_success"],
            "state_aware_test_success": d["state_aware_test_success"],
            "expected_success_gain": d["expected_success_gain"],
            "per_block_gain_variance": d["per_block_gain_variance"],
            "per_block_gain_nondegenerate": d["per_block_gain_nondegenerate"],
            "mixed_best_single_labels": d["mixed_best_single_labels"],
            "mixed_state_aware_labels": d["mixed_state_aware_labels"],
            "common_robust_action_exists": d["common_robust_action_exists"],
            "all_states_compensable": d["all_states_compensable"],
            "best_single_tie_on_success": d["best_single_tie_on_success"],
            "best_single_success_margin_top2": _top2_margin(bank),
            "gain_sensitivity_min": sens["gain_min"], "gain_sensitivity_max": sens["gain_max"],
            "nondegenerate_all_tau": sens["nondegenerate_all_tau"],
            "frozen_dist_gain": fd["gain"], "frozen_dist_best_single_success": fd["best_single_success"],
            "probe_K1_selected_success": pr["K_selected_success"][1],
            "probe_K2_selected_success": pr["K_selected_success"][2],
            "probe_first_distinguishes_test": pr["first_probe_distinguishes_test_nominals"],
            "gross_VOI_K1_success": pr["voi"][1]["gross_voi_success"],
            "net_VOI_K1": pr["voi"][1]["net_voi"],
            "criteria": crit, "all_criteria_met": all(crit.values()),
            "detail": {"sensitivity": sens, "frozen_dist": fd, "probe": pr},
        })
    return rows


def _top2_margin(bank):
    _, diag = R.best_single_offset(bank)
    ss = sorted((r["mean_success"] for r in diag["per_offset"]), reverse=True)
    return round(ss[0] - ss[1], 4) if len(ss) > 1 else 1.0


def _stable_margin(d, bank, min_margin=0.15):
    """Best-single is 'stable' if it wins train/val success by a comfortable margin over the runner-up
    (which, in the fine bank, is the degeneracy-inducing +-0.02). Thin margin -> fragile -> reject."""
    return _top2_margin(bank) >= min_margin


def geometry_validation():
    """Per-nominal per-residual-extreme compensability + common-action + operating-cell straddle."""
    out = {}
    for name, bank in R.CANDIDATE_BANKS.items():
        comp = R.all_states_compensable(bank)
        common = R.common_robust_action(bank)
        d = R.evaluate_design(bank)
        # operating cells: best-single abs_eff range at each test nominal
        cells = {}
        bs = d["best_single_offset"]
        for nb in R.TEST_NOMINALS:
            aes = [abs(nb + r + bs) for r in R.EMPIRICAL_RESIDUALS]
            cells[str(nb)] = {"best_single_offset": bs, "abs_eff_min": round(min(aes), 5),
                              "abs_eff_max": round(max(aes), 5),
                              "straddles_edge": bool(min(aes) <= R.TAU_POINT <= max(aes))}
        out[name] = {"all_states_compensable": comp["all_compensable"], "non_compensable": comp["non_compensable"],
                     "common_robust_action": common, "best_single_operating_cells": cells}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = evaluate_all()
    geom = geometry_validation()

    (out / "confirmatory_candidate_bank_comparison_v1.json").write_text(json.dumps(
        {"designs": rows, "empirical_edge": {"tau_point": R.TAU_POINT, "tau_interval": [R.TAU_LOWER, R.TAU_UPPER],
                                             "isotonic": R.TAU_ISOTONIC},
         "residual": {"sigma": R.RESIDUAL_SIGMA, "support": list(R.RESIDUAL_SUPPORT)},
         "criteria_thresholds": {"gain_min": GAIN_MIN, "bs_test_success_band": list(BS_TEST_SUCCESS_RANGE),
                                 "sa_test_success_min": SA_TEST_SUCCESS_MIN}}, indent=2))
    (out / "confirmatory_design_geometry_validation_v1.json").write_text(json.dumps(geom, indent=2))

    # comparison CSV
    keys = ["design_id", "candidate_bank", "best_single_offset", "best_single_test_success",
            "state_aware_test_success", "expected_success_gain", "per_block_gain_nondegenerate",
            "mixed_best_single_labels", "best_single_tie_on_success", "best_single_success_margin_top2",
            "common_robust_action_exists", "all_states_compensable", "probe_K1_selected_success",
            "gross_VOI_K1_success", "net_VOI_K1", "all_criteria_met"]
    with open(out / "confirmatory_candidate_bank_comparison_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k]) for k in keys})

    # verdict
    winners = [r for r in rows if r["all_criteria_met"]]
    a_ok = next((r for r in rows if r["design_id"] == "A_three_point"), None)
    if a_ok and a_ok["all_criteria_met"]:
        verdict = "SELECT_CANDIDATE_BANK_REDESIGN"; selected = "A_three_point"
    elif winners:
        verdict = "SELECT_CANDIDATE_BANK_REDESIGN"; selected = winners[0]["design_id"]
    else:
        verdict = "SELECT_CONTINUOUS_PRIMARY_REDESIGN"; selected = None
    vobj = {"verdict": verdict, "selected_design": selected,
            "needs_larger_residual": False,
            "needs_new_runtime_validation": _needs_runtime(selected),
            "next_stage_learned_selector_power_allowed": bool(selected is not None),
            "designs_meeting_all_criteria": [r["design_id"] for r in winners],
            "rejected": {r["design_id"]: [k for k, v in r["criteria"].items() if not v]
                         for r in rows if not r["all_criteria_met"]}}
    (out / "confirmatory_redesign_verdict_v1.json").write_text(json.dumps(vobj, indent=2))
    print(json.dumps({"verdict": verdict, "selected": selected,
                      "per_design": {r["design_id"]: {"best_single": r["best_single_offset"],
                                                       "bs_succ": r["best_single_test_success"],
                                                       "sa_succ": r["state_aware_test_success"],
                                                       "gain": r["expected_success_gain"],
                                                       "nondeg": r["per_block_gain_nondegenerate"],
                                                       "bs_margin": r["best_single_success_margin_top2"],
                                                       "all_met": r["all_criteria_met"]} for r in rows}}, indent=2))
    return 0


def _needs_runtime(selected):
    if selected is None:
        return True
    bank = set(R.CANDIDATE_BANKS.get(selected, ()))
    # all offsets already executed in the 135-cap-map / 306 band-edge (offsets +-0.04/0.0 etc. are covered)
    covered = set(round(o, 3) for o in R.CANDIDATE_BANKS["C_current_seven_point"])   # 306 used these
    return not bank.issubset(covered)


if __name__ == "__main__":
    import sys
    sys.exit(main())
