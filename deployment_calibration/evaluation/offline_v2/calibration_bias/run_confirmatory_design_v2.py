"""Emit confirmatory_design_validation_v2.json for the calibration-bias v2 preregistration.

Pre-data ONLY: uses the exploration-derived success geometry + effect size + variance. Reads NO
confirmatory data. Produces: the v1-flaw / v2-fix split validation, the predicted confirmatory
K-curve on the finer 9-bias grid (geometry), the power simulation, and the frozen block counts.

    python -m deployment_calibration.evaluation.offline_v2.calibration_bias.run_confirmatory_design_v2 --out <DIR>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import design_validation as DV
from deployment_calibration.offline_v2.calibration_bias import power as PW

OFFSETS = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
BAND = DV.COMPENSATION_BAND                     # 0.02
CONFIRM_BIASES = [round(-0.04 + 0.01 * i, 3) for i in range(9)]   # -0.04..+0.04 step 0.01
TRAIN = [-0.04, -0.02, 0.0, 0.02, 0.04]
VAL = [-0.01, 0.01]
TEST = [-0.03, 0.03]
PROBES = [("probe_m040", -0.04), ("probe_p040", 0.04)]   # first, second (by probe_index)
PROBE_TIME = {1: 11.78, 2: 23.22}               # measured cumulative probe time (s), from capability map
LAMBDA_TIME = 0.02


def _probe_succ(bias, probe_off):
    return abs(bias + probe_off) <= BAND + 1e-9


def _class_best_success(biases_in_class):
    """Geometric in-class ceiling: best single offset's mean success over the class biases."""
    return float(max(np.mean([1.0 if abs(b + o) <= BAND + 1e-9 else 0.0 for b in biases_in_class])
                     for o in OFFSETS))


def predicted_k_curve(biases):
    """Geometric selected-success by K on `biases` using the fixed probe order."""
    out = {}
    for K in (0, 1, 2):
        used = PROBES[:K]
        classes = defaultdict(list)
        for b in biases:
            classes[tuple(_probe_succ(b, po) for _, po in used)].append(b)
        sel = float(np.mean([_class_best_success(bs) for c, bs in classes.items() for _ in bs]))
        out[K] = {"probes": [p for p, _ in used], "n_classes": len(classes),
                  "selected_success": sel}
    return out


def net_voi(kcurve):
    base = kcurve[0]["selected_success"]
    out = {}
    for K in (0, 1, 2):
        gross = kcurve[K]["selected_success"] - base
        t = PROBE_TIME.get(K, 0.0)
        out[K] = {"gross_success_voi": gross, "probe_time_cumulative": t,
                  "probe_time_cost": LAMBDA_TIME * t, "net_voi": gross - LAMBDA_TIME * t}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sims", type=int, default=400)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    v1 = DV.validate_confirmatory_split(TRAIN, [-0.03, 0.03], [-0.01, 0.01], OFFSETS, BAND)
    v2 = DV.validate_confirmatory_split(TRAIN, VAL, TEST, OFFSETS, BAND)

    k_full = predicted_k_curve(CONFIRM_BIASES)
    k_test = predicted_k_curve(TEST)
    voi_full = net_voi(k_full)

    # power: baseline (sharp edge) + conservative sensitivity (softer edge, larger nuisance)
    base = PW.power_curve(candidates=(6, 9, 12, 15, 18), min_test_blocks=9, target_power=0.8,
                          max_halfwidth=0.20, n_sims=a.n_sims, n_boot=800, seed=0)
    sens = PW.power_curve(candidates=(6, 9, 12, 15, 18), min_test_blocks=9, target_power=0.8,
                          max_halfwidth=0.20, s=0.005, sigma=0.008, n_sims=a.n_sims, n_boot=800, seed=1)
    # freeze the conservative choice (>=9 floor; power>=0.8 under the softer-edge model)
    n_test = max(base["chosen_test_blocks"], sens["chosen_test_blocks"])
    n_train, n_val = n_test, max(6, n_test // 2)   # train pairs across 5 biases; val fewer
    sessions = len(TRAIN) * n_train + len(VAL) * n_val + len(TEST) * n_test
    total_episodes = sessions * (len(PROBES) + len(OFFSETS))   # 2 probes + 7 candidates per session

    report = {
        "pre_data_only": True,
        "band": BAND, "offsets": OFFSETS, "confirmatory_biases": CONFIRM_BIASES,
        "split_v1_flawed": {"test": [-0.01, 0.01], "validation": v1},
        "split_v2_fixed": {"train": TRAIN, "validation": VAL, "test": TEST, "validation_report": v2},
        "predicted_K_curve_full_grid": k_full,
        "predicted_K_curve_test_biases": k_test,
        "net_voi_full_grid": voi_full,
        "k1_note": "K=1 uses ONLY the fixed first probe (probe_m040). On the 9-bias confirmatory grid a "
                   "single probe does NOT saturate (fail-class spans biases needing different offsets): "
                   f"full-grid selected success K0={k_full[0]['selected_success']:.3f} -> "
                   f"K1={k_full[1]['selected_success']:.3f} -> K2={k_full[2]['selected_success']:.3f}. "
                   "On the TEST biases {-0.03,+0.03} the first probe already separates them, so K>=1 "
                   f"reaches {k_test[1]['selected_success']:.3f}. The exploration-grid K1=1.0 was the "
                   "fixed-first-probe in-class ceiling on 5 biases, NOT best-of-two.",
        "power_baseline": base, "power_sensitivity": sens,
        "frozen_block_counts": {
            "train_blocks": n_train, "validation_blocks": n_val, "test_blocks": n_test,
            "min_test_blocks_floor": 9, "rationale": "smallest N>=9 with power>=0.8 under the "
            "conservative softer-edge model; train mirrors test, val halved (threshold selection only)"},
        "total_sessions": sessions, "total_episodes": total_episodes,
        "block_bootstrap_ci_scheme": {
            "resample_unit": "independent nuisance BLOCK within the evaluated split",
            "n_boot": 2000, "ci": 0.95,
            "note": "each resampled block contributes all its (bias,offset) rows; blocks are disjoint "
                    "across train/val/test; CI is over TEST blocks for the held-out H3 gain."},
        "overall_ok": bool(v2["ok"] and not v1["ok"]),
    }
    (out / "confirmatory_design_validation_v2.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "v1_split_ok": v1["ok"], "v1_reason": v1["reason"],
        "v2_split_ok": v2["ok"], "v2_common_offset": v2["common_success_offset_across_test"],
        "v2_max_success_gain": v2["max_achievable_success_only_gain"],
        "predicted_K_full": {k: round(k_full[k]["selected_success"], 3) for k in (0, 1, 2)},
        "predicted_K_test": {k: round(k_test[k]["selected_success"], 3) for k in (0, 1, 2)},
        "net_voi_full": {k: round(voi_full[k]["net_voi"], 3) for k in (0, 1, 2)},
        "frozen_blocks": report["frozen_block_counts"], "total_episodes": total_episodes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
