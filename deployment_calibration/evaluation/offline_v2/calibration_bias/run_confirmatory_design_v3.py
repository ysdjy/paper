"""Emit confirmatory_design_validation_v3.json — runtime-aligned residual nuisance (v3).

Pre-data ONLY. v3 fixes the v2 power-model / runtime-implementation mismatch: the nuisance is now a
per-block RESIDUAL calibration bias (actual_bias = nominal + residual), which the runtime can inject and
which genuinely moves the success outcome — replacing v2's grasp-perturbation assumption. Split,
7-offset bank, probe order, K=1 stop rule, utility, AND gate, and instrumentation are unchanged unless
the residual-aware design validation proves otherwise. Reads NO confirmatory data.

    python -m deployment_calibration.evaluation.offline_v2.calibration_bias.run_confirmatory_design_v3 --out <DIR>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import design_validation as DV
from deployment_calibration.offline_v2.calibration_bias import power as PW
from deployment_calibration.offline_v2.calibration_bias.residual_nuisance import (
    DEFAULT_RESIDUAL, residual_std_effective)

OFFSETS = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
BAND = DV.COMPENSATION_BAND
CONFIRM_BIASES = [round(-0.04 + 0.01 * i, 3) for i in range(9)]
TRAIN = [-0.04, -0.02, 0.0, 0.02, 0.04]
VAL = [-0.01, 0.01]
TEST = [-0.03, 0.03]
PROBES = [("probe_m040", -0.04), ("probe_p040", 0.04)]
PROBE_TIME = {1: 11.78, 2: 23.22}
LAMBDA_TIME = 0.02
TARGET_POWER = 0.8      # frozen (from v2)


def _probe_succ(bias, po):
    return abs(bias + po) <= BAND + 1e-9


def _class_best_success(bs):
    return float(max(np.mean([1.0 if abs(b + o) <= BAND + 1e-9 else 0.0 for b in bs]) for o in OFFSETS))


def predicted_k_curve(biases):
    out = {}
    for K in (0, 1, 2):
        used = PROBES[:K]
        cl = defaultdict(list)
        for b in biases:
            cl[tuple(_probe_succ(b, po) for _, po in used)].append(b)
        out[K] = {"probes": [p for p, _ in used], "n_classes": len(cl),
                  "selected_success": float(np.mean([_class_best_success(bs) for c, bs in cl.items() for _ in bs]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sims", type=int, default=400)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT_RESIDUAL

    v2_split = DV.validate_confirmatory_split(TRAIN, VAL, TEST, OFFSETS, BAND)
    v3_split = DV.validate_confirmatory_split_residual(TRAIN, VAL, TEST, OFFSETS, cfg, BAND)

    k_full = predicted_k_curve(CONFIRM_BIASES)
    k_test = predicted_k_curve(TEST)

    base = PW.power_curve_residual(candidates=(6, 9, 12, 15, 18), min_test_blocks=9,
                                   target_power=TARGET_POWER, max_halfwidth=0.20,
                                   n_sims=a.n_sims, n_boot=800, seed=0)
    sens = PW.power_curve_residual(candidates=(6, 9, 12, 15, 18), min_test_blocks=9,
                                   target_power=TARGET_POWER, max_halfwidth=0.20, s=0.005,
                                   n_sims=a.n_sims, n_boot=800, seed=1)
    n_test = max(base["chosen_test_blocks"], sens["chosen_test_blocks"])
    retained = (n_test == 9)
    n_train, n_val = 9, 6
    sessions = len(TRAIN) * n_train + len(VAL) * n_val + len(TEST) * n_test
    total_episodes = sessions * (len(PROBES) + len(OFFSETS))

    report = {
        "pre_data_only": True,
        "revision": "v3 aligns the power model with an IMPLEMENTABLE runtime nuisance (per-block residual "
                    "calibration bias); it does not change the split / bank / probes / utility / gate.",
        "residual_nuisance": cfg.to_dict(),
        "residual_effective_sd": residual_std_effective(cfg),
        "band": BAND, "offsets": OFFSETS, "confirmatory_biases": CONFIRM_BIASES,
        "split": {"train": TRAIN, "validation": VAL, "test": TEST},
        "split_validation_nominal_v2": v2_split,
        "split_validation_residual_v3": v3_split,
        "predicted_K_curve_full_grid": k_full,
        "predicted_K_curve_test_biases": k_test,
        "power_baseline_residual": base,
        "power_sensitivity_residual": sens,
        "frozen_block_counts": {"train_blocks": n_train, "validation_blocks": n_val,
                                "test_blocks": n_test, "min_test_blocks_floor": 9},
        "retained_9_6_9": retained,
        "total_sessions": sessions, "total_episodes": total_episodes,
        "block_bootstrap_ci_scheme": {"resample_unit": "independent nuisance block within the split",
                                      "n_boot": 2000, "ci": 0.95,
                                      "note": "residual is fixed per block and reused across the split's "
                                              "bias levels; blocks disjoint across train/val/test"},
        "unchanged_from_v2": ["bias split", "7-offset matched bank", "probe order + K=1 stop rule",
                              "frozen utility", "success-only co-primary", "AND gate",
                              "6 instrumentation fields", "K=0/1/2 (no third probe)",
                              "Net VOI(K=1) > 0 final-GO condition"],
        "overall_ok": bool(v3_split["ok"] and retained),
    }
    (out / "confirmatory_design_validation_v3.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "residual_sd_effective": round(residual_std_effective(cfg), 5),
        "v3_residual_split_ok": v3_split["ok"],
        "compensable_over_residual": v3_split["compensable_over_residual_support"]["ok"],
        "no_common_offset_any_residual": v3_split["no_common_offset_any_residual"],
        "max_success_gain_residual": v3_split["max_achievable_success_only_gain_residual"],
        "power_N9_baseline": next(r["power"] for r in base["curve"] if r["n_test_blocks"] == 9),
        "power_N9_sensitivity": next(r["power"] for r in sens["curve"] if r["n_test_blocks"] == 9),
        "chosen_test_blocks": n_test, "retained_9_6_9": retained, "total_episodes": total_episodes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
