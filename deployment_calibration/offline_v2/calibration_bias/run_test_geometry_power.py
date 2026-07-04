"""Driver for the test-geometry learned-selector power study (offline; NO Isaac, no confirmatory data).

Runs the frozen protocol: primary test geometry +-0.035 over the block grid {9/6/9, 12/9/12, 18/9/18}
x primary taus {0.0342, 0.03425, 0.0343}, plus the pre-registered sensitivity geometries +-0.034 / +-0.036
(9/6/9, all three taus). Reuses test_geometry_power (which reuses the frozen learned_selector_power sim).

Sharded execution: each (design, tau, geometry) is one shard so the grid can run as parallel processes
(each pinned to 1 torch thread). `--shard` runs one config and writes a JSON; `--finalize` merges the
shard dir into results CSV/JSON, structural-vs-learned, seed stability, probe cost, and the verdict.

Usage:
  python -m deployment_calibration.offline_v2.calibration_bias.run_test_geometry_power \
      --shard TR VA TE TAU GEOMKEY REPS OUT.json
  python -m deployment_calibration.offline_v2.calibration_bias.run_test_geometry_power \
      --finalize SHARD_DIR
"""

from __future__ import annotations

import csv
import json
import os
import sys

# ---------------- frozen driver config ----------------
GRID = [(9, 6, 9), (12, 9, 12), (18, 9, 18)]
PRIMARY_TAUS = (0.0342, 0.03425, 0.0343)
TAU_ISO = 0.0325                      # sensitivity anchor only, not a gate
GEOMS = {"p035": (-0.035, 0.035), "s034": (-0.034, 0.034), "s036": (-0.036, 0.036)}
PRIMARY_GEOM = "p035"
SENS_GEOMS = ("s034", "s036")
REPS_PRIMARY = 500
REPS_SENS = 500
N_BOOT = 2000

# acceptance
GAIN_MIN = 0.15
LEARNED_POWER_MIN = 0.85
LEARNED_POWER_CI_LOWER_MIN = 0.80
ORACLE_MIN = 0.85

# probe cost (frozen from the prior phase; full-task deployment diagnostic trial)
PROBE_TIME_S = 16.58
LAMBDA_TIME = 0.02
HORIZONS = (1, 2, 5, 10)


def _run_shard(tr, va, te, tau, geomkey, reps, out):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T
    geom = GEOMS[geomkey]
    res = T.run_power(tr, va, te, tau, reps, geom, n_boot=N_BOOT)
    res["geom_key"] = geomkey
    res["is_primary"] = bool(geomkey == PRIMARY_GEOM)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out, "learned_power=%.3f struct_power=%.3f" % (res["learned_power"], res["struct_power"]))


def amortized_net_voi(gross_gain, probe_time=PROBE_TIME_S, lam=LAMBDA_TIME, horizons=HORIZONS):
    cost = lam * probe_time
    out = {"gross_voi_1task": gross_gain, "probe_cost": cost, "net_voi_1task": gross_gain - cost,
           "break_even_T": (cost / gross_gain) if gross_gain > 0 else None, "horizon": {}}
    for T in horizons:
        out["horizon"][str(T)] = T * gross_gain - cost
    return out


def _passes(shard):
    return bool(shard["learned_power"] >= LEARNED_POWER_MIN
                and shard["learned_power_ci95"][0] >= LEARNED_POWER_CI_LOWER_MIN)


def _finalize(shard_dir):
    shards = []
    for fn in sorted(os.listdir(shard_dir)):
        if fn.startswith("shard_") and fn.endswith(".json"):
            with open(os.path.join(shard_dir, fn)) as f:
                shards.append(json.load(f))
    docs = _docs_dir()
    os.makedirs(docs, exist_ok=True)

    # ---- results CSV + JSON ----
    fields = ["geom_key", "test_nominals", "train_blocks", "val_blocks", "test_blocks", "tau", "n_reps",
              "learned_power", "learned_power_ci95", "struct_power", "struct_power_ci95",
              "mean_learned_gain", "mean_struct_gain", "mean_k1_success", "mean_k0_success",
              "mean_best_single_success", "mean_oracle_success", "mean_history_increment",
              "frac_seed_stable", "frac_converged", "best_single_offset_counts"]
    with open(os.path.join(docs, "test_geometry_power_results_v1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for s in shards:
            w.writerow([s.get(k) for k in fields])
    with open(os.path.join(docs, "test_geometry_power_results_v1.json"), "w") as f:
        json.dump(shards, f, indent=2)

    # ---- verdict ----
    primary = [s for s in shards if s["geom_key"] == PRIMARY_GEOM]
    # minimal passing design: smallest total blocks passing at ALL three primary taus
    by_design = {}
    for s in primary:
        d = (s["train_blocks"], s["val_blocks"], s["test_blocks"])
        by_design.setdefault(d, {})[round(s["tau"], 6)] = s
    passing_designs = []
    for d, tmap in by_design.items():
        taus_ok = all(round(t, 6) in tmap and _passes(tmap[round(t, 6)]) for t in PRIMARY_TAUS)
        if taus_ok:
            passing_designs.append(d)
    passing_designs.sort(key=lambda d: (sum(d), -d[2], -d[0]))     # min total, then larger test, larger train
    min_design = passing_designs[0] if passing_designs else None

    # structural vs learned separation and mechanism health
    k1k0 = [s["mean_history_increment"] for s in primary]
    k1_beats_k0 = all(s["mean_k1_success"] - s["mean_k0_success"] > 0.05 for s in primary)
    struct_high = all(s["struct_power"] >= ORACLE_MIN for s in primary)
    any_learned_pass = any(_passes(s) for s in primary)

    # Net VOI on the minimal passing design (or the 9/6/9 primary point) — reported, secondary
    ref = None
    if min_design is not None:
        ref = by_design[min_design][round(0.03425, 6)]
    else:
        ref = next((s for s in primary if s["train_blocks"] == 9 and abs(s["tau"] - 0.03425) < 1e-9), primary[0])
    net_voi = amortized_net_voi(ref["mean_learned_gain"])

    if not k1_beats_k0:
        verdict = "STOP_CURRENT_CONFIRMATORY_CONCEPT"
    elif min_design is not None:
        verdict = "POWER_SUFFICIENT_FOR_PREREG_V4"
    elif any_learned_pass or struct_high:
        verdict = "INCREASE_BLOCK_COUNTS"
    else:
        verdict = "CLAIM_REDESIGN_REQUIRED"

    ninenine = by_design.get((9, 6, 9), {})
    verdict_obj = {
        "verdict": verdict,
        "primary_geometry": list(GEOMS[PRIMARY_GEOM]),
        "gain_threshold": GAIN_MIN,
        "learned_power_bar": {"power_min": LEARNED_POWER_MIN, "ci_lower_min": LEARNED_POWER_CI_LOWER_MIN},
        "min_passing_design": list(min_design) if min_design else None,
        "9_6_9_passes_all_primary_taus": bool((9, 6, 9) in [tuple(d) for d in passing_designs]),
        "9_6_9_per_tau": {str(t): {"learned_power": ninenine[round(t, 6)]["learned_power"],
                                   "learned_power_ci95": ninenine[round(t, 6)]["learned_power_ci95"],
                                   "struct_power": ninenine[round(t, 6)]["struct_power"],
                                   "passes": _passes(ninenine[round(t, 6)])}
                          for t in PRIMARY_TAUS if round(t, 6) in ninenine},
        "k1_beats_k0": bool(k1_beats_k0),
        "mean_history_increment_range": [min(k1k0), max(k1k0)],
        "struct_power_all_high": bool(struct_high),
        "threshold_not_lowered": True,
        "primary_comparator_unchanged": "B2_K1_success - train/val-only best_single_success",
        "net_voi_reference_design": list(min_design) if min_design else [9, 6, 9],
        "net_voi": net_voi,
        "no_preregistration_v4": True,
        "no_runtime_or_confirmatory_data": True,
    }
    with open(os.path.join(docs, "test_geometry_power_verdict_v1.json"), "w") as f:
        json.dump(verdict_obj, f, indent=2)

    # ---- seed stability machine form (primary geom) ----
    seed_json = {"gate": ">=4/5 seeds point-gain>=0.15 and none<0",
                 "per_config": [{"design": [s["train_blocks"], s["val_blocks"], s["test_blocks"]],
                                 "tau": s["tau"], "frac_seed_stable": s["frac_seed_stable"],
                                 "frac_converged": s["frac_converged"]}
                                for s in primary]}
    with open(os.path.join(docs, "test_geometry_seed_stability_v1.json"), "w") as f:
        json.dump(seed_json, f, indent=2)

    print("VERDICT:", verdict, "| min_design:", min_design)
    return verdict_obj


def _docs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))     # up to project root (paper/)
    return os.path.join(root, "docs", "offline_v2", "calibration_bias")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--shard":
        tr, va, te = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        tau = float(sys.argv[5]); geomkey = sys.argv[6]; reps = int(sys.argv[7]); out = sys.argv[8]
        _run_shard(tr, va, te, tau, geomkey, reps, out)
    elif len(sys.argv) >= 3 and sys.argv[1] == "--finalize":
        _finalize(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
