"""Read-only analysis of the authorized 306-episode band-edge exploratory run.

Uses ONLY the FROZEN analysis plan (`band_edge.py` estimators, 0.005 m bins, logistic + isotonic,
block-bootstrap n=2000, operating region, exit criteria). Nothing here chooses a bin width, model,
bootstrap unit/count, success definition, or exit condition from the data. The statistical unit is the
nuisance BLOCK (18); every CI is a block bootstrap (resample whole blocks, each 17 records).

Never mutates the source data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import band_edge as BE

BIN_WIDTH = BE.BIN_WIDTH            # 0.005 (frozen)
N_BOOT = BE.N_BOOT                  # 2000 (frozen)
CI = BE.CI                          # 0.95 (frozen)
OPERATING_ABS_EFF = BE.OPERATING_ABS_EFF
EFF03 = BE.EFF03
PRED_POINTS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05)
FUTURE_TEST_NOMINALS = (-0.03, 0.03)   # v3 frozen confirmatory test biases (mapping only, not observed here)


# --------------------------------------------------------------------- load / seed
def load_run(data_dir):
    d = Path(data_dir)
    eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
    man = json.loads((d / "manifest.json").read_text())
    meta = json.loads((d / "run_metadata.json").read_text())
    return eps, man, meta


def episodes_sha256(data_dir) -> str:
    return hashlib.sha256((Path(data_dir) / "episodes.jsonl").read_bytes()).hexdigest()


def derive_bootstrap_seed(data_dir) -> int:
    """Protocol froze n_boot=2000 but not a seed -> derive a deterministic seed from the input hash."""
    return int(episodes_sha256(data_dir)[:8], 16)


def to_records(eps) -> list:
    """BE-estimator record shape: block_id, eff (|actual+offset|), eff_signed, success (+ audit fields)."""
    out = []
    for e in eps:
        s = e["secret_deployment_state"]
        out.append({
            "block_id": e["block_id"], "offset_id": e["offset_id"],
            "offset": float(e["theta"]["grasp_offset_local_y"]),
            "residual": float(s["residual_bias_y"]), "actual_bias": float(s["actual_bias_y"]),
            "eff_signed": float(e["eff_signed"]), "eff": float(e["abs_eff"]),
            "success": int(e["y"]["success"]),
            "failure_reason": e["y"]["failure_reason"], "failure_phase": e.get("failure_phase"),
        })
    return out


# --------------------------------------------------------------------- general block bootstrap
def block_bootstrap(records, stat_fn, n_boot=N_BOOT, seed=0, ci=CI):
    """Resample the 18 blocks with replacement (each block = all its records); recompute stat_fn."""
    by_block = defaultdict(list)
    for r in records:
        by_block[r["block_id"]].append(r)
    blocks = list(by_block)
    rng = np.random.default_rng(seed)
    point = stat_fn(records)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(blocks), size=len(blocks))
        sample = [rr for i in idx for rr in by_block[blocks[i]]]
        v = stat_fn(sample)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            reps.append(v)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    reps = np.asarray(reps, float)
    return {"point": point,
            "ci_low": float(np.quantile(reps, lo)) if reps.size else float("nan"),
            "ci_high": float(np.quantile(reps, hi)) if reps.size else float("nan"),
            "n_valid": int(reps.size), "n_boot": n_boot}


# --------------------------------------------------------------------- 5.1 binned success
def binned_success(records, bin_width=BIN_WIDTH, direction=None, seed=0):
    """Per 0.005 bin: counts, block count, success rate + block-bootstrap CI, failure reasons/phases.
    direction: None (all), 'neg' (eff_signed<0), 'pos' (eff_signed>0)."""
    recs = records
    if direction == "neg":
        recs = [r for r in records if r["eff_signed"] < 0]
    elif direction == "pos":
        recs = [r for r in records if r["eff_signed"] > 0]
    by = defaultdict(list)
    for r in recs:
        by[round(np.floor(r["eff"] / bin_width) * bin_width, 4)].append(r)
    rows = []
    for b in sorted(by):
        es = by[b]
        n = len(es); s = sum(r["success"] for r in es)
        blks = len(set(r["block_id"] for r in es))
        fr = Counter(r["failure_reason"] for r in es if not r["success"])
        fp = Counter(r["failure_phase"] for r in es if not r["success"])
        # block-bootstrap CI on the success rate WITHIN this bin (resample blocks contributing to it)
        boot = block_bootstrap(es, lambda rr: float(np.mean([x["success"] for x in rr])) if rr else float("nan"),
                               seed=seed)
        rows.append({"bin_lo": round(b, 4), "bin_hi": round(b + bin_width, 4), "bin_center": round(b + bin_width / 2, 4),
                     "n_episodes": n, "n_blocks": blks, "n_success": s, "n_failure": n - s,
                     "success_rate": round(s / n, 4), "ci_low": round(boot["ci_low"], 4), "ci_high": round(boot["ci_high"], 4),
                     "failure_reasons": dict(fr), "failure_phases": dict(fp)})
    return rows


# --------------------------------------------------------------------- 5.2 logistic (frozen) + bootstrap
def logistic_analysis(records, seed=0):
    fit = BE.fit_logistic_edge(records)
    bb = BE.block_bootstrap_edge(records, n_boot=N_BOOT, ci=CI, seed=seed)
    preds = {}
    if fit.get("converged"):
        for p in PRED_POINTS:
            with np.errstate(over="ignore"):
                preds[str(p)] = round(float(1.0 / (1.0 + np.exp((p - fit["center"]) / fit["scale"]))), 4)
    # separation / identifiability check
    ys = sorted(set(r["success"] for r in records))
    sep = "complete_separation" if len(ys) < 2 else None
    return {"converged": fit.get("converged"), "edge_center": fit.get("center"), "edge_scale": fit.get("scale"),
            "center_ci": bb["center_ci"], "scale_ci": bb["scale_ci"],
            "n_valid_boot": bb["n_valid_boot"], "n_boot": N_BOOT,
            "predicted_success_prob": preds,
            "separation_note": sep,
            "scale_identifiable": bool(fit.get("converged") and fit.get("scale", 0) > BE.EDGE_SCALE_MIN_SOFT)}


# --------------------------------------------------------------------- 5.3 isotonic (frozen) + bootstrap
def isotonic_analysis(records, seed=0):
    iso = BE.fit_isotonic_edge(records)
    def cross_fn(rr):
        r = BE.fit_isotonic_edge(rr)
        return r.get("edge_center_cross0.5")
    bb = block_bootstrap(records, cross_fn, n_boot=N_BOOT, seed=seed)
    # count non-unique / unobtainable crossings across bootstraps
    return {"grid": iso["grid"], "success_pred": iso["success_pred"],
            "crossing_0.5": iso.get("edge_center_cross0.5"),
            "crossing_ci_low": round(bb["ci_low"], 5) if not np.isnan(bb["ci_low"]) else None,
            "crossing_ci_high": round(bb["ci_high"], 5) if not np.isnan(bb["ci_high"]) else None,
            "n_valid_boot_crossings": bb["n_valid"], "n_boot": N_BOOT,
            "crossing_unobtainable_count": N_BOOT - bb["n_valid"]}


# --------------------------------------------------------------------- 6 directional asymmetry
def directional_asymmetry(records, seed=0):
    neg = logistic_analysis([r for r in records if r["eff_signed"] < 0], seed=seed)
    pos = logistic_analysis([r for r in records if r["eff_signed"] > 0], seed=seed)
    diff = None
    if neg["edge_center"] is not None and pos["edge_center"] is not None:
        diff = abs(neg["edge_center"] - pos["edge_center"])
    # matched-bin success-rate difference (neg vs pos) with block bootstrap
    bins_neg = {r["bin_lo"]: r for r in binned_success(records, direction="neg", seed=seed)}
    bins_pos = {r["bin_lo"]: r for r in binned_success(records, direction="pos", seed=seed)}
    matched = []
    for b in sorted(set(bins_neg) & set(bins_pos)):
        matched.append({"bin_lo": b, "neg_rate": bins_neg[b]["success_rate"], "pos_rate": bins_pos[b]["success_rate"],
                        "abs_diff": round(abs(bins_neg[b]["success_rate"] - bins_pos[b]["success_rate"]), 4)})
    severe = bool(diff is not None and diff > BE.ASYMMETRY_MAX_CENTER_DIFF)
    return {"neg_center": neg["edge_center"], "neg_scale": neg["edge_scale"],
            "pos_center": pos["edge_center"], "pos_scale": pos["edge_scale"],
            "center_abs_diff": diff, "asymmetry_threshold": BE.ASYMMETRY_MAX_CENTER_DIFF,
            "matched_bin_rate_diff": matched,
            "verdict": "SEVERE_UNEXPLAINED_ASYMMETRY" if severe else "NO_SEVERE_ASYMMETRY"}


# --------------------------------------------------------------------- 7 boundary mixed labels
def boundary_mixed_labels(records):
    region = [r for r in records if 0.025 <= r["eff"] <= 0.035]
    s = sum(r["success"] for r in region); n = len(region)
    fr = Counter(r["failure_reason"] for r in region if not r["success"])
    fp = Counter(r["failure_phase"] for r in region if not r["success"])
    # the frozen bin containing 0.03 is [0.030, 0.035)
    b03 = [r for r in records if 0.030 <= r["eff"] < 0.035]
    s03 = sum(r["success"] for r in b03)
    neg = [r for r in region if r["eff_signed"] < 0]; pos = [r for r in region if r["eff_signed"] > 0]
    return {"region": "0.025 <= abs_eff <= 0.035", "n_episodes": n, "n_blocks": len(set(r["block_id"] for r in region)),
            "n_success": s, "n_failure": n - s, "success_rate": round(s / n, 4) if n else None,
            "failure_reasons": dict(fr), "failure_phases": dict(fp),
            "neg_success_rate": round(np.mean([r["success"] for r in neg]), 4) if neg else None,
            "pos_success_rate": round(np.mean([r["success"] for r in pos]), 4) if pos else None,
            "frozen_bin_containing_0.03": {"bin": "[0.030, 0.035)", "note": "the 0.005 bin that CONTAINS 0.03 "
                                           "(not exactly equal to 0.03)", "n": len(b03), "n_success": s03,
                                           "n_failure": len(b03) - s03,
                                           "success_rate": round(s03 / len(b03), 4) if b03 else None},
            "BOUNDARY_MIXED_LABELS": bool(0 < s < n)}


# --------------------------------------------------------------------- 8.1 per-offset residual label variation
def per_offset_variation(records):
    by_off = defaultdict(list)
    for r in records:
        by_off[round(r["offset"], 4)].append(r)
    rows = []
    any_mixed = False
    for off in sorted(by_off):
        es = by_off[off]
        succ_blocks = [r["block_id"] for r in es if r["success"]]
        fail_blocks = [r["block_id"] for r in es if not r["success"]]
        mixed = len(succ_blocks) > 0 and len(fail_blocks) > 0
        any_mixed = any_mixed or mixed
        succ_res = [r["residual"] for r in es if r["success"]]
        fail_res = [r["residual"] for r in es if not r["success"]]
        rows.append({"offset": off, "n_blocks": len(es),
                     "n_success": len(succ_blocks), "n_failure": len(fail_blocks), "mixed_labels": mixed,
                     "success_residual_range": [round(min(succ_res), 4), round(max(succ_res), 4)] if succ_res else None,
                     "failure_residual_range": [round(min(fail_res), 4), round(max(fail_res), 4)] if fail_res else None,
                     "abs_eff_range": [round(min(r["eff"] for r in es), 4), round(max(r["eff"] for r in es), 4)]})
    return {"per_offset": rows, "any_offset_mixed_across_blocks": any_mixed}


# --------------------------------------------------------------------- 8.2 future operating region mapping
def frozen_best_single_offset(offsets, train_nominals=(-0.04, -0.02, 0.0, 0.02, 0.04), band=0.02):
    """v3-frozen geometric best-single: offset covering the most train nominal biases (band 0.02). This is
    a DESIGN definition (independent of THIS run's outcomes)."""
    best, best_o = -1, offsets[0]
    for o in offsets:
        cov = sum(1 for b in train_nominals if abs(b + o) <= band + 1e-9)
        if cov > best:
            best, best_o = cov, o
    return best_o


def future_operating_region(records, logi, iso, res_lo=-0.01, res_hi=0.01):
    """Map the v3 future confirmatory test nominal biases (+/-0.03) + frozen residual support to abs_eff,
    then to the EMPIRICAL edge probability. Uses the v3-frozen best-single offset (NOT re-picked here)."""
    offsets = sorted(set(round(r["offset"], 4) for r in records))
    bs = frozen_best_single_offset(offsets)
    center, scale = logi.get("edge_center"), logi.get("edge_scale")

    def emp_logit(x):
        if not (center and scale):
            return None
        with np.errstate(over="ignore"):
            return round(float(1.0 / (1.0 + np.exp((x - center) / scale))), 4)

    def emp_iso(x):
        g = np.array(iso["grid"]); p = np.array(iso["success_pred"])
        return round(float(np.interp(x, g, p)), 4)

    cells = []
    for nb in FUTURE_TEST_NOMINALS:
        for policy, off in [("best_single_v3", bs), ("state_aware_compensating", -round(np.sign(nb) * 0.02, 4))]:
            lo = abs(nb + res_lo + off); hi = abs(nb + res_hi + off); mid = abs(nb + off)
            arange = [round(min(lo, hi, mid), 4), round(max(lo, hi, mid), 4)]
            cells.append({"nominal_bias": nb, "policy": policy, "offset": off, "abs_eff_range": arange,
                          "logistic_prob_range": [emp_logit(arange[0]), emp_logit(arange[1])],
                          "isotonic_prob_range": [emp_iso(arange[0]), emp_iso(arange[1])]})
    # non-degenerate = the best-single operating cells' probability strictly inside (eps, 1-eps)
    bs_cells = [c for c in cells if c["policy"] == "best_single_v3"]
    def nondeg(pr):
        return pr and all(p is not None and 0.02 < p < 0.98 for p in pr)
    enters_nondeg = any(nondeg(c["logistic_prob_range"]) or (c["abs_eff_range"][0] < 0.035 and c["abs_eff_range"][1] > 0.030)
                        for c in bs_cells)
    return {"v3_best_single_offset": bs, "future_test_nominals": list(FUTURE_TEST_NOMINALS),
            "residual_support": [res_lo, res_hi], "cells": cells,
            "best_single_enters_nondegenerate_edge": bool(enters_nondeg)}


# --------------------------------------------------------------------- 9 collision + threshold
def collision_analysis(records, eps_full):
    forces = [e["max_unintended_contact_force_N"] for e in eps_full]
    frames = [e["unintended_contact_frame_count"] for e in eps_full]
    avail = all(e.get("contact_sensor_available") for e in eps_full)
    n_contact = sum(1 for f in forces if f > 0)
    # frozen rule: threshold = above the clean-baseline noise floor from the band data distribution
    if n_contact == 0:
        thr = {"threshold_N": None, "status": "THRESHOLD_NOT_IDENTIFIABLE_FROM_FROZEN_RULE",
               "reason": "all 306 max unintended contact forces are exactly 0.0 N -> no positive distribution "
                         "to place a threshold; any epsilon>0 separates. No collision-confounded episodes exist."}
    else:
        thr = {"threshold_N": None, "status": "DISTRIBUTION_PRESENT_SEE_RULE",
               "reason": "positive contact forces present; apply the frozen rule to the distribution."}
    confounded = [e["episode_id"] for e in eps_full if e["max_unintended_contact_force_N"] > 0]
    # failure mechanism vs abs_eff
    by_mech = defaultdict(list)
    for r in records:
        if not r["success"]:
            by_mech[(r["failure_reason"], r["failure_phase"])].append(r["eff"])
    mech = {f"{k[0]}/{k[1]}": {"n": len(v), "abs_eff_min": round(min(v), 4), "abs_eff_max": round(max(v), 4),
                               "abs_eff_mean": round(float(np.mean(v)), 4)} for k, v in by_mech.items()}
    return {"contact_sensor_available_all": bool(avail), "max_force_N": max(forces), "min_force_N": min(forces),
            "n_episodes_with_contact": n_contact, "collision_confounded_episodes": confounded,
            "threshold_proposal": thr, "failure_mechanism_vs_abs_eff": mech,
            "EDGE_NOT_COLLISION_DRIVEN": bool(n_contact == 0)}


# --------------------------------------------------------------------- 10 instrumentation / confound
def instrumentation_confound(records, eps_full):
    from .band_edge_instrumentation import REQUIRED_FIELD_GROUPS
    req = [f for g in REQUIRED_FIELD_GROUPS.values() for f in g]
    missing = sum(1 for e in eps_full for f in req if f not in e)
    # signed joint margin: is any REAL joint-limit involvement (meaningful negative + limit clamp)?
    real_jl = []
    for e in eps_full:
        sgn = e.get("minimum_joint_limit_margin_rad_signed")
        lc = e.get("joint_limit_clamp_count", 0)
        if sgn is not None and sgn < -1e-3 and lc > 0:
            real_jl.append(e["episode_id"])
    # IK failures on failing edge episodes
    ik_fail_eps = [e["episode_id"] for e in eps_full if e.get("ik_failure_count", 0) > 0]
    # margin summary by success
    succ_margins = [e.get("minimum_joint_limit_margin_rad_signed") for e in eps_full if e["y"]["success"]]
    fail_margins = [e.get("minimum_joint_limit_margin_rad_signed") for e in eps_full if not e["y"]["success"]]
    neg_flags = sum(1 for e in eps_full if e.get("joint_limit_margin_negative_flag"))
    step_clamp = [e.get("joint_step_clamp_count", 0) for e in eps_full]
    limit_clamp = [e.get("joint_limit_clamp_count", 0) for e in eps_full]
    return {"instrumentation_missing_total": missing,
            "INSTRUMENTATION_COMPLETE": missing == 0,
            "signed_margin_negative_flag_count": neg_flags,
            "real_joint_limit_involvement_episodes": real_jl,
            "min_signed_margin_success": round(min([m for m in succ_margins if m is not None], default=0.0), 6),
            "min_signed_margin_failure": round(min([m for m in fail_margins if m is not None], default=0.0), 6),
            "ik_failure_episodes": ik_fail_eps,
            "max_ik_failure_count": max((e.get("ik_failure_count", 0) for e in eps_full), default=0),
            "step_clamp_range": [min(step_clamp), max(step_clamp)],
            "limit_clamp_range": [min(limit_clamp), max(limit_clamp)],
            "PRIMARY_EDGE_MECHANISM_NOT_EXPLAINED_BY_JOINT_OR_IK_CONFOUND":
                bool(len(real_jl) == 0 and len(ik_fail_eps) == 0)}


# --------------------------------------------------------------------- 11 continuous outcomes
def continuous_outcomes(eps_full, seed=0):
    recs = to_records(eps_full)
    by_id = {e["episode_id"]: e for e in eps_full}
    def col(e, path):
        if path[0] == "y":
            return e["y"].get(path[1])
        return e.get(path[0])
    fields = {"task_outcome_error": ("y", "task_outcome_error"), "skill_elapsed_time": ("y", "skill_elapsed_time"),
              "true_handle_error_at_close_3d": ("true_handle_error_at_close_3d",),
              "true_handle_error_at_close_local_y": ("true_handle_error_at_close_local_y",),
              "gripper_width_at_close": ("gripper_width_at_close",)}
    # by success
    out = {}
    for name, path in fields.items():
        vs_s = [col(e, path) for e in eps_full if e["y"]["success"] and col(e, path) is not None]
        vs_f = [col(e, path) for e in eps_full if not e["y"]["success"] and col(e, path) is not None]
        out[name] = {"success_mean": round(float(np.mean(vs_s)), 5) if vs_s else None,
                     "failure_mean": round(float(np.mean(vs_f)), 5) if vs_f else None,
                     "n_success_present": len(vs_s), "n_failure_present": len(vs_f)}
    return out
