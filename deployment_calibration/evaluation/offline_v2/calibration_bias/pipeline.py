"""Calibration-bias offline pipeline (validation + decision-value + held-out-bias evaluation).

Runs on a run dir (episodes.jsonl) OR an in-memory episode list (synthetic). Read-only on inputs.
Produces validation / independence / decision-value / split-manifest / held-out selection / VOI /
exploration-gate artifacts. Every main decision-value number uses the FROZEN utility; success-only
is reported as a co-primary. If any episode is synthetic, outputs carry synthetic_only=True.

The frozen EXPLORATION gate (GO-to-preregistration, NOT final significance):
  1. >=3 bias levels have a different best offset.
  2. no single offset is near-optimal at every bias (no robust generalist).
  3. state-aware oracle vs best-single: success-only gain >= 0.15 AND frozen VSI >= 0.05 (both).
  4. replicate-independence audit passes (no blocker).
  5. pairing / leakage / provenance pass.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from deployment_calibration.evaluation.offline_v2.pipeline import build_pairs, groups_from_pairs
from deployment_calibration.models_v2.train import ensemble_predict, train_seeds
from deployment_calibration.offline_v2 import metrics as MET
from deployment_calibration.offline_v2.utility import UtilityConfig, predicted_utility, true_utility
from deployment_calibration.offline_v2.calibration_bias import independence as IND
from deployment_calibration.offline_v2.calibration_bias import oracle as CBO
from deployment_calibration.offline_v2.calibration_bias import splits as CBS
from deployment_calibration.offline_v2.calibration_bias import voi as CBV
from deployment_calibration.offline_v2.calibration_bias.schema import (
    CONTROLLABLE_OFFSET, DEFAULT_FIELDS, FieldMap, ROLE_CANDIDATE, bias_level_of, offset_key)
from deployment_calibration.offline_v2.calibration_bias.validator import validate

FROZEN = UtilityConfig(lambda_error=1.0, lambda_time=0.02)
SUCCESS_ONLY = UtilityConfig(lambda_error=0.0, lambda_time=0.0)

# exploration-gate thresholds (FROZEN before any confirmatory data)
GATE = {"min_bias_levels_with_distinct_best_offset": 3, "robust_offset_tol": 0.02,
        "min_success_gain": 0.15, "min_frozen_vsi": 0.05}


def load_episodes(run_dir):
    p = Path(run_dir) / "episodes.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _distinct_best_offsets_across_bias(episodes, cfg, fm) -> int:
    """Marginal best offset per bias level (over all matched groups); count distinct values."""
    bank = CBO.build_bank(episodes, fm)
    by_bias = {}
    for mg in bank["matched_groups"]:
        for bias, cmap in mg["by_damping"].items():
            us = {ck: true_utility(ep, cfg) for ck, ep in cmap.items()}
            by_bias.setdefault(str(bias), {}).setdefault("acc", []).append(max(us, key=us.get))
    best_per_bias = {}
    for bias, d in by_bias.items():
        # modal best offset at this bias
        vals, counts = np.unique(np.array(d["acc"], dtype=object), return_counts=True)
        best_per_bias[bias] = vals[int(np.argmax(counts))]
    return len(set(best_per_bias.values())), best_per_bias


def held_out_selection(episodes, manifest, cfg, fm, seeds=(0, 1, 2, 3, 4)) -> dict:
    """Train on train-bias sessions, evaluate on UNSEEN test-bias sessions."""
    sess = manifest["sessions"]
    def eps_of(names):
        s = set(names)
        return [e for e in episodes if e.get(fm.session_id) in s]
    train_eps = eps_of(sess["train"] + sess["val"])
    test_eps = eps_of(sess["test"])
    val_eps = eps_of(sess["val"])
    kmax = max([e.get(fm.history_cutoff, 3) for e in episodes] + [3])
    tr = build_pairs(train_eps, set(sess["train"] + sess["val"]), k=kmax)
    va = build_pairs(val_eps, set(sess["val"]), k=kmax) or None
    te = build_pairs(test_eps, set(sess["test"]), k=kmax)
    test_groups = groups_from_pairs(te)
    train_groups = groups_from_pairs(tr)

    out = {"n_test_groups": len(test_groups), "test_bias_levels": manifest["split_level_ids"]["test"],
           "models": {}, "prediction": {}, "adaptation": {}}
    hist_by_ep = {id(e): H for e, H in te}
    for name in ("B0_baserate", "B1_static", "B2_mean", "B2_deepsets", "B2_gru", "OracleZ"):
        fit = train_seeds(name, tr, va, seeds=seeds)
        # prediction metrics on held-out bias
        y, ph, et_t, et_p, tt_t, tt_p = [], [], [], [], [], []
        for e, H in te:
            pr = ensemble_predict(fit, e, H)
            y.append(1.0 if e["y"]["success"] else 0.0); ph.append(pr["p_success"])
            et_t.append(float(e["y"]["task_outcome_error"])); et_p.append(pr["pred_error"])
            tt_t.append(float(e["y"]["skill_elapsed_time"])); tt_p.append(pr["pred_time"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["prediction"][name] = MET.prediction_metrics(y, ph, et_t, et_p, tt_t, tt_p)
        # selection
        chosen = _choose(fit, test_groups, hist_by_ep, cfg, fm)
        out["models"][name] = _regret(test_groups, chosen, cfg, fm)
    # baselines on the same test groups
    out["best_single_offset"] = CBO.best_single_offset(train_groups, test_groups, cfg, fm)
    # adaptation K sweep for the capacity-matched + history models
    for name in ("B1_static", "B2_mean", "B2_deepsets"):
        out["adaptation"][name] = {}
        for k in (0, 1, 2, 3):
            trk = build_pairs(train_eps, set(sess["train"] + sess["val"]), k=k)
            tek = build_pairs(test_eps, set(sess["test"]), k=k)
            fitk = train_seeds(name, trk, None, seeds=seeds)
            yk = [1.0 if e["y"]["success"] else 0.0 for e, _ in tek]
            pk = [ensemble_predict(fitk, e, H)["p_success"] for e, H in tek]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out["adaptation"][name][k] = {"auroc": MET.auroc(yk, pk), "brier": MET.brier(yk, pk)}
    return out


def _choose(fit, groups, hist_by_ep, cfg, fm):
    chosen = {}
    for gid, eps in groups.items():
        best_u, best_idx = -1e9, None
        for e in eps:
            pr = ensemble_predict(fit, e, hist_by_ep[id(e)])
            u = predicted_utility(pr["p_success"], pr["pred_error"], pr["pred_time"], cfg)
            if u > best_u:
                best_u, best_idx = u, e.get(fm.candidate_index)
        chosen[gid] = best_idx
    return chosen


def _regret(groups, chosen, cfg, fm):
    from deployment_calibration.offline_v2.oracle import oracle_candidate_per_group
    oc = oracle_candidate_per_group(groups, cfg)
    rows = []
    for gid, eps in groups.items():
        pick = next((e for e in eps if e.get(fm.candidate_index) == chosen[gid]), eps[0])
        u = true_utility(pick, cfg)
        rows.append({"regret": oc[gid]["best_true_utility"] - u,
                     "success": 1.0 if pick["y"]["success"] else 0.0,
                     "top1": 1.0 if u >= oc[gid]["best_true_utility"] - 1e-9 else 0.0})
    return {"mean_regret": float(np.mean([r["regret"] for r in rows])),
            "selected_success_rate": float(np.mean([r["success"] for r in rows])),
            "top1_acc": float(np.mean([r["top1"] for r in rows]))}


def voi_analysis(episodes, cfg, fm, seeds=(0, 1, 2, 3, 4)) -> dict:
    """Descriptive VOI ceiling: DeepSets at K probes vs best-single-offset baseline (frozen + success)."""
    cands = [e for e in episodes if e.get(fm.episode_role) == ROLE_CANDIDATE]
    all_sess = {e.get(fm.session_id) for e in cands}
    groups = CBO.selection_groups(episodes, fm)
    base = CBO.best_single_offset(groups, groups, cfg, fm)
    base_s = CBO.best_single_offset(groups, groups, SUCCESS_ONLY, fm)
    ptime = CBV.probe_time_by_k(episodes, fm, kmax=3)
    sel_u, sel_us = {}, {}
    for k in (0, 1, 2, 3):
        tek = build_pairs(episodes, all_sess, k=k)
        tg = groups_from_pairs(tek)
        hist_by_ep = {id(e): H for e, H in tek}
        fit = train_seeds("B2_deepsets", build_pairs(episodes, all_sess, k=k), None, seeds=seeds)
        ch = _choose(fit, tg, hist_by_ep, cfg, fm)
        sel_u[k] = float(np.mean([true_utility(next((e for e in eps if e.get(fm.candidate_index) == ch[gid]), eps[0]), cfg)
                                  for gid, eps in tg.items()]))
        sel_us[k] = float(np.mean([true_utility(next((e for e in eps if e.get(fm.candidate_index) == ch[gid]), eps[0]), SUCCESS_ONLY)
                                   for gid, eps in tg.items()]))
    return {"frozen": CBV.voi_curve(sel_u, base["mean_true_utility"], ptime, cfg),
            "success_only": CBV.voi_curve(sel_us, base_s["mean_true_utility"], ptime, SUCCESS_ONLY),
            "probe_time_by_k": ptime, "exploratory": True}


def exploration_gate(dv, ind, validation, n_distinct_best) -> dict:
    c1 = n_distinct_best >= GATE["min_bias_levels_with_distinct_best_offset"]
    c2 = not dv["robust_offset"].get("robust_generalist_exists", True)
    # criterion 3 is a CONJUNCTION: the success-only selected-success gain (physical decision value,
    # NECESSARY) AND the frozen-utility VSI (combined utility, NECESSARY). AND — not OR — so a design
    # cannot enter the confirmatory stage on the time/error term alone.
    succ_gain = (dv.get("success_only") or {}).get("state_aware_selected_success_gain") or 0.0
    c3a = succ_gain >= GATE["min_success_gain"]                      # success-only necessary condition
    c3b = (dv["VSI_frozen"] or 0) >= GATE["min_frozen_vsi"]          # frozen-utility necessary condition
    c3 = c3a and c3b
    c4 = not ind["blocker"]
    c5 = validation["ok"]
    passed = c1 and c2 and c3 and c4 and c5
    return {"gate_passed": bool(passed), "thresholds": GATE,
            "criteria": {
                "1_distinct_best_offsets>=3": {"ok": bool(c1), "value": n_distinct_best},
                "2_no_robust_generalist": {"ok": bool(c2),
                                           "worst_gap": dv["robust_offset"].get("worst_case_gap_of_most_robust")},
                "3_success_gain>=0.15_AND_vsi>=0.05": {
                    "ok": bool(c3),
                    "success_only_gain_ok(>=0.15)": bool(c3a), "success_gain": succ_gain,
                    "frozen_vsi_ok(>=0.05)": bool(c3b), "vsi_frozen": dv["VSI_frozen"],
                    "note": "AND: success-only gain is necessary for physical decision value; frozen "
                            "VSI is necessary for combined utility. Net VOI is NOT a capability-map "
                            "gate here but MUST be positive in the final confirmatory GO."},
                "4_replicate_independence": {"ok": bool(c4), "blocker": ind["blocker"]},
                "5_pairing_leakage_provenance": {"ok": bool(c5)},
            }}


def run(episodes, out_dir, *, fm: FieldMap = DEFAULT_FIELDS, seeds=(0, 1, 2, 3, 4),
        expect_bias_levels=None, provenance=None, do_models=True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    synthetic = any(e.get("synthetic_only") for e in episodes)

    validation = validate(episodes, fm, expect_bias_levels=expect_bias_levels)
    ind = IND.audit(episodes, fm)
    dv = CBO.decision_value(episodes, FROZEN, fm, success_only=SUCCESS_ONLY)
    n_distinct_best, best_per_bias = _distinct_best_offsets_across_bias(episodes, FROZEN, fm)
    dv["n_distinct_best_offsets_across_bias"] = n_distinct_best
    dv["marginal_best_offset_per_bias"] = {k: str(v) for k, v in best_per_bias.items()}

    lv = CBS.level_values(episodes, fm)
    manifest = None
    heldout = voi = None
    if len(lv) >= 5:
        split = CBS.split_bias_levels(lv)
        manifest = CBS.build_manifest(episodes, split, fm)
        if do_models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heldout = held_out_selection(episodes, manifest, FROZEN, fm, seeds=seeds)
                voi = voi_analysis(episodes, FROZEN, fm, seeds=seeds)

    gate = exploration_gate(dv, ind, validation, n_distinct_best)

    def dump(name, obj):
        (out / name).write_text(json.dumps(_tag(obj, synthetic), indent=2, default=str))
    dump("provenance.json", provenance or {"synthetic_only": synthetic})
    dump("validation_report.json", validation)
    dump("independence_audit.json", ind)
    dump("decision_value.json", dv)
    if manifest:
        dump("split_manifest.json", manifest)
    if heldout:
        dump("held_out_selection.json", heldout)
    if voi:
        dump("voi.json", voi)
    dump("exploration_gate.json", gate)

    return {"synthetic_only": synthetic, "validation_ok": validation["ok"],
            "blocker": ind["blocker"], "VSI_frozen": dv["VSI_frozen"],
            "success_only_VSI": (dv.get("success_only") or {}).get("VSI_success_only"),
            "switch_rate": dv["optimal_candidate_switch_rate"],
            "reversal_rate": dv["pairwise_rank_reversal_rate"],
            "n_distinct_best_offsets": n_distinct_best,
            "robust_generalist_exists": dv["robust_offset"].get("robust_generalist_exists"),
            "gate_passed": gate["gate_passed"], "out_dir": str(out)}


def _tag(obj, synthetic):
    if isinstance(obj, dict):
        return {"synthetic_only": synthetic, **obj} if synthetic and "synthetic_only" not in obj else obj
    return obj
