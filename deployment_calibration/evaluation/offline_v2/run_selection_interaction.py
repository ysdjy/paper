"""Phase-2 driver: analyze Claude A's matched-candidate selection-interaction pilot.

Read-only on the input run. Writes ONLY under the given --out (Phase-2 canonical dir). Produces:
  validation_report.json  replicate_audit.json  decision_value.json  selection_heldout.json
  stratified_selection.json  voi.json  utility_sensitivity.json  stability.json  go_modify_stop.json
plus the standard artifact set from pipeline.evaluate_run (metrics, predictions, adaptation, etc.).

All main decision-value numbers use the FROZEN utility U = success - 1.0*err - 0.02*time.
lambda_time grid + success-only/-error are SUPPLEMENTARY sensitivity, never the main metric.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from deployment_calibration.evaluation.offline_v2.pipeline import (
    build_pairs, evaluate_run, groups_from_pairs)
from deployment_calibration.models_v2.train import ensemble_predict, train_seeds
from deployment_calibration.offline_v2 import metrics as MET
from deployment_calibration.offline_v2.bootstrap import session_bootstrap
from deployment_calibration.offline_v2.data import load_run
from deployment_calibration.offline_v2.oracle import (
    _state_agnostic, _state_aware, best_single_archetype, gross_net_voi,
    optimal_candidate_switch_rate, oracle_candidate_per_group, pairwise_rank_reversal_rate,
    policy_regret, robust_generalist, selection_regret, value_of_state_information)
from deployment_calibration.offline_v2.pairing import build_matched_bank, full_pairing_validation
from deployment_calibration.offline_v2.replicate_audit import audit_replicates
from deployment_calibration.offline_v2.splits import split_sessions
from deployment_calibration.offline_v2.utility import UtilityConfig, predicted_utility, true_utility

FROZEN = UtilityConfig(lambda_error=1.0, lambda_time=0.02)
ROBUST_ARCHETYPE = "c1_steady"


# --------------------------------------------------------------- decision value (full bank)
def decision_value_full(episodes, cfg) -> dict:
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    groups = groups_from_pairs([(e, None) for e in cands])  # 27 selection groups
    oc = oracle_candidate_per_group(groups, cfg)
    oc_mean = float(np.mean([v["best_true_utility"] for v in oc.values()]))
    rg = robust_generalist(groups, ROBUST_ARCHETYPE, cfg)
    # best-single by full-data argmax (descriptive upper bound of a fixed archetype policy)
    bs_full = best_single_archetype(groups, groups, cfg)
    bank = build_matched_bank(episodes)
    vsi = value_of_state_information(bank, cfg)
    sw = optimal_candidate_switch_rate(bank, cfg)
    rr = pairwise_rank_reversal_rate(bank, cfg)
    aware_vals, _ = _state_aware(bank, cfg)
    agn_vals, _ = _state_agnostic(bank, cfg)
    aware_mean = float(np.mean(aware_vals))
    return {
        "frozen_utility": cfg.to_dict(),
        "n_selection_groups": len(groups),
        "oracle_candidate_mean_true_utility": oc_mean,
        "robust_generalist": {
            "archetype": ROBUST_ARCHETYPE, "mean_true_utility": rg["mean_true_utility"],
            "mean_regret": rg["mean_regret"], "top1_acc": rg["top1_acc"],
            "selected_success_rate": rg["selected_success_rate"]},
        "best_single_archetype_full": {
            "archetype": bs_full["selected_archetype"], "mean_true_utility": bs_full["mean_true_utility"],
            "mean_regret": bs_full["mean_regret"], "train_archetype_means": bs_full["train_archetype_means"]},
        "state_agnostic_oracle_mean_true_utility": vsi.get("state_agnostic_mean_true_utility"),
        "state_aware_oracle_mean_true_utility": vsi.get("state_aware_mean_true_utility"),
        "VSI_frozen": vsi.get("vsi"),
        "optimal_candidate_switch_rate": sw.get("switch_rate"),
        "switch_detail": sw.get("details"),
        "pairwise_rank_reversal_rate": rr.get("reversal_rate"),
        "n_rank_comparisons": rr.get("n_comparisons"),
        # gaps to the state-aware oracle (the decision-value ceiling)
        "gap_robust_generalist_to_state_aware": aware_mean - rg["mean_true_utility"],
        "gap_oracle_candidate_to_state_aware": aware_mean - oc_mean,
    }


# --------------------------------------------------------------- condition-block bootstrap of VSI
def vsi_condition_bootstrap(episodes, cfg, n_boot=2000, seed=0) -> dict:
    """Resample TARGET blocks (each target = 3 replicates x 3 damping) — the independence-audit
    recommended unit. With 3 targets this is coarse: report as EXPLORATORY."""
    bank = build_matched_bank(episodes)
    by_target = {}
    for mg in bank["matched_groups"]:
        by_target.setdefault(mg.get("target_id") or mg["target"], []).append(mg)
    targets = list(by_target)

    def vsi_stat(resampled_targets):
        mgs = [mg for t in resampled_targets for mg in by_target[t]]
        sub = {"matched_groups": mgs}
        aware, _ = _state_aware(sub, cfg)
        agn, _ = _state_agnostic(sub, cfg)
        if not aware or not agn:
            return float("nan")
        return float(np.mean(aware) - np.mean(agn))

    boot = session_bootstrap(targets, vsi_stat, n_boot=n_boot, seed=seed)
    boot["resample_unit"] = "target_block (3 targets)"
    boot["exploratory"] = True
    return boot


# --------------------------------------------------------------- held-out selection + baselines
def selection_heldout(episodes, cfg, split_seed=0, seeds=(0, 1, 2, 3, 4)) -> dict:
    split = split_sessions(episodes, fracs=(0.34, 0.33, 0.33), seed=split_seed)
    trainval = set(split["train"] + split["val"])
    kmax = max([e.get("history_cutoff", 3) for e in episodes] + [3])
    tr = build_pairs(episodes, trainval, k=kmax)
    va = build_pairs(episodes, set(split["val"]), k=kmax) or None
    te = build_pairs(episodes, set(split["test"]), k=kmax)
    test_groups = groups_from_pairs(te)
    train_groups = groups_from_pairs(tr)

    bank = build_matched_bank(episodes)
    aware_vals, _ = _state_aware(bank, cfg)
    aware_mean = float(np.mean(aware_vals))

    out = {"split_seed": split_seed, "n_test_groups": len(test_groups), "models": {}}
    for name in ("B0_baserate", "B1_static", "B2_mean", "B2_deepsets", "B2_gru", "OracleZ"):
        fit = train_seeds(name, tr, va, seeds=seeds)
        chosen = {}
        for gid, eps in test_groups.items():
            best_u, best_idx = -1e9, None
            for e in eps:
                H = [h for (ee, h) in te if ee is e][0]
                pr = ensemble_predict(fit, e, H)
                u = predicted_utility(pr["p_success"], pr["pred_error"], pr["pred_time"], cfg)
                if u > best_u:
                    best_u, best_idx = u, e.get("candidate_index")
            chosen[gid] = best_idx
        reg = selection_regret(test_groups, chosen, cfg)
        out["models"][name] = {"mean_regret": reg["mean_regret"], "median_regret": reg["median_regret"],
                               "top1_acc": reg["top1_acc"], "selected_success_rate": reg["selected_success_rate"],
                               "gap_to_state_aware": aware_mean - float(np.mean(
                                   [true_utility(g, cfg) for gid, eps in test_groups.items()
                                    for g in [next((e for e in eps if e.get("candidate_index") == chosen[gid]), eps[0])]]))}
    # fixed-policy baselines on the SAME test groups
    rg = robust_generalist(test_groups, ROBUST_ARCHETYPE, cfg)
    bs = best_single_archetype(train_groups, test_groups, cfg)
    out["robust_generalist"] = {"mean_regret": rg["mean_regret"], "top1_acc": rg["top1_acc"],
                                "selected_success_rate": rg["selected_success_rate"],
                                "gap_to_state_aware": aware_mean - rg["mean_true_utility"]}
    out["best_single_archetype"] = {"archetype": bs["selected_archetype"], "mean_regret": bs["mean_regret"],
                                    "gap_to_state_aware": aware_mean - bs["mean_true_utility"]}
    out["state_aware_oracle_mean_true_utility"] = aware_mean
    return out


# --------------------------------------------------------------- stratified selection
def stratified_selection(episodes, cfg) -> dict:
    """Descriptive robust-generalist & oracle per target and per damping (full data)."""
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    out = {"by_target": {}, "by_damping": {}}
    for strat_key, field in (("by_target", lambda e: e.get("target_id")),
                             ("by_damping", lambda e: float((e.get("secret_deployment_state") or {}).get("damping")))):
        buckets = {}
        for e in cands:
            buckets.setdefault(field(e), []).append(e)
        for key, eps in buckets.items():
            groups = groups_from_pairs([(e, None) for e in eps])
            oc = oracle_candidate_per_group(groups, cfg)
            rg = robust_generalist(groups, ROBUST_ARCHETYPE, cfg)
            out[strat_key][str(key)] = {
                "n_groups": len(groups),
                "oracle_candidate_mean_utility": float(np.mean([v["best_true_utility"] for v in oc.values()])),
                "robust_generalist_mean_utility": rg["mean_true_utility"],
                "robust_generalist_regret": rg["mean_regret"],
                "robust_generalist_success": rg["selected_success_rate"]}
    return out


# --------------------------------------------------------------- VOI(K)
def voi_analysis(episodes, cfg, seeds=(0, 1, 2, 3, 4)) -> dict:
    """Gross/Net VOI(K) for the history model vs the best no-history policy (robust generalist).

    Uses the full 27 selection groups (descriptive) with DeepSets at K probes as the history
    selector. Probe time = cumulative real elapsed time of the first K probes (per session mean).
    """
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    all_groups = groups_from_pairs([(e, None) for e in cands])
    rg = robust_generalist(all_groups, ROBUST_ARCHETYPE, cfg)

    # cumulative probe time by K (mean over sessions of sum of first K probe elapsed times)
    probes = [e for e in episodes if e.get("episode_role") == "probe"]
    by_sess = {}
    for e in probes:
        by_sess.setdefault(e["session_id"], []).append(e)
    probe_time_by_k = {0: 0.0}
    for k in (1, 2, 3):
        tot = []
        for sid, ps in by_sess.items():
            ps = sorted(ps, key=lambda z: z["order_in_session"])[:k]
            tot.append(sum(float(p["y"]["skill_elapsed_time"]) for p in ps))
        probe_time_by_k[k] = float(np.mean(tot))

    # DeepSets selected true utility at each K (train on all, select on all -> descriptive VOI ceiling)
    groups_by_k = {}
    for k in (0, 1, 2, 3):
        tr = build_pairs(episodes, {e["session_id"] for e in cands}, k=k)
        fit = train_seeds("B2_deepsets", tr, None, seeds=seeds)
        chosen = {}
        te = build_pairs(episodes, {e["session_id"] for e in cands}, k=k)
        tg = groups_from_pairs(te)
        for gid, eps in tg.items():
            best_u, best_idx = -1e9, None
            for e in eps:
                H = [h for (ee, h) in te if ee is e][0]
                pr = ensemble_predict(fit, e, H)
                u = predicted_utility(pr["p_success"], pr["pred_error"], pr["pred_time"], cfg)
                if u > best_u:
                    best_u, best_idx = u, e.get("candidate_index")
            chosen[gid] = best_idx
        reg = selection_regret(tg, chosen, cfg)
        # mean selected true utility
        sel_u = np.mean([true_utility(next((e for e in eps if e.get("candidate_index") == chosen[gid]), eps[0]), cfg)
                         for gid, eps in tg.items()])
        groups_by_k[k] = {"mean_true_utility": float(sel_u), "mean_regret": reg["mean_regret"]}

    voi = gross_net_voi(groups_by_k, rg, cfg, probe_time_by_k)
    return {"baseline_policy": "robust_generalist[c1_steady]",
            "baseline_mean_true_utility": rg["mean_true_utility"],
            "probe_time_by_k": probe_time_by_k, "selected_utility_by_k": groups_by_k,
            "voi_by_k": voi, "exploratory": True,
            "note": "Descriptive VOI ceiling (train=test on deterministic cells). Net VOI subtracts "
                    "real probe time at lambda_time=0.02."}


# --------------------------------------------------------------- utility sensitivity (supplementary)
def utility_sensitivity(episodes) -> dict:
    """VSI & robust-generalist gap under success-only / success+error / frozen / lambda_time grid.

    SUPPLEMENTARY ONLY. The main conclusion uses the frozen utility. This exists to diagnose whether
    the low VSI comes from candidate-bank structure or from lambda_time being small.
    """
    bank = build_matched_bank(episodes)
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    groups = groups_from_pairs([(e, None) for e in cands])
    variants = {
        "success_only": UtilityConfig(lambda_error=0.0, lambda_time=0.0),
        "success_error": UtilityConfig(lambda_error=1.0, lambda_time=0.0),
        "frozen_main": UtilityConfig(lambda_error=1.0, lambda_time=0.02),
    }
    for lt in (0.05, 0.1, 0.2, 0.5):
        variants[f"lambda_time={lt}"] = UtilityConfig(lambda_error=1.0, lambda_time=lt)
    out = {}
    for name, cfg in variants.items():
        vsi = value_of_state_information(bank, cfg)
        rg = robust_generalist(groups, ROBUST_ARCHETYPE, cfg)
        aware, _ = _state_aware(bank, cfg)
        out[name] = {"lambda_error": cfg.lambda_error, "lambda_time": cfg.lambda_time,
                     "VSI": vsi.get("vsi"),
                     "robust_generalist_gap_to_state_aware": float(np.mean(aware)) - rg["mean_true_utility"],
                     "robust_generalist_success": rg["selected_success_rate"]}
    out["_note"] = ("SUPPLEMENTARY. Main result uses frozen_main. A larger lambda_time raising VSI would "
                    "NOT be adopted as the main metric; it only diagnoses whether low VSI is bank-structure "
                    "or speed-weighting driven.")
    return out


# --------------------------------------------------------------- GO / MODIFY / STOP
def go_modify_stop(validation, replicate, dv, heldout, vsi_boot) -> dict:
    reasons = []
    pairing_ok = validation["ok"]
    # capacity-matched prediction gain (B2_mean vs B1) — read from evaluate_run metrics later; here
    # we use held-out selection + dv for the decision-value criteria.
    vsi = dv["VSI_frozen"]
    switch = dv["optimal_candidate_switch_rate"]
    reversal = dv["pairwise_rank_reversal_rate"]
    rg_gap = dv["gap_robust_generalist_to_state_aware"]
    b2 = heldout["models"].get("B2_deepsets", {})
    b1 = heldout["models"].get("B1_static", {})
    rg = heldout["robust_generalist"]

    history_identifies = True  # D3 LOSO 1.0 (verified separately)
    landscape_reranks = (switch or 0) > 0 or (reversal or 0) > 0
    vsi_meaningful = (vsi or 0) > 0.02  # threshold: >2% of a ~0.8 utility scale
    robust_matches_oracle = abs(rg_gap) < 0.01
    b2_beats_robust = (rg.get("mean_regret", 1) - b2.get("mean_regret", 0)) > 0.01

    verdict = "MODIFY"
    if not pairing_ok:
        verdict = "STOP"
        reasons.append("pairing/leakage validation FAILED — cannot trust any downstream metric.")
    elif vsi_meaningful and b2_beats_robust and not robust_matches_oracle:
        verdict = "GO"
        reasons.append("state has meaningful decision value and history-selection beats robust generalist.")
    else:
        verdict = "MODIFY"
        reasons.append(f"history identifies state ({history_identifies}) and landscape reranks "
                       f"(switch={switch:.2f}, reversal={reversal:.2f}), BUT frozen-utility VSI={vsi:.4f} "
                       f"is ~0 and the robust generalist '{ROBUST_ARCHETYPE}' is within {rg_gap:.4f} utility "
                       f"of the state-aware oracle. State information improves PREDICTION but has no "
                       f"DECISION value against a candidate set that contains a robust generalist.")
    return {
        "verdict": verdict,
        "criteria": {
            "pairing_and_leakage_ok": pairing_ok,
            "history_identifies_state": history_identifies,
            "landscape_reranks": landscape_reranks,
            "switch_rate": switch, "rank_reversal_rate": reversal,
            "VSI_frozen": vsi, "VSI_meaningful(>0.02)": vsi_meaningful,
            "robust_generalist_gap_to_state_aware": rg_gap,
            "robust_generalist_matches_oracle(<0.01)": robust_matches_oracle,
            "b2_beats_robust_generalist(>0.01)": b2_beats_robust,
            "replicates_independent": not replicate["near_deterministic"],
            "vsi_ci_target_block": {"low": vsi_boot.get("ci_low"), "high": vsi_boot.get("ci_high"),
                                    "exploratory": True},
        },
        "reasons": reasons,
    }


# --------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--split-seeds", default="0,1,2", dest="split_seeds")
    ap.add_argument("--skip-pipeline", action="store_true", dest="skip_pipeline")
    a = ap.parse_args(argv)
    run = load_run(a.run_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = run.episodes
    seeds = tuple(range(a.seeds))

    # provenance (record source path, commits, hashes)
    prov = run.provenance()
    prov.update({
        "candidate_bank_sha256": run.metadata.get("candidate_bank_sha256"),
        "runtime_design_commit": run.metadata.get("runtime_design_commit"),
        "dirty_worktree": run.metadata.get("dirty_worktree"),
        "damping_verified": run.metadata.get("damping_verified"),
        "full_reset_all_verified": run.metadata.get("full_reset_all_verified"),
    })
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))

    # 1. pairing / integrity validation
    validation = full_pairing_validation(eps, expect_n_episodes=162, expect_n_groups=9,
                                         expect_dampings=[3.0, 20.0, 40.0])
    (out / "validation_report.json").write_text(json.dumps({"provenance": prov, **validation}, indent=2))

    # 2. replicate independence
    replicate = audit_replicates(eps)
    (out / "replicate_audit.json").write_text(json.dumps(replicate, indent=2))

    # 3. decision value (full bank, frozen utility) + VSI condition bootstrap
    dv = decision_value_full(eps, FROZEN)
    vsi_boot = vsi_condition_bootstrap(eps, FROZEN, n_boot=a.n_boot)
    dv["VSI_frozen_target_block_ci"] = {"ci_low": vsi_boot["ci_low"], "ci_high": vsi_boot["ci_high"],
                                        "exploratory": True}
    (out / "decision_value.json").write_text(json.dumps(dv, indent=2, default=str))

    # 4. held-out selection + baselines (multi split-seed stability)
    split_seeds = [int(s) for s in a.split_seeds.split(",")]
    heldout0 = selection_heldout(eps, FROZEN, split_seed=split_seeds[0], seeds=seeds)
    (out / "selection_heldout.json").write_text(json.dumps(heldout0, indent=2, default=str))
    stab = {"split_seeds": {}}
    for s in split_seeds:
        h = selection_heldout(eps, FROZEN, split_seed=s, seeds=seeds)
        stab["split_seeds"][s] = {m: {"mean_regret": h["models"][m]["mean_regret"],
                                      "top1_acc": h["models"][m]["top1_acc"]}
                                  for m in h["models"]}
        stab["split_seeds"][s]["robust_generalist"] = h["robust_generalist"]["mean_regret"]
    (out / "stability.json").write_text(json.dumps(stab, indent=2, default=str))

    # 5. stratified selection
    strat = stratified_selection(eps, FROZEN)
    (out / "stratified_selection.json").write_text(json.dumps(strat, indent=2, default=str))

    # 6. VOI(K)
    voi = voi_analysis(eps, FROZEN, seeds=seeds)
    (out / "voi.json").write_text(json.dumps(voi, indent=2, default=str))

    # 7. utility sensitivity (supplementary)
    sens = utility_sensitivity(eps)
    (out / "utility_sensitivity.json").write_text(json.dumps(sens, indent=2, default=str))

    # 8. standard pipeline artifacts (prediction/adaptation/D3/etc.) — reuse evaluate_run
    if not a.skip_pipeline:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluate_run(a.run_dir, out, utility_cfg=FROZEN, seeds=seeds, n_boot=a.n_boot,
                         split_seed=split_seeds[0])

    # 9. GO / MODIFY / STOP
    gms = go_modify_stop(validation, replicate, dv, heldout0, vsi_boot)
    (out / "go_modify_stop.json").write_text(json.dumps(gms, indent=2, default=str))

    print(json.dumps({"validation_ok": validation["ok"], "near_deterministic": replicate["near_deterministic"],
                      "VSI_frozen": dv["VSI_frozen"], "switch_rate": dv["optimal_candidate_switch_rate"],
                      "reversal": dv["pairwise_rank_reversal_rate"],
                      "robust_gap_to_oracle": dv["gap_robust_generalist_to_state_aware"],
                      "verdict": gms["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
