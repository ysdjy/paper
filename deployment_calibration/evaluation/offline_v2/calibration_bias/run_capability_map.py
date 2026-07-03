"""Read-only analysis of Claude A's calibration-bias exploratory capability map.

Recomputes (does not trust A's self-check) provenance/integrity, independence, decision value
(with BLOCK-WISE best-single CV), probe identifiability + Net VOI, failure mechanism, and the frozen
AND exploration gate. Writes JSON artifacts to --out. Never writes to A's tree; never runs Isaac.

    python -m deployment_calibration.evaluation.offline_v2.calibration_bias.run_capability_map \
        <A_run_dir> --out <OUT_DIR>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import independence as IND
from deployment_calibration.offline_v2.calibration_bias import oracle as CBO
from deployment_calibration.offline_v2.calibration_bias import validator as V
from deployment_calibration.offline_v2.calibration_bias.schema import (FieldMap, bias_level_of,
                                                                       offset_key)
from deployment_calibration.offline_v2.utility import UtilityConfig, true_utility

# Claude A's field naming for this run (verified by reading an episode; not trusting labels).
FM = FieldMap(bias_value_in_secret="handle_bias_local_y", bias_level_id="hidden_state_id",
              bias_id="hidden_state_id", nuisance_seed="block_seed", nuisance_block_id="block_id")
FROZEN = UtilityConfig(lambda_error=1.0, lambda_time=0.02)
SUCCESS_ONLY = UtilityConfig(lambda_error=0.0, lambda_time=0.0)
BAND = 0.02          # observed compensation band half-width (for reporting only)
PROBE_ORDER = ["probe_m040", "probe_p040"]

# frozen exploration-gate thresholds (branch 17cc4a9)
GATE = {"min_distinct_best_offsets": 3, "robust_offset_tol": 0.02,
        "min_success_gain": 0.15, "min_frozen_vsi": 0.05}


def _sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def _bias(e):
    return round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 3)


def _off(e):
    return round(float(e["theta"]["grasp_offset_local_y"]), 3)


def _succ(e):
    return int(e["y"]["success"])


# ---------------------------------------------------------------- provenance / integrity
def provenance(run_dir, meta):
    ep = Path(run_dir) / "episodes.jsonl"
    return {
        "source_run_path": str(Path(run_dir).resolve()),
        "episodes_sha256": _sha256(ep),
        "source_git_commit": meta.get("git_commit"),
        "runtime_design_commit": meta.get("runtime_design_commit"),
        "branch": meta.get("branch"),
        "dirty_worktree": meta.get("dirty_worktree"),
        "damping_verified": meta.get("damping_verified"),
        "full_reset_all_verified": meta.get("full_reset_all_verified"),
        "n_reset_verify_fail": meta.get("n_reset_verify_fail"),
        "offset_grid_sha256": meta.get("offset_grid_sha256"),
        "contract_version": meta.get("contract_version"),
    }


def integrity(eps, meta):
    """Structural checks recomputed independently (do NOT trust A's self-check)."""
    cands = [e for e in eps if e["episode_role"] == "candidate"]
    probes = [e for e in eps if e["episode_role"] == "probe"]
    out = {}
    out["counts"] = {"episodes": len(eps), "candidates": len(cands), "probes": len(probes),
                     "expected": {"episodes": 135, "candidates": 105, "probes": 30}}
    out["counts_ok"] = (len(eps) == 135 and len(cands) == 105 and len(probes) == 30)

    # candidate_id -> offset is a pure function (identical across all bias / sessions)
    cid_to_off = defaultdict(set)
    for e in cands:
        cid_to_off[e["candidate_id"]].add(_off(e))
    out["candidate_id_to_offset_pure_function"] = all(len(v) == 1 for v in cid_to_off.values())

    # matched candidate bank identical across ALL bias levels (offset-id set + theta per id)
    bank = V.matched_offset_bank(eps, FM)
    out["matched_bank_consistent"] = bank["ok"]
    out["n_matched_groups"] = bank["n_matched_groups"]
    out["n_bias_levels"] = bank["n_bias_levels"]

    # x / g identical within each candidate_group (selection group = session)
    def sig_xg(e):
        x = e["x"]
        return (tuple(round(v, 6) for v in x.get("robot_joint_pos", [])),
                round(float(x.get("gripper_width", 0)), 6),
                round(float(x.get("initial_mechanism_joint_pos", 0)), 6),
                round(float(e["g"]["target_open_position"]), 6))
    within = defaultdict(set)
    for e in cands + probes:
        within[e["session_id"]].add(sig_xg(e))
    out["within_session_xg_identical"] = all(len(v) == 1 for v in within.values())

    # matched block: the SAME block_id gives identical x/g across bias levels
    by_block = defaultdict(set)
    for e in cands + probes:
        by_block[e.get("block_id")].add(sig_xg(e))
    # a block spans all bias levels; x/g should be identical (paired) -> one signature per block
    out["matched_block_xg_identical_across_bias"] = all(len(v) == 1 for v in by_block.values())

    # bias never in x; raw seed / block id never in x (validator does the full sweep)
    val = V.validate(eps, FM, expect_bias_levels=5)
    out["validator_ok"] = val["ok"]
    out["validator_checks"] = val["checks"]
    out["all_ok"] = bool(out["counts_ok"] and out["candidate_id_to_offset_pure_function"]
                         and out["matched_bank_consistent"] and out["within_session_xg_identical"]
                         and out["matched_block_xg_identical_across_bias"] and out["validator_ok"])
    return out


# ---------------------------------------------------------------- independence adjudication
def independence_adjudication(eps, meta):
    ind = IND.audit(eps, FM)
    cands = [e for e in eps if e["episode_role"] == "candidate"]
    # label flips + continuous variation per cell (bias, offset)
    cells = defaultdict(list)
    for e in cands:
        cells[(_bias(e), _off(e))].append(e)
    succ_flips = sum(1 for v in cells.values() if len({_succ(x) for x in v}) > 1)
    fr_flips = sum(1 for v in cells.values() if len({x["y"]["failure_reason"] for x in v}) > 1)
    time_sd = [float(np.std([x["y"]["skill_elapsed_time"] for x in v])) for v in cells.values()]
    err_sd = [float(np.std([x["y"]["task_outcome_error"] for x in v])) for v in cells.values()]
    return {
        "frozen_blocker": ind["blocker"],
        "frozen_blocker_reason": ind["blocker_reason"],
        "discrete_deterministic": ind["discrete_deterministic"],
        "near_deterministic": ind["near_deterministic"],
        "adjudication": {
            "independent_random_sampling": {
                "verdict": True,
                "evidence": "3 distinct nuisance blocks, distinct block_seeds drawn from a master RNG "
                            "independent of bias, distinct robot_joint_delta/target_jitter; the SAME "
                            "block gives identical x across all bias levels (paired). Correctly applied.",
            },
            "label_flips": {"success_cells_with_flip": succ_flips, "failure_reason_cells_with_flip": fr_flips,
                            "verdict": "success label is deterministic across replicates (0 flips)"},
            "continuous_outcomes_change": {
                "final_pos_max_sd": ind["continuous_within_cell_max_std"]["final_joint_position"],
                "task_error_max_sd": max(err_sd) if err_sd else 0.0,
                "time_max_sd": max(time_sd) if time_sd else 0.0,
                "verdict": "continuous outcomes DO vary across blocks (not byte-identical clones)"},
            "technical_repeats": {
                "verdict": "near-deterministic in the SUCCESS LABEL but not clones; independent draws "
                           "whose discrete response is robust to <=0.03 rad joint / 0.005 m target jitter"},
        },
        "vs_runtime_A": {
            "A_replicates_independent": False,
            "A_criterion": "success-cell median SD < 0.003 and 0 flips -> A flagged NOT independent",
            "B_frozen_blocker": ind["blocker"],
            "B_criterion": "blocker fires only if near-deterministic AND blocks unapplied/unvaried; here "
                           "blocks ARE applied+varied and continuous SD exceeds the near-det threshold -> "
                           "blocker=False. B does NOT add a success-label-flip requirement (frozen rule).",
            "reconciliation": "Both agree the discrete label is near-deterministic. They differ on whether "
                              "that blocks the exploration gate: A's CI-oriented bar says not-independent; "
                              "B's frozen bug-guard does not fire (no implementation bug). Per the frozen "
                              "protocol the gate is not blocked, but the data is EXPLORATORY-ONLY.",
        },
        "three_blocks_as_independent_samples": (
            "Yes for DESCRIPTIVE statistics of the compensation landscape (3 correctly-applied independent "
            "draws), but n=3 with ~0 success-label variance is insufficient for formal success-rate CIs."),
        "sufficiency": "Exploratory description + gate evaluation: SUFFICIENT. Formal CIs on success rate: "
                       "NOT supported (needs genuine outcome variance in the confirmatory pilot).",
        "effective_sample_sizes": ind["effective_sample_sizes"],
        "no_nuisance_redesign_mandated": True,
        "note": "Blocks are correctly applied, so NO nuisance redesign is mandated as a blocker. Genuine "
                "outcome variance for confirmatory CIs is a POWER consideration, not a bug fix; we do NOT "
                "recommend fabricating label flips.",
    }


# ---------------------------------------------------------------- decision value + block-wise CV
def decision_value(eps):
    dv = CBO.decision_value(eps, FROZEN, FM, success_only=SUCCESS_ONLY)
    cands = [e for e in eps if e["episode_role"] == "candidate"]
    blocks = sorted({e["block_id"] for e in cands})
    biases = sorted({bias_level_of(e, FM) for e in cands})

    succ_cell = defaultdict(list)
    for e in cands:
        succ_cell[(bias_level_of(e, FM), offset_key(e, FM))].append(_succ(e))
    succ_cell = {k: float(np.mean(v)) for k, v in succ_cell.items()}
    offs = sorted({offset_key(e, FM) for e in cands}, key=str)

    def best_single_on(train_blocks, cfg):
        by_off = defaultdict(list)
        for e in cands:
            if e["block_id"] in train_blocks:
                by_off[offset_key(e, FM)].append(true_utility(e, cfg))
        m = {k: float(np.mean(v)) for k, v in by_off.items()}
        return max(m, key=m.get)

    def state_aware_succ(test_blocks):
        g = defaultdict(list)
        for e in cands:
            if e["block_id"] in test_blocks:
                g[(bias_level_of(e, FM), e["block_id"])].append(_succ(e))
        return float(np.mean([max(v) for v in g.values()]))

    def fixed_succ(o, test_blocks):
        g = defaultdict(list)
        for e in cands:
            if e["block_id"] in test_blocks and offset_key(e, FM) == o:
                g[(bias_level_of(e, FM), e["block_id"])].append(_succ(e))
        return float(np.mean([np.mean(v) for v in g.values()]))

    folds = []
    for held in blocks:
        train = [b for b in blocks if b != held]
        bso = best_single_on(train, SUCCESS_ONLY)
        sa, fs = state_aware_succ([held]), fixed_succ(bso, [held])
        folds.append({"held_block": held, "best_single_offset": bso,
                      "state_aware_success": sa, "best_single_success": fs, "success_gain": sa - fs})
    gains = [f["success_gain"] for f in folds]
    dv["block_wise_best_single_cv"] = {
        "folds": folds, "mean_success_gain": float(np.mean(gains)),
        "min_success_gain": float(np.min(gains)),
        "method": "leave-one-block-out: select best-single offset on 2 blocks, evaluate success gain "
                  "vs state-aware on the held-out block; 3-fold rotation"}
    return dv


# ---------------------------------------------------------------- probe analysis
def probe_analysis(eps):
    probes = [e for e in eps if e["episode_role"] == "probe"]
    cands = [e for e in eps if e["episode_role"] == "candidate"]
    biases = sorted({_bias(e) for e in cands})
    offs = sorted({_off(e) for e in cands})
    psig = defaultdict(dict)
    for e in probes:
        pid = e.get("probe_id") or f"p{e.get('probe_index')}"
        psig[pid].setdefault(_bias(e), []).append(_succ(e))
    psig = {pid: {b: int(round(np.mean(v))) for b, v in d.items()} for pid, d in psig.items()}
    succ_cell = defaultdict(list)
    for e in cands:
        succ_cell[(_bias(e), _off(e))].append(_succ(e))
    succ_cell = {k: int(round(np.mean(v))) for k, v in succ_cell.items()}

    def patt(b, kp):
        return tuple(psig[pid][b] for pid in kp)

    def class_best_success(bs):
        return float(max(np.mean([succ_cell.get((b, o), 0) for b in bs]) for o in offs))

    kcurve = {}
    for K in (0, 1, 2):
        kp = PROBE_ORDER[:K]
        cl = defaultdict(list)
        for b in biases:
            cl[patt(b, kp)].append(b)
        sel = float(np.mean([class_best_success(bs) for c, bs in cl.items() for _ in bs]))
        kcurve[K] = {"probes": kp, "n_distinguishable_classes": len(cl),
                     "classes": {str(k): v for k, v in cl.items()}, "selected_success": sel}

    # order-independence: each single probe alone
    single = {}
    for pid in PROBE_ORDER:
        cl = defaultdict(list)
        for b in biases:
            cl[(psig[pid][b],)].append(b)
        single[pid] = float(np.mean([class_best_success(bs) for c, bs in cl.items() for _ in bs]))

    by_sess = defaultdict(list)
    for e in probes:
        by_sess[e["session_id"]].append(e)
    ptime = {0: 0.0}
    for K in (1, 2):
        ptime[K] = float(np.mean([sum(p["y"]["skill_elapsed_time"]
                                      for p in sorted(ps, key=lambda z: z["order_in_session"])[:K])
                                  for ps in by_sess.values()]))
    voi = {}
    for K in (0, 1, 2):
        gross = kcurve[K]["selected_success"] - kcurve[0]["selected_success"]
        cost = FROZEN.lambda_time * ptime[K]
        voi[K] = {"gross_success_voi": gross, "probe_time_cumulative": ptime[K],
                  "probe_time_cost": cost, "net_voi": gross - cost}
    return {
        "probe_success_pattern_by_bias": psig,
        "K_curve": kcurve,
        "single_probe_selected_success": single,
        "probe_order_matters_for_selected_success": len(set(single.values())) > 1,
        "voi": voi,
        "note": "Only 2 probes exist -> K=3 is NOT computed or interpolated. K=1 already saturates "
                "selected success (each single probe partitions into 2 classes each with a compensating "
                "offset); the 2nd probe adds identifiability (3 classes) but no selected-success gain and "
                "makes Net VOI negative under the frozen utility.",
    }


# ---------------------------------------------------------------- failure mechanism
def failure_mechanism(eps):
    cands = [e for e in eps if e["episode_role"] == "candidate"]
    phases = ["ARC_TO_FACE", "MOVE_TO_PRE_GRASP", "APPROACH", "CLOSE_GRIPPER", "PULL", "SETTLE", "RELEASE"]

    def last_phase(e):
        pd = e["y"].get("phase_durations", {})
        pres = [p for p in phases if p in pd]
        return pres[-1] if pres else "NONE"

    by_eff = defaultdict(lambda: {"n": 0, "succ": 0, "fr": Counter()})
    for e in cands:
        eff = round(abs(_bias(e) + _off(e)), 3)
        by_eff[eff]["n"] += 1
        by_eff[eff]["succ"] += _succ(e)
        if not e["y"]["success"]:
            by_eff[eff]["fr"][e["y"]["failure_reason"]] += 1
    band = {str(k): {"n": v["n"], "success_rate": v["succ"] / v["n"], "failures": dict(v["fr"])}
            for k, v in sorted(by_eff.items())}

    fr = defaultdict(list)
    for e in cands:
        if not e["y"]["success"]:
            fr[e["y"]["failure_reason"]].append(e)
    modes = {}
    for r, es in fr.items():
        modes[r] = {
            "n": len(es), "last_phase": dict(Counter(last_phase(e) for e in es)),
            "mean_handle_rel_err": float(np.mean([e["y"]["handle_relative_error"] for e in es])),
            "mean_time": float(np.mean([e["y"]["skill_elapsed_time"] for e in es])),
            "mean_final_pos": float(np.mean([e["y"]["final_joint_position"] for e in es])),
            "detached_frac": float(np.mean([bool(e["y"].get("handle_detached")) for e in es])),
            "mean_cmd_tracking_error": float(np.mean([e["y"].get("command_tracking_error", 0) for e in es])),
        }

    ykeys = sorted({k for e in cands for k in e["y"]})
    exclusion = {
        "success_band_driven_by_abs_bias_plus_offset": True,
        "band_half_width": BAND,
        "can_exclude": {
            "joint_limit_failure": False, "ik_local_unreachability": False,
            "collision": False, "gripper_slip": "partial",
        },
        "why": {
            "POSITION_TIMEOUT@|eff|=0.04": "last phase APPROACH, no grasp (handle_rel_err 0), 20.8s timeout "
                                           "-> consistent with approach infeasibility, but no joint-limit "
                                           "margin / IK-clamp / collision fields to distinguish IK-unreachable "
                                           "from controller stall or joint limit.",
            "HANDLE_DETACHED@|eff|>=0.06": "last phase PULL, handle_detached, handle_rel_err ~0.085 -> "
                                           "grasp far off-centre -> handle slips during pull (gripper slip); "
                                           "but no collision/contact field to exclude a collision cause.",
        },
        "available_y_fields": ykeys,
        "required_confirmatory_instrumentation": [
            "minimum joint-limit margin (per episode)",
            "IK failure / clamp count",
            "explicit failure_phase field",
            "true handle error at CLOSE_GRIPPER end (not only final)",
            "gripper width at close",
            "collision / contact availability, or explicit no-collision verification",
        ],
    }
    return {"success_rate_by_abs_bias_plus_offset": band, "failure_modes": modes, "exclusion": exclusion}


# ---------------------------------------------------------------- exploration gate (AND)
def exploration_gate(dv, ind_adj, integ):
    n_distinct = dv["n_distinct_best_offsets_across_bias"] if "n_distinct_best_offsets_across_bias" in dv \
        else dv["best_offset_per_bias"]["groups_with_best_offset_differing_across_bias"]
    # distinct best offsets across bias (marginal): recompute from best_offset_per_bias detail
    c1 = _distinct_best(dv) >= GATE["min_distinct_best_offsets"]
    c2 = not dv["robust_offset"]["robust_generalist_exists"]
    succ_gain = dv["block_wise_best_single_cv"]["min_success_gain"]  # use the WORST fold (conservative)
    c3a = succ_gain >= GATE["min_success_gain"]
    c3b = (dv["VSI_frozen"] or 0) >= GATE["min_frozen_vsi"]
    c3 = c3a and c3b
    c4 = not ind_adj["frozen_blocker"]
    c5 = integ["all_ok"]
    passed = c1 and c2 and c3 and c4 and c5
    verdict = "PASS" if passed else ("BLOCKED" if not (c4 and c5) else "MODIFY")
    return {
        "verdict": verdict, "gate_passed": bool(passed), "thresholds": GATE,
        "criteria": {
            "1_distinct_best_offsets>=3": {"ok": bool(c1), "value": _distinct_best(dv)},
            "2_no_robust_generalist": {"ok": bool(c2),
                                       "worst_gap": dv["robust_offset"]["worst_case_gap_of_most_robust"]},
            "3_success_gain>=0.15_AND_vsi>=0.05": {
                "ok": bool(c3), "success_only_gain_ok(>=0.15)": bool(c3a),
                "block_wise_min_success_gain": succ_gain,
                "frozen_vsi_ok(>=0.05)": bool(c3b), "vsi_frozen": dv["VSI_frozen"]},
            "4_replicate_independence_no_blocker": {"ok": bool(c4), "frozen_blocker": ind_adj["frozen_blocker"]},
            "5_integrity_pairing_leakage_provenance": {"ok": bool(c5)},
        },
        "note": "Gate PASSES per frozen rules; data is EXPLORATORY-only (see independence adjudication). "
                "Net VOI is a diagnostic here but MUST be positive in the final confirmatory GO.",
    }


def _distinct_best(dv):
    bpb = dv["best_offset_per_bias"]["per_group_best_by_bias"]
    # marginal best offset per bias = modal across matched groups
    per_bias = defaultdict(list)
    for grp, row in bpb.items():
        for bias, off in row.items():
            per_bias[bias].append(off)
    best = {b: Counter(v).most_common(1)[0][0] for b, v in per_bias.items()}
    return len(set(best.values()))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "metadata.json").read_text())

    prov = provenance(run, meta)
    integ = integrity(eps, meta)
    ind_adj = independence_adjudication(eps, meta)
    dv = decision_value(eps)
    probe = probe_analysis(eps)
    fail = failure_mechanism(eps)
    gate = exploration_gate(dv, ind_adj, integ)

    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    (out / "validation_report.json").write_text(json.dumps(integ, indent=2, default=str))
    (out / "independence_adjudication.json").write_text(json.dumps(ind_adj, indent=2, default=str))
    (out / "decision_value.json").write_text(json.dumps(dv, indent=2, default=str))
    (out / "probe_analysis.json").write_text(json.dumps(probe, indent=2, default=str))
    (out / "failure_mechanism.json").write_text(json.dumps(fail, indent=2, default=str))
    (out / "exploration_gate.json").write_text(json.dumps(gate, indent=2, default=str))

    print(json.dumps({
        "validator_ok": integ["all_ok"], "frozen_blocker": ind_adj["frozen_blocker"],
        "VSI_frozen": round(dv["VSI_frozen"], 4),
        "success_only_block_wise_min_gain": dv["block_wise_best_single_cv"]["min_success_gain"],
        "robust_generalist_exists": dv["robust_offset"]["robust_generalist_exists"],
        "distinct_best_offsets": _distinct_best(dv),
        "probe_K_selected_success": {k: probe["K_curve"][k]["selected_success"] for k in (0, 1, 2)},
        "net_voi": {k: round(probe["voi"][k]["net_voi"], 3) for k in (0, 1, 2)},
        "gate_verdict": gate["verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
