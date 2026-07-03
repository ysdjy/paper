"""Runtime self-check for the calibration-bias capability map v2 (BLOCK nuisance). RAW stats only.

Adds to the v1 read-outs: the block-level nuisance manifest, candidate-group x/g consistency, matched-
group nuisance/x consistency across bias, and replicate independence measured across the 3 blocks. NOT
Claude B's offline evaluation. Emits selfcheck.json in the run dir.

    python .../selfcheck_calibration_bias_capability_map_v2.py [run_dir]
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
DATA = _DC / "data"
GRID_PATH = _DC / "configs" / "calibration_bias_offset_grid_v1.json"

LAMBDA_ERR, LAMBDA_TIME = 1.0, 0.02
GATE_MIN_SUCCESS_GAP = 0.15
GATE_MIN_VSI = 0.05
GENERALIST_EPS = 0.05
REPLICATE_DET_SD = 3e-3   # success-cell within-block final-pos SD must exceed ~5x the damping float noise (6e-4)
TOL_X = 5e-3


def _u(s, err, t):
    return (1.0 if s else 0.0) - LAMBDA_ERR * float(err) - LAMBDA_TIME * float(t)


def _latest():
    runs = sorted(DATA.glob("calibration_bias_capability_map_v2_*"))
    if not runs:
        raise SystemExit("no calibration_bias_capability_map_v2 run found")
    return runs[-1]


def _max_x_spread(es):
    rj = np.array([e["x"]["robot_joint_pos"] for e in es], dtype=float)
    tc = np.array([e["x"]["tcp_pos"] for e in es], dtype=float)
    return float(max(np.ptp(rj, axis=0).max(), np.ptp(tc, axis=0).max()))


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest()
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "metadata.json").read_text())
    grid_sha = hashlib.sha256(GRID_PATH.read_bytes()).hexdigest()
    fails, warns = [], []
    cand = [e for e in eps if e["episode_role"] == "candidate"]
    probe = [e for e in eps if e["episode_role"] == "probe"]
    print(f"== calib-bias v2 self-check {run.name} :: {len(eps)} eps ({len(cand)} cand, {len(probe)} probe) ==")

    # ---- integrity ----
    if any(e.get("offset_grid_sha256") != grid_sha for e in eps):
        fails.append("offset_grid_sha256 mismatch")
    for e in eps:
        if any(k.lower().find(x) >= 0 for k in e["x"].keys() for x in ("bias", "secret", "hidden", "damping", "perceived")):
            fails.append(f"x carries privileged key in {e['episode_id']}"); break
    if meta.get("dirty_worktree") is not False:
        fails.append(f"dirty_worktree={meta.get('dirty_worktree')}")
    if not all(e.get("full_reset_verified") for e in eps):
        fails.append("some full_reset_verified=false")
    if meta.get("nuisance_level") != "replicate_block":
        fails.append("nuisance_level != replicate_block")

    biases = sorted({round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 6) for e in cand})
    offsets = sorted({round(float(e["candidate_grasp_offset_y"]), 6) for e in cand})

    # ---- BLOCK nuisance manifest + consistency ----
    manifest = meta.get("nuisance_contexts", {})
    by_session = defaultdict(list)
    for e in eps:
        by_session[e["session_id"]].append(e)
    worst_group_x = 0.0
    g_shared = True
    for sid, es in by_session.items():
        if len({round(float(e["g"]["target_open_position"]), 9) for e in es}) != 1:
            g_shared = False
        worst_group_x = max(worst_group_x, _max_x_spread(es))
    cand_group_ok = g_shared and worst_group_x <= TOL_X
    if not cand_group_ok:
        fails.append(f"candidate-group NOT clean (g_shared={g_shared}, worst x spread {worst_group_x:.2e})")

    by_matched = defaultdict(list)
    for e in eps:
        by_matched[e["matched_group_id"]].append(e)
    matched_ctx_ok = True
    worst_matched_x = 0.0
    for mg, es in by_matched.items():
        if len({e["block_seed"] for e in es}) != 1 or \
           len({tuple(round(v, 9) for v in e["nuisance_robot_joint_delta"]) for e in es}) != 1:
            matched_ctx_ok = False
        worst_matched_x = max(worst_matched_x, _max_x_spread(es))
    if not matched_ctx_ok:
        fails.append("matched-group nuisance context differs across bias")
    block_deltas = {e["block_id"]: tuple(round(v, 9) for v in e["nuisance_robot_joint_delta"]) for e in eps}
    blocks_distinct = len(set(block_deltas.values())) == len(block_deltas)
    if not blocks_distinct:
        fails.append("replicate blocks share identical nuisance")

    # candidate_id -> offset pure function of id (not bias)
    id2 = {}
    for e in cand:
        o = round(float(e["candidate_grasp_offset_y"]), 9)
        if e["candidate_id"] in id2 and id2[e["candidate_id"]] != o:
            fails.append(f"candidate_id {e['candidate_id']} offset depends on bias"); break
        id2[e["candidate_id"]] = o

    # ---- per (bias, offset) cells over the 3 blocks ----
    cell = defaultdict(list)
    for e in cand:
        cell[(round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 6),
              round(float(e["candidate_grasp_offset_y"]), 6))].append(e)

    def stats(es):
        fjp = [x["y"]["final_joint_position"] for x in es]
        return dict(n=len(es), p_succ=round(float(np.mean([int(x["y"]["success"]) for x in es])), 3),
                    err=round(float(np.mean([x["y"]["task_outcome_error"] for x in es])), 4),
                    time=round(float(np.mean([x["y"]["skill_elapsed_time"] for x in es])), 2),
                    final_pos_mean=round(float(np.mean(fjp)), 4), final_pos_sd=round(float(np.std(fjp)), 5),
                    utility=round(float(np.mean([_u(x["y"]["success"], x["y"]["task_outcome_error"], x["y"]["skill_elapsed_time"]) for x in es])), 4),
                    n_detached=int(sum(x["y"]["handle_detached"] for x in es)),
                    handle_rel_err=round(float(np.mean([x["y"].get("handle_relative_error", 0.0) for x in es])), 4),
                    success_flips=int(0 < sum(int(x["y"]["success"]) for x in es) < len(es)))

    per_cell = {f"b{b:+.2f}|o{o:+.2f}": stats(cell[(b, o)]) for (b, o) in cell}

    def util(b, o):
        c = per_cell.get(f"b{b:+.2f}|o{o:+.2f}"); return c["utility"] if c else -9.9

    def succ(b, o):
        c = per_cell.get(f"b{b:+.2f}|o{o:+.2f}"); return c["p_succ"] if c else 0.0

    best_off = {b: max(offsets, key=lambda o: util(b, o)) for b in biases}
    bo = [best_off[b] for b in biases]
    mono = all(bo[i + 1] <= bo[i] + 1e-9 for i in range(len(bo) - 1)) or all(bo[i + 1] >= bo[i] - 1e-9 for i in range(len(bo) - 1))
    corr = float(np.corrcoef(biases, bo)[0, 1]) if len(set(bo)) > 1 else 0.0
    comp_err = float(np.mean([abs(best_off[b] - (-b)) for b in biases]))
    per_bias_best_u = {b: max(util(b, o) for o in offsets) for b in biases}
    generalists = [o for o in offsets if all(per_bias_best_u[b] - util(b, o) <= GENERALIST_EPS for b in biases)]

    def best_single(metric):
        return max(offsets, key=lambda o: np.mean([metric(b, o) for b in biases]))

    bs_u = best_single(util)
    vsi_full = float(np.mean([per_bias_best_u[b] - util(b, bs_u) for b in biases]))
    per_bias_best_s = {b: max(succ(b, o) for o in offsets) for b in biases}
    bs_s = best_single(succ)
    vsi_succ = float(np.mean([per_bias_best_s[b] - succ(b, bs_s) for b in biases]))

    pairs = list(itertools.combinations(offsets, 2))
    rev = tot = 0
    for b1, b2 in itertools.combinations(biases, 2):
        for o1, o2 in pairs:
            tot += 1
            s1 = np.sign(util(b1, o1) - util(b1, o2)); s2 = np.sign(util(b2, o1) - util(b2, o2))
            if s1 and s2 and s1 != s2:
                rev += 1
    rank_reversal = round(rev / tot, 4) if tot else 0.0

    # Replicate independence, two layers (the all-cells median is polluted by clean-fail cells whose
    # final==0 gives SD==0, so evaluate the CONTINUOUS outcome variance on cells that actually grasp):
    #  (1) continuous: within-cell final-pos SD on success-bearing cells -> is there a REAL, nuisance-driven
    #      spread across the 3 blocks, well above the damping-stage float noise (~6e-4)?
    #  (2) label: do any cells flip success across blocks?
    FLOAT_NOISE = 6e-4                                   # damping-stage pure-float-noise reference
    all_sd = [c["final_pos_sd"] for c in per_cell.values()]
    succ_sd = [c["final_pos_sd"] for c in per_cell.values() if c["p_succ"] > 0.0]   # non-degenerate cells
    med_sd_all = float(np.median(all_sd))
    med_sd_succ = float(np.median(succ_sd)) if succ_sd else 0.0
    max_sd_succ = float(max(succ_sd)) if succ_sd else 0.0
    any_flips = sum(c["success_flips"] for c in per_cell.values())
    # non-technical-repeat = a real continuous replicate spread (>= REPLICATE_DET_SD, ~5x float noise)
    # on success-bearing cells, OR any success-label flip.
    replicate_continuous_ok = med_sd_succ >= REPLICATE_DET_SD
    replicate_ok = replicate_continuous_ok or any_flips > 0
    if not replicate_ok:
        warns.append(f"replicates ~deterministic (success-cell median SD {med_sd_succ:.2e} < {REPLICATE_DET_SD}, flips {any_flips})")
    elif any_flips == 0:
        warns.append(f"replicate spread is CONTINUOUS-only: success-cell median SD {med_sd_succ:.2e} "
                     f"(max {max_sd_succ:.2e}) >> float noise {FLOAT_NOISE}, but 0 success-label flips "
                     f"(nuisance moves final pos/time but is below the grasp-tolerance boundary, so it "
                     f"never flips a success label). Diagnosis: nuisance amplitude vs a sharp success edge.")
    med_sd = med_sd_succ   # keep the downstream summary key meaningful (success-cell continuous spread)

    cs = sum(int(e["y"]["success"]) for e in cand)
    if cs == 0 or cs == len(cand):
        warns.append(f"degenerate candidate outcomes ({cs}/{len(cand)})")

    probe_signal = {}
    for pid in sorted({e["probe_id"] for e in probe}):
        probe_signal[pid] = {f"{b:+.2f}": round(float(np.mean([x["y"]["final_joint_position"] for x in probe
                              if x["probe_id"] == pid and round(float(x["secret_deployment_state"]["handle_bias_local_y"]), 6) == b])), 4)
                             for b in biases if any(x["probe_id"] == pid and round(float(x["secret_deployment_state"]["handle_bias_local_y"]), 6) == b for x in probe)}

    gate = {
        "ge3_distinct_best_offset": bool(len(set(bo)) >= 3),
        "no_robust_generalist": bool(len(generalists) == 0),
        "success_gap_ge_0.15_or_vsi_ge_0.05": bool(vsi_succ >= GATE_MIN_SUCCESS_GAP or vsi_full >= GATE_MIN_VSI),
        "replicates_independent": bool(replicate_ok),
        "bias_not_leaked": bool(not any("privileged key" in f for f in fails)),
        "compensation_direction_correct": bool(corr < 0 and comp_err <= 0.02 + 1e-9),
        "candidate_group_clean": bool(cand_group_ok),
        "matched_group_clean": bool(matched_ctx_ok and worst_matched_x <= TOL_X),
    }
    gate_pass = all(gate.values())

    print("\n  block nuisance manifest:")
    for r, ctx in manifest.items():
        print(f"    block {r}: seed={ctx['block_seed']} jitter={ctx['target_jitter']:+.4f} "
              f"delta_absmax={max(abs(v) for v in ctx['robot_joint_delta']):.4f}")
    print(f"  candidate-group clean: {cand_group_ok} (worst within-session x spread {worst_group_x:.2e})")
    print(f"  matched-group clean: {matched_ctx_ok and worst_matched_x <= TOL_X} "
          f"(ctx identical={matched_ctx_ok}, worst x-across-bias {worst_matched_x:.2e}); blocks distinct={blocks_distinct}")
    print("  best offset per bias (ideal = -bias):")
    for b in biases:
        print(f"    bias {b:+.2f} -> best offset {best_off[b]:+.2f} (U {per_bias_best_u[b]:+.3f})")
    print(f"  distinct best offsets={len(set(bo))} monotone={mono} corr={corr:+.3f} comp_err={comp_err:.4f}")
    print(f"  robust generalist offset(s): {generalists if generalists else 'NONE'}")
    print(f"  best-single(full)={bs_u:+.2f}  VSI_full={vsi_full:+.4f}  VSI_succ={vsi_succ:+.4f}  rank_reversal={rank_reversal}")
    print(f"  replicate: success-cell median SD={med_sd_succ:.2e} (max {max_sd_succ:.2e}, all-cell median {med_sd_all:.2e}) "
          f"label-flips={any_flips}; continuous_indep={replicate_continuous_ok}; candidate success {cs}/{len(cand)}")
    print("\n  GO-to-preregistration gate:")
    for k, v in gate.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  >>> GATE {'PASS' if gate_pass else 'NOT MET'} <<<")

    summary = {
        "run": run.name, "n_episodes": len(eps), "n_candidates": len(cand), "n_probes": len(probe),
        "biases": biases, "offsets": offsets, "damping": meta["damping"], "nuisance_level": meta.get("nuisance_level"),
        "block_nuisance_manifest": manifest, "master_seed": meta.get("master_seed"),
        "execution_order_session_ids": meta.get("execution_order_session_ids"),
        "dirty_worktree": meta.get("dirty_worktree"), "full_reset_all_verified": all(e.get("full_reset_verified") for e in eps),
        "candidate_group_clean": cand_group_ok, "worst_within_session_x_spread": round(worst_group_x, 6),
        "matched_group_ctx_identical": matched_ctx_ok, "worst_matched_x_spread": round(worst_matched_x, 6),
        "blocks_distinct_nuisance": blocks_distinct,
        "candidate_success_overall": f"{cs}/{len(cand)}",
        "best_offset_by_bias": {f"{b:+.2f}": best_off[b] for b in biases}, "best_offset_monotone": mono,
        "corr_bias_bestoffset": round(corr, 4), "n_distinct_best_offsets": len(set(bo)),
        "compensation_error": round(comp_err, 4), "robust_generalist_offsets": generalists,
        "best_single_offset_full": bs_u, "vsi_full": round(vsi_full, 5), "vsi_success_only": round(vsi_succ, 5),
        "rank_reversal": rank_reversal,
        "replicate_success_cell_median_sd": round(med_sd_succ, 6),
        "replicate_success_cell_max_sd": round(max_sd_succ, 6),
        "replicate_all_cell_median_sd": round(med_sd_all, 6),
        "replicate_continuous_independent": replicate_continuous_ok,
        "replicate_success_label_flips": any_flips, "float_noise_reference": FLOAT_NOISE,
        "replicate_median_within_sd": round(med_sd, 6),
        "replicate_success_flips": any_flips, "replicates_independent": replicate_ok,
        "probe_final_pos_signal": probe_signal, "per_cell": per_cell,
        "gate": gate, "gate_pass": gate_pass, "fails": fails, "warns": warns,
    }
    json.dump(summary, open(run / "selfcheck.json", "w"), indent=1)
    print("\n== VERDICT ==")
    for w in warns:
        print("WARN:", w)
    for f in fails:
        print("FAIL:", f)
    ok = not fails
    print(f"self-check {'PASS' if ok else str(len(fails))+' FAIL'} (gate {'PASS' if gate_pass else 'NOT MET'}) -> {run/'selfcheck.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
