"""Runtime self-check for the exploratory calibration-bias capability map (v1). RAW stats only.

Data integrity + physical validity + the pre-set exploratory read-outs (best offset vs bias, robust-
generalist existence, success-only / frozen-utility VSI, rank reversal, replicate independence) and the
GO-to-preregistration gate. This is NOT Claude B's offline evaluation (no bootstrap CI, no held-out, no
model) -- it only tells us whether the exploratory grid is worth pre-registering a confirmatory run on.

    python .../selfcheck_calibration_bias_capability_map_v1.py [run_dir]
Emits selfcheck.json in the run dir.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
DATA = _DC / "data"
GRID_PATH = _DC / "configs" / "calibration_bias_offset_grid_v1.json"

LAMBDA_ERR, LAMBDA_TIME = 1.0, 0.02          # frozen utility (same as the damping stage)
# GO-to-preregistration thresholds
GATE_MIN_SUCCESS_GAP = 0.15
GATE_MIN_VSI = 0.05
GENERALIST_EPS = 0.05                          # a "robust generalist" is within this U of per-bias best everywhere
REPLICATE_DET_SD = 5e-3                        # below this within-cell final-pos SD => still ~deterministic


def _u(success, err, t):
    return (1.0 if success else 0.0) - LAMBDA_ERR * float(err) - LAMBDA_TIME * float(t)


def _latest():
    runs = sorted(DATA.glob("calibration_bias_capability_map_v1_*"))
    if not runs:
        raise SystemExit("no calibration_bias_capability_map_v1 run found")
    return runs[-1]


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest()
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "metadata.json").read_text())
    grid_sha = hashlib.sha256(GRID_PATH.read_bytes()).hexdigest()
    fails, warns = [], []
    cand = [e for e in eps if e["episode_role"] == "candidate"]
    probe = [e for e in eps if e["episode_role"] == "probe"]
    print(f"== calib-bias self-check {run.name} :: {len(eps)} eps ({len(cand)} cand, {len(probe)} probe) ==")

    # ---- integrity ----
    if any(e.get("offset_grid_sha256") != grid_sha for e in eps):
        fails.append("some episode offset_grid_sha256 != current grid file")
    for e in eps:
        for k in e["x"].keys():
            kl = str(k).lower()
            if "bias" in kl or "secret" in kl or "hidden" in kl or "perceived" in kl or "damping" in kl:
                fails.append(f"x carries privileged key {k} in {e['episode_id']}"); break
    ids = [e["episode_id"] for e in eps]
    if len(set(ids)) != len(ids):
        fails.append("duplicate episode_id")
    if meta.get("dirty_worktree") is not False:
        fails.append(f"dirty_worktree={meta.get('dirty_worktree')} (must be false)")
    if not all(e.get("full_reset_verified") for e in eps):
        fails.append("some full_reset_verified=false")
    dset = [abs(float(e["damping_eff"]) - float(e["secret_deployment_state"].get("handle_bias_local_y", 0) * 0 + meta["damping"])) for e in eps if "damping_eff" in e]
    # damping is fixed baseline; just confirm eff==post
    dpost = [abs(float(e["damping_post"]) - float(e["damping_eff"])) for e in eps if "damping_post" in e and "damping_eff" in e]
    if dpost and max(dpost) > 1e-3:
        warns.append(f"damping drifted (post vs eff) max {max(dpost):.2e}")

    # bias never determines theta offset: candidate_id -> offset pure function
    id2off = {}
    for e in cand:
        o = round(float(e["candidate_grasp_offset_y"]), 9)
        if e["candidate_id"] in id2off and id2off[e["candidate_id"]] != o:
            fails.append(f"candidate_id {e['candidate_id']} offset depends on bias"); break
        id2off[e["candidate_id"]] = o

    biases = sorted({float(e["secret_deployment_state"]["handle_bias_local_y"]) for e in cand})
    offsets = sorted(set(id2off.values()))

    # ---- per (bias, offset) cell aggregation over replicates ----
    cell = defaultdict(list)
    for e in cand:
        cell[(round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 6),
              round(float(e["candidate_grasp_offset_y"]), 6))].append(e)

    def stats(es):
        p = np.mean([int(x["y"]["success"]) for x in es])
        err = np.mean([x["y"]["task_outcome_error"] for x in es])
        t = np.mean([x["y"]["skill_elapsed_time"] for x in es])
        fjp = [x["y"]["final_joint_position"] for x in es]
        u = np.mean([_u(x["y"]["success"], x["y"]["task_outcome_error"], x["y"]["skill_elapsed_time"]) for x in es])
        det = [x["y"]["handle_detached"] for x in es]
        hre = np.mean([x["y"].get("handle_relative_error", 0.0) for x in es])
        return dict(n=len(es), p_succ=round(float(p), 3), err=round(float(err), 4), time=round(float(t), 2),
                    final_pos_mean=round(float(np.mean(fjp)), 4), final_pos_sd=round(float(np.std(fjp)), 5),
                    utility=round(float(u), 4), n_detached=int(sum(det)), handle_rel_err=round(float(hre), 4),
                    success_flips=int(0 < sum(int(x["y"]["success"]) for x in es) < len(es)))

    per_cell = {f"b{b:+.2f}|o{o:+.2f}": stats(cell[(b, o)]) for (b, o) in cell}

    # ---- best offset per bias + monotonicity + generalist ----
    best_off = {}
    for b in biases:
        row = {o: per_cell[f"b{b:+.2f}|o{o:+.2f}"] for o in offsets if f"b{b:+.2f}|o{o:+.2f}" in per_cell}
        best_off[b] = max(row, key=lambda o: row[o]["utility"])
    # monotone: best offset should DECREASE as bias increases (ideal = -bias)
    bo = [best_off[b] for b in biases]
    mono = all(bo[i + 1] <= bo[i] + 1e-9 for i in range(len(bo) - 1)) or \
           all(bo[i + 1] >= bo[i] - 1e-9 for i in range(len(bo) - 1))
    corr = float(np.corrcoef(biases, bo)[0, 1]) if len(set(bo)) > 1 else 0.0
    n_distinct_best = len(set(bo))
    # compensation direction: best offset ~ -bias
    comp_err = float(np.mean([abs(best_off[b] - (-b)) for b in biases]))

    # robust generalist: a single offset within GENERALIST_EPS of the per-bias-best utility at EVERY bias
    def util(b, o):
        c = per_cell.get(f"b{b:+.2f}|o{o:+.2f}")
        return c["utility"] if c else -9.9
    per_bias_best_u = {b: max(util(b, o) for o in offsets) for b in biases}
    generalists = [o for o in offsets if all(per_bias_best_u[b] - util(b, o) <= GENERALIST_EPS for b in biases)]
    has_generalist = len(generalists) > 0

    # ---- VSI (frozen utility + success-only) & success gap ----
    def best_single(metric):
        return max(offsets, key=lambda o: np.mean([metric(b, o) for b in biases]))

    def succ(b, o):
        c = per_cell.get(f"b{b:+.2f}|o{o:+.2f}")
        return c["p_succ"] if c else 0.0

    bs_u = best_single(util)
    vsi_full = float(np.mean([per_bias_best_u[b] - util(b, bs_u) for b in biases]))
    per_bias_best_s = {b: max(succ(b, o) for o in offsets) for b in biases}
    bs_s = best_single(succ)
    vsi_succ = float(np.mean([per_bias_best_s[b] - succ(b, bs_s) for b in biases]))
    succ_gap = vsi_succ  # state-aware oracle success - best-single success, averaged over bias

    # ---- pairwise rank reversal across bias (fraction of offset-pairs whose utility order flips) ----
    import itertools
    pairs = list(itertools.combinations(offsets, 2))
    rev = 0; tot = 0
    for b1, b2 in itertools.combinations(biases, 2):
        for o1, o2 in pairs:
            tot += 1
            s1 = np.sign(util(b1, o1) - util(b1, o2)); s2 = np.sign(util(b2, o1) - util(b2, o2))
            if s1 != 0 and s2 != 0 and s1 != s2:
                rev += 1
    rank_reversal = round(rev / tot, 4) if tot else 0.0

    # ---- replicate independence ----
    within_sd = [c["final_pos_sd"] for c in per_cell.values()]
    med_sd = float(np.median(within_sd))
    any_flips = sum(c["success_flips"] for c in per_cell.values())
    replicate_ok = med_sd > REPLICATE_DET_SD or any_flips > 0
    if not replicate_ok:
        warns.append(f"replicates still ~deterministic (median within-cell final-pos SD {med_sd:.2e}, "
                     f"flips {any_flips}) -- nuisance too small")

    # ---- degeneracy ----
    cs = sum(int(e["y"]["success"]) for e in cand)
    if cs == 0:
        warns.append("ALL candidates FAILED (degenerate)")
    if cs == len(cand):
        warns.append("ALL candidates SUCCEEDED (degenerate)")

    # ---- probe monotonicity (identifiability preview) ----
    probe_signal = {}
    for pid in sorted({e["probe_id"] for e in probe}):
        row = {}
        for b in biases:
            es = [e for e in probe if e["probe_id"] == pid and round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 6) == b]
            if es:
                row[b] = round(float(np.mean([x["y"]["final_joint_position"] for x in es])), 4)
        probe_signal[pid] = row

    # ---- GO-to-preregistration gate ----
    gate = {
        "ge3_distinct_best_offset": bool(n_distinct_best >= 3),
        "no_robust_generalist": bool(not has_generalist),
        "success_gap_ge_0.15_or_vsi_ge_0.05": bool(succ_gap >= GATE_MIN_SUCCESS_GAP or vsi_full >= GATE_MIN_VSI),
        "replicates_independent": bool(replicate_ok),
        "bias_not_leaked": bool(not any("privileged key" in f for f in fails)),
        "compensation_direction_correct": bool(corr < 0 and comp_err <= 0.02 + 1e-9),
    }
    gate_pass = all(gate.values())

    print("\n  best offset per bias (ideal = -bias):")
    for b in biases:
        print(f"    bias {b:+.2f} -> best offset {best_off[b]:+.2f}  (per-bias best U {per_bias_best_u[b]:+.3f})")
    print(f"  distinct best offsets: {n_distinct_best}; monotone={mono}; corr(bias,best)={corr:+.3f}; "
          f"comp_err(|best+bias|)={comp_err:.4f}")
    print(f"  robust generalist offset(s): {generalists if generalists else 'NONE'}")
    print(f"  best-single offset (full U)={bs_u:+.2f}  VSI_full={vsi_full:+.4f}  VSI_succ={vsi_succ:+.4f}  "
          f"success_gap={succ_gap:+.3f}")
    print(f"  rank_reversal={rank_reversal}  replicate median within-SD={med_sd:.2e} flips={any_flips}")
    print(f"  candidate success overall {cs}/{len(cand)}")
    print("\n  GO-to-preregistration gate:")
    for k, v in gate.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  >>> GATE {'PASS' if gate_pass else 'NOT MET'} <<<")

    summary = {
        "run": run.name, "n_episodes": len(eps), "n_candidates": len(cand), "n_probes": len(probe),
        "biases": biases, "offsets": offsets, "damping": meta["damping"],
        "dirty_worktree": meta.get("dirty_worktree"), "full_reset_all_verified": all(e.get("full_reset_verified") for e in eps),
        "offset_grid_sha256": grid_sha, "candidate_success_overall": f"{cs}/{len(cand)}",
        "best_offset_by_bias": {f"{b:+.2f}": best_off[b] for b in biases},
        "best_offset_monotone": mono, "corr_bias_bestoffset": round(corr, 4),
        "n_distinct_best_offsets": n_distinct_best, "compensation_error": round(comp_err, 4),
        "robust_generalist_offsets": generalists, "best_single_offset_full": bs_u,
        "vsi_full": round(vsi_full, 5), "vsi_success_only": round(vsi_succ, 5),
        "success_gap_oracle_vs_single": round(succ_gap, 4), "rank_reversal": rank_reversal,
        "replicate_median_within_sd": round(med_sd, 6), "replicate_success_flips": any_flips,
        "replicates_independent": replicate_ok, "probe_final_pos_signal": probe_signal,
        "per_cell": per_cell, "gate": gate, "gate_pass": gate_pass, "fails": fails, "warns": warns,
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
