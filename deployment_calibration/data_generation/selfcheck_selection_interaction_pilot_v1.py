"""Runtime self-check for the selection-interaction pilot (v1). RAW stats only.

Data-integrity + physical-validity + a RAW optimal-candidate-switch statistic. This does NOT implement
Claude B's offline models/eval (no AUROC, no held-out D3, no bootstrap regret) -- it only verifies the
run is complete/matched/valid and reports raw per-cell outcomes and a raw state-aware vs state-agnostic
utility gap, using the same FROZEN utility as the pilot: U = p_succ - 1.0*err - 0.02*time.

    python .../selfcheck_selection_interaction_pilot_v1.py [run_dir]
Emits selfcheck.json in the run dir and prints a report.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
DATA = _DC / "data"
BANK_PATH = _DC / "configs" / "selection_interaction_candidate_bank_v1.json"

LAMBDA_ERR, LAMBDA_TIME = 1.0, 0.02   # frozen utility weights (match damping pilot)
TKEYS = ("grasp_offset_local_y", "max_pos_step", "pull_lead")


def _utility(success, err, t):
    return (1.0 if success else 0.0) - LAMBDA_ERR * float(err) - LAMBDA_TIME * float(t)


def _latest_run():
    runs = sorted(DATA.glob("selection_interaction_pilot_v1_*"))
    if not runs:
        raise SystemExit("no selection_interaction_pilot_v1 run found")
    return runs[-1]


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_run()
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "metadata.json").read_text())
    bank_sha = hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()
    fails, warns = [], []
    print(f"== self-check {run.name} :: {len(eps)} episodes ==")

    # ---- 0. completeness ----
    exp = meta.get("n_episodes") or 162
    if len(eps) != exp:
        fails.append(f"episode count {len(eps)} != expected {exp}")
    cand = [e for e in eps if e["episode_role"] == "candidate"]
    probe = [e for e in eps if e["episode_role"] == "probe"]
    if len(cand) != 135:
        fails.append(f"candidate count {len(cand)} != 135")
    if len(probe) != 27:
        fails.append(f"probe count {len(probe)} != 27")

    # ---- 1. duplicates ----
    ids = [e["episode_id"] for e in eps]
    if len(set(ids)) != len(ids):
        fails.append("duplicate episode_id")
    # duplicate candidate_id within (session, group) should not happen
    seen = set()
    for e in cand:
        key = (e["session_id"], e["candidate_id"])
        if key in seen:
            fails.append(f"duplicate candidate_id {e['candidate_id']} in {e['session_id']}")
        seen.add(key)

    # ---- 2. bank pairing integrity + provenance ----
    if any(e.get("candidate_bank_sha256") != bank_sha for e in eps):
        fails.append("some episode candidate_bank_sha256 != current bank file hash")
    if meta.get("candidate_bank_sha256") != bank_sha:
        warns.append("metadata bank sha256 != current bank file hash (bank changed after run?)")
    bank = json.loads(BANK_PATH.read_bytes())
    id2theta = {c["candidate_id"]: tuple(round(float(c["theta"][k]), 9) for k in TKEYS)
                for t in bank["targets"].values() for c in t["candidates"]}
    for e in cand:
        tt = tuple(round(float(e["theta"][k]), 9) for k in TKEYS)
        if id2theta.get(e["candidate_id"]) != tt:
            fails.append(f"{e['episode_id']} theta != bank")
            break
    # theta is a pure function of candidate_id (not of damping)
    idth = {}
    for e in cand:
        tt = tuple(round(float(e["theta"][k]), 9) for k in TKEYS)
        if e["candidate_id"] in idth and idth[e["candidate_id"]] != tt:
            fails.append(f"candidate_id {e['candidate_id']} theta depends on damping"); break
        idth[e["candidate_id"]] = tt

    # ---- 3. matched groups present at all 3 damping ----
    dampings = sorted({float(l["value"]) for l in meta["damping_levels"]})
    mg = defaultdict(lambda: defaultdict(list))
    for e in cand:
        mg[e["matched_group_id"]][float(e["secret_deployment_state"]["damping"])].append(e)
    for g, byd in mg.items():
        if sorted(byd.keys()) != dampings:
            fails.append(f"matched_group {g} missing damping levels (has {sorted(byd)})")
        else:
            idsets = [frozenset(x["candidate_id"] for x in byd[d]) for d in dampings]
            if len(set(idsets)) != 1:
                fails.append(f"matched_group {g} candidate_id set differs across damping")
    print(f"  matched groups: {len(mg)} (expect 9), each x {len(dampings)} damping levels")

    # ---- 4. damping set + persisted ----
    deff = [abs(float(e["damping_eff"]) - float(e["secret_deployment_state"]["damping"])) for e in eps
            if "damping_eff" in e]
    dpost = [abs(float(e["damping_post"]) - float(e["damping_eff"])) for e in eps
             if "damping_post" in e and "damping_eff" in e]
    set_maxerr = max(deff) if deff else float("nan")
    post_maxerr = max(dpost) if dpost else float("nan")
    if set_maxerr > 1e-3:
        fails.append(f"damping set error {set_maxerr:.3e} > 1e-3")
    if post_maxerr > 1e-3:
        warns.append(f"damping drifted during episode (post vs eff) max {post_maxerr:.3e}")
    print(f"  damping set_maxerr={set_maxerr:.2e} post_drift_maxerr={post_maxerr:.2e}")

    # ---- 5. full reset + dirty_worktree ----
    if not all(e.get("full_reset_verified") for e in eps):
        fails.append(f"{sum(1 for e in eps if not e.get('full_reset_verified'))} eps full_reset_verified=false")
    if meta.get("dirty_worktree") is not False:
        fails.append(f"dirty_worktree={meta.get('dirty_worktree')} (must be false)")
    print(f"  full_reset_all_verified={all(e.get('full_reset_verified') for e in eps)} "
          f"dirty_worktree={meta.get('dirty_worktree')}")

    # ---- 6. physical label validity ----
    for e in eps:
        y, g = e["y"], e["g"]
        fp = y["final_joint_position"]
        expect = fp >= g["target_open_position"] - g["target_tolerance"]
        if bool(y["success"]) != bool(expect):
            fails.append(f"{e['episode_id']} success!=label ({fp:.3f} vs {g['target_open_position']}-{g['target_tolerance']})")
            break
        if abs(y["task_outcome_error"] - abs(fp - g["target_open_position"])) > 1e-4:
            fails.append(f"{e['episode_id']} task_outcome_error mismatch"); break

    # ---- 7. per (damping,target,archetype) cell aggregation + optimal-candidate switch ----
    def arche(e):
        return e.get("candidate_archetype") or e["candidate_id"].split("_", 1)[1]

    cell = defaultdict(list)   # (D, target_id, archetype) -> list of episodes (replicates)
    for e in cand:
        cell[(float(e["secret_deployment_state"]["damping"]), e["target_id"], arche(e))].append(e)

    def cell_stats(es):
        n = len(es)
        p = sum(int(x["y"]["success"]) for x in es) / n
        err = sum(x["y"]["task_outcome_error"] for x in es) / n
        t = sum(x["y"]["skill_elapsed_time"] for x in es) / n
        u = sum(_utility(x["y"]["success"], x["y"]["task_outcome_error"], x["y"]["skill_elapsed_time"])
                for x in es) / n
        fjp = sum(x["y"]["final_joint_position"] for x in es) / n
        return dict(n=n, p_succ=round(p, 3), err=round(err, 4), time=round(t, 2),
                    final_pos=round(fjp, 4), utility=round(u, 4))

    targets = meta["targets_order"]
    archetypes = [a for a in bank["archetypes"].keys()]
    per_cell = {}
    print("\n  === per (damping, target): utility by archetype [best*] ===")
    for D in dampings:
        for T in targets:
            row = {}
            for A in archetypes:
                es = cell.get((D, T, A))
                if es:
                    row[A] = cell_stats(es)
                    per_cell[f"D{D}|{T}|{A}"] = row[A]
            if not row:
                continue
            best = max(row, key=lambda a: row[a]["utility"])
            cells = " ".join(f"{a.split('_')[0]}={row[a]['utility']:+.3f}{'*' if a==best else ' '}"
                             for a in archetypes if a in row)
            print(f"  D={D:>4} {T}: {cells}  best={best}")

    # optimal-candidate switch: per target, best archetype at each damping
    switch = {}
    for T in targets:
        best_by_D = {}
        for D in dampings:
            row = {A: per_cell[f"D{D}|{T}|{A}"] for A in archetypes if f"D{D}|{T}|{A}" in per_cell}
            if row:
                best_by_D[D] = max(row, key=lambda a: row[a]["utility"])
        switch[T] = best_by_D
    n_switch = sum(1 for T in targets if len(set(switch[T].values())) > 1)
    print(f"\n  optimal-candidate SWITCH across damping: {n_switch}/{len(targets)} targets change best archetype")
    for T in targets:
        print(f"    {T}: " + " -> ".join(f"D{D}:{switch[T].get(D,'-')}" for D in dampings))

    # ---- 8. state-aware vs state-agnostic raw utility gap ----
    # state-aware: per (D,T) pick best archetype for that D.
    # state-agnostic: per T pick single archetype maximizing mean-over-D utility, evaluate at each D.
    gaps = []
    for T in targets:
        meanU = {}
        for A in archetypes:
            us = [per_cell[f"D{D}|{T}|{A}"]["utility"] for D in dampings if f"D{D}|{T}|{A}" in per_cell]
            if len(us) == len(dampings):
                meanU[A] = sum(us) / len(us)
        if not meanU:
            continue
        agn = max(meanU, key=lambda a: meanU[a])
        for D in dampings:
            aware = switch[T].get(D)
            if aware and f"D{D}|{T}|{agn}" in per_cell and f"D{D}|{T}|{aware}" in per_cell:
                gaps.append(per_cell[f"D{D}|{T}|{aware}"]["utility"] - per_cell[f"D{D}|{T}|{agn}"]["utility"])
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    print(f"\n  state-aware vs state-agnostic RAW utility gap: mean={mean_gap:+.4f} over {len(gaps)} (D,target) cells "
          f"(>=0 by construction; larger = more selection value)")

    # ---- degeneracy guard ----
    cs = sum(int(e["y"]["success"]) for e in cand)
    if cs == 0:
        warns.append("ALL candidates FAILED (degenerate)")
    if cs == len(cand):
        warns.append("ALL candidates SUCCEEDED (degenerate)")
    print(f"\n  candidate success overall: {cs}/{len(cand)}")

    summary = {
        "run": run.name, "n_episodes": len(eps), "n_candidates": len(cand), "n_probes": len(probe),
        "matched_groups": len(mg), "dampings": dampings,
        "damping_set_maxerr": set_maxerr, "damping_post_drift_maxerr": post_maxerr,
        "full_reset_all_verified": all(e.get("full_reset_verified") for e in eps),
        "dirty_worktree": meta.get("dirty_worktree"), "candidate_bank_sha256": bank_sha,
        "candidate_success_overall": f"{cs}/{len(cand)}",
        "optimal_candidate_switch": {T: switch[T] for T in targets},
        "n_targets_with_switch": n_switch,
        "state_aware_vs_agnostic_utility_gap_mean": round(mean_gap, 4), "n_gap_cells": len(gaps),
        "per_cell": per_cell, "fails": fails, "warns": warns,
    }
    json.dump(summary, open(run / "selfcheck.json", "w"), indent=1)

    print("\n== VERDICT ==")
    for w in warns:
        print("WARN:", w)
    for f in fails:
        print("FAIL:", f)
    ok = not fails
    print(f"self-check {'PASS' if ok else str(len(fails))+' FAIL'} -> {run/'selfcheck.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
