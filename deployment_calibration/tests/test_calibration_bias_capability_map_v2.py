"""Tests for the calibration-bias capability map v2 (BLOCK-level nuisance) grid/config and produced data.

The v2 checks that matter (the point of v2):
  A. candidate-group consistency: within a session, every episode (2 probes + 7 candidates) shares the
     SAME g (exact) and the SAME initial x (within tol) -> a candidate group differs ONLY in theta.
  B. matched-group consistency: within a matched_group (drawer__r{r}), across ALL bias levels, the
     nuisance context (block_seed, robot_joint_delta, target_jitter) is IDENTICAL, and g/x match -> a
     matched group differs across bias ONLY in the hidden bias.
  C. blocks differ: the 3 replicate blocks have DIFFERENT nuisance (robot_joint_delta).
  D. bias not in x; candidate_id -> offset identical across ALL bias; coverage; randomized exec order.

Grid/config integrity always runs. Run:
    python .../deployment_calibration/tests/test_calibration_bias_capability_map_v2.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
DATA = _DC / "data"
GRID = _DC / "configs" / "calibration_bias_offset_grid_v1.json"
CFG = _DC / "configs" / "calibration_bias_capability_map_v2.yaml"

TOL_X = 5e-3   # PhysX float noise tolerance for "same initial x"


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def check_grid_config(fails):
    g = json.loads(GRID.read_bytes())
    offs = [c["grasp_offset_local_y"] for c in g["candidate_offsets"]]
    if len(g["candidate_offsets"]) != 7 or len(g["bias_levels_y"]) != 5:
        fails.append("grid: expected 7 offsets and 5 bias levels")
    for b in g["bias_levels_y"]:
        if not any(abs(o - (-b)) < 1e-9 for o in offs):
            fails.append(f"grid: no compensating offset for bias {b}")
    cfg = _load_yaml(CFG)
    if int(cfg["sessions_per_bias"]) != 3:
        fails.append("config: sessions_per_bias != 3")
    if "master_seed" not in cfg:
        fails.append("config v2: master_seed missing (needed for block seeds + order shuffle)")
    nzc = cfg["nuisance"]
    if not (0 < nzc["robot_joint_sigma"] <= 0.05 and 0 < nzc["robot_joint_cap"] <= 0.06 and 0 <= nzc["target_jitter"] <= 0.02):
        fails.append("config: unsafe nuisance ranges")
    n_sess = len(g["bias_levels_y"]) * int(cfg["sessions_per_bias"])
    per = len(g["probe_offsets"]) + len(g["candidate_offsets"])
    if n_sess * per > 150:
        fails.append(f"config: budget {n_sess*per} > 150")
    return g, cfg


def _max_x_spread(episodes):
    """Max abs spread of robot_joint_pos and tcp_pos across a set of episodes."""
    import numpy as np
    rj = np.array([e["x"]["robot_joint_pos"] for e in episodes], dtype=float)
    tc = np.array([e["x"]["tcp_pos"] for e in episodes], dtype=float)
    return float(max(np.ptp(rj, axis=0).max(), np.ptp(tc, axis=0).max()))


def check_data(g, cfg, fails):
    runs = sorted(DATA.glob("calibration_bias_capability_map_v2_*"))
    if not runs:
        print("no capability-map v2 run yet; grid+config checks only")
        return
    run = runs[-1]
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "metadata.json").read_text())
    cand = [e for e in eps if e["episode_role"] == "candidate"]
    print(f"checking v2 run {run.name}: {len(eps)} eps ({len(cand)} cand)")
    id2off = {c["candidate_id"]: round(float(c["grasp_offset_local_y"]), 9) for c in g["candidate_offsets"]}

    # A. candidate-group (session) consistency: same g (exact), same x (within tol) across ALL episodes
    by_session = defaultdict(list)
    for e in eps:
        by_session[e["session_id"]].append(e)
    worst_x = 0.0
    for sid, es in by_session.items():
        gts = {round(float(e["g"]["target_open_position"]), 9) for e in es}
        if len(gts) != 1:
            fails.append(f"A: session {sid} has differing target_open_position {gts} (g not shared)")
        sp = _max_x_spread(es); worst_x = max(worst_x, sp)
        if sp > TOL_X:
            fails.append(f"A: session {sid} initial x spread {sp:.2e} > {TOL_X} (candidates differ in more than theta)")
    print(f"  A candidate-group: worst within-session x spread = {worst_x:.2e} (tol {TOL_X})")

    # B. matched-group consistency across bias: nuisance context identical; g/x match within tol
    by_matched = defaultdict(list)
    for e in eps:
        by_matched[e["matched_group_id"]].append(e)
    for mg, es in by_matched.items():
        seeds = {e["block_seed"] for e in es}
        deltas = {tuple(round(v, 9) for v in e["nuisance_robot_joint_delta"]) for e in es}
        jit = {round(float(e["nuisance_target_jitter"]), 9) for e in es}
        if len(seeds) != 1 or len(deltas) != 1 or len(jit) != 1:
            fails.append(f"B: matched_group {mg} nuisance context differs across bias "
                         f"(seeds={len(seeds)}, deltas={len(deltas)}, jitter={len(jit)})")
        # x/g must match across bias within tol (only the hidden bias should differ)
        sp = _max_x_spread(es)
        if sp > TOL_X:
            fails.append(f"B: matched_group {mg} x spread across bias {sp:.2e} > {TOL_X}")

    # C. blocks differ: distinct replicate blocks have distinct nuisance deltas
    block_delta = {}
    for e in eps:
        block_delta[e["block_id"]] = tuple(round(v, 9) for v in e["nuisance_robot_joint_delta"])
    if len(set(block_delta.values())) != len(block_delta):
        fails.append(f"C: replicate blocks do not all have distinct nuisance ({block_delta})")
    print(f"  C blocks: {len(block_delta)} blocks, distinct nuisance = {len(set(block_delta.values()))}")

    # D. bias not in x; offset purity; coverage; exec order recorded
    for e in eps:
        for k in e["x"].keys():
            if "bias" in k.lower() or "secret" in k.lower() or "hidden" in k.lower():
                fails.append(f"D: x carries privileged key {k}"); break
    seen = {}
    cover = defaultdict(int)
    for e in cand:
        cid = e["candidate_id"]; o = round(float(e["theta"]["grasp_offset_local_y"]), 9)
        if round(float(e["candidate_grasp_offset_y"]), 9) != id2off.get(cid):
            fails.append(f"D: {e['episode_id']} offset != grid"); break
        if cid in seen and seen[cid] != o:
            fails.append(f"D: candidate_id {cid} offset depends on bias/session")
        seen[cid] = o
        cover[(cid, round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 6))] += 1
    reps = int(cfg["sessions_per_bias"])
    for cid in id2off:
        for b in [round(float(x), 6) for x in g["bias_levels_y"]]:
            if cover.get((cid, b), 0) != reps:
                fails.append(f"D: {cid} x bias {b} appears {cover.get((cid,b),0)}x (expected {reps})"); break
    eo = meta.get("execution_order_session_ids", [])
    if sorted(eo) != sorted(meta.get("canonical_session_order", [])) or len(eo) != meta["n_sessions"]:
        fails.append("D: execution order is not a recorded permutation of the sessions")
    if meta.get("nuisance_level") != "replicate_block":
        fails.append("D: metadata nuisance_level != replicate_block")


def main() -> int:
    fails: list = []
    g, cfg = check_grid_config(fails)
    check_data(g, cfg, fails)
    for f in fails[:40]:
        print("FAIL:", f)
    ok = not fails
    print(f"[test_calibration_bias_capability_map_v2] {'PASS' if ok else str(len(fails))+' FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
