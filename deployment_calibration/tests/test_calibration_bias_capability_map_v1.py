"""Tests for the calibration-bias capability-map grid/config, and (if a run exists) the produced data.

Grid/config (always): 5 bias levels, 7 candidate offsets, unique candidate_ids, offsets in the skill's
valid grasp range, ideal-compensation offset (-bias) is present in the grid for every bias, probes fixed.
Config integrity: sessions_per_bias, nuisance ranges safe & bias-independent, robust theta present.
Data (if present): offset is a pure function of candidate_id (bias-independent), each candidate_id x bias
appears sessions_per_bias times, matched groups span all biases, bias absent from x, nuisance recorded.

Run: python .../deployment_calibration/tests/test_calibration_bias_capability_map_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
DATA = _DC / "data"
GRID = _DC / "configs" / "calibration_bias_offset_grid_v1.json"
CFG = _DC / "configs" / "calibration_bias_capability_map_v1.yaml"


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def check_grid(fails):
    g = json.loads(GRID.read_bytes())
    biases = g["bias_levels_y"]
    cand = g["candidate_offsets"]
    offs = [c["grasp_offset_local_y"] for c in cand]
    ids = [c["candidate_id"] for c in cand]
    if len(cand) != 7:
        fails.append(f"grid: {len(cand)} candidate offsets (expected 7)")
    if len(biases) != 5:
        fails.append(f"grid: {len(biases)} bias levels (expected 5)")
    if len(set(ids)) != len(ids):
        fails.append("grid: candidate_id not unique")
    for o in offs:
        if not (-0.06 - 1e-9 <= o <= 0.06 + 1e-9):
            fails.append(f"grid: offset {o} outside skill valid grasp range [-0.06,0.06]")
    # ideal compensation offset (-bias) must be a grid point for every bias
    for b in biases:
        if not any(abs(o - (-b)) < 1e-9 for o in offs):
            fails.append(f"grid: no offset compensates bias {b} (need {-b})")
    if len(g.get("probe_offsets", [])) < 2:
        fails.append("grid: need >=2 fixed probe offsets")
    return g


def check_config(g, fails):
    cfg = _load_yaml(CFG)
    if cfg["drawer"] != g["drawer"]:
        fails.append("config: drawer != grid drawer")
    if int(cfg["sessions_per_bias"]) != 3:
        fails.append(f"config: sessions_per_bias={cfg['sessions_per_bias']} (expected 3)")
    nz = cfg["nuisance"]
    if not (0 < nz["robot_joint_sigma"] <= 0.05 and 0 < nz["robot_joint_cap"] <= 0.06):
        fails.append("config: nuisance robot-joint ranges unsafe/absent")
    if not (0 <= nz["target_jitter"] <= 0.02):
        fails.append("config: target_jitter out of safe range")
    for k in ("max_pos_step", "pull_lead"):
        if k not in cfg["robust_theta"]:
            fails.append(f"config: robust_theta missing {k}")
    n_sess = len(g["bias_levels_y"]) * int(cfg["sessions_per_bias"])
    per = len(g.get("probe_offsets", [])) + len(g["candidate_offsets"])
    if n_sess * per > 150:
        fails.append(f"config: episode budget {n_sess*per} > 150")
    return cfg


def check_data(g, cfg, fails):
    runs = sorted(DATA.glob("calibration_bias_capability_map_v1_*"))
    if not runs:
        print("no capability-map run yet; grid+config checks only")
        return
    run = runs[-1]
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    cand = [e for e in eps if e["episode_role"] == "candidate"]
    print(f"checking run {run.name}: {len(eps)} eps ({len(cand)} cand)")
    id2off = {c["candidate_id"]: round(float(c["grasp_offset_local_y"]), 9) for c in g["candidate_offsets"]}
    seen = {}
    cover = {}
    for e in cand:
        cid = e["candidate_id"]
        if round(float(e["candidate_grasp_offset_y"]), 9) != id2off.get(cid):
            fails.append(f"data: {e['episode_id']} offset != grid for {cid}"); break
        o = round(float(e["theta"]["grasp_offset_local_y"]), 9)
        if cid in seen and seen[cid] != o:
            fails.append(f"data: candidate_id {cid} offset depends on session/bias")
        seen[cid] = o
        for k in e["x"].keys():
            if "bias" in k.lower() or "secret" in k.lower() or "hidden" in k.lower():
                fails.append(f"data: x carries privileged key {k}"); break
        if "nuisance_robot_joint_delta" not in e or "nuisance_seed" not in e:
            fails.append(f"data: {e['episode_id']} missing nuisance provenance"); break
        b = round(float(e["secret_deployment_state"]["handle_bias_local_y"]), 6)
        cover.setdefault((cid, b), 0)
        cover[(cid, b)] += 1
    # each candidate_id x bias appears sessions_per_bias times; matched groups span all biases
    reps = int(cfg["sessions_per_bias"])
    biases = sorted({round(float(b), 6) for b in g["bias_levels_y"]})
    for cid in id2off:
        for b in biases:
            if cover.get((cid, b), 0) != reps:
                fails.append(f"data: {cid} x bias {b} appears {cover.get((cid,b),0)}x (expected {reps})"); break


def main() -> int:
    fails: list = []
    g = check_grid(fails)
    cfg = check_config(g, fails)
    check_data(g, cfg, fails)
    for f in fails[:40]:
        print("FAIL:", f)
    ok = not fails
    print(f"[test_calibration_bias_capability_map_v1] {'PASS' if ok else str(len(fails))+' FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
