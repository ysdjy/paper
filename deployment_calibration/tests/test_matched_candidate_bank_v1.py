"""Matched-candidate-bank tests (v1) for the selection-interaction pilot.

Validates the FROZEN candidate bank + pilot config WITHOUT Isaac (pure python), and -- if a
selection_interaction_pilot_v1_* run exists -- validates the produced episodes:
  * bank integrity   : 5 candidates/target, unique ids, theta in frozen ranges, grasp offset in the
                       reliable zone, differentiation lives on (max_pos_step,pull_lead) not grasp offset,
                       archetype theta identical across targets, no duplicate/exact-copy candidates.
  * config integrity : yaml loads, 3 damping levels, 3 replicates, targets/candidate order match the bank.
  * id uniqueness    : candidate_id globally unique in the bank and in the produced data.
  * matched pairing  : same candidate_id -> identical theta across ALL damping levels and repeat sessions;
                       every matched_group_id appears at all 3 damping levels with the identical
                       candidate_id set; candidate theta is NOT determined by damping/hidden state.
  * data provenance  : every episode carries candidate_bank_sha256 == the current bank file hash; theta
                       matches the bank; no duplicate episode_id; candidate_index unique within a group.
  * leakage (data)   : history reconstructed for each candidate is same-session probes only (reuse
                       evaluation.build_history if importable).

Run: python projects/paper/deployment_calibration/tests/test_matched_candidate_bank_v1.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
DATA = _DC / "data"
BANK_PATH = _DC / "configs" / "selection_interaction_candidate_bank_v1.json"
CFG_PATH = _DC / "configs" / "selection_interaction_pilot_v1.yaml"

TKEYS = ("grasp_offset_local_y", "max_pos_step", "pull_lead")


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def _theta_tuple(theta):
    return tuple(round(float(theta[k]), 9) for k in TKEYS)


# ---------------------------------------------------------------------------- bank integrity
def check_bank(fails):
    bank_bytes = BANK_PATH.read_bytes()
    bank = json.loads(bank_bytes)
    ranges = bank["frozen_theta_ranges"]
    zone = bank["grasp_offset_reliable_zone"]
    all_ids = []
    archetype_theta = {}
    for tid, t in bank["targets"].items():
        cands = t["candidates"]
        if len(cands) != 5:
            fails.append(f"bank: target {tid} has {len(cands)} candidates (expected 5)")
        steps, pulls, offs, thetas = [], [], [], []
        for c in cands:
            cid = c["candidate_id"]
            all_ids.append(cid)
            th = c["theta"]
            for k in TKEYS:
                lo, hi = ranges[k]
                if not (lo - 1e-9 <= th[k] <= hi + 1e-9):
                    fails.append(f"bank: {cid} {k}={th[k]} out of frozen range [{lo},{hi}]")
            if not (zone[0] - 1e-9 <= th["grasp_offset_local_y"] <= zone[1] + 1e-9):
                fails.append(f"bank: {cid} grasp_offset {th['grasp_offset_local_y']} out of reliable zone {zone}")
            # candidate_id should encode its target
            if not cid.startswith(tid):
                fails.append(f"bank: {cid} does not start with target id {tid}")
            steps.append(th["max_pos_step"]); pulls.append(th["pull_lead"])
            offs.append(th["grasp_offset_local_y"]); thetas.append(_theta_tuple(th))
            # archetype theta must be identical across targets (matched archetypes)
            a = c["archetype"]
            tt = _theta_tuple(th)
            if a in archetype_theta and archetype_theta[a] != tt:
                fails.append(f"bank: archetype {a} theta differs across targets ({archetype_theta[a]} vs {tt})")
            archetype_theta[a] = tt
        # differentiation must live on (max_pos_step,pull_lead), not be a single grasp offset sweep
        if len(set(steps)) < 4:
            fails.append(f"bank: target {tid} max_pos_step not differentiated ({sorted(set(steps))})")
        if len(set(pulls)) < 4:
            fails.append(f"bank: target {tid} pull_lead not differentiated ({sorted(set(pulls))})")
        # not a single grasp offset controlling everything: ranking by grasp offset must NOT match the
        # aggressiveness ranking (by max_pos_step)
        order_by_step = [i for i, _ in sorted(enumerate(steps), key=lambda kv: kv[1])]
        order_by_off = [i for i, _ in sorted(enumerate(offs), key=lambda kv: kv[1])]
        if order_by_step == order_by_off:
            fails.append(f"bank: target {tid} grasp offset ordering == aggressiveness ordering "
                         f"(grasp offset would dominate)")
        # no exact-duplicate candidates within a target
        if len(set(thetas)) != len(thetas):
            fails.append(f"bank: target {tid} has duplicate candidate theta")
    if len(set(all_ids)) != len(all_ids):
        fails.append(f"bank: candidate_id not globally unique ({len(all_ids)} ids, {len(set(all_ids))} unique)")
    return bank, hashlib.sha256(bank_bytes).hexdigest()


# ---------------------------------------------------------------------------- config integrity
def check_config(bank, fails):
    cfg = _load_yaml(CFG_PATH)
    if cfg["drawer"] != bank["drawer"]:
        fails.append(f"config: drawer {cfg['drawer']} != bank {bank['drawer']}")
    if len(cfg["damping_levels"]) != 3:
        fails.append(f"config: {len(cfg['damping_levels'])} damping levels (expected 3)")
    if int(cfg["replicates_per_level"]) != 3:
        fails.append(f"config: replicates_per_level={cfg['replicates_per_level']} (expected 3)")
    if set(cfg["targets_order"]) != set(bank["targets"].keys()):
        fails.append(f"config: targets_order {cfg['targets_order']} != bank targets {list(bank['targets'])}")
    # candidate_order must match the archetype order present in every target
    for tid in cfg["targets_order"]:
        arche = [c["archetype"] for c in bank["targets"][tid]["candidates"]]
        if arche != list(cfg["candidate_order"]):
            fails.append(f"config: candidate_order {cfg['candidate_order']} != bank {tid} order {arche}")
    # expected episode budget sanity
    n_sess = len(cfg["damping_levels"]) * int(cfg["replicates_per_level"])
    per_sess = int(cfg["n_probes"]) + len(cfg["targets_order"]) * 5
    if n_sess * per_sess != 162:
        fails.append(f"config: episode budget {n_sess}x{per_sess}={n_sess*per_sess} (expected 162)")
    return cfg


# ---------------------------------------------------------------------------- produced-data checks
def _latest_run():
    runs = sorted(DATA.glob("selection_interaction_pilot_v1_*"))
    return runs[-1] if runs else None


def check_data(bank, bank_sha, cfg, fails):
    run = _latest_run()
    if run is None:
        print("no selection_interaction_pilot_v1 run yet; bank+config checks only")
        return
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    cand = [e for e in eps if e.get("episode_role") == "candidate"]
    print(f"checking run {run.name}: {len(eps)} eps ({len(cand)} candidate)")

    # bank theta lookup by candidate_id
    id2theta = {}
    for t in bank["targets"].values():
        for c in t["candidates"]:
            id2theta[c["candidate_id"]] = _theta_tuple(c["theta"])

    # 1) provenance: every episode carries the matching bank hash; candidate theta matches the bank
    for e in eps:
        if e.get("candidate_bank_sha256") != bank_sha:
            fails.append(f"data: {e['episode_id']} bank sha256 mismatch"); break
    for e in cand:
        cid = e.get("candidate_id")
        if cid not in id2theta:
            fails.append(f"data: unknown candidate_id {cid} ({e['episode_id']})"); continue
        if _theta_tuple(e["theta"]) != id2theta[cid]:
            fails.append(f"data: {e['episode_id']} theta {_theta_tuple(e['theta'])} != bank {id2theta[cid]}")

    # 2) no duplicate episode_id
    ids = [e["episode_id"] for e in eps]
    if len(set(ids)) != len(ids):
        fails.append("data: duplicate episode_id present")

    # 3) candidate_index unique within each within-session group
    by_group = {}
    for e in cand:
        by_group.setdefault(e["candidate_group"], []).append(e)
    for gid, es in by_group.items():
        idx = [e["candidate_index"] for e in es]
        if len(set(idx)) != len(idx):
            fails.append(f"data: duplicate candidate_index in group {gid}")

    # 4) matched pairing: each matched_group_id appears at all 3 damping levels with the identical
    #    candidate_id set, and each candidate_id has identical theta across damping (NOT determined by it)
    dampings = {float(l["value"]) for l in cfg["damping_levels"]}
    by_matched = {}
    for e in cand:
        by_matched.setdefault(e["matched_group_id"], {}).setdefault(
            float(e["secret_deployment_state"]["damping"]), []).append(e)
    for mg, bydamp in by_matched.items():
        if set(bydamp.keys()) != dampings:
            fails.append(f"data: matched_group {mg} damping levels {sorted(bydamp)} != {sorted(dampings)}")
            continue
        idsets = {d: {e["candidate_id"] for e in es} for d, es in bydamp.items()}
        ref = None
        for d, s in idsets.items():
            if ref is None:
                ref = s
            elif s != ref:
                fails.append(f"data: matched_group {mg} candidate_id set differs across damping")
                break
    # theta strictly a function of candidate_id (never of damping/hidden state)
    id_theta_seen = {}
    for e in cand:
        cid = e["candidate_id"]; tt = _theta_tuple(e["theta"])
        if cid in id_theta_seen and id_theta_seen[cid] != tt:
            fails.append(f"data: candidate_id {cid} has damping-dependent theta {id_theta_seen[cid]} vs {tt}")
        id_theta_seen[cid] = tt

    # 5) leakage: reconstructed history for each candidate = same-session probes before it, only
    try:
        sys.path.insert(0, str(_DC))
        from evaluation.run_session_eval_v2 import build_history  # noqa: E402
        for e in cand:
            Hh = build_history(eps, e["session_id"], e["order_in_session"])
            legal = [p for p in eps if p["session_id"] == e["session_id"]
                     and p.get("episode_role") == "probe" and p["order_in_session"] < e["order_in_session"]]
            if len(Hh) != len(legal):
                fails.append(f"data/leakage: {e['episode_id']} history {len(Hh)} != legal probes {len(legal)}")
                break
    except Exception as ex:
        print(f"(leakage reuse skipped: {ex})")


def main() -> int:
    fails: list[str] = []
    bank, bank_sha = check_bank(fails)
    cfg = check_config(bank, fails)
    check_data(bank, bank_sha, cfg, fails)
    for f in fails[:40]:
        print("FAIL:", f)
    ok = not fails
    print(f"[test_matched_candidate_bank_v1] bank_sha256={bank_sha[:16]}... "
          f"{'PASS' if ok else str(len(fails))+' FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
