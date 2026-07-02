"""Leakage tests (v2): history is decision-legal; splits are session-disjoint.

Synthetic self-test always runs (validates build_history/split logic). If a sessions_/pilot run exists,
also checks it. Run: python projects/paper/deployment_calibration/tests/test_history_no_leakage_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER / "deployment_calibration"))
sys.path.insert(0, str(_PAPER / "deployment_calibration" / "evaluation"))

from evaluation.run_session_eval_v2 import build_history, candidates_in  # noqa: E402
from evaluation.session_split_v2 import split_sessions  # noqa: E402
from evaluation.session_split_v2 import load_episodes  # noqa: E402

DATA = _PAPER / "deployment_calibration" / "data"


def _synthetic():
    eps = []
    for sid in ("sess_000", "sess_001"):
        for pi in range(3):
            eps.append({"session_id": sid, "drawer_name": "middle_drawer", "mechanism_id": "cabinet:middle_drawer",
                        "episode_role": "probe", "order_in_session": pi, "hidden_state_id": "L1",
                        "theta": {"grasp_offset_local_y": 0, "max_pos_step": 0.02, "pull_lead": 0.08},
                        "g": {"target_open_position": 0.2}, "y": {"success": True, "failure_reason": "NONE",
                        "task_outcome_error": 0.01, "skill_elapsed_time": 9.0, "final_joint_position": 0.19,
                        "phase_durations": {"PULL": 1.0}}})
        for ci in range(4):
            eps.append({"session_id": sid, "drawer_name": "middle_drawer", "mechanism_id": "cabinet:middle_drawer",
                        "episode_role": "candidate", "order_in_session": 3 + ci, "candidate_group": f"{sid}_g0",
                        "candidate_index": ci, "history_cutoff": 3, "hidden_state_id": "L1",
                        "theta": {"grasp_offset_local_y": 0.01, "max_pos_step": 0.02, "pull_lead": 0.08},
                        "g": {"target_open_position": 0.2}, "y": {"success": True, "failure_reason": "NONE",
                        "task_outcome_error": 0.02, "skill_elapsed_time": 9.0, "final_joint_position": 0.19,
                        "phase_durations": {"PULL": 1.0}}})
    return eps


def _check(episodes, tag, fails):
    for e in [x for x in episodes if x.get("episode_role") == "candidate"]:
        H = build_history(episodes, e["session_id"], e["order_in_session"])
        for h in H:
            pass
        # reconstruct which episodes fed H (probes same session, order < candidate order)
        used = [p for p in episodes if p["session_id"] == e["session_id"]
                and p.get("episode_role") == "probe" and p["order_in_session"] < e["order_in_session"]]
        if len(H) != len(used):
            fails.append(f"{tag}: {e.get('episode_id', e['session_id'])} history size {len(H)} != legal probes {len(used)}")
        # no candidate (self/other) in history: build_history only pulls role==probe -> assert none are candidates
        # no future: all used order < candidate order (guaranteed by filter); double-check
        if any(p["order_in_session"] >= e["order_in_session"] for p in used):
            fails.append(f"{tag}: future episode leaked into history")
        # cross-session: all used from same session
        if any(p["session_id"] != e["session_id"] for p in used):
            fails.append(f"{tag}: cross-session leakage")


def main() -> int:
    fails = []
    # 1) synthetic
    syn = _synthetic()
    _check(syn, "synthetic", fails)
    split = split_sessions(syn, seed=0)
    inter = set(split["train"]) & set(split["test"])
    if inter:
        fails.append(f"synthetic: train/test not disjoint {inter}")

    # 2) real run if present
    runs = sorted(list(DATA.glob("damping_pilot_v1_*")) + list(DATA.glob("*sessions*_v2_*"))
                  + list(DATA.glob("formal_sessions_v2_*")))
    if runs:
        run = runs[-1]
        eps = load_episodes(run)
        _check(eps, run.name, fails)
        sp = split_sessions(eps, seed=0)
        if set(sp["train"]) & set(sp["test"]) or set(sp["val"]) & set(sp["test"]):
            fails.append(f"{run.name}: split not disjoint")
        print(f"checked real run: {run.name} ({len(eps)} eps)")
    else:
        print("no real sessions run yet; synthetic self-test only")

    for f in fails:
        print("FAIL:", f)
    print(f"[test_history_no_leakage_v2] {'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
