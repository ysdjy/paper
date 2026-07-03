"""Unit tests for offline_v2.history — deliberately inject poison and confirm rejection."""

import pytest

from deployment_calibration.offline_v2 import history as H


def _session_episodes():
    eps = []
    sid = "s0"
    for o in range(3):  # 3 probes
        eps.append({"session_id": sid, "episode_role": "probe", "order_in_session": o,
                    "probe_index": o, "mechanism_id": "m", "g": {"target_open_position": 0.2},
                    "theta": {"grasp_offset_local_y": 0.0, "max_pos_step": 0.02, "pull_lead": 0.08},
                    "y": {"success": True, "failure_reason": "NONE", "final_joint_position": 0.19,
                          "task_outcome_error": 0.01, "skill_elapsed_time": 9.0,
                          "phase_durations": {"PULL": 0.7}, "handle_relative_error": 0.01}})
    # a candidate at order 5 (history_cutoff 3)
    cand = {"session_id": sid, "episode_role": "candidate", "order_in_session": 5,
            "history_cutoff": 3, "candidate_group": "s0_g", "candidate_index": 0}
    # a probe from ANOTHER session
    other = {"session_id": "s1", "episode_role": "probe", "order_in_session": 0, "probe_index": 0,
             "mechanism_id": "m", "g": {"target_open_position": 0.2},
             "theta": {"grasp_offset_local_y": 0.0, "max_pos_step": 0.02, "pull_lead": 0.08},
             "y": {"success": False, "failure_reason": "OTHER", "final_joint_position": 0.0,
                   "task_outcome_error": 0.2, "skill_elapsed_time": 20.0,
                   "phase_durations": {"PULL": 0.0}, "handle_relative_error": 0.0}}
    return eps, cand, other


def test_legal_history_passes():
    eps, cand, _ = _session_episodes()
    hist = H.build_history(eps, cand, k=3)
    assert len(hist) == 3
    H.assert_history_legal(cand, hist, all_episodes=eps + [cand])


def test_truncation_respects_k():
    eps, cand, _ = _session_episodes()
    hist = H.build_history(eps, cand, k=1)
    assert len(hist) == 1
    assert hist[0]["order_in_session"] == 0


def test_reject_future_probe():
    eps, cand, _ = _session_episodes()
    hist = H.build_history(eps, cand, k=3)
    # forge a future probe (order >= candidate order)
    hist.append({**hist[0], "order_in_session": 6})
    with pytest.raises(ValueError, match="future"):
        H.assert_history_legal(cand, hist)


def test_reject_cross_session_probe():
    eps, cand, other = _session_episodes()
    hist = H.build_history(eps, cand, k=3)
    poisoned = hist[:2] + [H.history_entry(other)]  # entry from session s1
    with pytest.raises(ValueError, match="not a legal probe"):
        H.assert_history_legal(cand, poisoned, all_episodes=eps + [cand])


def test_reject_candidate_outcome_field():
    eps, cand, _ = _session_episodes()
    hist = H.build_history(eps, cand, k=3)
    hist[0]["candidate_index"] = 2  # inject a candidate-only field
    with pytest.raises(ValueError, match="candidate field"):
        H.assert_history_legal(cand, hist)


def test_reject_privileged_key():
    eps, cand, _ = _session_episodes()
    hist = H.build_history(eps, cand, k=3)
    hist[0]["damping"] = 3.0
    with pytest.raises(ValueError, match="privileged"):
        H.assert_history_legal(cand, hist)


def test_reject_exceeding_cutoff():
    eps, cand, _ = _session_episodes()
    hist = H.build_history(eps, cand, k=None)  # all 3
    cand2 = dict(cand, history_cutoff=2)
    with pytest.raises(ValueError, match="exceeds cutoff"):
        H.assert_history_legal(cand2, hist)
