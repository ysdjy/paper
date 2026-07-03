"""Unit tests for offline_v2.oracle + pairing on a SYNTHETIC matched bank.

We construct a small matched bank where damping deterministically changes the best candidate,
so switch-rate, rank-reversal, and VSI have known non-trivial values. Also verify the pilot-
style non-matched data is correctly rejected.
"""

from deployment_calibration.offline_v2 import oracle as O
from deployment_calibration.offline_v2 import pairing as P
from deployment_calibration.offline_v2.utility import UtilityConfig

CFG = UtilityConfig(lambda_error=1.0, lambda_time=0.0)


def _cand(sid, idx, theta_y, damping, success, err, gid):
    return {
        "session_id": sid, "episode_role": "candidate", "order_in_session": 10 + idx,
        "drawer_name": "middle_drawer", "candidate_group": gid, "candidate_index": idx,
        "x": {"initial_mechanism_joint_pos": 0.0},
        "g": {"target_open_position": 0.2},
        "theta": {"grasp_offset_local_y": theta_y, "max_pos_step": 0.02, "pull_lead": 0.08},
        "secret_deployment_state": {"damping": damping},
        "y": {"success": success, "failure_reason": "NONE" if success else "OTHER",
              "final_joint_position": 0.2 if success else 0.0,
              "task_outcome_error": err, "skill_elapsed_time": 10.0},
    }


def _matched_episodes():
    """Two candidates A(theta_y=-0.05) and B(theta_y=0.05) tested at damping 3 and 40.
    At low damping A wins; at high damping B wins -> switch + reversal + positive VSI."""
    eps = []
    # low damping session: A success(err .01), B fail(err .2)
    eps.append(_cand("sL", 0, -0.05, 3.0, True, 0.01, "sL_g"))
    eps.append(_cand("sL", 1, 0.05, 3.0, False, 0.20, "sL_g"))
    # high damping session: A fail(err .2), B success(err .01)
    eps.append(_cand("sH", 0, -0.05, 40.0, False, 0.20, "sH_g"))
    eps.append(_cand("sH", 1, 0.05, 40.0, True, 0.01, "sH_g"))
    return eps


def test_matched_bank_builds():
    bank = P.build_matched_bank(_matched_episodes())
    assert bank["matched"]
    assert bank["n_matched_groups"] == 1
    mg = bank["matched_groups"][0]
    assert sorted(mg["dampings"]) == [3.0, 40.0]
    assert len(mg["candidate_keys"]) == 2


def test_switch_rate_is_one():
    bank = P.build_matched_bank(_matched_episodes())
    r = O.optimal_candidate_switch_rate(bank, CFG)
    assert r["available"]
    assert r["switch_rate"] == 1.0


def test_rank_reversal_positive():
    bank = P.build_matched_bank(_matched_episodes())
    r = O.pairwise_rank_reversal_rate(bank, CFG)
    assert r["available"]
    assert r["reversal_rate"] == 1.0  # the single A-vs-B pair reverses across the two dampings


def test_vsi_positive():
    bank = P.build_matched_bank(_matched_episodes())
    r = O.value_of_state_information(bank, CFG)
    assert r["available"]
    # state-aware always picks the winner (U ~ 1-.01=.99); state-agnostic must commit to one
    # candidate and eats a failure at the other damping -> aware > agnostic.
    assert r["vsi"] > 0.0


def test_non_matched_bank_rejected():
    # two sessions, DIFFERENT theta sets -> not matched
    eps = [
        _cand("s0", 0, -0.05, 3.0, True, 0.01, "s0_g"),
        _cand("s0", 1, 0.05, 3.0, False, 0.2, "s0_g"),
        _cand("s1", 0, -0.03, 40.0, True, 0.01, "s1_g"),
        _cand("s1", 1, 0.04, 40.0, False, 0.2, "s1_g"),
    ]
    bank = P.build_matched_bank(eps)
    assert not bank["matched"]
    assert O.value_of_state_information(bank, CFG)["available"] is False
    assert O.optimal_candidate_switch_rate(bank, CFG)["available"] is False


def test_oracle_candidate_and_regret():
    eps = _matched_episodes()
    groups = {}
    for e in eps:
        groups.setdefault(e["candidate_group"], []).append(e)
    oc = O.oracle_candidate_per_group(groups, CFG)
    # in sL_g the best is candidate 0 (A succeeds); a model choosing index 1 incurs regret
    reg = O.selection_regret(groups, {"sL_g": 1, "sH_g": 1}, CFG)
    rows = {r["group_id"]: r for r in reg["rows"]}
    assert rows["sL_g"]["regret"] > 0  # picked the loser at low damping
    assert rows["sH_g"]["regret"] == 0  # picked the winner at high damping
