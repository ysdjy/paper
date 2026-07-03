"""Tests for the (target, replicate) matched-bank keying + decision-value policies.

Builds a synthetic bank mirroring the selection-interaction structure: candidate_id/target_id/
replicate_id present, a robust generalist that wins everywhere, and aggressive candidates that
collapse at high damping. Confirms 9-group keying, robust-generalist == VSI gap, switch>0, VSI~0.
"""

import numpy as np

from deployment_calibration.offline_v2 import oracle as O
from deployment_calibration.offline_v2 import pairing as P
from deployment_calibration.offline_v2.utility import UtilityConfig

CFG = UtilityConfig(lambda_error=1.0, lambda_time=0.02)


def _ep(target, rep, arch, damping, success, err, time, order):
    cid = f"{target}_{arch}"
    theta = {"c1_steady": (-0.015, 0.015, 0.06), "c4_aggressive": (0.01, 0.034, 0.14)}[arch]
    return {
        "session_id": f"s_{target}_{rep}_{int(damping)}", "episode_role": "candidate",
        "order_in_session": order, "drawer_name": "middle_drawer",
        "candidate_group": f"s_{target}_{rep}_{int(damping)}__{target}",
        "candidate_index": 0 if arch == "c1_steady" else 1,
        "candidate_id": cid, "candidate_archetype": arch, "target_id": target, "replicate_id": rep,
        "matched_group_id": f"{target}__r{rep}", "history_cutoff": 0,
        "x": {"initial_mechanism_joint_pos": 0.0},
        "g": {"target_open_position": 0.2, "target_tolerance": 0.02},
        "theta": {"grasp_offset_local_y": theta[0], "max_pos_step": theta[1], "pull_lead": theta[2]},
        "secret_deployment_state": {"damping": damping},
        "y": {"success": success, "failure_reason": "NONE" if success else "PULL_TIMEOUT",
              "final_joint_position": 0.2 if success else 0.0, "task_outcome_error": err,
              "skill_elapsed_time": time, "phase_durations": {"PULL": 0.5}},
    }


def _bank_episodes():
    eps = []
    o = 0
    for target in ("T012", "T020"):
        for rep in (0, 1, 2):
            for damping in (3.0, 20.0, 40.0):
                # c1_steady: succeeds everywhere, modest time
                eps.append(_ep(target, rep, "c1_steady", damping, True, 0.01, 9.5, o)); o += 1
                # c4_aggressive: fast at low damping, fails at mid/high
                if damping == 3.0:
                    eps.append(_ep(target, rep, "c4_aggressive", damping, True, 0.02, 8.0, o))
                else:
                    eps.append(_ep(target, rep, "c4_aggressive", damping, False, 0.2, 10.0, o))
                o += 1
    return eps


def test_matched_bank_keys_by_target_replicate():
    bank = P.build_matched_bank(_bank_episodes())
    assert bank["matched"]
    assert bank["n_matched_groups"] == 6  # 2 targets x 3 replicates
    for mg in bank["matched_groups"]:
        assert sorted(mg["dampings"]) == [3.0, 20.0, 40.0]
        assert len(mg["candidate_keys"]) == 2


def test_full_validation_passes():
    v = P.full_pairing_validation(_bank_episodes(), expect_n_groups=6, expect_dampings=[3.0, 20.0, 40.0])
    assert v["ok"], [c for c in v["checks"] if not c["ok"]]
    assert v["n_selection_groups"] == 18  # 2 targets x 3 replicates x 3 damping sessions


def test_switch_positive_but_vsi_small():
    bank = P.build_matched_bank(_bank_episodes())
    sw = O.optimal_candidate_switch_rate(bank, CFG)
    vsi = O.value_of_state_information(bank, CFG)
    # at low damping c4 (fast, err .02, time 8) vs c1 (err .01, time 9.5):
    #   U_c4 = 1-.02-.16=0.82 ; U_c1 = 1-.01-.19=0.80 -> c4 wins low, c1 wins mid/high -> switch
    assert sw["switch_rate"] == 1.0
    # but state-agnostic (always c1) is near-optimal -> tiny VSI
    assert 0.0 <= vsi["vsi"] < 0.05


def test_robust_generalist_gap_equals_vsi():
    eps = _bank_episodes()
    bank = P.build_matched_bank(eps)
    groups = {}
    for e in eps:
        groups.setdefault(e["candidate_group"], []).append(e)
    rg = O.robust_generalist(groups, "c1_steady", CFG)
    aware, _ = O._state_aware(bank, CFG)
    gap = float(np.mean(aware)) - rg["mean_true_utility"]
    vsi = O.value_of_state_information(bank, CFG)["vsi"]
    # c1_steady IS the state-agnostic optimum here -> its gap to the state-aware oracle == VSI
    assert abs(gap - vsi) < 1e-9


def test_best_single_archetype_picks_robust():
    eps = _bank_episodes()
    groups = {}
    for e in eps:
        groups.setdefault(e["candidate_group"], []).append(e)
    bs = O.best_single_archetype(groups, groups, CFG)
    assert bs["selected_archetype"] == "c1_steady"
