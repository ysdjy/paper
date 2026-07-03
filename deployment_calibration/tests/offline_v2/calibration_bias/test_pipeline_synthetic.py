"""Synthetic pipeline recovery tests.

Confirms the offline pipeline recovers the RIGHT verdict on controllable synthetic worlds:
  * positive world (compensation, no robust offset)   -> gate PASSES, VSI>0, switch>0, history gain
  * negative world (wide tolerance, one offset robust) -> gate FAILS, VSI~0, robust offset exists
  * deterministic replicates (no nuisance noise, shared seed) -> independence BLOCKER
All synthetic; never a paper result.
"""

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import independence as IND
from deployment_calibration.offline_v2.calibration_bias import oracle as CBO
from deployment_calibration.offline_v2.calibration_bias import synthetic
from deployment_calibration.offline_v2.calibration_bias.schema import DEFAULT_FIELDS as FM
from deployment_calibration.offline_v2.utility import UtilityConfig

FROZEN = UtilityConfig(lambda_error=1.0, lambda_time=0.02)
SO = UtilityConfig(lambda_error=0.0, lambda_time=0.0)


def test_positive_world_has_decision_value():
    eps = synthetic.make_synthetic(grasp_tol=0.02)
    dv = CBO.decision_value(eps, FROZEN, success_only=SO)
    assert dv["VSI_frozen"] > 0.05
    assert dv["optimal_candidate_switch_rate"] > 0.0
    assert dv["pairwise_rank_reversal_rate"] > 0.0
    assert dv["robust_offset"]["robust_generalist_exists"] is False
    assert dv["success_only"]["VSI_success_only"] > 0.05
    assert dv["best_offset_per_bias"]["groups_with_best_offset_differing_across_bias"] > 0


def test_negative_world_no_decision_value():
    # wide grasp tolerance + flat (tiny-capped) error -> the same offset (fastest, ~0) is best at
    # every bias -> a robust generalist exists and knowing bias buys ~nothing.
    eps = synthetic.make_synthetic(grasp_tol=0.5, err_cap=0.005, noise=0.0)
    dv = CBO.decision_value(eps, FROZEN, success_only=SO)
    assert dv["robust_offset"]["robust_generalist_exists"] is True
    assert dv["VSI_frozen"] < 0.05
    assert dv["success_only"]["VSI_success_only"] < 0.05


def test_history_reveals_bias_via_deepsets():
    # capacity control that WORKS for the nonlinear compensation task: DeepSets K=0 vs K>0.
    from deployment_calibration.evaluation.offline_v2.pipeline import build_pairs, groups_from_pairs
    from deployment_calibration.models_v2.train import ensemble_predict, train_seeds
    from deployment_calibration.offline_v2.metrics import auroc
    eps = synthetic.make_synthetic(grasp_tol=0.02)
    sess = {e["session_id"] for e in eps if e.get("episode_role") == "candidate"}
    y = [1.0 if e["y"]["success"] else 0.0 for e in eps if e.get("episode_role") == "candidate"]

    def auroc_at(k):
        te = build_pairs(eps, sess, k=k)
        fit = train_seeds("B2_deepsets", te, None, seeds=(0, 1))
        yk = [1.0 if e["y"]["success"] else 0.0 for e, _ in te]
        pk = [ensemble_predict(fit, e, H)["p_success"] for e, H in te]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return auroc(yk, pk)
    a0, a2 = auroc_at(0), auroc_at(2)
    # with no history the band centre (bias) is unknown -> weak; with probes it is revealed
    assert a2 > a0 + 0.2


def test_independence_blocker_on_deterministic_replicates():
    # zero nuisance noise AND a shared seed per bias -> technical repeats -> blocker
    eps = synthetic.make_synthetic(noise=0.0, replicates=3)
    for e in eps:
        e["nuisance_seed"] = e["bias_level_id"]  # same seed for all sessions of a bias
    ind = IND.audit(eps)
    assert ind["near_deterministic"]
    assert ind["blocker"]


def test_synthetic_flag_present():
    eps = synthetic.make_synthetic(replicates=2)
    assert all(e.get("synthetic_only") for e in eps)


# ---- exploration gate criterion 3 is a CONJUNCTION (success-only AND frozen VSI) ----
from deployment_calibration.evaluation.offline_v2.calibration_bias import pipeline as PIPE


def _dv(succ_gain, vsi, robust=False):
    return {"success_only": {"state_aware_selected_success_gain": succ_gain},
            "VSI_frozen": vsi, "robust_offset": {"robust_generalist_exists": robust,
                                                  "worst_case_gap_of_most_robust": 0.5}}


_OKV = {"ok": True}
_NOBLOCK = {"blocker": False}


def test_gate_requires_both_success_gain_and_vsi():
    # both above threshold -> criterion 3 passes
    g = PIPE.exploration_gate(_dv(0.3, 0.2), _NOBLOCK, _OKV, n_distinct_best=4)
    assert g["criteria"]["3_success_gain>=0.15_AND_vsi>=0.05"]["ok"]
    assert g["gate_passed"]


def test_gate_fails_on_time_only_vsi_without_success_gain():
    # frozen VSI high (from time/error) but success-only gain ~0 -> criterion 3 FAILS (AND, not OR)
    g = PIPE.exploration_gate(_dv(0.0, 0.2), _NOBLOCK, _OKV, n_distinct_best=4)
    c3 = g["criteria"]["3_success_gain>=0.15_AND_vsi>=0.05"]
    assert c3["frozen_vsi_ok(>=0.05)"] and not c3["success_only_gain_ok(>=0.15)"]
    assert not c3["ok"]
    assert not g["gate_passed"]


def test_gate_fails_on_success_gain_without_vsi():
    # success gain high but frozen VSI below threshold -> criterion 3 FAILS
    g = PIPE.exploration_gate(_dv(0.3, 0.01), _NOBLOCK, _OKV, n_distinct_best=4)
    c3 = g["criteria"]["3_success_gain>=0.15_AND_vsi>=0.05"]
    assert c3["success_only_gain_ok(>=0.15)"] and not c3["frozen_vsi_ok(>=0.05)"]
    assert not g["gate_passed"]
