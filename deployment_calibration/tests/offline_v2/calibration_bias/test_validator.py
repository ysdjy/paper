"""Poison tests for the calibration-bias validator — each injects one leakage and asserts reject."""

import copy

from deployment_calibration.offline_v2.calibration_bias import synthetic
from deployment_calibration.offline_v2.calibration_bias import validator as V
from deployment_calibration.offline_v2.calibration_bias.schema import DEFAULT_FIELDS as FM


def _base():
    return synthetic.make_synthetic(replicates=2)


def test_clean_data_passes():
    v = V.validate(_base(), expect_bias_levels=7)
    assert v["ok"], [c for c in v["checks"] if not c["ok"]]


def test_reject_bias_in_x():
    eps = _base()
    eps[0]["x"]["bias_y"] = 0.03  # leak the hidden state into x
    v = V.validate(eps)
    assert not v["ok"]
    assert any(c["check"] == "no_bias_or_seed_in_x" and not c["ok"] for c in v["checks"])


def test_reject_nuisance_seed_in_x():
    eps = _base()
    eps[0]["x"]["nuisance_seed"] = 1234  # raw seed must not be a feature
    v = V.validate(eps)
    assert not v["ok"]
    assert any(c["check"] == "no_bias_or_seed_in_x" and not c["ok"] for c in v["checks"])


def test_reject_ground_truth_calibration_in_x():
    eps = _base()
    eps[3]["x"]["effective_error"] = 0.01
    v = V.validate(eps)
    assert not v["ok"]


def test_reject_candidate_outcome_in_history():
    # duplicate a candidate as if it were a probe with a lower order -> history would carry an outcome
    eps = _base()
    cand = next(e for e in eps if e["episode_role"] == "candidate")
    fake_probe = copy.deepcopy(cand)
    fake_probe["episode_role"] = "probe"
    fake_probe["order_in_session"] = -1  # earlier than everything
    fake_probe["candidate_index"] = 7    # a candidate-only field leaking in
    fake_probe.pop("probe_index", None)
    eps.append(fake_probe)
    v = V.validate(eps)
    assert not v["ok"]
    # a candidate masquerading as a probe is caught at source (carries candidate-only fields)
    assert any(c["check"] == "probes_have_no_candidate_fields" and not c["ok"] for c in v["checks"])


def test_reject_seed_bias_one_to_one():
    # make nuisance_seed a deterministic function of bias (seed == bias level index) -> rejected
    eps = _base()
    lvl_to_idx = {}
    for e in eps:
        lvl = e["bias_level_id"]
        lvl_to_idx.setdefault(lvl, len(lvl_to_idx))
        e["nuisance_seed"] = lvl_to_idx[lvl]  # every session at a bias shares one seed == 1:1
    ind = V.nuisance_bias_independence(eps)
    assert ind["deterministic_map"] or not ind["ok"]
    v = V.validate(eps)
    assert not v["ok"]
    assert any(c["check"] == "nuisance_seed_independent_of_bias" and not c["ok"] for c in v["checks"])
