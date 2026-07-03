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


def test_reject_block_bias_one_to_one():
    # make the nuisance BLOCK a deterministic function of bias (block == bias level) -> rejected
    eps = _base()
    for e in eps:
        e["nuisance_block_id"] = e["bias_level_id"]  # one block per bias, 1:1 -> bias-encoding
    ind = V.nuisance_bias_independence(eps)
    assert ind["deterministic_map"]
    assert not ind["ok"]
    v = V.validate(eps)
    assert not v["ok"]
    assert any(c["check"] == "nuisance_seed_independent_of_bias" and not c["ok"] for c in v["checks"])


def test_exploration_paired_blocks_are_independence_legal():
    # exploration PAIRED design: the same block reused across ALL bias levels (matched context).
    # This must NOT be flagged as leakage — reuse-across-bias is healthy, only 1:1 is bad.
    eps = synthetic.make_synthetic(replicates=3, paired_blocks=True)
    ind = V.nuisance_bias_independence(eps)
    assert ind["paired_reuse_across_bias"]
    assert not ind["deterministic_map"]
    assert ind["ok"]
    v = V.validate(eps, expect_bias_levels=7)
    assert v["ok"], [c for c in v["checks"] if not c["ok"]]
