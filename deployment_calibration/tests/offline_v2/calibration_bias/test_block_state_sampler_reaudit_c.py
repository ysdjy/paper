"""Claude C block-state sampler re-audit — independent, read-only.

Narrow regression of the GENERATOR_IMPLEMENTATION_BLOCKED_MISSING_FROZEN_BLOCK_STATE_SAMPLER closure.
PASS = the resolved parts (sampler determinism, KAT recompute, validator exact-recompute, reference,
production uniqueness, power invariance). strict-xfail = the two remaining gaps of the SAME blocker:
  * §4 no mandatory production known-answer/version gate (drift protection is test-only).
  * §9.1 the authoritative deep_manifest_validator.checks omits the residual/nuisance exact-recompute.
Do NOT modify B files or prior audits. Run in env_isaaclab (numpy+torch).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[4] / "docs" / "offline_v2" / "calibration_bias"


def _BS():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
    return BS


def _ID():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    return ID


def _MI():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    return MI


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


_BLOCKS = [("train", i) for i in range(9)] + [("validation", i) for i in range(6)] + [("test", i) for i in range(9)]


# ============================ RESOLVED parts ============================
def test_law_consistent_across_three_sources():
    from deployment_calibration.offline_v2.calibration_bias.residual_nuisance import DEFAULT_RESIDUAL
    BS = _BS()
    cr = _cfg()["design"]["residual"]
    assert (BS.RESIDUAL_MEAN, BS.RESIDUAL_SIGMA, BS.RESIDUAL_LO, BS.RESIDUAL_HI) == \
        (DEFAULT_RESIDUAL.mean, DEFAULT_RESIDUAL.sigma, DEFAULT_RESIDUAL.lo, DEFAULT_RESIDUAL.hi) == \
        (cr["mean"], cr["sigma"], cr["support"][0], cr["support"][1])


def test_kat_six_vectors_recompute_exact():
    BS, ID = _BS(), _ID()
    ka = json.loads((_DOCS / "confirmatory_v4_block_state_known_answers.json").read_text())
    for v in ka["vectors"]:
        rs = ID.block_residual_subseed(v["split"], v["block_index"])
        ns = ID.block_nuisance_subseed(v["split"], v["block_index"])
        assert rs == v["residual_subseed"] and ns == v["nuisance_subseed"]
        assert BS.residual_value_from_subseed(rs).hex() == v["residual_float_hex"]
        assert BS.nuisance_values_from_subseed(ns) == {}


def test_24_block_determinism_order_and_global_rng_independent():
    import numpy as np
    BS = _BS()
    a = {b: BS.residual_value(*b) for b in _BLOCKS}
    assert len(a) == 24 and all(-0.01 <= x <= 0.01 for x in a.values())
    np.random.seed(999)
    [np.random.random() for _ in range(50)]
    b = {k: BS.residual_value(*k) for k in reversed(_BLOCKS)}
    assert a == b


def test_subseed_domain():
    BS = _BS()
    for bad in (True, 1.0, None, -1, 2 ** 63 - 1):
        with pytest.raises(BS.BlockStateSamplerError):
            BS.validate_subseed(bad)
    assert BS.validate_subseed(0) == 0 and BS.validate_subseed(2 ** 63 - 2) == 2 ** 63 - 2


def test_nuisance_none_v1():
    BS = _BS()
    assert BS.NUISANCE_POLICY_VERSION == "none_v1"
    d1 = BS.nuisance_values_from_subseed(0)
    d2 = BS.nuisance_values_from_subseed(0)
    assert d1 == {} and d1 is not d2  # fresh dict each call


def test_validator_exact_recompute_rejects_tampering():
    MI, BS = _MI(), _BS()
    m = MI.reference_phase_manifest("test")
    MI.validate_fully_resolved_phase_manifest(m)
    rv = m["blocks"][0]["residual_value"]
    assert type(rv) is float and rv == BS.residual_value_from_subseed(m["blocks"][0]["residual_subseed"])

    def rejects(mut):
        mm = copy.deepcopy(m); mut(mm)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)
    rejects(lambda mm: mm["blocks"][0].__setitem__("residual_value", math.nextafter(rv, 1.0)))  # 1 ULP
    rejects(lambda mm: mm["blocks"][0].__setitem__("residual_value", 0.005))                     # other in support
    rejects(lambda mm: mm["blocks"][0].__setitem__("residual_value", 0))                         # int 0
    rejects(lambda mm: mm["blocks"][0].__setitem__("residual_value", True))                      # bool
    rejects(lambda mm: mm["blocks"][0].__setitem__("nuisance_values", {"joint_delta": 0.0}))     # extra key
    rejects(lambda mm: mm["blocks"][0].__setitem__("residual_subseed", m["blocks"][0]["residual_subseed"] + 1))


def test_reference_uses_sampler_no_placeholders():
    MI = _MI()
    for phase, n in (("train_validation", 228), ("test", 72)):
        m = MI.reference_phase_manifest(phase)
        MI.validate_fully_resolved_phase_manifest(m)
        assert len(m["trials"]) == n
        assert all(b["nuisance_values"] == {} for b in m["blocks"])
        assert all(b["residual_value"] != 0.001 for b in m["blocks"])  # exact placeholder gone
    assert "joint_delta" not in json.dumps(MI.reference_phase_manifest("test"))


def test_production_sampler_uniqueness():
    # the validator + reference both bind to confirmatory_v4_block_state; config designates one module
    src = (Path(__file__).resolve().parents[3] / "offline_v2" / "calibration_bias"
           / "confirmatory_v4_manifest_integrity.py").read_text()
    assert "confirmatory_v4_block_state" in src and "resolved_block_state" in src
    assert _cfg()["block_state_sampler"]["module"].endswith("confirmatory_v4_block_state")


def test_power_invariants_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert d["residual"] == {"dist": "TruncatedNormal", "mean": 0.0, "sigma": 0.005,
                             "support": [-0.01, 0.01], "unit": "m"}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"


# ============================ INCOMPLETE (same blocker) ============================
@pytest.mark.xfail(reason="§4 INCOMPLETE: no MANDATORY production known-answer/version gate. There is no "
                          "validate_known_answer_vectors (or equivalent) that the generator must run before "
                          "manifest construction, and no BLOCKING NumPy-version freeze; drift protection is "
                          "test-only, and the validator's self-recompute uses the same NumPy so it cannot "
                          "detect a Generator.normal transform change.", strict=True)
def test_mandatory_known_answer_or_version_gate_exists():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    names = set(dir(BS)) | set(dir(MI))
    has_kat_fn = any("known_answer" in n.lower() or "kat" in n.lower() for n in names)
    cfg = _cfg()
    blob = json.dumps(cfg).lower()
    has_version_freeze = "numpy_version_frozen" in blob or "require_numpy" in blob or \
        cfg.get("block_state_sampler", {}).get("numpy_version_frozen") is not None
    assert has_kat_fn or has_version_freeze


@pytest.mark.xfail(reason="§9.1 INCOMPLETE: the authoritative deep_manifest_validator.checks still reads only "
                          "'block-shared residual finite in [-0.01,+0.01]; one nuisance set per block' and "
                          "omits 'residual_value == frozen block-state sampler output' and 'nuisance_values "
                          "== none_v1 {}' (the code does exact recompute; the authoritative description does "
                          "not reflect it).", strict=True)
def test_authoritative_checks_describe_exact_recompute():
    ch = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert ("residual_value ==" in ch and ("sampler" in ch or "residual_value_from_subseed" in ch))
    assert ("nuisance_values ==" in ch and ("none_v1" in ch or "{}" in ch))
