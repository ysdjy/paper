"""Claude C block-state KAT-gate final re-audit — independent, read-only.

Regresses ONLY the two previously-INCOMPLETE items: the mandatory known-answer compatibility gate and the
authoritative-description sync. All PASS -> the sampler blocker is closed. Do NOT modify B files or prior
audits. Run in env_isaaclab (numpy+torch).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[4] / "docs" / "offline_v2" / "calibration_bias"


def _BS():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
    return BS


def _MI():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
    return MI


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


# ============================ §2 frozen literals + JSON parity ============================
def test_2_frozen_known_answer_literals():
    BS = _BS()
    assert BS.KNOWN_ANSWER_VERSION == "pcg64_normal_floathex_kat_v1"
    assert BS.RESIDUAL_SAMPLER_VERSION == "pcg64_rejection_v1"
    assert len(BS.KNOWN_ANSWER_VECTORS) == 6
    ids = {(s, b) for s, b, *_ in BS.KNOWN_ANSWER_VECTORS}
    assert ids == {("train", 0), ("train", 8), ("validation", 0), ("validation", 5), ("test", 0), ("test", 8)}
    ka = json.loads((_DOCS / "confirmatory_v4_block_state_known_answers.json").read_text())
    jmap = {(v["split"], v["block_index"]): (v["residual_subseed"], v["residual_float_hex"], v["nuisance_subseed"])
            for v in ka["vectors"]}
    for s, b, rs, hx, ns in BS.KNOWN_ANSWER_VECTORS:  # module literals == JSON, field-by-field
        assert jmap[(s, b)] == (rs, hx, ns)


# ============================ §3 unchecked core private, gated public ============================
def test_3_unchecked_core_private_and_gated():
    BS = _BS()
    assert "_residual_value_from_subseed_unchecked" not in BS.__all__
    for public in ("residual_value_from_subseed", "residual_value", "resolved_block_state",
                   "require_known_answer_compatibility", "validate_known_answer_vectors"):
        assert public in BS.__all__
    # resolved_block_state / residual_value go through the gated API (KAT runs)
    assert BS.resolved_block_state("test", 0)["nuisance_values"] == {}


# ============================ §4 KAT + drift blocking ============================
def test_4_normal_report():
    import numpy as np
    r = _BS().validate_known_answer_vectors()
    assert r["passed"] and r["known_answer_version"] == "pcg64_normal_floathex_kat_v1"
    assert r["sampler_version"] == "pcg64_rejection_v1" and r["vectors_checked"] == 6
    assert r["numpy_version"] == np.__version__


def _inject_and_check_all_blocked(patch, restore):
    """Apply a drift patch, assert every production path raises, then restore."""
    BS, MI = _BS(), _MI()
    m = MI.reference_phase_manifest("test")  # valid under normal KAT (restore happens after)
    MI.validate_fully_resolved_phase_manifest(m)
    patch()
    try:
        for fn in (lambda: BS.validate_known_answer_vectors(),
                   lambda: BS.require_known_answer_compatibility(),
                   lambda: BS.residual_value("test", 0),
                   lambda: BS.resolved_block_state("test", 0),
                   lambda: MI.reference_phase_manifest("test"),
                   lambda: MI.validate_fully_resolved_phase_manifest(m),
                   lambda: MI.fully_resolved_phase_manifest_hash(m)):
            with pytest.raises((BS.BlockStateCompatibilityError, BS.BlockStateSamplerExhausted,
                                MI.ManifestIntegrityError)):
                fn()
    finally:
        restore()


def test_41_hex_mismatch_blocks_all_and_diagnostics():
    import numpy as np
    BS = _BS()
    orig = BS.KNOWN_ANSWER_VECTORS
    bad = list(orig); s, b, rs, hx, ns = bad[0]; bad[0] = (s, b, rs, "0x1.5555555555555p-9", ns)
    _inject_and_check_all_blocked(lambda: setattr(BS, "KNOWN_ANSWER_VECTORS", tuple(bad)),
                                  lambda: setattr(BS, "KNOWN_ANSWER_VECTORS", orig))
    BS.KNOWN_ANSWER_VECTORS = tuple(bad)
    try:
        BS.validate_known_answer_vectors()
        assert False
    except BS.BlockStateCompatibilityError as e:
        msg = str(e)
        assert f"({s},{b})" in msg and "0x1.5555555555555p-9" in msg and "actual" in msg
        assert np.__version__ in msg and "pcg64_rejection_v1" in msg and "pcg64_normal_floathex_kat_v1" in msg
    finally:
        BS.KNOWN_ANSWER_VECTORS = orig


def test_42_subseed_literal_mismatch_blocks():
    BS = _BS()
    orig = BS.KNOWN_ANSWER_VECTORS
    bad = list(orig); s, b, rs, hx, ns = bad[0]; bad[0] = (s, b, rs + 1, hx, ns)
    _inject_and_check_all_blocked(lambda: setattr(BS, "KNOWN_ANSWER_VECTORS", tuple(bad)),
                                  lambda: setattr(BS, "KNOWN_ANSWER_VECTORS", orig))


def test_43_version_drift_blocks():
    BS = _BS()
    for attr, bad in (("RESIDUAL_SAMPLER_VERSION", "x"), ("KNOWN_ANSWER_VERSION", "y")):
        orig = getattr(BS, attr)
        _inject_and_check_all_blocked(lambda a=attr, v=bad: setattr(BS, a, v),
                                      lambda a=attr, o=orig: setattr(BS, a, o))
    assert BS.NUISANCE_POLICY_VERSION == "none_v1" and BS.nuisance_values_from_subseed(0) == {}


def test_44_nuisance_drift_blocks():
    BS = _BS()
    orig = BS.nuisance_values_from_subseed
    _inject_and_check_all_blocked(lambda: setattr(BS, "nuisance_values_from_subseed", lambda ns: {"x": 1.0}),
                                  lambda: setattr(BS, "nuisance_values_from_subseed", orig))


def test_45_core_exhaustion_blocks_with_invalid_verdict():
    BS, MI = _BS(), _MI()
    orig = BS._residual_value_from_subseed_unchecked

    def boom(subseed):
        raise BS.BlockStateSamplerExhausted("forced")
    BS._residual_value_from_subseed_unchecked = boom
    try:
        with pytest.raises((BS.BlockStateSamplerExhausted, BS.BlockStateCompatibilityError)) as ei:
            BS.validate_known_answer_vectors()
        assert getattr(ei.value, "verdict", "") == "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"
        with pytest.raises((BS.BlockStateSamplerExhausted, MI.ManifestIntegrityError)):
            MI.fully_resolved_phase_manifest_hash(MI.reference_phase_manifest("test"))
    finally:
        BS._residual_value_from_subseed_unchecked = orig


# ============================ §5 mandatory, no cache ============================
def test_5_no_success_cache():
    BS = _BS()
    orig = BS.KNOWN_ANSWER_VECTORS
    assert BS.require_known_answer_compatibility()["passed"]
    bad = list(orig); s, b, rs, hx, ns = bad[0]; bad[0] = (s, b, rs, "0x1.0p-9", ns)
    BS.KNOWN_ANSWER_VECTORS = tuple(bad)
    try:
        with pytest.raises(BS.BlockStateCompatibilityError):
            BS.require_known_answer_compatibility()   # not masked by the earlier success
    finally:
        BS.KNOWN_ANSWER_VECTORS = orig
    assert BS.require_known_answer_compatibility()["passed"]


# ============================ §6 reference 228/72 + gate in validator/hash ============================
def test_6_reference_valid_and_reproducible():
    MI = _MI()
    for phase, n in (("train_validation", 228), ("test", 72)):
        m = MI.reference_phase_manifest(phase)
        MI.validate_fully_resolved_phase_manifest(m)
        assert len(m["trials"]) == n
        assert MI.fully_resolved_phase_manifest_hash(m) == MI.fully_resolved_phase_manifest_hash(
            MI.reference_phase_manifest(phase))


# ============================ §7 authoritative description sync ============================
def test_7_authoritative_checks_and_config_gate():
    d = _cfg()
    ch = " ".join(d["deep_manifest_validator"]["checks"]).lower()
    assert "mandatory block-state kat" in ch and "require_known_answer_compatibility" in ch
    assert "residual_value == bs.residual_value_from_subseed" in ch
    assert "nuisance_values == bs.nuisance_values_from_subseed" in ch and "none_v1 {}" in ch
    g = d["block_state_sampler"]["compatibility_gate"]
    assert g["function"].endswith("require_known_answer_compatibility") and g["vectors"] == 6
    assert g["comparison"] == "Python float.hex() exact" and g["numpy_version_alone_is_not_the_gate"] is True
    assert g["on_mismatch"] == "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"
    assert set(("deep manifest validation", "full manifest hash")).issubset(set(g["mandatory_before"]))


# ============================ §9 invariance ============================
def test_9_invariants_unchanged():
    BS = _BS()
    assert BS.MAX_ATTEMPTS == 1000 and BS.NUISANCE_POLICY_VERSION == "none_v1"
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert d["residual"] == {"dist": "TruncatedNormal", "mean": 0.0, "sigma": 0.005,
                             "support": [-0.01, 0.01], "unit": "m"}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
