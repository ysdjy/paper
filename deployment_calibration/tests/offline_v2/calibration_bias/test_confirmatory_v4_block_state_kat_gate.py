"""B-owned regression for the mandatory block-state KAT gate (offline). No Isaac, no generator, no data."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def test_status():
    assert P.STATUS == "PREREGISTRATION_V4_BLOCK_STATE_KAT_GATE_READY_FOR_C_FINAL_REAUDIT"


# ---- 8.1 normal KAT ----
def test_kat_passes_report():
    import numpy as np
    r = BS.validate_known_answer_vectors()
    assert r["passed"] is True and r["vectors_checked"] == 6
    assert r["known_answer_version"] == "pcg64_normal_floathex_kat_v1"
    assert r["sampler_version"] == "pcg64_rejection_v1"
    assert r["numpy_version"] == np.__version__
    assert BS.require_known_answer_compatibility()["passed"] is True


def test_known_answers_json_matches_module_constants():
    ka = json.loads((_DOCS / "confirmatory_v4_block_state_known_answers.json").read_text())
    assert ka["known_answer_version"] == BS.KNOWN_ANSWER_VERSION
    assert ka["production_gate_function"].endswith("require_known_answer_compatibility")
    assert ka["mandatory_before_phase_construction"] is True
    mod = {(s, b): (rs, hx, ns) for s, b, rs, hx, ns in BS.KNOWN_ANSWER_VECTORS}
    assert len(ka["vectors"]) == len(BS.KNOWN_ANSWER_VECTORS) == 6
    for v in ka["vectors"]:
        rs, hx, ns = mod[(v["split"], v["block_index"])]
        assert (v["residual_subseed"], v["residual_float_hex"], v["nuisance_subseed"]) == (rs, hx, ns)
        assert v["nuisance_values"] == {}


# ---- 8.2 hex mismatch blocks everything ----
def _corrupt_vectors():
    v = BS.KNOWN_ANSWER_VECTORS
    return v[:1] + ((v[1][0], v[1][1], v[1][2], "0x0.0p+0", v[1][4]),) + v[2:]


def test_hex_mismatch_blocks_all_entrypoints(monkeypatch):
    monkeypatch.setattr(BS, "KNOWN_ANSWER_VECTORS", _corrupt_vectors())
    with pytest.raises(BS.BlockStateCompatibilityError):
        BS.validate_known_answer_vectors()
    with pytest.raises(BS.BlockStateCompatibilityError):
        BS.residual_value_from_subseed(BS.KNOWN_ANSWER_VECTORS[0][2])
    with pytest.raises(BS.BlockStateCompatibilityError):
        BS.resolved_block_state("train", 0)
    # validator + full hash: fail (KAT gate at top) -> ManifestIntegrityError or the compat error before it
    with pytest.raises((MI.ManifestIntegrityError, BS.BlockStateCompatibilityError)):
        MI.validate_fully_resolved_phase_manifest(MI.reference_phase_manifest("test"))


def test_full_hash_blocked_on_kat_failure(monkeypatch):
    # build a valid manifest FIRST, then break the KAT; hashing must fail (validate precedes hash)
    m = MI.reference_phase_manifest("test")
    monkeypatch.setattr(BS, "KNOWN_ANSWER_VECTORS", _corrupt_vectors())
    with pytest.raises((MI.ManifestIntegrityError, BS.BlockStateCompatibilityError)):
        MI.fully_resolved_phase_manifest_hash(m)


# ---- 8.3 gate not bypassable ----
def test_public_api_gated_and_core_private():
    assert "_residual_value_from_subseed_unchecked" not in BS.__all__
    # public residual functions call the gate; they do NOT sample PCG64 directly
    for fn in (BS.residual_value_from_subseed, BS.residual_value):
        src = inspect.getsource(fn)
        assert "require_known_answer_compatibility" in src or "residual_value_from_subseed" in src
        assert "PCG64" not in src
    # resolved_block_state goes through the gated public API, not the unchecked core
    rsrc = inspect.getsource(BS.resolved_block_state)
    assert "residual_value_from_subseed" in rsrc and "_residual_value_from_subseed_unchecked" not in rsrc
    # validator explicitly invokes the gate
    vsrc = inspect.getsource(MI.validate_fully_resolved_phase_manifest)
    assert "require_known_answer_compatibility" in vsrc


def test_active_spec_forbids_unchecked_core_and_documents_gate():
    bss = _cfg()["block_state_sampler"]
    cg = bss["compatibility_gate"]
    assert cg["function"].endswith("require_known_answer_compatibility")
    assert cg["numpy_version_alone_is_not_the_gate"] is True
    assert cg["on_mismatch"] == "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"
    for step in ("residual draw", "phase manifest construction", "deep manifest validation", "full manifest hash"):
        assert step in cg["mandatory_before"]
    assert "unchecked core" in bss["compatibility_gate"]["note"].lower()


# ---- 8.4 no-cache re-check behaviour ----
def test_no_cache_recheck(monkeypatch):
    assert BS.require_known_answer_compatibility()["passed"] is True   # passes now
    monkeypatch.setattr(BS, "KNOWN_ANSWER_VECTORS", _corrupt_vectors())
    with pytest.raises(BS.BlockStateCompatibilityError):              # immediately re-checked, not cached
        BS.require_known_answer_compatibility()
    monkeypatch.undo()
    assert BS.require_known_answer_compatibility()["passed"] is True   # restored -> passes again


# ---- authoritative description (§9.1) ----
def test_authoritative_checks_describe_kat_and_exact_recompute():
    ch = " ".join(_cfg()["deep_manifest_validator"]["checks"]).lower()
    assert "kat gate" in ch or "known_answer_compatibility" in ch
    assert "residual_value ==" in ch and "residual_value_from_subseed" in ch
    assert "nuisance_values ==" in ch and ("none_v1" in ch or "{}" in ch)
    assert "block-shared residual finite in [-0.01,+0.01]; one nuisance set per block" not in ch


# ---- 9. invariance ----
def test_algorithm_and_power_unchanged():
    from deployment_calibration.offline_v2.calibration_bias.residual_nuisance import DEFAULT_RESIDUAL
    assert BS.MAX_ATTEMPTS == 1000 and BS.NUISANCE_POLICY_VERSION == "none_v1"
    assert (BS.RESIDUAL_MEAN, BS.RESIDUAL_SIGMA, BS.RESIDUAL_LO, BS.RESIDUAL_HI) == \
        (DEFAULT_RESIDUAL.mean, DEFAULT_RESIDUAL.sigma, DEFAULT_RESIDUAL.lo, DEFAULT_RESIDUAL.hi)
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["blocks"] == {"train": 9, "val": 6, "test": 9}
    assert json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())["verdict"] \
        == "POWER_SUFFICIENT_FOR_PREREG_V4"
    # reference still valid (24 residuals in support, all recomputable)
    m = MI.reference_phase_manifest("train_validation")
    MI.validate_fully_resolved_phase_manifest(m)
    assert all(-0.01 <= b["residual_value"] <= 0.01 for b in m["blocks"])
