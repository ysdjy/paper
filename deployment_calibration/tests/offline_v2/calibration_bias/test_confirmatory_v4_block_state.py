"""B-owned regression for the frozen block-state sampler (offline). No Isaac, no generator, no data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
from deployment_calibration.offline_v2.calibration_bias.residual_nuisance import DEFAULT_RESIDUAL

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def test_status():
    assert P.STATUS == "PREREGISTRATION_V4_BLOCK_STATE_SAMPLER_FROZEN_READY_FOR_C_REAUDIT"
    assert json.loads((_DOCS / "preregistration_v4.json").read_text())["power_recertification_required"] is False


# ---- 1. illegal subseeds rejected ----
def test_validate_subseed_rejects_illegal():
    for bad in (True, False, 1.0, "5", None, -1, 2 ** 63 - 1, 2 ** 63):
        with pytest.raises(BS.BlockStateSamplerError):
            BS.validate_subseed(bad)
    assert BS.validate_subseed(0) == 0 and BS.validate_subseed(2 ** 63 - 2) == 2 ** 63 - 2


# ---- 2. known-answer vectors exact (float.hex) ----
def test_known_answers_exact():
    ka = json.loads((_DOCS / "confirmatory_v4_block_state_known_answers.json").read_text())["vectors"]
    assert len(ka) >= 6
    for v in ka:
        rs = ID.block_residual_subseed(v["split"], v["block_index"])
        ns = ID.block_nuisance_subseed(v["split"], v["block_index"])
        assert rs == v["residual_subseed"] and ns == v["nuisance_subseed"]
        rv = BS.residual_value_from_subseed(rs)
        assert rv.hex() == v["residual_float_hex"]          # bit-exact known answer
        assert rv == v["residual_value"]
        assert BS.nuisance_values_from_subseed(ns) == v["nuisance_values"] == {}


# ---- 3. determinism ----
def test_repeat_bit_identical():
    for split, bi in (("train", 0), ("test", 8), ("validation", 3)):
        assert BS.residual_value(split, bi).hex() == BS.residual_value(split, bi).hex()


# ---- 4. order independence ----
def test_order_independent():
    a1 = BS.residual_value("test", 8)
    _ = [BS.residual_value(s, b) for s in ("train", "validation") for b in ID.BLOCKS[s]]
    a2 = BS.residual_value("test", 8)
    assert a1 == a2


# ---- 5. all 24 residuals in support ----
def test_all_24_in_support():
    n = 0
    for s in ("train", "validation", "test"):
        for b in ID.BLOCKS[s]:
            r = BS.residual_value(s, b)
            assert BS.RESIDUAL_LO <= r <= BS.RESIDUAL_HI and type(r) is float
            n += 1
    assert n == 24


# ---- 6. does not depend on global numpy RNG state ----
def test_independent_of_global_numpy_rng():
    np.random.seed(0); a = BS.residual_value("train", 0)
    np.random.seed(12345); [np.random.random() for _ in range(37)]
    b = BS.residual_value("train", 0)
    assert a == b


# ---- 7. nuisance exactly {} and a fresh dict each call ----
def test_nuisance_none_v1_fresh_dict():
    d1 = BS.nuisance_values("train", 0)
    d2 = BS.nuisance_values("train", 0)
    assert d1 == {} and d2 == {} and d1 is not d2
    assert BS.NUISANCE_POLICY_VERSION == "none_v1"


# ---- 8. reference manifests valid ----
def test_reference_manifests_valid():
    for phase in ("train_validation", "test"):
        m = MI.reference_phase_manifest(phase)
        MI.validate_fully_resolved_phase_manifest(m)
        assert len(m["trials"]) == MI.PHASES[phase]["trials"]


# ---- 9/10/11/12. validator recompute rejections ----
def _test_manifest():
    return MI.reference_phase_manifest("test")


def test_residual_tamper_rejected():
    import copy
    m = _test_manifest()
    good = m["blocks"][0]["residual_value"]
    # 1-ULP change
    m2 = copy.deepcopy(m); m2["blocks"][0]["residual_value"] = np.nextafter(good, 1.0)
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m2)
    # another in-support float that isn't the sampler output
    other = 0.001 if good != 0.001 else 0.002
    m3 = copy.deepcopy(m); m3["blocks"][0]["residual_value"] = other
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m3)
    # int 0 and bool
    for bad in (0, True):
        m4 = copy.deepcopy(m); m4["blocks"][0]["residual_value"] = bad
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(m4)


def test_nuisance_tamper_rejected():
    import copy
    m = _test_manifest()
    for nv in ({"joint_delta": 0.0}, {"x": 1}):
        mm = copy.deepcopy(m); mm["blocks"][0]["nuisance_values"] = nv
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)


def test_subseed_tamper_rejected():
    import copy
    m = _test_manifest()
    for key in ("residual_subseed", "nuisance_subseed"):
        mm = copy.deepcopy(m); mm["blocks"][0][key] = mm["blocks"][0][key] + 1
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(mm)


# ---- 13. reproducible canonical JSON + hash ----
def test_reference_reproducible_hash():
    for phase in ("train_validation", "test"):
        h1 = MI.fully_resolved_phase_manifest_hash(MI.reference_phase_manifest(phase))
        h2 = MI.fully_resolved_phase_manifest_hash(MI.reference_phase_manifest(phase))
        assert h1 == h2


# ---- 14. law matches DEFAULT_RESIDUAL + active config ----
def test_law_matches_default_and_config():
    assert (BS.RESIDUAL_MEAN, BS.RESIDUAL_SIGMA, BS.RESIDUAL_LO, BS.RESIDUAL_HI) == \
        (DEFAULT_RESIDUAL.mean, DEFAULT_RESIDUAL.sigma, DEFAULT_RESIDUAL.lo, DEFAULT_RESIDUAL.hi)
    cr = _cfg()["design"]["residual"]
    assert cr["mean"] == BS.RESIDUAL_MEAN and cr["sigma"] == BS.RESIDUAL_SIGMA
    assert cr["support"] == [BS.RESIDUAL_LO, BS.RESIDUAL_HI]
    bss = _cfg()["block_state_sampler"]
    assert bss["residual"]["version"] == "pcg64_rejection_v1" and bss["nuisance"]["version"] == "none_v1"
    assert bss["nuisance"]["exact_values"] == {} and bss["residual"]["fallback"] is None


# ---- unique production sampler (no second copy) ----
def test_no_second_sampler_in_other_modules():
    import inspect
    # the reference builder must call the sampler, not re-sample
    src = inspect.getsource(MI.reference_phase_manifest) + inspect.getsource(MI.validate_fully_resolved_phase_manifest)
    assert "confirmatory_v4_block_state" in src or "resolved_block_state" in src or \
        "residual_value_from_subseed" in src
    assert "0.001" not in inspect.getsource(MI.reference_phase_manifest)   # placeholder gone
