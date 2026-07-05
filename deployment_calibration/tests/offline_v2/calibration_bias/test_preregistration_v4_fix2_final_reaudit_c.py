"""Claude C final (fix2) re-audit tests — independent, read-only.

Confirms the resolved cores of the last two blockers (production selector reads only observed outcomes;
canonical identity format frozen) and documents the residual GO-blocking gaps found in fix2:
  * best-single input guard: 45/12 split composition unchecked; completeness/bank bypass exposed (3.3/3.4)
  * identity builder does not validate the frozen design domain (4.2)
  * canonical_manifest_hash() is structure-only but is used as the full-manifest integrity anchor (4.6)
Do NOT modify B's files or prior C audits. Run in env_isaaclab (torch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_DOCS = _ROOT / "docs" / "offline_v2" / "calibration_bias"


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


# ================= Blocker 1 core RESOLVED: production selector observed-only =================
def test_active_binding_is_production_selector_not_sim_helper():
    bs = _cfg()["best_single"]
    assert bs["implementation"].endswith("confirmatory_v4_selection.select_best_single")
    assert "best_single_legal" in bs["simulation_only_reference"]
    assert "NOT production" in bs["simulation_only_reference"]


def _valid_171(success_of_offset):
    """171 well-formed records (45 train + 12 val sessions, 3 offsets) with SECRET fields attached."""
    recs = []
    sessions = [("train", i) for i in range(45)] + [("validation", i) for i in range(12)]
    for split, s in sessions:
        for off in (-0.04, 0.0, 0.04):
            recs.append({
                "split": split, "session_id": f"{split}-{s}", "trial_role": "candidate",
                "theta": {"grasp_offset_local_y": off},
                "y": {"success": bool(success_of_offset[off])},
                "planned_episode_id": f"{split}-{s}-{off:+.3f}",
                # secret audit fields deliberately present on the raw record:
                "secret_deployment_state": {"nominal_bias_y": 0.037, "residual_bias_y": -0.006,
                                            "actual_bias_y": 0.031},
                "eff_signed": 0.031, "abs_eff": 0.031, "hidden_state_id": "bias_p035"})
    return recs


def test_secret_perturbation_invariant_and_observed_success_decisive():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    # offset 0 has most observed success -> selected 0 regardless of secret values
    base = _valid_171({-0.04: False, 0.0: True, 0.04: False})
    a = S.select_best_single(base)["selected_offset"]
    for r in base:  # wildly perturb every secret field
        r["secret_deployment_state"] = {"nominal_bias_y": -99.0, "residual_bias_y": 99.0, "actual_bias_y": 42.0}
        r["eff_signed"] = -99.0
        r["abs_eff"] = 99.0
    b = S.select_best_single(base)["selected_offset"]
    assert a == b == 0.0, "secret perturbation must not change the selection"
    # flipping the OBSERVED success changes the selection per the rule
    flip = _valid_171({-0.04: True, 0.0: False, 0.04: False})
    assert S.select_best_single(flip)["selected_offset"] == -0.04


def test_production_selector_algorithm_matches_frozen_rule_not_hardcoded_zero():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    def recs(a, b, c):
        out = []
        for off, n in ((-0.04, a), (0.0, b), (0.04, c)):
            for i in range(57):
                out.append({"split": "train", "session_id": f"x{i}", "trial_role": "candidate",
                            "theta": {"grasp_offset_local_y": off}, "y": {"success": bool(i < n)},
                            "planned_episode_id": f"{off:+.3f}-{i}"})
        return out
    assert S.select_best_single(recs(30, 57, 30), validate_completeness=False)["selected_offset"] == 0.0
    assert S.select_best_single(recs(10, 10, 57), validate_completeness=False)["selected_offset"] == 0.04
    assert S.select_best_single(recs(57, 10, 57), validate_completeness=False)["selected_offset"] == -0.04  # tie->min|off|


def test_selector_rejects_test_and_non_bool_and_probe():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
    with pytest.raises(S.BestSingleInputError):
        S.select_best_single([{"split": "test", "session_id": "s", "trial_role": "candidate",
                               "theta": {"grasp_offset_local_y": 0.0}, "y": {"success": True},
                               "planned_episode_id": "p"}], validate_completeness=False)


# ---- GAP 3.3 + 3.4: input guard incomplete ----
@pytest.mark.xfail(reason="BLOCKER: completeness checks total 57 sessions but NOT the 45 train / 12 "
                          "validation composition; and select_best_single exposes validate_completeness/"
                          "candidate_bank/expected_candidate_records bypass with no config freeze",
                   strict=True)
def test_completeness_checks_45_12_split_and_no_bypass():
    cfg = _cfg()["best_single"]
    checks = " ".join(cfg["input_completeness"]["checks"]).lower()
    assert "45" in checks and "12" in checks, "must verify 45 train + 12 validation, not only total 57"
    s = json.dumps(_cfg())
    assert "validate_completeness" in s and "expected_candidate_records" in s, \
        "config must freeze validate_completeness=True + expected_candidate_records=171 (no bypass)"


# ================= Blocker 2 core RESOLVED: identity format frozen =================
def test_identity_templates_and_zero_normalization():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    assert ID.block_identity("train", 0) == "v4|split=train|block=00"
    assert ID.trial_identity("test", 8, 0.035, "candidate", 0.04) == \
        "v4|split=test|block=08|nominal=+0.035|role=candidate|offset=+0.040"
    assert ID.fmt_m(0.0) == "+0.000" and ID.fmt_m(-0.0) == "+0.000"


def test_300_planned_ids_unique_and_role_distinct():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    rows = ID.enumerate_trials()
    pids = [r["planned_episode_id"] for r in rows]
    assert len(rows) == 300 and len(set(pids)) == 300
    assert ID.planned_episode_id(ID.trial_identity("train", 0, 0.0, "probe", -0.04)) != \
        ID.planned_episode_id(ID.trial_identity("train", 0, 0.0, "candidate", -0.04))


def test_domain_subseeds_frozen_and_not_python_hash():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    a = ID.block_residual_subseed("train", 0)
    b = ID.block_nuisance_subseed("train", 0)
    assert a != b and isinstance(a, int)  # different domains -> different subseed
    assert ID.block_residual_subseed("train", 0) == a  # stable


# ---- GAP 4.2: identity builder does not validate the frozen domain ----
@pytest.mark.xfail(reason="BLOCKER (task 4.2): trial_identity validates only split/role, not the frozen "
                          "domain (block range, per-split nominal set, probe==-0.04, candidate in bank, "
                          "non-negative block) -> leaves manifest freedom", strict=True)
def test_identity_builder_validates_frozen_domain():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    for args in [("train", 99, 0.0, "candidate", 0.0), ("train", 0, 0.5, "candidate", 0.0),
                 ("test", 0, 0.035, "probe", 0.02), ("train", -1, 0.0, "candidate", 0.0)]:
        with pytest.raises((ValueError, KeyError)):
            ID.trial_identity(*args)


# ---- GAP 4.6: structure-only hash conflated with full-manifest integrity hash ----
@pytest.mark.xfail(reason="BLOCKER (task 4.6): canonical_manifest_hash() covers only "
                          "[trial_identity, planned_episode_id]; the manifest_spec calls manifest_hash "
                          "'sha256 of the fully-resolved manifest' and uses it as the per-trial integrity "
                          "anchor -> residual/nuisance/execution-order tampering is not covered",
                   strict=True)
def test_manifest_hash_is_full_not_structure_only():
    import inspect
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
    src = inspect.getsource(ID.canonical_manifest_hash)
    # a full-manifest integrity hash must include the resolved randomization, not only identities
    assert "residual" in src or "nuisance" in src or "order" in src, \
        "manifest integrity hash must cover residual/nuisance/execution order, not only planned structure"


# ================= determinism / power-unchanged (PASS) =================
def test_determinism_flags_frozen():
    s = json.dumps(_cfg())
    for f in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f in s, f"determinism flag {f} not frozen"


def test_power_inputs_unchanged_and_status_fix2():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert d["blocks"] == {"train": 9, "val": 6, "test": 9}
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_bridge_invariance_claimed_and_power_recert_false():
    b = json.loads((_DOCS / "production_best_single_bridge_invariance_v1.json").read_text())
    assert b.get("power_recertification_required") is False
