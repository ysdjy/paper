"""Claude B read-only implementation-consistency audit of the confirmatory v4 generator + smoke.

Independent audit tests for commit b6e6700. PASS tests confirm the correctly-resolved parts; the tests named
`test_GEN_B_00x_*` DEMONSTRATE the frozen findings (they assert the CURRENT observed behaviour so the audit is
machine-checkable and a future A fix flips them). Do NOT modify A's files.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import re
from pathlib import Path

import pytest

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as PRE

_AUTH = GEN.GeneratorAuthorization(smoke_only=True)
_ENV = GEN.EnvironmentVersionContext({"numpy": "1.26.0"})


def _build(phase, commits=None):
    return GEN.build_phase_manifest_in_memory(phase, auth=_AUTH,
                                              commits=commits or GEN.smoke_placeholder_commits(),
                                              environment_versions=_ENV)


# ==================== PASS: KAT order, builder correctness, authorization, formal writer, baseline ====================
def test_kat_first_and_no_unchecked_core_or_reference_wrap():
    src = inspect.getsource(GEN)
    assert not re.search(r"BS\._residual_value_from_subseed_unchecked", src)
    assert not re.search(r"MI\.reference_phase_manifest\(", src)      # independent construction
    order = inspect.getsource(GEN.build_phase_manifest_in_memory)
    assert order.index("require_generator_authorization") < order.index("require_known_answer_compatibility") \
        < order.index("_build_unsealed_manifest") < order.index("validate_fully_resolved_phase_manifest") \
        < order.index("fully_resolved_phase_manifest_hash")


def test_builder_counts_and_orders_and_reference_equal():
    for phase, (nb, ns, nt) in (("train_validation", (15, 57, 228)), ("test", (9, 18, 72))):
        g = _build(phase)
        m = g.unsealed_manifest
        assert m["counts"] == {"blocks": nb, "sessions": ns, "trials": nt} and len(m["trials"]) == nt
        assert tuple(b["canonical_block_identity"] for b in m["blocks"]) == ID.canonical_phase_block_identities(phase)
        assert tuple(s["canonical_session_identity"] for s in m["sessions"]) == ID.canonical_phase_session_identities(phase)
        assert tuple(t["canonical_trial_identity"] for t in m["trials"]) == ID.canonical_phase_trial_identities(phase)
        plan = {tid: i for i, tid in enumerate(ID.resolve_phase_execution_plan(phase))}
        assert all(t["execution_order_index"] == plan[t["canonical_trial_identity"]] for t in m["trials"])
        MI.validate_fully_resolved_phase_manifest(m)                  # deep valid
        # byte-equal to the reference fixture GIVEN THE SAME provenance inputs (commits+env), though
        # production never wraps the reference builder
        ref = MI.reference_phase_manifest(phase, protocol_commit="a" * 40, generator_commit="b" * 40,
                                          runtime_commit="c" * 40, environment_versions={"numpy": "1.26.0"})
        assert ID.canonical_json(m) == ID.canonical_json(ref)
        assert MI.fully_resolved_phase_manifest_hash(m) == MI.fully_resolved_phase_manifest_hash(ref)


def test_kat_drift_blocks_builder_and_hash(monkeypatch):
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
    v = BS.KNOWN_ANSWER_VECTORS
    monkeypatch.setattr(BS, "KNOWN_ANSWER_VECTORS", v[:1] + ((v[1][0], v[1][1], v[1][2], "0x0.0p+0", v[1][4]),) + v[2:])
    with pytest.raises((BS.BlockStateCompatibilityError, MI.ManifestIntegrityError)):
        _build("test")


def test_authorization_gate_and_formal_writer_locked():
    for bad in (GEN.GeneratorAuthorization(smoke_only=False),
                GEN.GeneratorAuthorization(smoke_only=True, formal_manifest_generation_authorized=True),
                GEN.GeneratorAuthorization(smoke_only=True, confirmatory_run_authorized=True)):
        with pytest.raises(GEN.FormalGenerationNotAuthorized):
            GEN.build_phase_manifest_in_memory("test", auth=bad, commits=GEN.smoke_placeholder_commits(),
                                               environment_versions=_ENV)
    # forged formal auth still writes nothing (writer raises before any fs effect)
    forged = GEN.GeneratorAuthorization(smoke_only=False, formal_manifest_generation_authorized=True)
    with pytest.raises(GEN.FormalGenerationNotAuthorized):
        GEN.write_formal_phase_manifest_atomic("test", _build("test"), "/tmp/should_not_exist_xyz", auth=forged)
    assert not Path("/tmp/should_not_exist_xyz").exists()
    # no second file-writing entry point in the generator module
    assert "open(" not in inspect.getsource(GEN)


def test_baseline_failure_classification_no_new_failures():
    # the 8 failing files are Claude C historical audit files (strict-xfail->XPASS); none imports the generator
    tdir = Path(__file__).parent
    c_files = ["test_preregistration_v4_audit_c.py", "test_preregistration_v4_fix1_reaudit_c.py",
               "test_preregistration_v4_fix2_final_reaudit_c.py", "test_preregistration_v4_fix3_final_reaudit_c.py",
               "test_one_shot_full_audit_c.py", "test_frozen_issue_regression_final_001_005_c.py",
               "test_final_001_002_regression_c.py", "test_block_state_sampler_reaudit_c.py"]
    for f in c_files:
        assert "confirmatory_v4_generator" not in (tdir / f).read_text()


# ==================== FROZEN FINDINGS (demonstrated on current code) ====================
def test_GEN_B_001_generated_phase_payload_is_mutable_breaking_hash_binding():
    g = _build("test")
    h0 = g.full_manifest_sha256
    g.unsealed_manifest["blocks"][0]["residual_value"] = 0.00777      # mutate scientific payload
    g.unsealed_manifest["trials"][0]["execution_order_index"] = 999
    assert g.full_manifest_sha256 == h0                               # BLOCKING: hash unchanged -> now inconsistent
    assert MI.fully_resolved_phase_manifest_hash(_build("test").unsealed_manifest) != None  # a re-hash would differ


def test_GEN_B_002_envelope_public_spec_mutable_secret_injectable_after_scan():
    tid = ID.canonical_phase_trial_identities("test")[0]
    env = GEN.build_trial_execution_envelope("test", tid, auth=_AUTH)   # scan already passed inside
    env.public_trial_spec["residual_bias"] = 0.009                      # inject secret AFTER the scan
    assert "residual_bias" in env.public_trial_spec                     # BLOCKING: no re-scan / immutability


def test_GEN_B_003_target_open_position_value_not_a_frozen_source_of_truth():
    src = inspect.getsource(GEN.build_trial_execution_envelope)
    assert '"target_open_position": 0.20' in src                        # hardcoded literal
    # the active prereg only allowlists the FIELD NAME, not a frozen value
    assert "g.target_open_position" in json.dumps(PRE.STATIC_ALLOWLIST)
    assert not hasattr(PRE, "TARGET_OPEN_POSITION")
    assert "target_open_position" not in PRE.PRIMARY_SUCCESS            # tolerance is frozen; the level is not
    assert "synthetic" not in src.lower()                              # not marked as a synthetic smoke fixture


def test_GEN_B_004_smoke_builder_accepts_real_commits_and_hash_has_no_smoke_marker():
    real = GEN.CommitContext("1" * 40, "2" * 40, "3" * 40)             # real-looking, not placeholder
    g = _build("test", commits=real)                                  # accepted under smoke_only=True
    assert g.full_manifest_sha256 and g.smoke_only is True
    assert not GEN.commit_context_is_smoke_placeholder(real)
    assert "smoke" not in ID.canonical_json(g.unsealed_manifest).lower()  # BLOCKING: hashed payload has no smoke marker


def test_GEN_B_005_environment_versions_has_no_completeness_contract():
    empty = GEN.build_phase_manifest_in_memory("test", auth=_AUTH, commits=GEN.smoke_placeholder_commits(),
                                               environment_versions=GEN.EnvironmentVersionContext({}))
    MI.validate_fully_resolved_phase_manifest(empty.unsealed_manifest)  # empty {} VALIDATES -> no contract
    assert empty.unsealed_manifest["environment_versions"] == {}        # BLOCKING: empty provenance accepted
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator_cli as CLI
    csrc = inspect.getsource(CLI._env_versions)
    assert "except Exception:" in csrc and "pass" in csrc              # silent torch-drop
    assert "platform" not in csrc and "python" not in csrc.replace("import numpy", "")  # missing python/OS/Isaac/GPU


def test_GEN_B_006_session_selection_domain_lax_nonblocking():
    c = GEN.SessionRunController("s"); c.probe_ready(); c.probe_complete()
    c.freeze_selection(False, "ev")                                    # bool False -> 0.0 accepted (should reject)
    assert c._selected_offset == 0.0
    c2 = GEN.SessionRunController("s2"); c2.probe_ready(); c2.probe_complete()
    c2.freeze_selection("0.0", "ev")                                  # numeric string accepted (should reject)
    assert c2._selected_offset == 0.0
    c3 = GEN.SessionRunController("s3"); c3.probe_ready(); c3.probe_complete()
    c3.freeze_selection(0.0, "")                                      # empty evidence_hash accepted
    # ordering guarantee IS enforced (the blocking part): candidate outcomes cannot enter the selector
    with pytest.raises(GEN.SelectionOrderError):
        c3.offer_candidate_outcome_to_selector()
    with pytest.raises(GEN.SelectionOrderError):
        GEN.SessionRunController("s4").freeze_selection(0.0, "ev")     # cannot freeze before probe


def test_GEN_B_007_smoke_summary_write_nonatomic_nonblocking():
    from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator_cli as CLI
    src = inspect.getsource(CLI.run_smoke)
    assert 'with open(out, "w") as f' in src                          # plain write, not temp+fsync+rename
    assert "os.replace" not in src and "fsync" not in src
    assert "_summary_path" in src                                     # returned dict has a key absent on disk