"""Post-fix4 PASS tests for preregistration v4 FIX4 (offline). No Isaac, no generator, no data.

Covers the four closed blockers: strict best-single offset domain; deep full-manifest validator (valid +
many invalid in-memory fixtures); combined-hash required fields/formats; active-doc cleanup. C's audit files
unmodified. Fixtures are in-memory only (no manifest file written).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_manifest_integrity as MI
from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _pre():
    return json.loads((_DOCS / "preregistration_v4.json").read_text())


def test_status_is_fix4():
    assert P.STATUS == "PREREGISTRATION_V4_VALIDATOR_DESCRIPTION_CONSOLIDATED_READY_FOR_C_REGRESSION"
    assert _pre()["status"] == P.STATUS and _pre()["power_recertification_required"] is False


# ================= BLOCKER a: strict offset domain =================
def _recs(zero):
    out = []
    for sp, n in (("train", 45), ("validation", 12)):
        for i in range(n):
            for o in (-0.04, zero, 0.04):
                out.append({"split": sp, "session_id": f"{sp}-{i}", "trial_role": "candidate",
                            "theta": {"grasp_offset_local_y": o}, "y": {"success": o == 0.0},
                            "planned_episode_id": f"{sp}-{i}-{o!r}"})
    return out


def test_offset_strict_accepts_exact_and_tiny_error():
    assert S._canonical_candidate_offset(-0.04) == -0.04
    assert S._canonical_candidate_offset(0.0 + 1e-13) == 0.0
    assert S._canonical_candidate_offset(0.04 - 3e-13) == 0.04
    art = S.select_best_single_confirmatory(_recs(0.0 + 1e-13))
    assert art["selected_offset"] == 0.0


def test_offset_strict_rejects_smuggled_and_bad_types():
    for bad in (0.0004, -0.0004, -0.0396, 0.0404, "0.0", True, False, float("nan"),
                float("inf"), float("-inf"), None):
        with pytest.raises(S.BestSingleInputError):
            S._canonical_candidate_offset(bad)
        with pytest.raises(S.BestSingleInputError):
            S.select_best_single_confirmatory(_recs(bad))


def test_offset_strict_numpy_bool_rejected():
    with pytest.raises(S.BestSingleInputError):
        S._canonical_candidate_offset(np.bool_(True))


def test_no_round_offset_deciding_legality():
    import re
    src = inspect.getsource(S._canonical_candidate_offset)
    code = re.sub(r'""".*?"""', "", src, flags=re.S)          # strip docstring (it mentions round(.,3))
    assert "round(" not in code and "isclose" in code


def test_bridge_invariance_still_holds_subset():
    for tau in (0.0342, 0.03425, 0.0343):
        for s in range(20):
            rng = np.random.default_rng(s)
            TR, VA, _ = T.make_dataset(9, 6, 9, tau, rng, (-0.035, 0.035))
            recs = []
            for si, sess in enumerate(TR + VA):
                split = "train" if sess in TR else "validation"
                for o in L.BANK:
                    recs.append({"split": split, "session_id": f"s{si}", "trial_role": "candidate",
                                 "theta": {"grasp_offset_local_y": o},
                                 "y": {"success": bool(L.succ(sess["nominal"], sess["residual"], o, tau))},
                                 "planned_episode_id": f"ep-{si}-{o}"})
            assert S._select_best_single_core(recs, validate_completeness=False)["selected_offset"] == \
                L.best_single_legal(TR, VA, tau)


# ================= BLOCKER b: deep manifest validator =================
def _valid_manifest(phase):
    # FINAL-001/002: use the shared deep-valid reference builder (resolved order + frozen values)
    return MI.reference_phase_manifest(phase)


def test_valid_phase_manifests_pass_and_hash():
    for phase in ("train_validation", "test"):
        m = _valid_manifest(phase)
        MI.validate_fully_resolved_phase_manifest(m)          # no raise
        h = MI.fully_resolved_phase_manifest_hash(m)
        assert len(h) == 64 and len(m["trials"]) == MI.PHASES[phase]["trials"]


def test_full_hash_covers_resolved_fields_valid_fixture():
    m = _valid_manifest("test")
    h0 = MI.fully_resolved_phase_manifest_hash(m)
    m2 = _valid_manifest("test"); m2["blocks"][0]["residual_value"] = 0.00777
    m2["blocks"][0]["nuisance_values"] = {"joint_delta": 0.0}   # keep others; only residual changed
    assert MI.fully_resolved_phase_manifest_hash(m2) != h0


def test_deep_validator_rejects_many_violations():
    def bad(fn, phase="test"):
        m = _valid_manifest(phase); fn(m)
        with pytest.raises(MI.ManifestIntegrityError):
            MI.validate_fully_resolved_phase_manifest(m)
    # duplicate trial (overwrite trial 1 with trial 0) -> uniqueness / structure
    bad(lambda m: m["trials"].__setitem__(1, copy.deepcopy(m["trials"][0])))
    # all-train composition in test phase
    bad(lambda m: [b.__setitem__("split", "train") for b in m["blocks"]])
    # PID != identity
    bad(lambda m: m["trials"][0].__setitem__("planned_episode_id", "v4ep-" + "0" * 24))
    # resume_key != PID
    bad(lambda m: m["trials"][0].__setitem__("resume_key", "v4ep-" + "0" * 24))
    # duplicate execution index
    bad(lambda m: m["trials"][1].__setitem__("execution_order_index", 0))
    # execution index gap
    bad(lambda m: m["trials"][0].__setitem__("execution_order_index", 999))
    # candidate order not a permutation
    bad(lambda m: m["sessions"][0].__setitem__("resolved_candidate_order", [-0.04, -0.04, 0.04]))
    # probe offset wrong
    bad(lambda m: [t.__setitem__("offset", 0.02) for t in m["trials"] if t["role"] == "probe"][:1])
    # trial references missing session
    bad(lambda m: m["trials"][0].__setitem__("session_ref", "nope"))
    # residual out of range
    bad(lambda m: m["blocks"][0].__setitem__("residual_value", 0.5))
    # wrong subseed
    bad(lambda m: m["blocks"][0].__setitem__("residual_subseed", 12345))
    # wrong planned structure hash
    bad(lambda m: m.__setitem__("planned_structure_sha256", "f" * 64))
    # nested integrity object
    bad(lambda m: m.__setitem__("integrity", {"x": 1}))
    # unexpected scientific field
    bad(lambda m: m["trials"][0].__setitem__("extra", 1))
    # NaN
    bad(lambda m: m["blocks"][0].__setitem__("residual_value", float("nan")))
    # wrong frozen seed
    bad(lambda m: m["frozen_seeds"].__setitem__("master_seed", 1))
    # device not cpu
    bad(lambda m: m["deterministic_environment"].__setitem__("device", "cuda"))


def test_probe_before_candidates_enforced():
    m = _valid_manifest("test")
    # find a session and make its probe execute AFTER a candidate
    probe = next(t for t in m["trials"] if t["role"] == "probe")
    sref = probe["session_ref"]
    cand = next(t for t in m["trials"] if t["role"] == "candidate" and t["session_ref"] == sref)
    probe["execution_order_index"], cand["execution_order_index"] = \
        cand["execution_order_index"], probe["execution_order_index"]
    with pytest.raises(MI.ManifestIntegrityError):
        MI.validate_fully_resolved_phase_manifest(m)


def test_full_hash_requires_deep_validation():
    m = _valid_manifest("train_validation"); m["trials"][5] = copy.deepcopy(m["trials"][0])
    with pytest.raises(MI.ManifestIntegrityError):
        MI.fully_resolved_phase_manifest_hash(m)


# ================= BLOCKER c: combined hash required fields =================
def _combined(**over):
    base = dict(train_validation_manifest_sha256="a" * 64, model_analysis_freeze_sha256="b" * 64,
                test_manifest_sha256="c" * 64, config_sha256="d" * 64, protocol_commit="e" * 40,
                generator_commit="f" * 40, analysis_code_sha256="a" * 64, bootstrap_seed=9014517173581927929)
    base.update(over)
    return MI.combined_experiment_plan_hash(**base)


def test_combined_hash_valid_and_deterministic():
    h1 = _combined()
    assert len(h1) == 64 and h1 == _combined()
    assert _combined(config_sha256="e" * 64) != h1        # mutation sensitivity


def test_combined_hash_rejects_bad_fields():
    for over in (dict(analysis_code_sha256=None), dict(bootstrap_seed=None), dict(bootstrap_seed=0),
                 dict(bootstrap_seed=True), dict(bootstrap_seed=9), dict(protocol_commit="e" * 10),
                 dict(config_sha256="A" * 64), dict(test_manifest_sha256="g" * 64),
                 dict(model_analysis_freeze_sha256="")):
        with pytest.raises(MI.ManifestIntegrityError):
            _combined(**over)


def test_combined_hash_fields_have_no_defaults():
    params = inspect.signature(MI.combined_experiment_plan_hash).parameters
    for name in ("analysis_code_sha256", "bootstrap_seed"):
        assert params[name].default is inspect.Parameter.empty


# ================= BLOCKER d: active doc cleanup =================
def test_active_docs_no_stale_semantics():
    spec = (_DOCS / "confirmatory_v4_manifest_spec.md").read_text()
    assert "block=<i>" not in spec
    assert "== frozen `manifest_hash`" not in spec
    assert '"manifest_hash"' not in json.dumps(_cfg())
    assert '"manifest_hash"' not in json.dumps(_pre())


def test_config_has_layered_manifest_fields():
    mf = _cfg()["manifest_fields"]
    for k in ("planned_structure_sha256", "train_validation_manifest_sha256", "test_manifest_sha256",
              "model_analysis_freeze_sha256", "combined_experiment_plan_sha256"):
        assert k in mf
    assert "manifest_hash" not in mf


def test_per_record_phase_anchor_documented():
    mi = _cfg()["manifest_integrity"]["per_record_anchor"]
    assert "train_validation_manifest_sha256" in mi["train_validation"]
    assert "test_manifest_sha256" in mi["test"]


# ================= protocol / power unchanged =================
def test_power_inputs_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04] and d["test_nominals"] == [-0.035, 0.035]
    assert _cfg()["trial_counts"]["full_task_trials_total"] == 300
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_no_generator_manifest_checkpoint_data():
    hc = _pre()["hard_constraints"]
    assert "no confirmatory generator" in hc and "no run authorization" in hc
    names = [p.name for p in _DOCS.iterdir()]
    assert not any(n.endswith(".pt") or "manifest_instance" in n or "confirmatory_data" in n for n in names)


def test_md_json_consistency():
    P.emit()
    md = (_DOCS / "preregistration_v4.md").read_text()
    assert "PREREGISTRATION_V4_VALIDATOR_DESCRIPTION_CONSOLIDATED_READY_FOR_C_REGRESSION" in md


def test_306_hash_unchanged():
    p = "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl"
    if not Path(p).exists():
        pytest.skip("306 not present")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
        "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
