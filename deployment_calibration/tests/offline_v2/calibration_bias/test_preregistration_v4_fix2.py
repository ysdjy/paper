"""Post-fix2 PASS tests for preregistration v4 FIX2 (offline). No Isaac, no generator, no data.

Covers the two closed blockers (production observed-only best-single; frozen canonical planned identity),
the determinism env, and protocol consistency. C's audit files are preserved unmodified.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as P
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_selection as S
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as I
from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L
from deployment_calibration.offline_v2.calibration_bias import test_geometry_power as T

_DOCS = Path(P._docs_dir())


def _cfg():
    return json.loads((_DOCS / "confirmatory_v4_config.json").read_text())


def _pre():
    return json.loads((_DOCS / "preregistration_v4.json").read_text())


# ---------------- status ----------------
def test_status_is_fix2():
    assert P.STATUS == "PREREGISTRATION_V4_ONE_SHOT_BATCH_FIX_READY_FOR_FROZEN_ISSUE_REGRESSION"
    assert _pre()["status"] == P.STATUS
    assert _pre()["power_recertification_required"] is False


# ================= BLOCKER 1: production best-single is observed-only =================
def test_active_config_points_to_production_not_simulation():
    impl = _cfg()["best_single"]["implementation"]
    # fix3: the production entry is the guarded single-param confirmatory selector
    assert impl.endswith("confirmatory_v4_selection.select_best_single_confirmatory")
    assert "best_single_legal" not in impl


def test_production_selector_signature_has_no_tau():
    import re
    sig = inspect.signature(S.select_best_single_confirmatory)
    assert "tau" not in sig.parameters
    src = "\n".join(inspect.getsource(fn) for fn in
                    (S.select_best_single_confirmatory, S._project_records, S._count_select,
                     S._check_confirmatory_completeness))
    code = re.sub(r'""".*?"""', "", src, flags=re.S)          # strip docstrings (they name the forbidden fields)
    # the production code must never READ the secret hidden state or call the sim success model
    assert '["nominal"]' not in code and '["residual"]' not in code
    assert "succ(" not in code
    assert '["theta"]["grasp_offset_local_y"]' in code and '["y"]["success"]' in code


def _mk_records(offset_success, n_sessions=57):
    """Build a valid 171-record input; offset_success maps offset->success fraction (count out of n)."""
    recs = []
    for s in range(n_sessions):
        for o in (-0.04, 0.0, 0.04):
            suc = s < int(round(offset_success[o] * n_sessions))
            recs.append({"split": "train" if s < 45 else "validation", "session_id": f"sess{s:02d}",
                         "trial_role": "candidate", "theta": {"grasp_offset_local_y": o},
                         "y": {"success": bool(suc)}, "planned_episode_id": f"v4ep-{s:02d}{o:+.3f}"})
    return recs


def test_production_selects_max_count_then_min_abs_then_bank():
    # offset 0 has the most successes -> selected
    recs = _mk_records({-0.04: 0.3, 0.0: 0.9, 0.04: 0.3})
    art = S.select_best_single(recs)
    assert art["selected_offset"] == 0.0
    assert art["n_trials_per_offset"] == 57
    assert art["success_count_per_offset"]["+0.000"] == int(round(0.9 * 57))
    # tie between -0.04 and +0.04 (equal counts, 0 lower) -> min|offset| both 0.04 -> bank order -> -0.04
    recs2 = _mk_records({-0.04: 0.8, 0.0: 0.2, 0.04: 0.8})
    assert S.select_best_single(recs2)["selected_offset"] == -0.04


def test_secret_perturbation_invariance():
    recs = _mk_records({-0.04: 0.3, 0.0: 0.8, 0.04: 0.3})
    base = S.select_best_single(recs)
    # inject secret fields; selection must not change
    for r in recs:
        r["secret_deployment_state"] = {"actual_bias_y": 0.033}
        r["nominal"] = 0.99; r["residual"] = -0.007; r["eff_signed"] = 0.5; r["tau"] = 0.01
    pert = S.select_best_single(recs)
    assert pert["selected_offset"] == base["selected_offset"]
    assert pert["input_projection_hash"] == base["input_projection_hash"]   # projection ignores secret


def test_observed_outcome_sensitivity():
    # flipping enough observed successes to +0.04 changes the selection
    a = S.select_best_single(_mk_records({-0.04: 0.3, 0.0: 0.8, 0.04: 0.3}))["selected_offset"]
    b = S.select_best_single(_mk_records({-0.04: 0.3, 0.0: 0.3, 0.04: 0.9}))["selected_offset"]
    assert a == 0.0 and b == 0.04


def test_incomplete_input_fail_fast():
    good = _mk_records({-0.04: 0.3, 0.0: 0.8, 0.04: 0.3})
    with pytest.raises(S.BestSingleInputError):        # missing a record
        S.select_best_single(good[:-1])
    with pytest.raises(S.BestSingleInputError):        # a test record
        bad = list(good); bad[0] = dict(bad[0], split="test"); S.select_best_single(bad)
    with pytest.raises(S.BestSingleInputError):        # a probe record
        bad = list(good); bad[0] = dict(bad[0], trial_role="probe"); S.select_best_single(bad)
    with pytest.raises(S.BestSingleInputError):        # non-bool success
        bad = list(good); bad[0] = dict(bad[0], y={"success": 1}); S.select_best_single(bad)
    with pytest.raises(S.BestSingleInputError):        # duplicate (session, offset)
        bad = list(good); bad[1] = dict(bad[0]); S.select_best_single(bad)


def test_bridge_invariance_recompute_subset():
    # production observed selector == simulation reference on a slice of the frozen grid
    for tau in (0.0342, 0.03425, 0.0343):
        for s in range(30):
            rng = np.random.default_rng(s)
            TR, VA, _ = T.make_dataset(9, 6, 9, tau, rng, (-0.035, 0.035))
            recs = []
            for si, sess in enumerate(TR + VA):
                split = "train" if sess in TR else "validation"
                for o in L.BANK:
                    recs.append({"split": split, "session_id": f"s{si:02d}", "trial_role": "candidate",
                                 "theta": {"grasp_offset_local_y": o},
                                 "y": {"success": bool(L.succ(sess["nominal"], sess["residual"], o, tau))},
                                 "planned_episode_id": f"v4ep-{si:02d}{o:+.3f}"})
            prod = S.select_best_single(recs)["selected_offset"]
            assert prod == L.best_single_legal(TR, VA, tau)


def test_bridge_invariance_evidence_present():
    j = json.loads((_DOCS / "production_best_single_bridge_invariance_v1.json").read_text())
    assert j["total_configs_checked"] == 4500 and j["mismatch_count"] == 0
    assert j["agree_all"] is True and j["power_recertification_required"] is False


def test_best_single_legal_marked_simulation_only():
    src = inspect.getsource(L.best_single_legal)
    assert "SIMULATION_ONLY_REFERENCE" in src


# ================= BLOCKER 2: canonical planned identity frozen =================
def test_identity_templates_exact():
    assert I.block_identity("train", 0) == "v4|split=train|block=00"
    assert I.session_identity("validation", 5, 0.01) == "v4|split=validation|block=05|nominal=+0.010"
    assert I.trial_identity("test", 8, 0.035, "probe", -0.04) == \
        "v4|split=test|block=08|nominal=+0.035|role=probe|offset=-0.040"


def test_float_format_and_zero_normalization():
    assert I.fmt_m(0.0) == "+0.000"
    assert I.fmt_m(-0.0) == "+0.000"
    assert I.fmt_m(-0.035) == "-0.035"
    assert I.fmt_m(0.04) == "+0.040"


def test_planned_episode_id_template():
    tid = I.trial_identity("test", 8, 0.035, "candidate", -0.04)
    pid = I.planned_episode_id(tid)
    assert pid.startswith("v4ep-") and len(pid) == len("v4ep-") + 24
    assert pid == "v4ep-" + hashlib.sha256(tid.encode("utf-8")).hexdigest()[:24]
    # role change changes the id (probe vs candidate at -0.04)
    assert I.planned_episode_id(I.trial_identity("test", 8, 0.035, "probe", -0.04)) != pid


def test_300_planned_ids_unique():
    rows = I.enumerate_trials()
    assert len(rows) == 300
    pids = [r["planned_episode_id"] for r in rows]
    assert len(set(pids)) == 300


def test_manifest_hash_order_independent():
    h1 = I.canonical_manifest_hash()
    # recompute after shuffling enumeration order -> same canonical hash
    rows = I.enumerate_trials()[::-1]
    key = I._sort_key
    resorted = sorted(rows, key=key)
    pair = [[r["canonical_trial_identity"], r["planned_episode_id"]] for r in resorted]
    h2 = hashlib.sha256(I.canonical_json(pair).encode("utf-8")).hexdigest()
    assert h1 == h2


def test_domain_subseeds_distinct_and_stable():
    a = I.block_residual_subseed("train", 0)
    assert a == I.block_residual_subseed("train", 0)                       # stable
    assert a != I.block_nuisance_subseed("train", 0)                       # domain label matters
    assert a != I.block_residual_subseed("train", 1)                       # identity matters
    assert I.trial_init_subseed("test", 8, 0.035, "probe", -0.04) != \
        I.trial_init_subseed("test", 8, 0.035, "candidate", -0.04)         # role matters


def test_attempt_not_in_resume_key():
    tid = I.trial_identity("train", 0, -0.04, "candidate", 0.0)
    pid = I.planned_episode_id(tid)
    assert I.attempt_id(pid, 0) == f"{pid}|attempt=00"
    assert I.attempt_id(pid, 1) == f"{pid}|attempt=01"
    # resume key is the planned_episode_id, invariant to attempt
    assert I.IDENTITY_TEMPLATE["resume_key"].startswith("planned_episode_id")


def test_identity_template_in_config():
    c = _cfg()
    assert "identity_template" in c or "planned_identity_format" in c
    t = c["planned_identity_format"]["identity_template"]
    assert t["trial"].startswith("v4|split=") and "role=" in t["trial"] and "offset=" in t["trial"]


# ================= determinism =================
def test_determinism_env_frozen():
    d = _cfg()["determinism_env"]
    assert d["device"].lower() == "cpu"
    assert d["torch.set_num_threads"] == 1 and d["torch.set_num_interop_threads"] == 1
    assert d["torch.use_deterministic_algorithms"] is True
    assert d["env"]["OMP_NUM_THREADS"] == "1" and d["env"]["MKL_NUM_THREADS"] == "1"
    assert d["explicit_hyperparameters_required"] is True


# ================= protocol / power unchanged =================
def test_power_inputs_unchanged():
    d = _cfg()["design"]
    assert d["candidate_bank"] == [-0.04, 0.0, 0.04]
    assert d["test_nominals"] == [-0.035, 0.035]
    assert _cfg()["trial_counts"]["full_task_trials_total"] == 300
    v = json.loads((_DOCS / "test_geometry_power_verdict_v1.json").read_text())
    assert v["verdict"] == "POWER_SUFFICIENT_FOR_PREREG_V4"


def test_no_generator_or_data_produced():
    hc = _pre()["hard_constraints"]
    assert "no confirmatory generator" in hc and "no run authorization" in hc
    names = [p.name for p in _DOCS.iterdir()]
    assert not any("manifest_instance" in n or "confirmatory_data" in n or n.endswith(".pt") for n in names)


def test_md_json_consistency():
    P.emit()
    md = (_DOCS / "preregistration_v4.md").read_text()
    assert "PREREGISTRATION_V4_ONE_SHOT_BATCH_FIX_READY_FOR_FROZEN_ISSUE_REGRESSION" in md
    assert "confirmatory_v4_selection.select_best_single" in (_DOCS / "preregistration_v4.json").read_text()


def test_306_hash_unchanged():
    p = "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl"
    if not Path(p).exists():
        pytest.skip("306 not present")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
        "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
