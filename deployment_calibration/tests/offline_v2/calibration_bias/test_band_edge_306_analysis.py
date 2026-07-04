"""Tests for the read-only 306-episode band-edge analysis (frozen plan). Pure python; data read-only.

Covers: input hash, block-bootstrap resamples whole blocks, bin assignment, logistic parameterization,
isotonic monotonicity, signed split, collision filtering, source-not-modified, determinism, verdict gate.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from deployment_calibration.offline_v2.calibration_bias import band_edge as BE
from deployment_calibration.offline_v2.calibration_bias import band_edge_analysis as A

_REPO = Path(__file__).resolve().parents[4]
DATA = _REPO / "deployment_calibration" / "data" / "band_edge_full_306_v1"
FROZEN_EP_SHA = "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
needs_data = pytest.mark.skipif(not (DATA / "episodes.jsonl").exists(),
                                reason="authorized band-edge data not present here")


def _synth_hard(center=0.034, seed=0):
    """18 blocks x frozen offset grid; hard threshold success iff |offset+residual| <= center."""
    rng = np.random.default_rng(seed)
    recs = []
    for b in range(18):
        r = float(rng.uniform(-0.008, 0.009))
        for o in BE.OFFSET_GRID:
            eff = abs(o + r)
            recs.append({"block_id": b, "offset_id": f"o{o:+.3f}", "offset": o, "residual": r,
                         "actual_bias": r, "eff_signed": o + r, "eff": round(eff, 6),
                         "success": int(eff <= center),
                         "failure_reason": "NONE" if eff <= center else "HANDLE_DETACHED",
                         "failure_phase": "NONE" if eff <= center else "PULL"})
    return recs


@needs_data
def test_input_episodes_hash_matches_frozen():
    assert A.episodes_sha256(DATA) == FROZEN_EP_SHA


def test_block_bootstrap_uses_whole_blocks():
    recs = _synth_hard()
    seen = {}

    def stat(sample):
        from collections import Counter
        seen["counts"] = Counter(r["block_id"] for r in sample)
        return float(np.mean([r["success"] for r in sample]))

    A.block_bootstrap(recs, stat, n_boot=5, seed=1)
    assert all(v % 17 == 0 for v in seen["counts"].values())   # whole 17-record blocks only


def test_block_bootstrap_deterministic():
    recs = _synth_hard()
    f = lambda rr: float(np.mean([r["success"] for r in rr]))
    a = A.block_bootstrap(recs, f, n_boot=200, seed=7)
    b = A.block_bootstrap(recs, f, n_boot=200, seed=7)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_bin_assignment_frozen_width():
    rows = A.binned_success(_synth_hard(), seed=0)
    assert A.BIN_WIDTH == 0.005
    assert all(round(r["bin_hi"] - r["bin_lo"], 6) == 0.005 for r in rows)
    assert round(np.floor(0.032 / 0.005) * 0.005, 4) == 0.030


def test_logistic_form_and_hard_edge():
    out = A.logistic_analysis(_synth_hard(center=0.034), seed=0)
    assert out["converged"] and abs(out["edge_center"] - 0.034) < 0.004
    assert not out["scale_identifiable"]        # perfect separation -> scale unidentifiable (honest)


def test_isotonic_monotone_nonincreasing():
    p = A.isotonic_analysis(_synth_hard(), seed=0)["success_pred"]
    assert all(p[i] >= p[i + 1] - 1e-9 for i in range(len(p) - 1))


def test_signed_split_partitions():
    recs = _synth_hard()
    n_neg = sum(r["n_episodes"] for r in A.binned_success(recs, direction="neg"))
    n_pos = sum(r["n_episodes"] for r in A.binned_success(recs, direction="pos"))
    n_zero = sum(1 for r in recs if r["eff_signed"] == 0)
    assert n_neg + n_pos + n_zero == len(recs)


def test_collision_zero_force_no_confound():
    eps = [{"episode_id": f"e{i}", "max_unintended_contact_force_N": 0.0, "unintended_contact_frame_count": 0,
            "contact_sensor_available": True, "y": {"success": True}} for i in range(10)]
    recs = [{"block_id": 0, "eff": 0.01, "success": 1, "failure_reason": "NONE", "failure_phase": "NONE"}]
    c = A.collision_analysis(recs, eps)
    assert c["EDGE_NOT_COLLISION_DRIVEN"]
    assert c["threshold_proposal"]["status"] == "THRESHOLD_NOT_IDENTIFIABLE_FROM_FROZEN_RULE"
    assert c["collision_confounded_episodes"] == []


def test_exit_gate_modify_on_hard_threshold():
    v = BE.evaluate_exit_criteria(
        eff025_035_both_labels=True, residual_block_variation=False, edge_center=0.034, edge_scale=1e-5,
        edge_estimable=True, severe_asymmetry=False, collision_sensor_available=True,
        instrumentation_complete=True, result_collision_dominated=False)
    assert v["verdict"] == "MODIFY_RESIDUAL_OR_DESIGN"


def test_exit_gate_pass_when_all_met():
    v = BE.evaluate_exit_criteria(
        eff025_035_both_labels=True, residual_block_variation=True, edge_center=0.03, edge_scale=0.004,
        edge_estimable=True, severe_asymmetry=False, collision_sensor_available=True,
        instrumentation_complete=True, result_collision_dominated=False)
    assert v["verdict"] == "PASS_TO_V4"


@needs_data
def test_source_data_not_modified():
    before = hashlib.sha256((DATA / "episodes.jsonl").read_bytes()).hexdigest()
    A.to_records(A.load_run(DATA)[0])
    after = hashlib.sha256((DATA / "episodes.jsonl").read_bytes()).hexdigest()
    assert before == after == FROZEN_EP_SHA


@needs_data
def test_real_data_hard_edge_and_no_collision():
    eps, _, _ = A.load_run(DATA)
    recs = A.to_records(eps)
    logi = A.logistic_analysis(recs, seed=A.derive_bootstrap_seed(DATA))
    assert 0.030 < logi["edge_center"] < 0.038 and not logi["scale_identifiable"]
    assert A.collision_analysis(recs, eps)["EDGE_NOT_COLLISION_DRIVEN"]
