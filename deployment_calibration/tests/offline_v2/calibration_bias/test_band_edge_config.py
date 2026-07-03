"""Frozen band-edge config self-consistency + geometry tests."""

import json
from pathlib import Path

from deployment_calibration.offline_v2.calibration_bias import band_edge as BE

CONFIG_JSON = (Path(BE.__file__).parent / "band_edge_characterization_config_v1.json")


def test_config_validates():
    v = BE.validate_config()
    assert v["ok"], [c for c in v["checks"] if not c["ok"]]
    assert v["n_episodes"] == 306


def test_counts_frozen():
    cfg = BE.DEFAULT_BAND_EDGE
    assert cfg.n_blocks == 18
    assert len(cfg.offsets) == 17
    assert cfg.n_episodes == 18 * 17 == 306
    assert cfg.run_probes is False
    assert cfg.nominal_bias_y == 0.0


def test_offset_grid_symmetric_and_covers_edge():
    cov = BE.abs_eff_coverage_nominal()
    # required unmeasured odd points + known edge + fail region + success region
    for p in (0.0, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050):
        assert round(p, 4) in cov, p
    pos = sorted(o for o in BE.OFFSET_GRID if o > 0)
    neg = sorted(-o for o in BE.OFFSET_GRID if o < 0)
    assert pos == neg


def test_effective_error_geometry():
    # nominal 0, residual +0.005, offset -0.02 -> eff_signed -0.015, |eff| 0.015
    assert abs(BE.effective_error_signed(0.005, -0.02) - (-0.015)) < 1e-12
    assert abs(BE.effective_error(0.005, -0.02) - 0.015) < 1e-12


def test_does_not_use_future_test_nominal_bias():
    # only nominal 0 is used; the future confirmatory test biases must not appear
    assert BE.NOMINAL_BIAS_Y == 0.0
    d = json.loads(CONFIG_JSON.read_text())
    assert d["design"]["uses_future_test_nominal_bias"] is False
    assert d["design"]["future_test_nominal_bias_excluded"] == [-0.03, 0.03]


def test_config_json_matches_module():
    d = json.loads(CONFIG_JSON.read_text())
    assert d["design"]["n_episodes"] == BE.DEFAULT_BAND_EDGE.n_episodes
    assert d["design"]["offsets"] == list(BE.OFFSET_GRID)
    assert d["residual_nuisance"]["sigma"] == BE.DEFAULT_BAND_EDGE.residual.sigma
    assert d["analysis_plan_frozen"]["bin_width_m"] == BE.BIN_WIDTH
    assert d["not_confirmatory"] is True
