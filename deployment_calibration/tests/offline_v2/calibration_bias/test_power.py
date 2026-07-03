"""Pre-data power-simulation tests (light — small n_sims for speed)."""

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import power as PW


def test_simulate_once_shape_and_mean():
    rng = np.random.default_rng(0)
    pb = PW.simulate_once(rng, [-0.03, 0.03], [-0.04, -0.02, 0.0, 0.02, 0.04], 9, PW.DEFAULT_OFFSETS,
                          PW.BAND_CENTER, PW.EDGE_SCALE, PW.SIGMA_NUIS)
    assert pb.shape == (9,)
    assert np.all((pb >= -1.0) & (pb <= 1.0))
    # over many blocks the per-block gain concentrates at ~0.5 (state-aware compensates each test bias;
    # best-single covers at most one, and lands near the ~50% band edge at |eff|=0.03)
    big = PW.simulate_once(rng, [-0.03, 0.03], [-0.04, -0.02, 0.0, 0.02, 0.04], 600, PW.DEFAULT_OFFSETS,
                           PW.BAND_CENTER, PW.EDGE_SCALE, PW.SIGMA_NUIS)
    assert 0.35 < big.mean() < 0.75


def test_power_meets_target_at_floor():
    r = PW.power_at(9, n_sims=120, n_boot=400, seed=1)
    assert r["power"] >= 0.8          # well above target at the 9-block floor
    assert 0.3 < r["mean_gain"] < 0.7


def test_power_nondecreasing_and_choice():
    pc = PW.power_curve(candidates=(6, 9, 18), min_test_blocks=9, target_power=0.8,
                        max_halfwidth=0.25, n_sims=120, n_boot=400)
    powers = [r["power"] for r in pc["curve"]]
    # more blocks -> not worse power (allow tiny MC noise)
    assert powers[-1] >= powers[0] - 0.05
    assert pc["chosen_test_blocks"] >= 9


def test_ci_excludes_threshold_direction():
    # the CI lower bound at the gain ~0.5 should sit above the 0.15 H3 threshold on average
    rng = np.random.default_rng(2)
    pb = PW.simulate_once(rng, [-0.03, 0.03], [-0.04, -0.02, 0.0, 0.02, 0.04], 12, PW.DEFAULT_OFFSETS,
                          PW.BAND_CENTER, PW.EDGE_SCALE, PW.SIGMA_NUIS)
    pt, lo, hi = PW.block_bootstrap_ci(pb, rng, n_boot=500)
    assert lo <= pt <= hi
