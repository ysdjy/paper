"""Pre-data power simulation for the calibration-bias confirmatory pilot (offline).

Uses ONLY exploration-derived effect size + continuous variance to size the number of independent
nuisance blocks. No confirmatory data is read. Deterministic given a seed (numpy Generator).

Success model (calibrated to the capability map: success at |eff|=0.02, fail at |eff|>=0.04):
    p_succ(eff) = sigmoid((c - |eff|) / s),  c=0.03 (50% band), s=0.0025 (sharp edge observed)
A per-block grasp perturbation delta ~ N(0, sigma_nuis) shifts the effective offset for all candidates
in that block-session (the nuisance block is shared within a session). sigma_nuis from the exploration
within-cell continuous SD (~0.005 m).

Statistic: success-only gain on the TEST biases = mean over (test bias, test block) of
    [state-aware selected success - best-single-offset success].
CI: block bootstrap over the TEST blocks (resample blocks with replacement; each block contributes its
test-bias rows). Power = P(95% CI lower bound >= threshold) over simulated confirmatory datasets.
"""

from __future__ import annotations

import numpy as np

DEFAULT_OFFSETS = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)
BAND_CENTER = 0.03       # |eff| at 50% success
EDGE_SCALE = 0.0025      # logistic scale (sharp edge, matches exploration)
SIGMA_NUIS = 0.005       # per-block grasp perturbation SD (from exploration continuous SD)


def p_succ(eff, c=BAND_CENTER, s=EDGE_SCALE):
    return 1.0 / (1.0 + np.exp((np.abs(eff) - c) / s))


def _best_single_offset_on_train(train_biases, offsets, c, s):
    """Deterministic train-selected best-single offset (max mean p_succ over train biases)."""
    best, best_o = -1.0, offsets[0]
    for o in offsets:
        m = float(np.mean([p_succ(b + o, c, s) for b in train_biases]))
        if m > best:
            best, best_o = m, o
    return best_o


def simulate_once(rng, test_biases, train_biases, n_test_blocks, offsets, c, s, sigma):
    """One simulated confirmatory TEST set; returns per-block gain array (len n_test_blocks)."""
    bs_off = _best_single_offset_on_train(train_biases, offsets, c, s)
    per_block = []
    for _ in range(n_test_blocks):
        delta = rng.normal(0.0, sigma)          # block grasp perturbation, shared in the session
        gains = []
        for b in test_biases:
            # state-aware: pick the offset with max true p_succ for this bias, observe Bernoulli
            effs = np.array([abs(b + o + delta) for o in offsets])
            o_star_p = p_succ(effs, c, s).max()
            sa = rng.random() < o_star_p
            bs = rng.random() < p_succ(b + bs_off + delta, c, s)
            gains.append(float(sa) - float(bs))
        per_block.append(float(np.mean(gains)))
    return np.array(per_block)


def block_bootstrap_ci(per_block, rng, n_boot=1000, ci=0.95):
    n = len(per_block)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[i] = per_block[idx].mean()
    lo = np.quantile(reps, (1 - ci) / 2)
    hi = np.quantile(reps, 1 - (1 - ci) / 2)
    return float(per_block.mean()), float(lo), float(hi)


def power_at(n_test_blocks, *, test_biases=(-0.03, 0.03), train_biases=(-0.04, -0.02, 0.0, 0.02, 0.04),
             offsets=DEFAULT_OFFSETS, threshold=0.15, c=BAND_CENTER, s=EDGE_SCALE, sigma=SIGMA_NUIS,
             n_sims=400, n_boot=800, seed=0):
    """Fraction of simulated datasets whose 95% CI lower bound >= threshold, + mean CI half-width."""
    rng = np.random.default_rng(seed)
    hits, halfw, points = 0, [], []
    for _ in range(n_sims):
        pb = simulate_once(rng, list(test_biases), list(train_biases), n_test_blocks, list(offsets), c, s, sigma)
        pt, lo, hi = block_bootstrap_ci(pb, rng, n_boot=n_boot)
        points.append(pt)
        halfw.append((hi - lo) / 2)
        if lo >= threshold:
            hits += 1
    return {"n_test_blocks": n_test_blocks, "power": hits / n_sims,
            "mean_gain": float(np.mean(points)), "mean_ci_halfwidth": float(np.mean(halfw)),
            "threshold": threshold}


def power_curve(candidates=(6, 9, 12, 15, 18), min_test_blocks=9, target_power=0.8,
                max_halfwidth=0.10, **kw):
    rows = [power_at(n, **kw) for n in candidates]
    # smallest N meeting power AND precision AND the >=9 floor
    chosen = None
    for r in rows:
        if r["n_test_blocks"] >= min_test_blocks and r["power"] >= target_power \
                and r["mean_ci_halfwidth"] <= max_halfwidth:
            chosen = r["n_test_blocks"]
            break
    if chosen is None:
        chosen = max(min_test_blocks, max(c for c in candidates))
    return {"curve": rows, "chosen_test_blocks": chosen, "min_test_blocks_floor": min_test_blocks,
            "target_power": target_power, "max_ci_halfwidth": max_halfwidth,
            "model": {"band_center": kw.get("c", BAND_CENTER), "edge_scale": kw.get("s", EDGE_SCALE),
                      "sigma_nuisance": kw.get("sigma", SIGMA_NUIS)}}
