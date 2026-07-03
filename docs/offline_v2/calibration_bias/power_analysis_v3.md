# Pre-data power analysis v3 — runtime-aligned residual nuisance

Claude B, branch `experiment/offline-calibration-bias-v1`. **Pre-data only.** v3 re-runs the block-count
power analysis under a nuisance model the runtime can actually implement, fixing the v2 mismatch.
Reproducible: `offline_v2/calibration_bias/{power,residual_nuisance}.py`; artifact
`confirmatory_design_validation_v3.json`.

## The v2 → v3 fix (model / implementation alignment)
- **v2 assumed** a per-block grasp perturbation `δ` that moved the *effective offset*. The actual A
  runtime injects a calibration bias and adds only initial-joint (σ=0.015 rad, cap 0.03) + target jitter
  (0.005 m); the capability map showed these produced **0 success-label flips**. So v2's "band-edge cells
  produce natural variance" assumption was not backed by the generator.
- **v3 replaces it** with a per-block **residual calibration bias** the runtime CAN inject:
  `actual_bias_y = nominal_bias_y + residual_bias_y`, `residual ~ TruncNormal(0, σ=0.005, [-0.01,+0.01])`,
  fixed per block, reused across the split's bias levels, blocks independent, splits isolated. Because the
  residual is on the SAME axis as the hidden state, it enters `|actual_bias + offset|` — exactly the
  quantity the success band is defined on — so it genuinely flips labels near the edge. Effective SD after
  truncation ≈ **0.0044 m**. residual/actual bias are secret/audit-only; a model reads neither.
- Mathematically the residual enters the success quantity the same way v2's δ did (same ~0.005 scale), so
  the numbers are close — but v3 is now **runtime-faithful and implementable**, which v2 was not.

## Model
`p_succ(eff) = sigmoid((0.03 − |eff|)/s)` (edge s=0.0025 baseline; 0.005 sensitivity), with
`eff = nominal_bias + residual_block + offset`. state-aware picks the best offset per test bias
(deep in band, ~deterministic success); best-single is the train-selected fixed offset (= 0.0), which at
the test biases ±0.03 sits near |eff|=0.03 (the ~50% edge) → genuine per-block variance. Statistic =
success-only gain on the TEST biases; CI = block bootstrap over test blocks; power = P(95% CI lower ≥ 0.15).

## Results (`confirmatory_design_validation_v3.json`)
Baseline (edge s=0.0025):

| N_test | power | mean gain | CI half-width |
|---|---|---|---|
| 6 | 0.94 | 0.51 | 0.18 |
| **9** | **0.97** | 0.50 | 0.16 |
| 12 | 1.00 | 0.51 | 0.15 |
| 18 | 1.00 | 0.50 | 0.12 |

Conservative sensitivity (softer edge s=0.005):

| N_test | power | mean gain | CI half-width |
|---|---|---|---|
| 6 | 0.86 | 0.51 | 0.22 |
| **9** | **0.89** | 0.48 | 0.19 |
| 12 | 0.97 | 0.50 | 0.18 |
| 18 | 0.99 | 0.48 | 0.15 |

At **N_test = 9** the power is **0.97 (baseline) / 0.89 (conservative)** — both above the frozen 0.8
target. The structural gain (~0.5) is large relative to the 0.15 threshold.

## Frozen block counts (v3 — UNCHANGED from v2)
Power under the runtime-aligned residual model still supports the v2 allocation, so it is **retained**:

| split | blocks |
|---|---|
| train | **9** |
| validation | **6** |
| **test** | **9** (≥9 floor; power ≥ 0.89 at N=9 under the conservative model) |

- Blocks disjoint across train/val/test; residual fixed per block, reused across the split's bias levels.
- **Total sessions** = 5·9 + 2·6 + 2·9 = **75**; **total episodes** = **675** — unchanged.
- Block-bootstrap CI: 2000 reps over the (independent) test blocks.
- Optional precision margin: N_test = 12 raises the conservative-model power to 0.97 (half-width 0.18);
  not required to meet the frozen target.

## Honest notes
- The variance is now produced by a mechanism the generator will actually implement (residual moving
  `actual_bias` across the edge), not by an un-injected grasp perturbation. This is the substantive fix.
- Extreme nominal biases ±0.04 with residual ±0.01 give actual ±0.05, still inside the 7-offset
  compensable range (checked in `design_validation.validate_confirmatory_split_residual`); no truncation
  of nominal biases is needed.
- Edge scale s is exploration-uncertain (no grid cell at |eff|=0.03); the sensitivity row shows the
  freeze is robust to a 2× softer edge.
