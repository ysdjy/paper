# Pre-data power analysis — calibration-bias confirmatory v2

Claude B, branch `experiment/offline-calibration-bias-v1`. Sizes the number of independent nuisance
blocks for the v2 confirmatory pilot. **Pre-data only** — uses the exploration-derived effect size and
continuous variance; reads NO confirmatory data. Reproducible:
`offline_v2/calibration_bias/power.py`; artifact `confirmatory_design_validation_v2.json`.

## Statistic and model
- **Quantity**: success-only H3 gain on the TEST biases {−0.03, +0.03} =
  mean over (test bias, test block) of [state-aware selected success − train-selected best-single-offset
  success]. On this split a fixed offset covers at most one of the two test biases (empty success-offset
  intersection), so the structural gain ≈ **0.5**.
- **Success model** (calibrated to the capability map: success at |eff|=0.02, fail at |eff|≥0.04):
  `p_succ(eff) = sigmoid((0.03 − |eff|)/s)`, sharp edge `s = 0.0025`.
- **Nuisance**: a per-block grasp perturbation `δ ~ N(0, σ)` shared within a session shifts the effective
  offset; `σ = 0.005 m` from the exploration within-cell continuous SD.
- **CI**: block bootstrap over the TEST blocks (2000 reps, 95%); each block contributes its test-bias rows.
- **Power**: fraction of simulated confirmatory datasets whose 95% CI lower bound ≥ **0.15** (the frozen
  H3 threshold).

## Results (`confirmatory_design_validation_v2.json`)
Baseline (sharp edge s=0.0025, σ=0.005):

| N_test blocks | power (CI_low ≥ 0.15) | mean gain | CI half-width |
|---|---|---|---|
| 6 | 0.97 | 0.50 | 0.16 |
| **9** | **0.97** | 0.49 | 0.16 |
| 12 | 0.99 | 0.50 | 0.14 |
| 15 | 0.99 | 0.50 | 0.13 |
| 18 | 1.00 | 0.50 | 0.12 |

Conservative sensitivity (softer edge s=0.005, larger σ=0.008):

| N_test blocks | power | mean gain | CI half-width |
|---|---|---|---|
| 6 | 0.94 | 0.50 | 0.18 |
| **9** | **0.94** | 0.49 | 0.18 |
| 12 | 0.96 | 0.49 | 0.15 |
| 18 | 1.00 | 0.50 | 0.13 |

The structural gain (~0.5) is large relative to the 0.15 threshold, so **power ≥ 0.94 at N_test = 9**
under both models — comfortably above the 0.8 target. The CI half-width (~0.16–0.18 at N=9) is driven by
the discreteness of the per-block gain (each block averages two Bernoulli test biases); it still excludes
0.15 with margin (CI ≈ [0.33, 0.66] at gain 0.5).

## Frozen block counts (v2)
| split | blocks | rationale |
|---|---|---|
| **train** | 9 | mirrors test; pairs across the 5 train biases |
| **validation** | 6 | threshold / hyperparameter selection only |
| **test** | **9** | smallest N ≥ 9 (protocol floor) with power ≥ 0.8 under the conservative model |

- Blocks are **disjoint** across train/val/test (a block never crosses a split); within a split a block
  pairs across that split's bias levels.
- **Total sessions** = 5·9 (train) + 2·6 (val) + 2·9 (test) = **75**; **total episodes** =
  75 × (2 probes + 7 candidates) = **675**.
- Power analysis does **not** require more than the 9-block floor; a reviewer wanting a tighter CI may
  raise test blocks to 12–18 (half-width 0.15→0.12) — this is optional precision, not a power requirement.

## Honest caveats
- The success label is near-deterministic mid-band (as in exploration); the success-gain variance the CI
  captures comes from the **band-edge** cells and the two-Bernoulli test average. The confirmatory grid
  (0.01-bias × 0.02-offset) places several cells near |eff| = 0.02, giving genuine edge variance — this is
  a design property, **not** injected noise.
- Secondary continuous metrics (task_error, time, handle error) carry more per-block variance and their
  CIs will be looser; they are reported as supporting evidence, not the H3 gate.
- The model parameters (edge scale, σ) are exploration-calibrated estimates; the sensitivity row shows the
  freeze is robust to a 2× softer edge and 1.6× larger nuisance.
