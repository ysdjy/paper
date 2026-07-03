# Offline evaluation summary — damping_pilot_v2_20260703_000056

- source: `/home1/banghai/Documents/IsaacLab/projects/paper_offline_v2/deployment_calibration/data/damping_pilot_v2_20260703_000056`
- source git commit: `ec18e4cdd347a64cc5e5cb0f2a7950de7f0c87fa`  sha256(episodes): `a732b2dc66bb45b7…`
- episodes: 72  sessions: 9  probes: 27  candidates: 45
- split (by session): {'train': 3, 'val': 3, 'test': 3}  audit ok: **True**  pairwise-disjoint: True  groups-within-split: True

- matched candidate bank: **False** (matched groups: 0)
  - reason: no candidate theta-set is shared across >=2 damping levels; this run is NOT a matched candidate bank, so switch-rate / rank-reversal / VSI are not computable

## Prediction metrics (test)

| model | AUROC (95% CI) | Brier | failMacroF1 | errMAE | timeMAE | meanRegret (CI) | top1 | selSucc |
|---|---|---|---|---|---|---|---|---|
| B0_baserate | 0.500 [0.500, 0.500] | 0.259 | 0.348 | 0.083 | 5.75 | 0.480 [0.000, 1.422] | 0.33 | 0.67 |
| B1_static | 0.536 [0.000, 0.750] | 0.276 | 0.550 | 0.084 | 6.02 | 0.580 [0.000, 1.406] | 0.33 | 0.67 |
| B2_mean | 0.696 [0.000, 0.778] | 0.210 | 0.785 | 0.073 | 6.03 | 0.580 [0.000, 1.406] | 0.33 | 0.67 |
| B2_deepsets | 0.893 [0.250, 1.000] | 0.076 | 0.932 | 0.030 | 4.37 | 0.000 [0.000, 0.000] | 1.00 | 1.00 |
| B2_gru | 0.893 [0.250, 1.000] | 0.069 | 0.932 | 0.027 | 4.16 | 0.000 [0.000, 0.000] | 1.00 | 1.00 |
| OracleZ | 0.714 [0.000, 0.778] | 0.217 | 0.722 | 0.072 | 6.03 | 0.580 [0.000, 1.406] | 0.33 | 0.67 |

## Decision-value / oracle analysis

- VSI / switch / reversal: **not computable** — no candidate theta-set is shared across >=2 damping levels; this run is NOT a matched candidate bank, so switch-rate / rank-reversal / VSI are not computable

## D3 hidden-state identification (held out)

- LOSO accuracy: 1.000 [1.000, 1.000]  chance 0.333  balAcc 1.000  macroF1 1.000
- confusion labels ['L1_low', 'L2_mid', 'L3_high']: [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

_Small-N pilot: treat all numbers as EXPLORATORY; CIs are wide by construction._
