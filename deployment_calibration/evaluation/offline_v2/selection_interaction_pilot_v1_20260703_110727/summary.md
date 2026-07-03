# Offline evaluation summary — selection_interaction_pilot_v1_20260703_110727

- source: `/home1/banghai/Documents/IsaacLab/projects/paper/deployment_calibration/data/selection_interaction_pilot_v1_20260703_110727`
- source git commit: `1921641acec5a7e153f45f909c69613ed6036a12`  sha256(episodes): `8f88cb79063a73bd…`
- episodes: 162  sessions: 9  probes: 27  candidates: 135
- split (by session): {'train': 3, 'val': 3, 'test': 3}  audit ok: **True**  pairwise-disjoint: True  groups-within-split: True

- matched candidate bank: **True** (matched groups: 9)

## Prediction metrics (test)

| model | AUROC (95% CI) | Brier | failMacroF1 | errMAE | timeMAE | meanRegret (CI) | top1 | selSucc |
|---|---|---|---|---|---|---|---|---|
| B0_baserate | 0.500 [0.500, 0.500] | 0.235 | 0.384 | 0.060 | 0.47 | 0.026 [0.025, 0.028] | 0.00 | 1.00 |
| B1_static | 0.872 [0.796, 1.000] | 0.152 | 0.669 | 0.038 | 0.44 | 0.026 [0.025, 0.028] | 0.00 | 1.00 |
| B2_mean | 1.000 [1.000, 1.000] | 0.022 | 0.977 | 0.032 | 0.38 | 0.021 [0.013, 0.026] | 0.11 | 1.00 |
| B2_deepsets | 1.000 [1.000, 1.000] | 0.000 | 1.000 | 0.002 | 0.03 | 0.000 [0.000, 0.000] | 0.89 | 1.00 |
| B2_gru | 1.000 [1.000, 1.000] | 0.000 | 1.000 | 0.002 | 0.02 | 0.000 [0.000, 0.000] | 0.89 | 1.00 |
| OracleZ | 0.968 [0.905, 1.000] | 0.072 | 0.855 | 0.031 | 0.44 | 0.021 [0.013, 0.026] | 0.11 | 1.00 |

## Decision-value / oracle analysis

- VSI = 0.0006  (state-aware 0.7958 − state-agnostic 0.7952)
- optimal-candidate switch rate: 0.6666666666666666
- pairwise rank-reversal rate: 0.17037037037037037

## D3 hidden-state identification (held out)

- LOSO accuracy: 1.000 [1.000, 1.000]  chance 0.333  balAcc 1.000  macroF1 1.000
- confusion labels ['L1_low', 'L2_mid', 'L3_high']: [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

_Small-N pilot: treat all numbers as EXPLORATORY; CIs are wide by construction._
