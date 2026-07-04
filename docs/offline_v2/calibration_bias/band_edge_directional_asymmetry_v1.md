# Band-edge directional asymmetry v1 (read-only)

Claude B. Whether the success boundary is symmetric in the sign of `eff_signed = actual_bias + offset`.
Block-aware throughout (no episode-level independence). Plot: `band_edge_directional_v1.png`.

## Verdict: **NO_SEVERE_ASYMMETRY**

## Method (frozen)
Fit the frozen logistic separately on `eff_signed < 0` (149 eps) and `eff_signed > 0` (155 eps) vs `|eff|`;
compare edge centers; also compare matched 0.005-bin success rates. Threshold for "severe":
|Δcenter| > 0.005 m (frozen).

## Results
| direction | edge center | edge scale |
|---|---|---|
| eff_signed < 0 | 0.03386 | hard (unidentifiable) |
| eff_signed > 0 | 0.03452 | hard (unidentifiable) |

- **|Δcenter| = 0.00067 m < 0.005 m** → not severe.
- Matched-bin success-rate differences are small; both directions are ~100% success below ~0.033 and 0%
  above ~0.035, with the mixed transition bin around 0.030–0.035 on both sides.
- Both directions show the same **hard** edge (scale unidentifiable), consistent with the overall boundary.

The ~0.0007 m center difference is well within the 0.005 m tolerance and within the 0.005 m grid/residual
resolution; there is no unexplained physical asymmetry between compensating a negative vs positive handle
bias.
