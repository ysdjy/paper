# Structural vs learned power v1 — test geometry ±0.035 (offline)

Claude B. Reports the **structural/oracle** decision-value power and the **learned** B2/K1 power
**separately** for every design/τ at the primary ±0.035 geometry. Per protocol, oracle power is **never**
substituted for learned power. Contrast with the prior phase, where *both* were low (≤0.11).

## Per design/τ (primary geometry ±0.035)
| design | τ | struct gain | **struct power** | learned gain | **learned power** | K1−K0 |
|---|---|---|---|---|---|---|
| 9/6/9 | 0.0342 | 0.568 | 1.000 | 0.523 | 0.986 | 0.523 |
| 9/6/9 | 0.03425 | 0.564 | 1.000 | 0.520 | 0.986 | 0.520 |
| 9/6/9 | 0.0343 | 0.560 | 1.000 | 0.517 | 0.984 | 0.517 |
| 12/9/12 | 0.03425 | 0.565 | 1.000 | 0.528 | 0.994 | 0.528 |
| 18/9/18 | 0.03425 | 0.563 | 1.000 | 0.531 | 1.000 | 0.531 |

## Three distinct facts
1. **The design is now powerable in principle (structural power = 1.000).** With the test nominal at ±0.035
   the oracle gain is ~0.56 and its per-block variance is small relative to the effect, so the block-bootstrap
   `CI_lower ≥ 0.15` event fires in essentially every replicate — at the smallest 9/6/9 grid. In the prior
   ±0.03 phase the oracle gain was only ~0.18 and structural power ≤ 0.11; the fix is entirely in the test
   geometry, not the model or the threshold.
2. **The learned model realizes almost all of the structural power (0.984–1.000).** Learned power tracks
   structural power closely now (gap ≤ 0.05), because the large effect leaves the model far from the 0.15
   bar — the model-seed instability that capped learned power below structural in the prior phase is gone
   (seed-stable ≈ 0.99, vs 0.44 before). The tiny residual gap is finite-sample training noise, not a
   structural deficit.
3. **The mechanism is the same real one, now with a decisive margin.** K1 test success ~0.955–0.968 vs
   K0 = best-single ~0.43–0.44; the history increment K1−K0 ~0.52 is ~3× the prior phase's ~0.17 because the
   probe now flips the *outcome* over a region where the state-agnostic action genuinely fails.

## Consequence (protocol-mandated separation)
Structural power (1.000) and learned power (0.984–1.000) are reported **separately** and are both high; the
oracle is **not** used as a stand-in for the learned result. Because the learned model independently clears
the bar at the smallest design across all primary τ, the honest reading is
`POWER_SUFFICIENT_FOR_PREREG_V4`: the pre-registered strict CI event, unchanged and un-weakened, is now
powerable **and** realized by the real DeepSets.
