# Learned-selector threshold sensitivity v1 (offline)

Claude B. Sensitivity of the learned-selector power to the empirical hard-edge threshold τ over the
identified primary interval {0.0342, 0.03425, 0.0343}; τ=0.0325 (isotonic) is a sensitivity anchor only.
No τ is chosen to maximize power.

## Learned power & gain vs τ (9/6/9)
| τ | learned power | learned gain | struct power | struct gain |
|---|---|---|---|---|
| 0.0342 (lower) | 0.075 | 0.169 | 0.065 | 0.184 |
| 0.03425 (point) | 0.075 | 0.167 | 0.060 | 0.182 |
| 0.0343 (upper) | 0.065 | 0.165 | 0.060 | 0.179 |

## Robustness reading
- The conclusion is **robust across the whole primary τ interval**: learned power is ~0.065–0.075 and the
  gain ~0.165–0.169 at every τ; struct power ~0.06–0.065; nowhere near the 0.85 bar.
- The **worst τ is the upper bound 0.0343** (most permissive edge → best-single succeeds slightly more →
  smallest gain 0.165 and lowest power 0.065 at 9/6/9, 0.040 at 18/9/18). Requiring power at **all three**
  primary τ (not the best τ) is what the protocol demands — and the design fails at all three.
- Larger designs are also τ-robust in their failure: 18/9/18 gives learned power 0.053 (0.03425) / 0.040
  (0.0343).
- τ=0.0325 (isotonic) is not run as a gate; the identified interval [0.0342, 0.0343] is the primary basis
  and the point estimate 0.03425 is representative. The tight interval (0.0001 m wide, from the completely-
  separated 306 data) means the edge location itself is essentially certain; it is the **effect size vs the
  0.15 threshold** — not τ uncertainty — that drives the low power.

## No τ-cherry-picking
Per protocol, power is reported at every τ and the verdict requires the bar at **all** primary τ. We do NOT
report only the most favorable τ. Since the design fails at every τ (and even the oracle does), τ is not the
lever — the claim/threshold is (`learned_selector_power_verdict_v1.json`).
