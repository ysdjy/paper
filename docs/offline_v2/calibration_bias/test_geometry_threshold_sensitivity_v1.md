# Test-geometry threshold & geometry sensitivity v1 (offline)

Claude B. Two sensitivity axes for the `POWER_SUFFICIENT_FOR_PREREG_V4` conclusion: (a) the empirical edge
threshold τ over the primary interval {0.0342, 0.03425, 0.0343} (τ=0.0325 isotonic is an anchor only), and
(b) the pre-registered test-nominal geometry anchors ±0.034 / ±0.036 (9/6/9). Neither is used to *select*
the primary; the primary is frozen at ±0.035 (protocol §10).

## (a) τ sensitivity at the primary geometry ±0.035, 9/6/9
| τ | learned power [CI95 lo] | struct power | learned gain | best-single |
|---|---|---|---|---|
| 0.0342 (lower) | 0.986 [0.976] | 1.000 | 0.523 | 0.432 |
| 0.03425 (point) | 0.986 [0.976] | 1.000 | 0.520 | 0.436 |
| 0.0343 (upper) | 0.984 [0.973] | 1.000 | 0.517 | 0.440 |

- **Robust across the whole primary τ interval**: learned power 0.984–0.986, all clearing the 0.85 bar with
  CI lower ≥ 0.973 (≥ 0.80 required). The worst τ is the upper 0.0343 (most permissive edge → best-single
  succeeds slightly more → smallest gain 0.517), and it still passes comfortably.
- The tight edge interval (0.0001 m wide, from the completely-separated 306 data) means τ uncertainty is
  negligible; the pass is not τ-fragile.

## (b) Geometry sensitivity ±0.034 / ±0.036 (9/6/9, all primary τ)
| geometry | τ | learned power [CI95 lo] | struct power | learned gain | best-single |
|---|---|---|---|---|---|
| **±0.034** (tight) | 0.0342 | 0.966 [0.950] | 1.000 | 0.445 | 0.516 |
| **±0.034** | 0.03425 | 0.964 [0.948] | 1.000 | 0.442 | 0.520 |
| **±0.034** | 0.0343 | 0.958 [0.940] | 1.000 | 0.438 | 0.525 |
| **±0.035** (primary) | 0.03425 | 0.986 [0.976] | 1.000 | 0.520 | 0.436 |
| **±0.036** (wide) | 0.0342 | 0.998 [0.994] | 1.000 | 0.597 | 0.351 |
| **±0.036** | 0.03425 | 0.998 [0.994] | 1.000 | 0.596 | 0.354 |
| **±0.036** | 0.0343 | 0.996 [0.990] | 1.000 | 0.592 | 0.359 |

- **The conclusion is monotone and robust in the geometry**: pushing the test nominal farther from the edge
  raises best-single's failure rate → larger gain → higher power. ±0.034 (closest to the edge, hardest)
  still passes at **0.958–0.966** with CI lower ≥ 0.940; ±0.036 passes at **0.996–0.998**.
- The primary ±0.035 sits between the two anchors (0.984–0.986), exactly as expected. So the pass does
  **not** hinge on the precise ±0.035 choice — every geometry in [±0.034, ±0.036] clears the bar at the
  smallest 9/6/9 design.

## No post-hoc geometry selection
Per protocol §10, the primary is **±0.035 only**; ±0.034/±0.036 are sensitivity anchors and cannot replace
it. Here the point is moot in the favorable direction: the primary passes on its own, and the anchors merely
show the result is not knife-edge. Had the primary failed and only a wider geometry passed, we would **not**
have swapped — a new freeze would be required. That contingency did not arise.

## Reading
The `POWER_SUFFICIENT_FOR_PREREG_V4` verdict holds at every primary τ **and** across the entire pre-allowed
geometry band, at the smallest evaluated design. τ is not a lever here (all pass); geometry moved the effect
from ~0.18 (old ±0.03, unpowerable) to ~0.44–0.60 (powerable), which is the whole finding.
