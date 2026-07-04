# Learned-selector seed stability v1 (offline)

Claude B. Per-model-seed behaviour of the learned DeepSets K=1 on the primary 9/6/9 design at τ=0.03425
(50 replicates). Every seed reported — never cherry-picked. Machine form:
`learned_selector_seed_stability_v1.json`.

## Frozen seeds {1103, 2207, 3301, 4409, 5519}
| seed | mean gain | frac gain ≥ 0.15 | frac gain < 0 |
|---|---|---|---|
| 1103 | 0.154 | 0.50 | 0.0 |
| 2207 | 0.139 | 0.44 | 0.0 |
| 3301 | 0.139 | 0.44 | 0.0 |
| 4409 | 0.142 | 0.42 | 0.0 |
| 5519 | 0.162 | 0.54 | 0.0 |

- mean seeds passing (≥0.15) per replicate: **~2.3 / 5**; fraction of replicates with **≥4/5** seeds
  passing (the frozen seed-stability gate): **0.44**.
- mean K1 ≈ 0.985, mean K0 ≈ 0.82.

## Reading
- **No seed is broken** (all mean gains 0.14–0.16, none negative on average): every seed learns to use the
  probe and beats K=0. So the mechanism is robust across seeds.
- **But every seed sits right at the 0.15 threshold** (mean gain ~0.14–0.16), so on any single replicate a
  seed clears 0.15 only ~42–54% of the time. The seed-stability gate (≥4/5 seeds ≥0.15) therefore holds in
  only **44%** of replicates.
- The remaining seed-to-seed variance traces to the **probe=fail train tie** (offset 0 vs +0.04 both succeed
  at 3/4 of probe=fail train nominals), so a seed may or may not learn to prefer +0.04 for test −0.03. More
  train blocks raise the mean gain (0.16→0.18) but do **not** remove this tie-driven variance.

## Consequence
The seed instability is a **second-order** contributor: the primary blocker is that the effect (~0.18) is
too close to the 0.15 threshold to power (even the oracle fails the CI event). The seed variance simply
pushes the learned power (~0.05) below the already-low structural power (~0.10). Reported per seed, not
aggregated to hide it.
