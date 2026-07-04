# Structural vs learned power v1 (offline)

Claude B. Reports the **structural/oracle** decision-value power and the **learned** B2 power **separately**
for every design/τ. Per the protocol, oracle power is **never** substituted for learned power.

## Per design/τ (learned vs structural)
| design | τ | struct gain | **struct power** | learned gain | **learned power** | K1−K0 |
|---|---|---|---|---|---|---|
| 9/6/9 | 0.0342 | 0.184 | 0.065 | 0.169 | 0.075 | 0.169 |
| 9/6/9 | 0.03425 | 0.182 | 0.060 | 0.167 | 0.075 | 0.167 |
| 9/6/9 | 0.0343 | 0.179 | 0.060 | 0.165 | 0.065 | 0.165 |
| 12/9/12 | 0.03425 | 0.194 | 0.107 | 0.161 | 0.053 | 0.161 |
| 12/9/12 | 0.0343 | 0.191 | 0.100 | 0.160 | 0.053 | 0.160 |
| 18/9/18 | 0.03425 | 0.180 | 0.073 | 0.166 | 0.053 | 0.166 |
| 18/9/18 | 0.0343 | 0.178 | 0.073 | 0.162 | 0.040 | 0.162 |

## Three distinct facts
1. **Structural/oracle power is itself low (≤ 0.11).** The oracle gain (~0.18) exceeds the 0.15 threshold,
   yet the block-bootstrap `CI_lower ≥ 0.15` event holds in only ~6–11% of replicates because the per-block
   gain is high-variance and 9–18 test blocks give a wide CI. **The event is unreachable at reasonable block
   counts even with perfect state knowledge.** This is the primary blocker and is design-level, not model-level.
2. **Learned power is even lower than structural** at matched design (e.g. 12/9/12: learned 0.053 vs struct
   0.107). The gap is the DeepSets' finite-sample + **model-seed instability** (see
   `learned_selector_seed_stability_v1.md`): the probe=fail class train-tie (offset 0 vs +0.04) makes a
   fraction of seeds mis-pick for test −0.03, so per-seed gains scatter around and often below 0.15.
3. **The mechanism is real regardless of power.** K1 test success ~0.98–0.99 vs K0 ~0.82; the history
   increment K1−K0 ~0.16–0.17 is large and consistent. B2 clearly learns to exploit the probe; it just
   cannot make the strict, high-precision CI event fire.

## Consequence (protocol-mandated)
Because **structural power is high-effect-but-low (≤0.11) and learned power is even lower**, the honest
reading is **not** "model too weak, add blocks" (the oracle also fails → block escalation won't reach the
bar) and **not** "substitute oracle power" (forbidden). It is a **threshold/claim** problem →
`REDESIGN_MODEL_OR_CLAIM`. The reportable, robust quantity is the **history increment (K1 vs K0)** and the
**success** decision value, not the strict gain-over-best-single CI event.
