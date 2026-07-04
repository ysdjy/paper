# Test-geometry power results v1 (offline)

Claude B. Monte-Carlo results for the frozen protocol (`test_geometry_redesign_protocol_v1.md`,
`test_geometry_power_config_v1.json`). Real `models_v2.DeepSets` K=0/K=1, 5 model seeds, empirical hard-edge
labels, block bootstrap (`n_boot=2000`) over test nuisance blocks, **500 replicates per config**. Machine
form: `test_geometry_power_results_v1.{json,csv}`. **No Isaac, no confirmatory data, no preregistration v4.**

## Headline
Moving the test nominal outward to `±0.035` (the only change from the prior phase) makes the pre-registered
power event — **unchanged**, `CI_lower(B2_K1 − train/val-only best-single) ≥ 0.15** — reach **learned power
0.984–0.986 at the smallest design 9/6/9**, at all three primary τ. → **`POWER_SUFFICIENT_FOR_PREREG_V4`**
(`test_geometry_power_verdict_v1.json`).

## Primary geometry ±0.035 (bar: learned power ≥ 0.85 AND MC 95% CI lower ≥ 0.80)
| design | τ | **learned power** [CI95 lo,hi] | struct power | learned gain | struct gain | K1 | K0 | best-single | seed-stable |
|---|---|---|---|---|---|---|---|---|---|
| **9/6/9** | 0.0342 | **0.986** [0.976, 0.996] | 1.000 | 0.523 | 0.568 | 0.955 | 0.432 | 0.432 | 0.99 |
| **9/6/9** | 0.03425 | **0.986** [0.976, 0.996] | 1.000 | 0.520 | 0.564 | 0.956 | 0.436 | 0.436 | 0.99 |
| **9/6/9** | 0.0343 | **0.984** [0.973, 0.995] | 1.000 | 0.517 | 0.560 | 0.957 | 0.440 | 0.440 | 0.99 |
| 12/9/12 | 0.0342 | 0.994 [0.987, 1.000] | 1.000 | 0.531 | 0.570 | 0.961 | 0.430 | 0.430 | 0.99 |
| 12/9/12 | 0.03425 | 0.994 [0.987, 1.000] | 1.000 | 0.528 | 0.565 | 0.963 | 0.435 | 0.435 | 0.99 |
| 12/9/12 | 0.0343 | 0.994 [0.987, 1.000] | 1.000 | 0.526 | 0.561 | 0.965 | 0.439 | 0.439 | 0.99 |
| 18/9/18 | 0.0342 | 1.000 [1.000, 1.000] | 1.000 | 0.534 | 0.567 | 0.967 | 0.433 | 0.433 | 1.00 |
| 18/9/18 | 0.03425 | 1.000 [1.000, 1.000] | 1.000 | 0.531 | 0.563 | 0.968 | 0.437 | 0.437 | 1.00 |
| 18/9/18 | 0.0343 | 1.000 [1.000, 1.000] | 1.000 | 0.527 | 0.558 | 0.968 | 0.442 | 0.442 | 1.00 |

**Minimal passing design = 9/6/9** (passes at all three primary τ; smallest total blocks). Power increases
monotonically with blocks (9/6/9 → 12/9/12 → 18/9/18: 0.986 → 0.994 → 1.000).

## What the numbers say
1. **The geometry fix works exactly as predicted.** best-single (offset 0) drops from ~0.82 (old ±0.03) to
   **~0.43** at ±0.035 because `|eff| = |0.035 + residual| ∈ [0.025, 0.045]` now straddles the empirical
   edge; the oracle/state-aware ±0.04 stays at ~1.0. So the **structural gain is ~0.56** (was ~0.18), well
   clear of 0.15, and its per-block variance is now small relative to the effect → the `CI_lower ≥ 0.15`
   event fires almost always. Structural power = **1.000** everywhere.
2. **The learned model nearly matches the oracle.** K1 test success **0.955–0.968** vs K0 = best-single
   **0.43–0.44**; the **history increment K1−K0 ≈ 0.52** is large and consistent. Learned power **0.984–1.000**.
3. **Threshold NOT lowered, comparator NOT changed.** The 0.15 bar and the primary contrast
   (B2_K1 − train/val-only best-single) are identical to the prior phase. Only the test nominal moved.
4. **best-single is perfectly stable**: offset 0 in **500/500** replicates at every design (see
   `best_single_offset_counts`), so the contrast is clean and non-degenerate.
5. **Seed stability is now ~0.97–1.00** (was 0.44): every model seed clears the point-gain bar in almost
   every replicate, because the effect (~0.52) sits far above 0.15 rather than on top of it.

## Structural vs learned (reported separately, never substituted)
See `test_geometry_structural_vs_learned_v1.md`. Structural power is 1.000 (the design is now powerable in
principle); learned power 0.984–1.000 (the real DeepSets realizes almost all of it). We report both; the
oracle is not used as a stand-in.

## Net VOI (now positive one-shot — but kept secondary)
gross success VOI ≈ 0.520; one-shot probe cost 0.02×16.58 ≈ 0.332 → **one-shot Net VOI ≈ +0.188**;
break-even at **T ≈ 0.64 tasks** (i.e. a single repeated-task deployment already amortizes the probe). Even
so, the primary claim remains the success/history-increment mechanism, not VOI
(`test_geometry_probe_cost_v1.md`).

## Verdict → `POWER_SUFFICIENT_FOR_PREREG_V4`
The mechanism is real and learnable (K1 ≫ K0, increment ~0.52), the design is powerable at the **smallest**
9/6/9 grid across all primary τ (learned power ≥ 0.984, CI lower ≥ 0.973), best-single and seeds are stable,
no leakage, and neither the 0.15 threshold nor the comparator was weakened. **This phase issues no
preregistration v4 and generates no runtime/confirmatory data** — it certifies that a v4 confirmatory design
at ±0.035 / 9-6-9 would be adequately powered.
