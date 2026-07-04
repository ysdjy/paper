# Learned-selector power results v1 (offline)

Claude B. Monte-Carlo results for the frozen protocol (`learned_selector_power_protocol_v1.md`,
`learned_selector_power_config_v1.json`). Real `models_v2.DeepSets` K=0/K=1, 5 model seeds, empirical
hard-edge labels, block-bootstrap over test nuisance blocks. Machine form:
`learned_selector_power_results_v1.{json,csv}`. **No Isaac, no confirmatory data, no preregistration v4.**

## Headline
The learned DeepSets K=1 **does** use history (mechanism confirmed) but the pre-registered power event is
**not** met at any design — and **neither is the oracle's**, so the block-bootstrap `CI_lower ≥ 0.15` bar is
unreachable for this ~0.18 effect. → **`REDESIGN_MODEL_OR_CLAIM`** (see `learned_selector_power_verdict_v1.json`).

## Results table (reps: 200 primary / 150 grid; n_boot 2000)
| design (tr/va/te) | τ | learned power | struct (oracle) power | learned gain | K1 succ | K0 succ | K1−K0 |
|---|---|---|---|---|---|---|---|
| 9/6/9 | 0.0342 | 0.075 | 0.065 | 0.169 | 0.984 | 0.815 | 0.169 |
| 9/6/9 | 0.03425 | 0.075 | 0.060 | 0.167 | 0.985 | 0.818 | 0.167 |
| 9/6/9 | 0.0343 | 0.065 | 0.060 | 0.165 | 0.985 | 0.820 | 0.165 |
| 12/9/12 | 0.03425 | 0.053 | 0.107 | 0.161 | 0.987 | 0.826 | 0.161 |
| 12/9/12 | 0.0343 | 0.053 | 0.100 | 0.160 | 0.988 | 0.828 | 0.160 |
| 18/9/18 | 0.03425 | 0.053 | 0.073 | 0.166 | 0.992 | 0.826 | 0.166 |
| 18/9/18 | 0.0343 | 0.040 | 0.073 | 0.162 | 0.992 | 0.831 | 0.162 |

(acceptance bar: learned power ≥ 0.85 AND its 95% MC CI lower ≥ 0.80. None approach it.)

## What the numbers say
1. **B2/K1 genuinely learns to use history.** K1 test success ≈ **0.98–0.99** vs the capacity-matched K=0
   ≈ **0.82** (which equals best-single, since with empty history all sessions look identical). The
   **history increment K1−K0 ≈ 0.16–0.17** is large and consistent across every design and τ. The mechanism
   — "one probe lets the selector pick the compensating action" — is demonstrated with the real model.
2. **But learned power ≈ 0.04–0.08 (target 0.85).** Two compounding reasons:
   - **The gain (~0.16–0.18) is barely above the 0.15 threshold.** With fresh residuals from the frozen
     TruncNormal(0,0.005,[-0.01,0.01]) the best-single (offset 0) already succeeds ~0.82 (more residuals sit
     near 0 than the lucky empirical-18 set gave in the redesign phase, where best-single was 0.75). So the
     honest expected gain is ~0.18, not 0.25.
   - **The per-block gain is a high-variance discrete quantity** (mean ~0.18, SD ~0.35–0.5, values in
     {−1,…,+1} over 2 test biases). A 95% CI **lower** bound ≥ 0.15 for a mean of 0.18 needs SE ≲ 0.015 →
     **hundreds of test blocks**. Escalation 9→12→18 test blocks moves struct power only 0.06→0.10→0.07 —
     nowhere near 0.85.
3. **Even the ORACLE fails the event** (structural power ≤ 0.11). So this is **not** a learned-model or
   block-count deficiency — the pre-registered `CI_lower ≥ 0.15` event is **structurally unreachable** for a
   ~0.18-magnitude, high-variance decision-value effect. Hence not `INCREASE_BLOCK_COUNTS`.
4. **Learned < structural** at matched design (e.g. 12/9/12: learned 0.053 vs struct 0.10) because of
   **model-seed instability**: the probe=fail class has an offset-0-vs-+0.04 train tie, so a fraction of
   seeds mis-pick for test −0.03 (`learned_selector_seed_stability_v1.json`). More train raises the mean
   gain (0.16→0.18) but does not remove the seed instability.

## Structural vs learned (reported separately, never substituted)
See `learned_selector_structural_vs_learned_power_v1.md`. Both are low; the learned is further capped by
seed instability. Structural power is **not** used as a stand-in for learned power.

## Net VOI (one-shot honestly negative)
gross success VOI ≈ 0.167; one-shot probe cost 0.02×16.58 ≈ 0.332 → **one-shot Net VOI ≈ −0.16**. Amortized
break-even ≈ T ≈ 2 tasks; T must come from deployment semantics, not these results
(`probe_cost_and_claim_scope_v1.md`).

## Verdict → `REDESIGN_MODEL_OR_CLAIM`
The mechanism is real and learnable (K1 ≫ K0), but the pre-registered power event cannot be met — the
effect (~0.18) is too close to the 0.15 threshold and too high-variance per block for any reasonable block
count (the oracle can't meet it either). The **claim/threshold** must change (recommended: make the
**history increment K1 vs K0** the primary quantity, and/or move the test biases further from the edge, e.g.
±0.035, for a larger lower-variance gain — a future redesign). **This phase issues no preregistration v4 and
generates no runtime/confirmatory data.**
