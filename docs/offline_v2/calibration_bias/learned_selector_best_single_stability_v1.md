# Learned-selector best-single stability v1 (offline)

Claude B. Distribution of the train/val-only best-single offset across Monte-Carlo replicates for the
primary 9/6/9 design (τ=0.03425, 50 replicates). Machine form in `learned_selector_power_verdict_v1.json`
(`best_single_stability`).

## Result: best-single = 0.00 in **50 / 50** replicates
| best-single offset | count |
|---|---|
| **0.00** | **50** |
| −0.04 | 0 |
| +0.04 | 0 |

- `mostly_zero` = true; `switches_to_±0.04` = **false**.

## Reading
- Design A's best-single is **perfectly stable at offset 0** — it never switches to ±0.04 under fresh
  residual draws. This confirms the redesign-phase finding that Design A's best-single wins by a wide
  train/val margin (~0.33 over ±0.04), unlike the current 7-point bank (Design C), whose best-single is a
  thin tie (0 vs ±0.02) that flips to the degeneracy-inducing ±0.02.
- So the **decision geometry is sound**: the state-agnostic operating point is robustly the edge-straddling
  offset 0, and the primary contrast (state-aware vs best-single) is non-degenerate — the residual flips
  best-single labels at the test biases as intended.
- The remaining problem is purely the **effect size vs the 0.15 CI bar** (see the results / structural-vs-
  learned docs), not best-single instability. A stable best-single is exactly what a clean confirmatory
  contrast needs; the bank redesign delivered that, but the resulting gain (~0.18) is too marginal for the
  strict pre-registered power event.
