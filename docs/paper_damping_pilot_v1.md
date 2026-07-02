# Damping pilot — v1  (Go / Modify / Stop)

Run `damping_pilot_v2_20260703_000056`: middle_drawer, hidden drawer-joint DAMPING fixed per session
at 3 levels **L1=3 / L2=20 / L3=40**, 3 sessions/level = 9 sessions × (3 probes + 1 group × 5 candidates)
= **72 episodes**. Damping **set + persist VERIFIED** (max set error 0.00; not overridden — the adapter
path does not call the GUI controller's dynamic damping). Frozen theta from the capability map.

## Per-level success (monotonic, non-degenerate)
| level (damping) | probes | candidates |
|---|---|---|
| L1 low (3) | 9/9 | 10/15 |
| L2 mid (20) | 6/9 | 5/15 |
| L3 high (40) | 3/9 | 3/15 |

Damping cleanly shifts both probe and candidate outcomes; no level is all-success or all-fail.

## D1–D5
Eval: `evaluation/run_session_eval_v2.py` (split BY SESSION 3/3/3; history built only from same-session
probes before the decision; leakage self-test PASS). Frozen utility U = p_succ − 1.0·err − 0.02·time.

- **D1 — damping changes y? YES.** Same (x,g,θ): mean final spread across levels **0.118 m**, mean
  pull-duration spread **1.14 s**. Strong, physical, monotonic.
- **D2 — does damping leak from initial x? NO.** `x_std across all = 0.0` — initial state is identical
  across levels (drawer at 0, robot home), so damping is genuinely hidden. A no-history model cannot
  read it off x (B1 success-AUROC 0.54 ≈ chance).
- **D3 — does probe history identify the hidden state? YES, strongly.** Nearest-centroid on probe-history
  features classifies the damping level with **acc = 1.0** (chance 0.33).
- **D4 — does history improve candidate prediction? YES.** Success AUROC **B1 0.54 → B2 0.71**, matching
  **Oracle-Z 0.71**; Brier **0.28 → 0.21**; error-MAE 0.084 → 0.072. Adaptation curve (B2-Seq):
  **K=0: 0.54 → K=1: 0.71 → K≥2: ~0.70** — a single probe lifts prediction to the Oracle ceiling.
- **D5 — does the prediction gain reduce selection regret? INCONCLUSIVE at pilot scale.** Regret is flat
  (B1 = B2 = Oracle-Z = 0.58) and selected-success = 0.667 for **every** predictor including Oracle-Z.
  With only **3 test candidate groups**, the selection metric is quantized and near its ceiling (one of
  the three groups appears to have no clearly-best candidate), so regret cannot separate the models here.

## Verdict: **MODIFY** (borderline-GO)

Go conditions met: damping effect (D1) ✓, not leaked from x (D2) ✓, history has incremental info (D3) ✓,
B2 > B1 up to the Oracle ceiling (D4) ✓, no leakage ✓, reset/label/leakage tests ✓.

Not met: **candidate-selection regret improvement (D5)** — but this maps to Modify condition #4
("prediction improves, selection does not") *driven by pilot scale*: 3 test groups is too few to measure
regret, and Oracle-Z shows the same flat regret, i.e. the ceiling itself is quantized here, not that the
signal is absent. This is **not** a Stop: Oracle-Z genuinely improves prediction, so the calibration
signal is real; nothing needs the execution system rebuilt.

### Modifications before/with the formal run
1. **More candidate groups per session** (≥3) and **multiple targets** so selection regret has enough
   independent groups to be significant (formal: 24 sessions × 3 groups × 5 candidates → ~72 test groups).
2. Keep the utility frozen (U above); re-fit λ on validation only in the formal run.
3. Keep L1/L2/L3 = 3/20/40 (clean monotonic band) for the single-drawer core; add sektion_top at higher
   damping for cross-mechanism generalization once single-drawer regret is demonstrated.

## Bottom line
The core paper claim is supported at pilot scale: **a hidden deployment state (damping) changes execution
outcomes, is invisible in x, and is recoverable from a few probe outcomes — lifting candidate-outcome
prediction from chance (0.54) to the Oracle ceiling (0.71) with a single probe.** The selection-regret
payoff needs the formal-scale run (more groups) to be measured; the pilot is otherwise GO-quality.
