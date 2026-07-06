# Scientific pilot report — probe history, hidden state, and decision value

**EXPLORATORY / PILOT — NOT CONFIRMATORY.** Offline re-analysis of the existing 306-episode band-edge run
(`band_edge_full_306_v1`, sha256 `131750e1…`); no new simulation, no Isaac, no confirmatory seeds. Machine
summary: `scientific_pilot_summary.json`. Resampling and data-split unit = **block** (18 blocks). n=18 is small
— all CIs are exploratory.

## Headline
> The hidden deployment state **is strongly identifiable from probe execution history** (residual-bias
> regression R² ≈ 0.86; sign accuracy 0.87), but it carries **zero decision value**: a single fixed offset
> already succeeds on **every** block (oracle headroom = 0.00), so K1 (probe history) does not beat K0 or
> best-single (K1−K0 = 0.00). This is an *identifiability-without-decision-value* regime, driven by the hidden
> residual (±9 mm) being far smaller than the grasp success tolerance band (offsets −0.025…+0.025 succeed
> everywhere). **Status: `SCIENTIFIC_PILOT_REQUIRES_HYPOTHESIS_OR_TASK_REVISION`.**

## Q1 — Does the task need hidden-state-adaptive selection? (oracle headroom) — **NO**
- oracle success = **1.000** (≥1 offset works in every block); best-single (leave-one-block-out global offset)
  = **1.000**. **oracle headroom = 0.000, 95% block-bootstrap CI [0.000, 0.000].**
- Per-offset success shows a wide central plateau (offsets −0.025…+0.025 = 100%) and a band edge only at the
  extremes (0.03 ≈ 0.72–0.78, 0.035 ≈ 0.39–0.56, 0.04 ≈ 0.11–0.17, ±0.05 = 0). The residual variation never
  moves a block out of the central plateau.
- **Go/No-Go 1: FAIL** (0.00 ≪ 0.15). *Figures:* `fig_pilot_oracle_headroom.png`,
  `fig_pilot_candidate_success_by_residual.png`. *Table:* `pilot_session_table.csv`.

## Q2 — Does probe history contain hidden-residual information? — **YES (strong)**
- K1 (static + probe-offset execution history) predicts the continuous residual with **R² = 0.862**
  (MAE ≈ 0.0011 m) vs. K0 (static only) **R² = −0.125** (no signal). Residual-sign balanced accuracy:
  **K1 = 0.867 vs K0 = 0.717** (5 seeds, GroupKFold by block), **K1−K0 = +0.150**.
- Most informative probe features (|corr| with residual): `probe_phasedur_MOVE_TO_PRE_GRASP` (−0.97),
  `probe_phase_goal_error` (−0.96), `probe_task_outcome_error` (+0.53), `probe_skill_elapsed_time` (−0.53).
- **Leakage check: PASS** (`scientific_pilot_leakage_check.json`) — no residual/actual/nominal-bias,
  eff/abs_eff, true-handle-error, oracle, or candidate-outcome field enters K0/K1.
- **Go/No-Go 2: PASS** (K1 ≥ 0.65 and K1−K0 ≥ 0.10). The K0 static "0.717" is small-sample noise (its
  regression R² is negative); the real signal is the probe history. *Figures:*
  `fig_pilot_probe_feature_vs_residual.png`, `fig_pilot_hidden_state_identifiability.png`.

## Q3 — Does using probe history improve candidate selection? — **NO**
- Selected native success: best-single **1.00**, K0 **1.00**, K1 **1.00**, oracle **1.00**.
  **K1−K0 = 0.000**, K1−best-single = 0.000, K1 regret-to-oracle = 0.000.
- Because a robust central offset already wins on every block, there is nothing for the identifiable hidden
  state to improve. **Go/No-Go 3: FAIL.** *Figures:* `fig_pilot_k0_k1_selected_success.png`,
  `fig_pilot_selector_regret.png`. *Tables:* `pilot_fold_metrics.csv`, `pilot_bootstrap_metrics.csv`.

## Mechanism ablation — which feedback identifies the hidden state?
Residual-sign balanced accuracy by probe-history subset: final-error-only **0.917**, success+error **0.917**,
full-history **0.917**, probe-success-only **0.633**, time-only **0.583**. **The continuous final/phase-goal
execution error — not the binary success label or execution time — carries the hidden-state information.**
*Figure:* `fig_pilot_history_ablation.png`. *Table:* `pilot_ablation_metrics.csv`.

## Answers to the 12 required questions
1. **306 sufficient for session/block-level analysis?** Yes for oracle-headroom + probe-emulated
   identifiability/selection; but it has no native probe+3-candidate design and all blocks are nominal_bias_y=0.
2. **oracle vs best-single headroom?** 0.000 (CI [0,0]).
3. **probe history predicts hidden residual?** Yes — R²=0.86, sign acc 0.87.
4. **K1 > K0?** For *identification* yes (+0.15 sign acc, R² −0.13→0.86); for *selection* no (0.00).
5. **K1 > best-single?** No (0.00).
6. **Which history features help?** Final/phase-goal execution error (and MOVE_TO_PRE_GRASP duration); not
   binary success or time.
7. **Where effective?** Identification works across the full residual range; **selection is never improved**.
8. **Where does it fail?** Everywhere for *decision value* — the residual never exceeds the success tolerance.
9. **Worth a formal 228+72 run now?** **No** — under this geometry a confirmatory run would reproduce
   headroom≈0 and VSI≈0 (consistent with the earlier damping-stage MODIFY verdict).
10. **Minimal modification?** Make the hidden state exceed the tolerance band: use non-zero nominal bias
    conditions (e.g. nominal ∈ {−0.04, 0, +0.04} as in the confirmatory candidate bank / the 135-episode
    capability map that showed VSI≈0.52) and/or tighten the effective grasp tolerance, so no single offset wins
    on every block. Then re-run Q1; only if oracle headroom ≥ 0.15 does adaptive selection become testable.

## Go/No-Go summary
| Gate | Metric | Result | Threshold | Pass |
|---|---|---|---|---|
| G1 oracle headroom | 0.000 [0,0] | no adaptive space | ≥0.15 | ✗ |
| G2 identifiability | K1 0.867, +0.150 | hidden state readable | ≥0.65 & +0.10 | ✓ |
| G3 selection gain | K1−K0 0.000 | no decision value | ≥0.10 | ✗ |

**Final status: `SCIENTIFIC_PILOT_REQUIRES_HYPOTHESIS_OR_TASK_REVISION`.** The pilot does not support launching
the formal 228+72 confirmatory run; the minimal next action is a **task-geometry revision** (non-zero nominal
bias range / tighter tolerance) followed by a small re-pilot, not more model complexity and not a confirmatory
run. Formal-writer / authorization / manifest machinery is explicitly NOT part of this conclusion.

## Minimal re-pilot (only if the geometry is revised)
6–8 exploratory blocks at the revised bias range × the candidate offsets, 1 emulated probe + candidates per
block (~96–128 trials), **exploratory seeds distinct from the confirmatory block seeds**. Enter the formal
228+72 run only if the re-pilot clears all three Go/No-Go gates.
