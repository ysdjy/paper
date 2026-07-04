# Confirmatory candidate-bank comparison v1 (offline)

Claude B. Geometric comparison of candidate banks under the empirical HARD edge (`success =
1[|actual_bias + offset| ≤ τ]`, τ ≈ 0.0343) and the FROZEN residual (σ=0.005, |·|≤0.01), evaluated on the
18 empirical block residuals from the 306 run (+ a frozen-distribution robustness check). Machine form:
`confirmatory_candidate_bank_comparison_v1.{json,csv}`. Test nominals {−0.03, +0.03}; train
{−0.04,−0.02,0,+0.02,+0.04}; val {−0.01,+0.01}. Best-single is chosen on **train+val only**.

## Comparison table
| design | bank | best-single | bs top-2 margin | best-single test succ | state-aware test succ | gain | per-block non-degenerate | K1 sel-succ | all criteria |
|---|---|---|---|---|---|---|---|---|---|
| **A** | {−0.04, 0, +0.04} | **0.0** | **0.333** | 0.75 (frozen-dist 0.81) | 1.00 | **0.25** (dist 0.19) | **yes** | 1.00 | **PASS** |
| B | {−0.06,−0.04,0,+0.04,+0.06} | 0.0 | 0.333 | 0.75 | 1.00 | 0.25 | yes | 1.00 | PASS |
| C | {−0.06,−0.04,−0.02,0,+0.02,+0.04,+0.06} | 0.0 | **0.048** | 0.75 | 1.00 | 0.25 | yes | 1.00 | **FAIL (fragile best-single)** |

## Reading each design
- **Design A (3-point) — RECOMMENDED.** state-aware compensates ±0.03 with ∓0.04 → |eff| ∈ [0, 0.02] (deep,
  deterministic success, 1.00). Best-single is robustly **offset 0** (wins train/val by a 0.333 margin over
  ±0.04); at the test biases offset 0 gives |eff| ∈ [0.021, 0.037] / [0.023, 0.039] — **straddles the 0.0343
  edge** → residual flips best-single labels across blocks → **non-degenerate** (gain 0.25 empirical / 0.185
  frozen-dist; per-block gain variance 0.0625). No robust common action (offset 0 succeeds at BOTH test
  biases for only 50% of residuals). All 12 redesign criteria met.
- **Design B (5-point).** Design A + redundant ±0.06. Identical primary contrast; the extra ±0.06 offsets
  introduce **no** robust generalist or utility tie (they only compensate |bias|≈0.06, outside the range).
  Passes, but adds nothing over A — prefer A for parsimony.
- **Design C (current 7-point) — FAILURE CONTROL.** Under success-only + the measured τ, best-single is
  *nominally* 0 (non-degenerate) — **but only by a 0.048 margin** over ±0.02. Offset −0.02 gives +0.03 always
  succeed / −0.03 always fail (**degenerate**). So the current bank's operating point is a **knife-edge**: a
  continuous-utility term, a threshold shift, or a tie-break change flips best-single to ±0.02 and the
  primary contrast collapses to a degenerate CI (the v3 failure). **Rejected** on the stability criterion
  (`best_single_train_val_stable`).
- **Design D (continuous primary, current bank).** Not tabulated as a bank change: it keeps the 7-point bank
  and residual but makes continuous utility / task-error the primary endpoint, with success-only secondary.
  Only a **fallback** if success-only cannot be made robustly non-degenerate — it weakens the paper's core
  "selected action improves **success**" claim, so it is not preferred while Design A works.

## Non-degeneracy criteria (Design A, all met — section 9)
A state-aware > best-single ✓; B gain 0.25 ≥ 0.15 ✓; C per-block gain variance > 0 ✓; D both best-single
outcomes appear across test blocks ✓; E residual flips the primary operating cell ✓; F state-aware ≥ 0.85
(=1.00) ✓; G no robust common action ✓. Extra (redesign selection, **not** original prereg numbers):
best-single test success ∈ [0.20, 0.80] (0.75) ✓; state-aware ≥ 0.85 ✓; robust best-single margin ≥ 0.15
(0.333) ✓; non-degenerate across the whole τ sensitivity interval ✓; all states compensable ✓.

## Threshold sensitivity (τ ∈ {0.0342, 0.03425, 0.0343, 0.0325})
Design A: gain ∈ [0.222, 0.333], non-degenerate at **every** τ, best-single = 0 with **no tie** at every τ.
Design C: non-degenerate under success-only at these τ too, but the best-single margin stays ~0.05 → the
fragility (not the point estimate) is what disqualifies it.

## Honest scope
This is a **constructed positive-control** design: the coarse discrete bank is chosen so that a
state-agnostic policy is forced to the empirical edge, demonstrating that history-conditioned action
selection **has** decision value on grasp success. It is **not** a claim that an arbitrary continuous action
space would yield the same gain. The bank is a pre-discretized skill-parameter library (common on real
robots), frozen before any confirmatory data; all offsets are executed in a matched design per block; the
bank change leaks no hidden state; the 306 band-edge data will not enter any confirmatory train/test set.
