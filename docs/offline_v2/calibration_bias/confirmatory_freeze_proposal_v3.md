# Confirmatory freeze proposal v3 — calibration bias

> **v2 → v3 revision (pre-data, minimal).** v3 fixes a **power-model / runtime mismatch**, not a design
> flaw: `power_analysis_v2` assumed a per-block grasp perturbation that moved the effective offset, but the
> actual runtime injects only a calibration bias + small joint/target jitter that produced **0 success-
> label flips** in the capability map. v3 freezes an **implementable** block-level residual calibration
> nuisance that genuinely moves the outcome, then re-runs design validation and power. **The split,
> 7-offset bank, probe order, K=1 stop rule, utility, AND gate, and 6 instrumentation fields are
> UNCHANGED** — the residual-aware design validation confirms they still hold. This revision is entirely
> **before any confirmatory data** and is **not** post-hoc tuning. v1/v2 retained for the record.

Single freeze proposal (v3). Pre-data. Sources: `power_analysis_v3.md`,
`confirmatory_design_validation_v3.json`. B does **not** request the run.

## 1. NEW in v3 — block-level residual calibration nuisance (FROZEN)
    actual_bias_y = nominal_bias_y + residual_bias_y
- `residual_bias_y ~ TruncatedNormal(mean = 0, sigma = 0.005 m, range [−0.01, +0.01] m)` (effective SD
  ≈ 0.0044 m). Physical rationale: a realistic few-millimetre handle-pose recalibration error on top of
  the commanded bias; on the same axis as the hidden state, so it perturbs `|actual_bias + offset|`.
- **One residual per nuisance block**, reused across all bias levels within that block's split; blocks
  independent; **train/val/test block ids + seeds + residual draws fully isolated**.
- `nominal_bias_y` and `actual_bias_y` (and the residual) live **only** in secret/audit fields. A
  deployable model may read **neither** the residual **nor** the actual bias. Real handle geometry is
  unchanged; **no artificial noise is added to the candidate action.**
- Cap ±0.01 m keeps the extreme nominal biases ±0.04 within the compensable range (actual ≤ ±0.05, still
  covered by the 7-offset bank — verified).

## 2. Bias split (UNCHANGED from v2, re-validated under residual)
train {−0.04, −0.02, 0, +0.02, +0.04}; validation {−0.01, +0.01}; **test {−0.03, +0.03}** (unseen
interior, bracketed). Residual-aware validation (`validate_confirmatory_split_residual`):
- every nominal bias compensable by the 7-offset bank for **every** residual in support ✓ (max actual bias 0.05);
- for **no** residual does a single offset satisfy the success band for BOTH test biases in the same block
  ✓ ⇒ best-single test success ≤ 0.5, state-aware 1.0, **max achievable success-only gain = 0.5**;
- candidate-group (= session) and matched-block (per replicate across bias) definitions unchanged ✓.

## 3. Offset bank (UNCHANGED) — matched 7-offset grid {−0.06 … +0.06} step 0.02; no post-hoc removal.

## 4. Probes / K / operating point (UNCHANGED)
First = `probe_m040` (−0.04, idx 0); second = `probe_p040` (+0.04, idx 1). K=1 = first probe only; K=2 =
both; report K=0/1/2; **no third probe**; **stop rule K=1**. Predicted K-curve (geometry, residual-
insensitive): full 9-bias grid K0 0.556 → K1 0.889 → K2 1.0; test biases K0 0.5 → K1 1.0.

## 5. Nuisance blocks (FROZEN — retained via v3 power)
Runtime-aligned residual power still supports the v2 allocation:

| split | blocks |
|---|---|
| train | 9 |
| validation | 6 |
| **test** | **9** (power 0.97 baseline / 0.89 conservative at N=9 ≥ 0.8 target) |

- Blocks disjoint across splits; residual fixed per block; total **75 sessions**, **675 episodes**.
- **Block-bootstrap CI**: 2000 reps over the independent test blocks (each block = one residual, all its
  bias×offset rows). Optional precision margin: N_test = 12 (conservative power 0.97).

## 6. Utility (UNCHANGED) `U = success − 1.0·err − 0.02·time`; success-only co-primary; λ frozen.

## 7. Required instrumentation (UNCHANGED) — the 6 fields (joint-limit margin, IK clamp/fail, failure_phase,
handle error at close, gripper width at close, collision/contact or no-collision verification), PLUS the
residual/actual/nominal bias recorded in **secret/audit** only.

## 8. Final confirmatory GO conditions (UNCHANGED from v2)
pairing/leakage/provenance; H1 (DeepSets K=0 vs K>0) stable; **success-only gain ≥ 0.15 AND frozen VSI ≥
0.05 with block-bootstrap CIs**; history selector beats best-single toward state-aware oracle on unseen
interior test biases; stability across split + model seeds; replicate independence acceptable (genuine
residual-driven variance, no blocker, ≥ 9 test blocks); **Net VOI(K=1) > 0**.

_All values frozen. Any change requires a new proposal before data generation._
