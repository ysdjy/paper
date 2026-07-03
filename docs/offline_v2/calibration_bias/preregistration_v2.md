# Preregistration v2 (FINAL) — calibration-bias confirmatory experiment

> **v2 supersedes v1.** Before ANY confirmatory data was generated, v1 was found to contain a
> **test-split decision-value flaw**: with test = {−0.01, +0.01} and success ⇔ |bias + offset| ≤ 0.02, the
> single offset **0.0** satisfies the success band for BOTH test biases, so a state-agnostic best-single
> offset attains full test success and the success-only H3 gain is **0 by construction** — H3 could not
> be established regardless of outcome. v2 fixes the split. **This revision is entirely pre-data and is
> NOT post-hoc tuning**: it repairs an un-provable hypothesis before any confirmatory outcome exists.
> `preregistration_v1.md` is retained for the record.

Claude B, branch `experiment/offline-calibration-bias-v1`. Frozen confirmatory protocol. Concrete values:
`confirmatory_freeze_proposal_v2.md`; pre-data checks: `confirmatory_design_validation_v2.json`,
`power_analysis_v2.md`. **No confirmatory run is requested.**

## What changed v1 → v2 (and what did NOT)
**Changed:** the bias split. v1 test {−0.01, +0.01} → **v2 test {−0.03, +0.03}**, v2 validation
{−0.01, +0.01}. Plus: exact nuisance-block counts frozen via a pre-data power simulation; probe order /
K-definitions / operating point written explicitly; the predicted confirmatory K-curve reported honestly
on the finer 9-bias grid.

**Unchanged (retained from v1):** the 7-offset matched candidate bank; the frozen utility
`U = success − 1.0·err − 0.02·time`; success-only as a co-primary; the AND exploration/GO gate; the 6
required failure-instrumentation fields; K = 0/1/2 with no third probe; Net VOI(K=1) > 0 as a final-GO
condition.

## 1. Frozen hypotheses (report prediction / ranking / selection / decision value separately)
- **H1 Prediction** — history improves candidate-outcome prediction under a capacity control that can
  represent the compensation band. Primary control: **DeepSets K=0 vs K>0**; secondary: feature-enriched
  linear baseline.
- **H2 Ranking** — a different bias re-orders candidate offsets (switch / rank reversal).
- **H3 Decision value** — a state-aware / history selector beats the train-selected **best-single offset**
  on the **unseen interior test biases {−0.03, +0.03}**, under **frozen utility AND success-only**
  (co-primary), toward the state-aware oracle. The v2 split guarantees this is *measurable* (no fixed
  offset covers both test biases; max achievable success-only gain 0.5).
- **H4 Probe value** — Gross VOI(K) > 0 and **Net VOI(K=1) > 0** after charging real probe time.

## 2. Frozen main metric
`U = success − 1.0·task_error − 0.02·time`; **success-only** (`U = success`) co-primary. λ frozen.

## 3. Frozen design (see `confirmatory_freeze_proposal_v2.md`)
- **Bias split (v2)**: train {−0.04, −0.02, 0, +0.02, +0.04}, val **{−0.01, +0.01}**, test **{−0.03, +0.03}**
  (unseen interior, bracketed; empty success-offset intersection).
- **Offset bank**: matched 7-offset grid {−0.06 … +0.06} step 0.02; no post-hoc removal.
- **Probes**: first = `probe_m040` (−0.04, index 0); second = `probe_p040` (+0.04, index 1). **K=1 = first
  probe only**; K=2 = both; report K=0/1/2; **no third probe**. **Operating point / stop rule = K=1** (no
  best-of-two). Predicted K-curve (geometry): full grid K0 0.556→K1 0.889→K2 1.0; test biases K0 0.5→K1 1.0.
- **Nuisance blocks (frozen via power analysis)**: train 9, val 6, **test 9** (≥9 floor; power ≥ 0.94 at
  N=9). Blocks disjoint across splits; pair across bias within a split; deterministic seed never crosses a
  split. Total 75 sessions, **675 episodes**. Block-bootstrap CI (2000 reps, over blocks).
- **Split isolation**: bias level + session + nuisance block, audited by `splits.py`.

## 4. Frozen final GO / MODIFY / STOP (evaluated ONLY after confirmatory data)
GO requires ALL of: (1) pairing/leakage/provenance; (2) H1 prediction gain (DeepSets K=0 vs K>0) stable
across seeds on held-out biases; (3) **success-only selected-success gain ≥ 0.15 AND frozen VSI ≥ 0.05,
with block-bootstrap CIs**; (4) history selector beats best-single offset toward the state-aware oracle on
the unseen interior test biases; (5) stable across split + model seeds; (6) replicate independence
acceptable — genuine variance, `blocker == False`, ≥ 9 independent test blocks; (7) **Net VOI(K=1) > 0**.

## 5. Required instrumentation (frozen)
min joint-limit margin; IK failure/clamp count; explicit `failure_phase`; true handle error at end of
CLOSE_GRIPPER; gripper width at close; collision/contact availability or explicit no-collision
verification (see `failure_mechanism_audit_v1.md`).

## 6. Analysis tooling (frozen, implemented + tested)
`offline_v2/calibration_bias/{schema,validator,splits,independence,oracle,voi,design_validation,power}.py`,
`evaluation/offline_v2/calibration_bias/{pipeline,run_capability_map,run_confirmatory_design_v2}.py`,
`tests/offline_v2/calibration_bias/`. Confirmatory data will be ingested read-only by the same code.

## 7. Immutability
Bias split, offset bank, probe set + order + K-definitions + stop rule, nuisance-block counts, utility,
thresholds, and required instrumentation are frozen by this document. Any change requires a new
preregistration BEFORE data generation. B will not request the confirmatory run.
