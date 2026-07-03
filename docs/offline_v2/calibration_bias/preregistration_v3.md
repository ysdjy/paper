# Preregistration v3 (FINAL) — calibration-bias confirmatory experiment

> **v3 supersedes v2 (minimal, pre-data).** v3 corrects a **power-model / runtime-implementation
> mismatch**, not a scientific design flaw. `power_analysis_v2` assumed a per-block grasp perturbation
> that moved the effective offset; the actual A runtime injects only a calibration bias + small
> joint/target jitter, which produced **0 success-label flips** in the capability map — so the v2 power
> estimate was not aligned with the generator. v3 freezes an **implementable** block-level residual
> calibration nuisance and re-runs design validation + power. The split, 7-offset bank, probe order, K=1
> stop rule, utility, AND gate, and 6 instrumentation fields are **unchanged** (the residual-aware
> validation confirms they still hold). This is entirely **before any confirmatory data** and is **not**
> post-hoc tuning. `preregistration_v1.md` and `preregistration_v2.md` are retained.

Claude B, branch `experiment/offline-calibration-bias-v1`. Frozen confirmatory protocol. Concrete values:
`confirmatory_freeze_proposal_v3.md`; pre-data checks: `confirmatory_design_validation_v3.json`,
`power_analysis_v3.md`. **No confirmatory run is requested.**

## What changed v2 → v3 (and what did NOT)
**Changed (only the nuisance model + its consequences):** a per-block **residual calibration bias**
`actual_bias_y = nominal_bias_y + residual_bias_y`, `residual ~ TruncNormal(0, 0.005, [−0.01,+0.01])`,
one per block, reused across the split's bias levels, splits isolated, secret/audit-only. Power re-run
under this runtime-aligned model.

**Unchanged (re-validated under residual):** bias split (train {−0.04,−0.02,0,+0.02,+0.04} / val
{−0.01,+0.01} / test {−0.03,+0.03}); 7-offset matched bank; probe order + K=1 stop rule; utility
`U = success − 1.0·err − 0.02·time` + success-only co-primary; the AND exploration/GO gate; the 6 required
failure-instrumentation fields; K=0/1/2 with no third probe; Net VOI(K=1) > 0 as a final-GO condition;
block counts **train 9 / val 6 / test 9**, **675 episodes**.

## 1. Hypotheses (unchanged; report prediction / ranking / selection / decision value separately)
- **H1 Prediction** — DeepSets K=0 vs K>0 (capacity-matched); secondary feature-enriched linear.
- **H2 Ranking** — bias re-orders candidate offsets.
- **H3 Decision value** — state-aware/history selector beats the train-selected best-single offset on the
  **unseen interior test biases {−0.03, +0.03}**, under **frozen utility AND success-only**. Guaranteed
  measurable: under the full residual support no single offset covers both test biases (best-single test
  success ≤ 0.5; state-aware 1.0; max gain 0.5).
- **H4 Probe value** — Gross VOI(K) > 0 and **Net VOI(K=1) > 0**.

## 2. Main metric (unchanged) `U = success − 1.0·task_error − 0.02·time`; success-only co-primary.

## 3. Design (see `confirmatory_freeze_proposal_v3.md`)
- Split: train/val/test as above; residual-aware validation passes (compensable over residual support; no
  common offset at any residual; candidate-group + matched-block unchanged).
- Offsets: matched 7-offset grid step 0.02; no post-hoc removal.
- Probes: first `probe_m040` (−0.04, idx 0), second `probe_p040` (+0.04, idx 1); K=1 = first probe only;
  report K=0/1/2; no third probe; stop rule K=1.
- **Nuisance (v3)**: block-level residual calibration bias (§1 of the freeze proposal); residual/actual
  secret-only; a model reads neither. Blocks disjoint across splits; residual fixed per block.
- Blocks: train 9 / val 6 / test 9 (power ≥ 0.89 at N=9 under the conservative model); 75 sessions,
  675 episodes; block-bootstrap CI (2000 reps, over test blocks).
- Split isolation: bias level + session + nuisance block (audited).

## 4. Final GO / MODIFY / STOP (unchanged, evaluated ONLY after confirmatory data)
GO requires ALL: (1) pairing/leakage/provenance; (2) H1 gain (DeepSets K=0 vs K>0) stable across seeds;
(3) **success-only selected-success gain ≥ 0.15 AND frozen VSI ≥ 0.05, with block-bootstrap CIs**;
(4) history selector beats best-single offset toward the state-aware oracle on the unseen interior test
biases; (5) stable across split + model seeds; (6) replicate independence acceptable — genuine
residual-driven variance, `blocker == False`, ≥ 9 independent test blocks; (7) **Net VOI(K=1) > 0**.

## 5. Required instrumentation (unchanged + residual provenance)
The 6 failure-mechanism fields (see `failure_mechanism_audit_v1.md`), PLUS residual/actual/nominal bias
recorded in secret/audit only.

## 6. Analysis tooling (frozen, implemented + tested)
`offline_v2/calibration_bias/{schema,validator,splits,independence,oracle,voi,design_validation,power,
residual_nuisance}.py`, `evaluation/offline_v2/calibration_bias/{pipeline,run_capability_map,
run_confirmatory_design_v2,run_confirmatory_design_v3}.py`, `tests/offline_v2/calibration_bias/` (69 tests
green). Confirmatory data will be ingested read-only by the same code.

## 7. Immutability
Nuisance model + all frozen v2 items are fixed by this document. Any change requires a new preregistration
BEFORE data generation. B will not request the confirmatory run.
