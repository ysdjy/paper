# Confirmatory freeze proposal v2 — calibration bias

> **v1 → v2 revision (pre-data).** Before any confirmatory data was generated, a **test-split
> decision-value flaw** was found in v1: with test = {−0.01, +0.01} and success ⇔ |bias+offset| ≤ 0.02,
> the offset **0.0** satisfies the success band for BOTH test biases, so a state-agnostic best-single
> offset attains full test success and the success-only H3 gain is **0 by construction**. **v2 supersedes
> v1.** This revision happened **entirely before confirmatory data exists** and is **not** post-hoc tuning
> — it fixes an un-provable hypothesis, not a disappointing result. `confirmatory_freeze_proposal_v1.md`
> is retained for the record.

Single freeze proposal (v2). Concrete confirmatory values frozen BEFORE data. Grounded in the capability
map (`capability_map_analysis_v1.md`) and pre-data design validation
(`confirmatory_design_validation_v2.json`, `power_analysis_v2.md`). B does **not** request the run.

## 1. Hidden state & compensable range (unchanged)
Hidden `handle_bias_local_y` (m); controllable `grasp_offset_local_y`. Validated compensable range
|bias| ≤ 0.04 m; success ⇔ |bias + offset| ≤ 0.02 m. Do not exceed ±0.04.

## 2. Bias levels & split (REVISED in v2)
9 continuous levels, 0.01 m spacing across [−0.04, +0.04]:

| split | bias levels (m) | role |
|---|---|---|
| **train** | −0.04, −0.02, 0.00, +0.02, +0.04 | seen; anchors + extremes |
| **validation** | **−0.01, +0.01** | seen intermediate (threshold/hyperparameter selection) |
| **test** | **−0.03, +0.03** | **UNSEEN interior**, bracketed by train (interpolation) |

Validation (`design_validation.validate_confirmatory_split`, pre-data geometry):
- −0.03 bracketed by train {−0.04, −0.02}; +0.03 bracketed by train {+0.02, +0.04} → interpolation ✓
- test success-offset sets: −0.03 → {+0.02, +0.04}; +0.03 → {−0.04, −0.02}; **intersection = ∅** ✓
- ⇒ no fixed offset covers both test biases; max achievable success-only gain = **0.5** (state-aware 1.0 −
  best-single ≤ 0.5). (v1 intersection was {0.0} ⇒ gain 0 ⇒ rejected.)

## 3. Offset bank (unchanged, matched)
7-offset grid {−0.06 … +0.06} step 0.02, identical `candidate_id → offset` across every bias/session/
block. A compensating offset exists for every bias in [−0.04, +0.04]. No post-hoc removal. best-single
offset selected on train/val only, block-wise.

## 4. Probe order, K definitions, operating point (FROZEN explicitly)
- **First probe** = `probe_m040` (offset −0.04), `probe_index 0`, executed first.
- **Second probe** = `probe_p040` (offset +0.04), `probe_index 1`, executed second.
- **K=0** = no probe history. **K=1** = the **first probe only** (`probe_m040`). **K=2** = the first two
  probes. Report **K = 0, 1, 2** only; **no third probe**.
- **Operating point / fixed stop rule**: execute the first probe, select, **stop at K=1** (the selector
  does NOT choose the better of the two single probes post-hoc — K=1 is the fixed first probe).

Verification (capability map): the exploration-grid "K=1 selected success = 1.0" came from the **fixed
first probe** (`probe_index 0`), **not** best-of-two — confirmed by recomputation.

Predicted confirmatory K-curve (`confirmatory_design_validation_v2.json`, geometry, pre-data):
- full 9-bias grid: **K0 0.556 → K1 0.889 → K2 1.000** (on the finer grid a single probe does NOT
  saturate; the second probe adds real selection value).
- test biases {−0.03, +0.03}: **K0 0.5 → K1 1.0 → K2 1.0** (the first probe already separates them).
- **Net VOI** (frozen U): K1 **+0.098**, K2 **−0.020** ⇒ K=1 is the economical operating point; the 2nd
  probe is retained for full-grid identifiability but is marginally net-negative, so the **stop rule is
  K=1**.

## 5. Nuisance blocks — EXACT counts (FROZEN via power analysis)
From `power_analysis_v2.md` (pre-data simulation, exploration-calibrated, + conservative sensitivity):

| split | **frozen blocks** |
|---|---|
| train | **9** |
| validation | **6** |
| test | **9** (≥9 floor; power ≥ 0.94 at N=9 under the conservative model) |

- Blocks **disjoint** across train/val/test; a block never crosses a split; within a split a block pairs
  across that split's bias levels (matched analysis). Deterministic block seed never crosses a split.
- **Total sessions** = 5·9 + 2·6 + 2·9 = **75**; **total episodes** = 75 × 9 = **675**.
- **Block-bootstrap CI**: resample independent blocks within the evaluated split (2000 reps, 95%); each
  block contributes all its (bias, offset) rows; the held-out H3 CI is over the 9 TEST blocks.
- Genuine outcome variance comes from the band-edge cells the 0.01×0.02 grid creates — **not** injected
  noise. No nuisance redesign is mandated (exploration blocks were correctly applied).

## 6. Utility (unchanged, frozen)
`U = success − 1.0·task_error − 0.02·time`; **success-only** co-primary. λ never re-tuned on test.

## 7. Required instrumentation (unchanged, frozen)
1. min joint-limit margin; 2. IK failure/clamp count; 3. explicit `failure_phase`; 4. true handle error at
end of CLOSE_GRIPPER; 5. gripper width at close; 6. collision/contact availability or explicit
no-collision verification.

## 8. Final confirmatory GO conditions (unchanged, frozen — evaluated only after data)
1. pairing/leakage/provenance; 2. B2 beats capacity-matched B1 (DeepSets K=0 vs K>0), stable across seeds
on held-out biases; 3. **success-only selected-success gain ≥ 0.15 AND frozen VSI ≥ 0.05, with block-
bootstrap CIs**; 4. history selector beats best-single offset toward the state-aware oracle on the **unseen
interior** test biases; 5. stable across split + model seeds; 6. replicate independence acceptable
(genuine variance, no blocker); 7. **Net VOI(K=1) > 0** at the operating point.

_All values frozen. Any change requires a new proposal before data generation._
