# Calibration-bias capability map — offline analysis v1

Claude B, branch `experiment/offline-calibration-bias-v1` @ `17cc4a9` (frozen rules). Read-only
analysis of Claude A's **exploratory** capability map. No Isaac, no runtime edits, no writes to A's
tree. Every number below is independently recomputed from `episodes.jsonl` — A's self-check labels are
NOT trusted. This is an **exploration-stage** analysis to decide GO-to-preregistration; it is not a
confirmatory result.

## Provenance (`provenance.json`)
- source (read-only): `.../paper_calibration_bias_runtime_v1/.../calibration_bias_capability_map_v2_20260703_162715/`
- episodes.jsonl sha256: `de21417390a80b5b…`
- runtime branch `experiment/runtime-calibration-bias-v1`; data-gen / git commit `4fcfeba`;
  runtime_design_commit `4fcfeba`; offset_grid sha256 `b3e14ff8c00d3f36…`
- `dirty_worktree=false`, `damping_verified=true`, `full_reset_all_verified=true`, `n_reset_verify_fail=0`
- (reference commits from task: data commit `13b152b`, runtime report `24919cd`, design `d65960e`.)

## Design (recomputed)
- 5 discrete bias levels `bias_y ∈ {−0.04, −0.02, 0.0, +0.02, +0.04}` (hidden = `handle_bias_local_y`).
- 7 matched grasp offsets `{−0.06 … +0.06}` step 0.02 (`candidate_id → offset` a pure function).
- 2 probes at fixed offsets: `probe_m040` (−0.04), `probe_p040` (+0.04).
- 3 replicate **nuisance blocks** (block 0/1/2), each **shared across all 5 bias levels** (paired).
- 135 = 15 sessions × (2 probes + 7 candidates); 105 candidates, 30 probes.

## Integrity (`validation_report.json`) — ALL PASS
- counts 135 / 105 / 30 ✓; `candidate_id → offset` pure function ✓; matched offset bank identical across
  all 5 bias levels (offset-id set + θ per id) ✓ (3 matched groups, 5 bias levels).
- within-session `x`/`g` identical across probe+candidates ✓; the SAME block gives identical `x`/`g`
  across all bias levels (paired) ✓.
- bias / secret / raw seed / block id never in `x`; leakage-safe history; no probe carries candidate
  fields ✓ (validator all-pass).

## Decision value (frozen U = success − 1.0·err − 0.02·time) (`decision_value.json`)
| quantity | value |
|---|---|
| best offset per bias | −0.04→+0.06, −0.02→+0.02, 0→0, +0.02→−0.02, +0.04→−0.04 (monotone, corr −0.99) |
| distinct best offsets across bias | **5** |
| robust generalist exists (tol 0.02) | **No** (most-robust offset worst-case gap 1.42) |
| frozen-utility **VSI** | **+0.524** |
| optimal-candidate switch rate | 1.0 |
| pairwise rank-reversal rate | 0.581 |
| **success-only VSI** | **+0.40** |
| success-only state-aware vs best-single gain (full data) | +0.40 |
| **block-wise best-single CV success gain** (leave-one-block-out, 3-fold) | **0.40 / 0.40 / 0.40** (min 0.40) |

The hidden bias **directly moves the optimal offset** (best offset ≈ −bias, monotone), there is **no
robust generalist**, and knowing the bias is worth a large success gain (0.40) and frozen VSI (0.524).
This is the decision-value structure the damping stage lacked. Best-single is selected **block-wise**
(train on 2 blocks, evaluate on the held-out block), not on all 135 rows — stable 0.40 across folds.

## Independence (summary; full adjudication in `independence_adjudication_v1.md`)
- Frozen **blocker = False** — the 3 nuisance blocks are correctly applied (distinct seeds/deltas, varied
  within each bias level) and continuous outcomes vary (time SD up to 1.2 s).
- BUT the **success label is deterministic across replicates (0 flips)**, n=3 blocks → **EXPLORATORY-only**;
  formal success-rate CIs are not supported. No nuisance redesign is mandated (blocks are not clones).

## Probes (summary; details in the gate + `probe_analysis.json`)
- Selected success by K: **K=0 0.60 → K=1 1.00 → K=2 1.00**. A single probe already saturates selection.
- **Net VOI(K)**: K=1 **+0.164** (positive), K=2 **−0.064** (negative — the 2nd probe costs 11.4 s for no
  selection gain). Only 2 probes exist → **K=3 not computed / not interpolated**.

## Failure mechanism (summary; details in `failure_mechanism_audit_v1.md`)
- Success iff **|bias + offset| ≤ 0.02** (15/15 and 30/30 succeed at |eff| 0.00/0.02; 0% at ≥0.04).
- |eff|=0.04 → POSITION_TIMEOUT in APPROACH; |eff|≥0.06 → HANDLE_DETACHED in PULL.
- Cannot exclude joint-limit / IK-unreachability / collision (fields absent) → confirmatory instrumentation required.

## Exploration gate: **PASS** (`exploration_gate_result_v1.md`)
All five frozen criteria met (distinct best offsets, no robust generalist, success-gain 0.40 ≥0.15 **AND**
VSI 0.524 ≥0.05, no blocker, integrity ok). Because the gate passes, `preregistration_v1.md` and
`confirmatory_freeze_proposal_v1.md` are produced. No confirmatory run is requested.
