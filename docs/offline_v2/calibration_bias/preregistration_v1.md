# Preregistration v1 (FINAL) — calibration-bias confirmatory experiment

Claude B, branch `experiment/offline-calibration-bias-v1`. This supersedes `preregistration_draft_v1.md`
and is the **frozen** confirmatory protocol, issued because the exploratory capability map **passed** the
frozen exploration gate (`exploration_gate_result_v1.md`). It is committed BEFORE any confirmatory data
exists. Concrete frozen values live in `confirmatory_freeze_proposal_v1.md`; this document fixes the
hypotheses, metrics, thresholds, and analysis plan. **No confirmatory run is requested** — that awaits
explicit user confirmation.

## 0. What the capability map established (exploration, not confirmatory)
- Hidden `handle_bias_local_y` moves the optimal grasp offset **monotonically** (best offset ≈ −bias; 5
  distinct best offsets, corr −0.99); **no robust generalist** exists.
- Frozen VSI **+0.524**, success-only VSI **+0.40**, block-wise best-single success gain **0.40**.
- Success ⇔ **|bias + offset| ≤ 0.02** (clean compensation band).
- A single probe saturates selected success (K=0 0.60 → K=1 1.00); Net VOI +0.164 at K=1, −0.064 at K=2.
- Data is **exploratory-only**: the success label is deterministic across 3 correctly-applied nuisance
  blocks; formal CIs require the confirmatory design below.

## 1. Frozen hypotheses (report prediction / ranking / selection / decision value separately)
- **H1 Prediction** — history improves candidate-outcome prediction under a capacity control that can
  represent the compensation band. Primary control: **DeepSets K=0 vs K>0** (same nonlinear capacity,
  isolating history); a feature-enriched linear baseline (`|offset|`, `offset²`, offset×history) is
  secondary. (Linear B1 vs B2-mean cannot represent the band and would understate history — see the draft.)
- **H2 Ranking** — a different bias re-orders the candidate offsets (switch / rank reversal).
- **H3 Decision value** — a state-aware / history selector beats the **train-selected best-single offset**
  on **unseen interior** biases, under **frozen utility AND success-only** (co-primary), moving toward the
  state-aware oracle.
- **H4 Probe value** — Gross VOI(K) > 0 and, after charging real probe time at the frozen λ_time,
  **Net VOI(K) > 0** at the operating point.

## 2. Frozen main metric
`U = success − 1.0·task_error − 0.02·time`, with **success-only** (`U = success`) as a co-primary. λ are
frozen; never re-tuned on confirmatory test. Predicted error/time clipped to the fixed physical ranges.

## 3. Frozen design (see `confirmatory_freeze_proposal_v1.md` for exact values)
- **Bias levels**: 9 continuous in [−0.04, +0.04]; train {±0.04, ±0.02, 0}, val {±0.03}, **test {±0.01}
  unseen interior** (bracketed → interpolation).
- **Offset bank**: matched 7-offset grid {−0.06 … +0.06} step 0.02 (compensating offset exists for every
  bias in range). No post-hoc removal.
- **Probes**: two fixed (−0.04, +0.04); report **K=0/1/2 only** (no third probe); K=1 is the economical
  operating point.
- **Nuisance**: ≥9 independent blocks partitioned by split (disjoint block ids + seeds; a block never
  crosses a split; may pair across bias levels within a split); level grid touches the band edge for
  genuine outcome variance. No artificial noise to manufacture flips.
- **Split**: triple isolation (bias level + session + nuisance block); deterministic seed never crosses a
  split; audited by `splits.py`.

## 4. Frozen final GO / MODIFY / STOP (evaluated ONLY after confirmatory data)
GO requires ALL of:
1. pairing / leakage / provenance pass;
2. H1 prediction gain (DeepSets K=0 vs K>0) stable across seeds on held-out biases;
3. **success-only selected-success gain ≥ 0.15 AND frozen VSI ≥ 0.05**, reported **with CIs** bootstrapped
   over independent nuisance blocks (not raw sessions);
4. history selector beats best-single offset and approaches the state-aware oracle on **unseen interior**
   biases;
5. stability across split seeds and model seeds;
6. replicate independence acceptable — genuine outcome variance, `independence.blocker == False`, and
   sufficient independent blocks for the CIs in (3);
7. **Net VOI(K) > 0** at the operating point (K=1).

MODIFY if the effect is present but (e.g.) carried only by continuous outcomes with success deterministic,
or CIs are too wide. STOP if compensation fails to reproduce or leakage is found.

## 5. Required instrumentation (frozen — must be recorded per episode)
minimum joint-limit margin; IK failure/clamp count; explicit `failure_phase`; true handle error at end of
CLOSE_GRIPPER; gripper width at close; collision/contact availability or explicit no-collision
verification. Needed to exclude joint-limit / IK / collision confounds (see `failure_mechanism_audit_v1.md`).

## 6. Analysis tooling (frozen, already implemented + tested)
`offline_v2/calibration_bias/{schema,validator,splits,independence,oracle,voi}.py`,
`evaluation/offline_v2/calibration_bias/{pipeline,run_capability_map}.py`,
`tests/offline_v2/calibration_bias/` (54 offline_v2 tests green). Confirmatory data will be ingested
read-only; the same validator/independence/oracle/gate code produces the final result.

## 7. Immutability
Bias levels, offset bank, probe set, nuisance-block scheme, utility, split, thresholds, and required
instrumentation are frozen by this document. Any change requires a new preregistration BEFORE data
generation. B will not request the confirmatory run; it awaits user confirmation.
