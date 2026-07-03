# Confirmatory freeze proposal v1 — calibration bias

Claude B. This is the **single** freeze proposal (issued once, per protocol). It fixes the concrete
confirmatory values BEFORE any confirmatory data exists. Grounded in the capability-map analysis
(`capability_map_analysis_v1.md`). Nothing here has seen confirmatory outcomes. B does **not** request
the run; it awaits user confirmation. Claude A owns runtime implementation.

## 1. Hidden state & compensable range (frozen)
- Hidden state: `handle_bias_local_y` (m); model-legal controllable: `grasp_offset_local_y`.
- Validated compensable range: **|bias| ≤ 0.04 m** (every bias in [−0.04, +0.04] has a compensating
  offset in the grid; success ⇔ |bias + offset| ≤ 0.02). Do NOT exceed ±0.04 (beyond it the ±0.06
  offset grid cannot compensate and failures become uninformative).

## 2. Bias levels (frozen) — continuous, held-out interior
9 levels, 0.01 m spacing across [−0.04, +0.04]:

| split | bias levels (m) | role |
|---|---|---|
| **train** | −0.04, −0.02, 0.00, +0.02, +0.04 | seen (the validated grid); anchors + extremes |
| **val** | −0.03, +0.03 | seen intermediate (hyperparameter/threshold selection) |
| **test** | **−0.01, +0.01** | **UNSEEN interior**, each bracketed by train levels (interpolation) |

Rationale: test levels −0.01/+0.01 are interior and bracketed (−0.02 < −0.01 < 0.0 < +0.01 < +0.02),
measuring interpolation, not extrapolation. Confirmed selectable: bias ±0.01 falls in the two-probe
"zero" class and offset 0.0 compensates (|±0.01| ≤ 0.02).

## 3. Offset bank (frozen, matched)
Keep the 7-offset matched grid **{−0.06, −0.04, −0.02, 0.0, +0.02, +0.04, +0.06}** (step 0.02), identical
`candidate_id → offset` across every bias level, session, and block. The 0.02 step guarantees a
compensating offset (|bias+offset| ≤ 0.02) for every bias in [−0.04, +0.04], including the interpolated
test levels. **No offset may be removed after the confirmatory test.** best-single offset selected on
train/val only.

## 4. Probe set (frozen: TWO probes, report K=0/1/2, NO third probe)
Freeze the two probes **`probe_m040` (−0.04)** and **`probe_p040` (+0.04)**, executed before candidates,
identical across all sessions. Report the **K = 0, 1, 2** adaptation curve only.

Justification (from `probe_analysis.json`):
- K=0 selected success 0.60 → K=1 **1.00** → K=2 1.00. A single probe already saturates selection.
- **Net VOI**: K=1 **+0.164**, K=2 **−0.064** under the frozen utility — the 2nd probe costs ~11.4 s for
  no selection gain.
- We keep 2 probes because the 2-probe 3-class partition gives finer bias identification useful for the
  interpolated test levels, and designate **K=1 as the economical operating point** (the selector may
  adaptively stop after one probe). We do **NOT** add a third probe: selected success is already
  saturated and a third probe would push Net VOI further negative. (Protocol option chosen: *freeze two
  probes, report K=0/1/2* — not *pre-register a third probe*.)

## 5. Nuisance blocks & independence (frozen) — fixes the exploratory-only limitation
The capability map used 3 paired blocks with a near-deterministic success label. For confirmatory CIs:
- **≥ 9 independent nuisance blocks**, partitioned **by split**: disjoint block-id + seed sets for
  train / val / test; a block must **not** cross a split (enforced by the split audit
  `nuisance_blocks_disjoint`). Within a split, a block may pair across that split's bias levels.
- To obtain genuine outcome variance (this is a POWER fix, not a bug fix — the exploration blocks were
  correctly applied): keep the current nuisance distribution but ensure the level grid places cells at the
  **success-band edge** (|bias + offset| ≈ 0.02), where the injected nuisance naturally flips outcomes;
  the 0.01-bias × 0.02-offset grid does this at several cells. Do **NOT** inject artificial noise to
  manufacture success-label flips.
- **Primary confirmatory inference** is framed on the **continuous outcomes** (task_error, time, handle
  error) and **selection regret** — which vary across blocks — plus success where band-edge variance
  exists; success-rate CIs are reported only where genuine variance is present.

## 6. Utility (frozen — unchanged)
`U = success − 1.0 · task_error − 0.02 · time`. **Success-only** (`U = success`) is a co-primary. λ are
frozen from the damping stage; not re-tuned on confirmatory test.

## 7. Split protocol (frozen)
Triple isolation: **bias level** (§2), **session**, and **nuisance block** never cross train/val/test.
Test = unseen interior bias levels (interpolation). Audited by
`offline_v2/calibration_bias/splits.py` (pairwise-disjoint levels/sessions/seeds/blocks + interpolation
bracketing). Deterministic block/seed never crosses a split.

## 8. Required instrumentation (frozen — must be recorded per episode)
From the failure-mechanism audit (fields currently absent):
1. minimum joint-limit margin; 2. IK failure / clamp count; 3. explicit `failure_phase`; 4. true handle
error at end of CLOSE_GRIPPER; 5. gripper width at close; 6. collision / contact availability or an
explicit no-collision verification. Without these, the confirmatory cannot exclude joint-limit / IK /
collision confounds for the APPROACH-timeout regime.

## 9. Final confirmatory GO conditions (frozen — evaluated only after confirmatory data)
1. pairing / leakage / provenance pass; 2. B2 beats capacity-matched B1 (DeepSets K=0 vs K>0) with stable
prediction gain on held-out biases; 3. success-only selected-success gain ≥ 0.15 AND frozen VSI ≥ 0.05,
**with CIs** over independent blocks; 4. B2 selection beats best-single offset and moves toward the
state-aware oracle on **unseen interior** biases; 5. results stable across split + model seeds;
6. replicate independence acceptable (genuine variance; no blocker); 7. **Net VOI(K) > 0** at the frozen
operating point (K=1). Report K=0/1/2 only.

_All values above are frozen. Any change requires a new proposal before data generation._
