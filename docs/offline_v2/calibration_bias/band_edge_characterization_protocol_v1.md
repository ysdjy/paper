# Band-edge characterization protocol v1 (FROZEN, exploratory)

Claude B, branch `experiment/offline-calibration-bias-v4` (from the scientific-audit commit `e21aa14`,
verdict **MODIFY_PREREGISTRATION**). This freezes a small **exploratory mechanism-characterization
experiment** that resolves the audit's must-fix items **before** any confirmatory data or preregistration
v4 exists. It is **NOT** confirmatory data and must never be used as a paper confirmatory result.

**No runtime is implemented here, no Isaac is launched, no data is generated.** This document + its config
/ schema / power-plan siblings are the frozen contract a later runtime-implementation phase must satisfy.
v1/v2/v3 files are untouched.

## Why this experiment (from Claude C's audit)
C's MODIFY_PREREGISTRATION rests on four fixable pre-data gaps, all at the band edge:
- every confirmatory val/test operating point sits at an **unmeasured** `|eff| ∈ {0.01, 0.03, 0.05}`
  (the 135-ep capability map only sampled even multiples of 0.02);
- `power.py` (soft edge, center 0.03) and `design_validation` (hard band 0.02) **disagree** about what
  happens at `|eff|=0.03`;
- under the hard band the v3 residual gives **~0 label variance** at the operating cells → the block CI
  risks being **degenerate** (the v2 failure the residual was meant to fix);
- the collision instrumentation is **off and under-specified**, so the band-edge (APPROACH-timeout)
  failures could not be attributed to grasp compensation vs a cabinet collision.

This experiment measures the true edge, its softness, and whether the residual induces genuine
success-**label** variation, and validates the six instrumentation fields + a concrete collision sensor.

## 1. Purpose (exploratory only)
1. measure the true success boundary; 2. measure edge softness (hard vs soft, and the scale);
3. verify the v3 residual causes block-level success-label variation; 4. validate the six instrumentation
fields; 5. obtain exploratory collision-criterion calibration data; 6. produce an empirical success model
for the learned-selector power simulation. **Not** a confirmatory result.

## 2. Design (FROZEN — see `band_edge_characterization_config_v1.json`)
- **nominal_bias_y = 0.00 ONLY.** The future confirmatory test nominal biases (−0.03 / +0.03) are **not**
  observed here (no peeking at future test conditions).
- **Residual (v3, unchanged):** `actual_bias_y = nominal_bias_y + residual_bias_y`,
  `residual_bias_y ~ TruncatedNormal(0, σ=0.005 m, [−0.01, +0.01])`, **one draw per block**, shared by all
  17 offsets in that block; blocks independent.
- **18 independent blocks.**
- **17 grasp offsets** (fine grid): −0.050, −0.040, −0.035, −0.030, −0.025, −0.020, −0.015, −0.010, 0.000,
  +0.010, +0.015, +0.020, +0.025, +0.030, +0.035, +0.040, +0.050.
- **Episodes = 18 × 17 = 306.** No probes (this characterizes the physical boundary, not a selector).
- `eff = actual_bias_y + grasp_offset_local_y`. At nominal 0, residual 0, `|eff| = |offset|`, covering the
  **clear-success** (0, 0.01), **known edge** (0.02), **unmeasured** (0.025, 0.03, 0.035), and
  **clear-fail** (0.04, 0.05) regions, symmetric in ±. The residual continuously fills the edge between.

## 3. Nuisance & pairing (FROZEN)
Within one block the 17 offset episodes share: residual, initial joint perturbation, target-open-position
jitter, mechanism, initial state, task target — **only the candidate offset varies**. Different blocks are
independent draws. **Block execution order and within-block offset order are randomized and saved to a
manifest** (seeds + realized orders) for reproducibility.

## 4. Six instrumentation fields (FROZEN definitions — see `band_edge_characterization_schema_v1.md`)
Joint-limit margin (rad + normalized + joint index + phase, per-episode min over steps/joints); IK failure
(solve/failure/max-consecutive counts + per-phase); **two DISTINCT clamp counters** (step-clamp vs
limit-clamp, never merged); failure phase (+ transition); true handle error at CLOSE_GRIPPER→PULL (3D +
local-Y + TCP pose + TRUE handle pose, using the real not perceived handle); gripper width + command at
close. residual/actual/nominal bias are secret/audit-only; no instrumentation field is model-legal.

## 5. Collision protocol (FROZEN mechanism — no vague wording)
An **Isaac Lab / PhysX ContactSensor** (or equivalent contact-report) monitors Franka non-gripper arm
links + hand body vs the cabinet, **excluding** the intended finger↔handle grasp contact. Raw fields:
`max_unintended_contact_force_N`, `unintended_contact_frame_count`, `first_unintended_contact_phase`,
`contact_force_by_phase`, `contact_sensor_available`. A **sensor-validation smoke** runs first: (1) clean
baseline → force ~0; (2) intentional-contact positive control → large force; (3) confirm the sensor
separates them. The numeric force **threshold is chosen from the band-edge data distribution AFTER this run
and frozen before confirmatory v4 — never from any confirmatory data.**

## 6. Frozen analysis plan (B runs read-only after A returns data)
Implemented + tested in `offline_v2/calibration_bias/band_edge.py`; parameters fixed now:
1. success probability vs `|actual_bias + offset|`; 2. fit ± directions separately and test symmetry;
3. block-level success rate per **0.005 m** `|eff|` bin; 4. fit **logistic** `p=sigmoid((center−|eff|)/scale)`
**and isotonic** (decreasing); 5. **block bootstrap** (resample blocks, 2000 reps, 95% CI) for edge center
& scale; 6. test whether `|eff|=0.03` has genuine success-probability variation (both labels present);
7. test whether the residual produces block-level label variation in the operating region
`|eff| ∈ {0.025, 0.030, 0.035}`; 8. flag collision-confounded episodes (do **not** silently delete);
9. report **both** the full result and the collision-excluded sensitivity result.

Chosen **before** seeing data and immutable: logistic center/init, bin width (0.005), edge/success
threshold definition, collision-threshold *selection method*.

## 7. Frozen exit criteria (`band_edge.evaluate_exit_criteria`)
**PASS_TO_V4** (ALL): `|eff|∈0.025–0.035` shows both success and failure; residual produces real
block-level label variation; edge center + scale estimable (finite CI); no severe unexplained ± asymmetry
(|Δcenter| ≤ 0.005 m); collision sensor available; six instrumentation groups complete; the main edge
result is not driven by collision-confounded episodes (edge-center shift when excluding them ≤ 0.005 m).

**MODIFY_RESIDUAL_OR_DESIGN** (ANY): success is still a fully hard threshold (edge scale ≈ 0); residual
causes no edge variation; `|eff|=0.03` is all one label; collision dominates the edge result;
instrumentation unavailable. In that case only the residual or the confirmatory design may change — the
confirmatory run does **not** proceed.

## 8. Hidden-state & probe semantics — frozen for the v4 to-MODIFY list (Audit A / D)
When preregistration v4 is written, it MUST adopt this wording (this experiment does not use probes, but
freezes the language):
- outcome-determining hidden state = **`actual_bias_y`**; `actual_bias_y = nominal_bias_y + residual_bias_y`;
  the residual is a **random component of the actual hidden state**.
- H3 tests **"generalization to unseen *nominal* deployment conditions, while the actual state remains
  within training support"** — NOT "unseen hidden state / unseen physical bias" (test actual-bias support
  ⊂ train actual-bias support).
- probes must be called **"repeated-task deployment diagnostic trial"** / **"full-task diagnostic
  attempt"** — never "low-cost probe", "non-destructive probe", or "one-shot in-task online probing".

## 9. What stays unchanged (Audit: scientific core is sound)
Bias split, 7-offset matched bank, probe order + K=1 stop rule, utility, success-only co-primary, AND gate,
VSI/oracle/VOI machinery, leakage guards, and block counts (9/6/9, 675 eps) are **retained** — this
experiment only re-derives the band/power model and the collision criterion. If item 7 shows genuine
operating-point variance, the confirmatory design is unchanged; if not, the residual / operating offsets
(not the block counts) are what move.

## 10. Scope & authorization
This commit produces ONLY: this protocol, `band_edge_characterization_config_v1.json`,
`band_edge_characterization_schema_v1.md`, `learned_selector_power_plan_v1.md`, and their tests. It does
**not** produce preregistration v4, a confirmatory generator, or any data. A full 306-episode band run
(and, later, any 675-episode confirmatory run) requires A to implement the generator + residual + six
instrumentation fields + the collision sensor, pass schema/leakage/instrumentation/safety smoke tests, and
obtain **explicit user approval**. No such approval is requested here.
