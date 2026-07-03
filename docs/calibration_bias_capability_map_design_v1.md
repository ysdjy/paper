# Calibration-bias capability map — design (v1, EXPLORATORY)

Exploratory only. This grid exists to (a) find a usable hidden-bias range and candidate-offset window,
(b) check whether the best offset moves with bias and whether a robust generalist offset exists, and
(c) prove **replicate independence** (fixing the damping stage's technical-repeat flaw) — **before** any
confirmatory run is pre-registered by Claude B. No result here is a paper claim.

## Hidden state & decision
- Hidden: controller handle local-Y calibration bias `bias_y` (fixed per session; audit-only).
- Decision: candidate `grasp_offset_local_y`. Ideal compensation ≈ `-bias_y` (verified in the injection
  smoke: `bias=−0.04` → success only at offset `+0.04`; true link geometry drift `0.00`).
- Injection: `handle_calibration_bias_v1.biased_spec` (perceived handle only). See
  `calibration_bias_injection_design_v1.md`.

## Grid (`configs/calibration_bias_offset_grid_v1.json`, sha-pinned)
- `bias_levels_y = [-0.04, -0.02, 0.00, 0.02, 0.04]` (exploratory).
- 7 `candidate_offsets = [-0.06 … +0.06]` step 0.02 — `candidate_id → offset` is a pure function, identical
  across all bias levels and sessions (matched). The compensating offset `-bias` is a grid point for every
  bias.
- 2 fixed `probe_offsets = [-0.04, +0.04]` (identical across all sessions; never read bias) for an
  identifiability preview.

## Run structure (`configs/calibration_bias_capability_map_v1.yaml`)
- `middle_drawer`, target 0.20 (jittered), baseline damping 3.0, robust fixed dynamics
  (`max_pos_step 0.020`, `pull_lead 0.080`) so the only structured driver is `bias × offset`.
- 5 bias × 3 independent-seed sessions = **15 sessions**; each = 2 probes + 7 candidates = 9 → **135
  episodes** (< 150). Within-session group = session (one bias). Matched group `{drawer}__r{replicate}`
  spans the 5 biases (no bias in the id).

## Replicate independence (the damping-stage fix)
Every episode gets an independent **nuisance** draw, seeded `session_seed*100000 + order_in_session` with
`session_seed` a running counter **independent of the bias level**:
- small robot arm-joint perturbation (Gaussian σ=0.015 rad, clip ±0.03) applied after reset → **recorded
  in `x`** (build_x reads the actual perturbed joints);
- `target_open_position` jitter (uniform ±0.005 m) → **recorded in `g`**;
- the applied delta vector, jitter, and seed are stored per episode; distributions in metadata.
The self-check computes within-cell final-position SD and success-label flips; if replicates are still
~deterministic (median SD < 5·10⁻³ and no flips) it **warns/pauses** — nuisance too small.

## Safety
Nuisance ranges are small (robot ±0.03 rad ≈ a few cm TCP; target ±5 mm). The run phase begins with a
reduced **nuisance safety smoke** (`--max_sessions`) to confirm no collisions before the full grid. Stop
if any unsafe/colliding behaviour appears.

## Self-check read-outs (`selfcheck_calibration_bias_capability_map_v1.py`, RAW)
Per (bias, offset): success, failure reason, final pos, task error, time, handle_relative_error,
handle_detached, frozen utility `U = p − 1.0·err − 0.02·time` (descriptive). Then: best offset per bias;
best-offset monotonicity + `corr(bias, best)`; compensation error `|best+bias|`; robust-generalist search;
success-only VSI and frozen-utility VSI; state-aware-oracle vs best-single success/utility gap; pairwise
rank reversal; replicate independence; probe monotonicity preview.

## GO-to-preregistration gate (exploratory → hand to Claude B for pre-registration)
1. ≥3 bias levels with clearly different best offset;
2. no offset is a robust generalist (within 0.05 U of the per-bias best at every bias);
3. state-aware oracle success ≥ best-single + 0.15, **or** frozen-utility VSI ≥ 0.05;
4. replicates are genuine independent samples;
5. bias not leaked into `x`;
6. compensation direction physically correct (`corr(bias,best) < 0`, `|best+bias| ≤ 0.02`).

If the gate is **not** met, the map does not license expanding to a confirmatory run; the report states
whether the cause is bias range too small/large, nuisance too small, or the skill tolerance window too
wide, and proposes the next **versioned** exploratory grid — no result-driven hand-tuning.

## Guardrail (carried from the damping audit)
This capability map may only be used to choose an exploratory range, never to fit values to a desired VSI.
Any bias/offset/utility change is versioned and motivated independently of the observed outcome.
