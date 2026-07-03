# Calibration-bias capability map — design (v2, EXPLORATORY)

**Supersedes the design in `calibration_bias_capability_map_design_v1.md`.** The v1 code and its
design-freeze commit (`013cb43`) are kept intact (not rewritten); v2 adds new versioned files.

## Why v2: the v1 candidate-group nuisance confound
v1 drew the nuisance (robot joint perturbation + target jitter) **per episode**. So within one candidate
group (same session, same bias) the 7 candidate offsets and 2 probes each ran under **different initial
conditions** — candidates differed in more than `theta`, and cross-bias matched groups differed in more
than the hidden bias. That reintroduces exactly the kind of confound the stage is meant to avoid.

## v2 fix: BLOCK-level (replicate/session) nuisance
- **One nuisance context per replicate block** `r`: `{robot arm-joint perturbation (7-vector), target
  jitter}`.
- **Reused by all 9 episodes in a session** (2 probes + 7 candidates) → within a candidate group ONLY
  `theta` (grasp offset) varies.
- **Reused by the same replicate `r` across all 5 bias levels** → within a matched group
  (`drawer__r{r}`) the `x`/`g` conditions are identical across bias; ONLY the hidden bias differs (a
  proper matched block).
- **3 blocks, 3 independent seeds from a master RNG** (`master_seed`), *not* a bias-ordered increment.
- **Randomized execution order** of the 15 sessions; canonical order, execution order, `block_id`, and
  seeds are saved in `metadata.json`.

Perturbation is applied as `default_joint_pos + block_delta` (base = default, reproducible), so two
episodes in the same block start from the same pose modulo PhysX float noise (checked within `TOL_X=5e-3`).

## Run structure (`configs/calibration_bias_capability_map_v2.yaml`)
5 bias `[-0.04..0.04]` × 7 offsets `[-0.06..0.06]` × 3 replicate blocks + 2 probes/session = **135 eps**.
`middle_drawer`, baseline damping 3.0, robust fixed dynamics. Grid unchanged (`calibration_bias_offset_grid_v1.json`).

## New automatic checks (`tests/test_calibration_bias_capability_map_v2.py`)
A. **candidate-group consistency** — within a session, `g` identical (exact) and initial `x` identical
   (≤ `TOL_X`) across all episodes → only `theta` differs.
B. **matched-group consistency** — within `drawer__r{r}`, the nuisance context (`block_seed`,
   `robot_joint_delta`, `target_jitter`) is identical across bias, and `x`/`g` match → only bias differs.
C. **blocks differ** — the 3 replicate blocks have distinct nuisance deltas.
D. **bias not in `x`**; `candidate_id → offset` identical across all bias; coverage (each id × bias =
   `sessions_per_bias`); execution order is a recorded permutation; `nuisance_level == replicate_block`.
These also surface in `selfcheck_calibration_bias_capability_map_v2.py` (block manifest + consistency).

## Stratified safety smoke (replaces `--max_sessions`)
`configs/calibration_bias_stratified_smoke_v2.yaml`: bias `{-0.04, 0.00, +0.04}` × 2 blocks `{r0, r1}` ×
(2 probes + extreme/center offsets `{-0.06, 0.00, +0.06}`) = **30 eps**. Covers negative/zero/positive
bias AND ≥2 replicate blocks, using the **same** `master_seed` and block-nuisance scheme as the full run,
so it exercises the exact nuisance application + reset + collision safety the full run will use. It runs
BEFORE the full map; if it shows a collision, reset anomaly, or nuisance-application error, stop.

## Read-outs & GO-to-preregistration gate
Per (bias, offset): success, failure reason, final pos, task error, time, handle_relative_error, detach,
frozen utility. Then best offset per bias, monotonicity/`corr`/`|best+bias|`, robust-generalist search,
success-only & frozen VSI, oracle-vs-single gap, rank reversal, replicate independence (across the 3
blocks), probe monotonicity. Gate = v1 criteria **plus** `candidate_group_clean` and `matched_group_clean`.

## Guardrail
Exploratory only; used to choose a range and prove the design is clean, never to fit values to a desired
VSI. Any change is versioned and independently motivated.
