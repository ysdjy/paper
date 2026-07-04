# Confirmatory v4 — manifest & provenance specification (frozen)

Claude B. Specifies the run manifest that the generator (Claude A, only after GO) must produce **before**
any execution. This phase produces the SPEC only — **no manifest instance** is generated. Machine form:
`MANIFEST_FIELDS` in `preregistration_v4.json`.

## 1. Purpose
All randomization is drawn and committed **before** the run, so the entire experiment is reproducible and
auditable, and so no choice can be made after seeing outcomes. The manifest is hashed; the hash is checked
on every trial and mismatch is a technical-invalid condition.

## 2. Required manifest fields (frozen)
```
master_seed          # top-level seed; all sub-seeds derive from it deterministically
residual_seed        # draws one residual per block (train/val/test independent streams)
block_order          # order of nuisance/deployment blocks
session_order        # session execution order (randomized by master seed)
nominal_order        # per split, the nominal-bias order
candidate_order      # per session, the frozen randomized candidate order (probe always first)
model_seeds          # {1103,2207,3301,4409,5519}  (fixed, not drawn)
bootstrap_seed       # the 2000-replicate test-block bootstrap seed
manifest_hash        # sha256 of the fully-resolved manifest (pre-run)
config_hash          # sha256 of confirmatory_v4_config.json
code_commit          # git commit of the generator + analysis code
environment_versions # python/torch/numpy/isaac versions, GPU/driver, OS
```

## 3. Construction order (pre-run)
1. Fix `master_seed`; derive `residual_seed`, `bootstrap_seed`, and per-split residual streams from it.
2. Draw **one residual per block** per split from `TruncatedNormal(0, 0.005, [-0.01,0.01])`; splits use
   independent streams (no shared residual across train/val/test).
3. Resolve `block_order`, `session_order`, `nominal_order`, `candidate_order` (probe forced first in each
   session; candidates randomized).
4. Serialize the fully-resolved plan; compute `manifest_hash` and `config_hash`.
5. Record `code_commit` and `environment_versions`.
6. Freeze. The manifest is now immutable; execution reads from it only.

## 4. Per-trial provenance (recorded, not chosen at run time)
Each executed trial records: `episode_id`, `planned_episode_id`, `session_id`, `block_id`,
`nuisance_block_id`, `episode_role` (probe|candidate), `candidate_index`, `probe_index`, `reset_index`,
`order_in_session`, `git_commit`, `dirty_worktree`, `full_reset_verified`, `config_sha256`,
`science_manifest_sha256`, `code_commit`. (These fields already exist in the 306 schema, confirming the
runtime can emit them.)

## 5. Integrity checks (any failure → technical-invalid; see analysis plan §9)
- `science_manifest_sha256` on each trial == frozen `manifest_hash`.
- `config_sha256` == frozen `config_hash`.
- `git_commit` == frozen `code_commit`; `dirty_worktree == False`.
- `full_reset_verified == True` before each trial.
- ContactSensor available and reporting.

## 6. Separation guarantees encoded in the manifest
- Train / validation / test blocks are disjoint ID ranges; no residual/seed/nuisance shared across splits.
- Within a block, the single residual is reused for the nominal, probe, and all candidate trials (shared
  deployment state), while drawer/task dynamic state is reset between trials.
- The bootstrap resamples **test blocks** only; the manifest fixes the block grouping the bootstrap uses.

## 7. Not produced in this phase
No manifest instance, no seeds materialized, no generator. The generator that consumes this spec is
implemented by Claude A **only after** Claude C returns GO.
