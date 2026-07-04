# Confirmatory v4 — manifest & provenance specification (frozen)

Claude B. Specifies the run manifest that the generator (Claude A, only after GO) must produce **before**
any execution. This phase produces the SPEC only — **no manifest instance** is generated. Machine form:
`MANIFEST_FIELDS` in `preregistration_v4.json`.

## 1. Purpose
All randomization is drawn and committed **before** the run, so the entire experiment is reproducible and
auditable, and so no choice can be made after seeing outcomes. The manifest is hashed; the hash is checked
on every trial and mismatch is a technical-invalid condition.

## 2. FIX1 — all seeds frozen NOW by a deterministic rule (BLOCKER B)
Randomization is **not** chosen when the generator runs. Every seed is derived from a fixed entropy source
that predates any confirmatory data (the power-certification commit), by a frozen algorithm whose string,
hash, byte encoding, hex-slice length, base, and modulus are all pinned:
```
SEED_ROOT = "confirmatory-v4|2bf7217907f24ae54a08db71bbdcf624b110ccf0"
seed(label)               = int( sha256(f"{SEED_ROOT}|{label}".encode()).hexdigest()[:16], 16 ) % (2**63 - 1)
subseed(domain_seed, id)  = int( sha256(f"{domain_seed}|{id}".encode()).hexdigest()[:16], 16 ) % (2**63 - 1)
```

### 2.1 The 15 exact frozen seeds (stored in `confirmatory_v4_config.json → frozen_seeds`)
```
master_seed                     = 6341914557047805261
train_residual_seed             = 21089512851089425
validation_residual_seed        = 460814074379885831
test_residual_seed              = 3711293848599206733
train_block_order_seed          = 1149828801064435573
validation_block_order_seed     = 3299593555457530435
test_block_order_seed           = 1822308122877636532
train_session_order_seed        = 2515327920114602329
validation_session_order_seed   = 1299366601310815075
test_session_order_seed         = 6018004042777158871
train_nominal_order_seed        = 6034848631996342392
validation_nominal_order_seed   = 138566264689194917
test_nominal_order_seed         = 6188830238780111281
candidate_order_seed            = 4072632114177748606
bootstrap_seed                  = 9014517173581927929
```
`model_seeds = {1103,2207,3301,4409,5519}` are fixed (not drawn). These integers are re-derivable by anyone
from the rule above; a test recomputes them.

### 2.2 Identity-addressed sub-randomization (BLOCKER B)
Per-item randomness uses `subseed(domain_seed, planned_identity)` — e.g.
`subseed(train_residual_seed, "block=03")`, `subseed(candidate_order_seed, "session=<id>")`,
`subseed(test_residual_seed, "block=07")`, `subseed(master_seed, "trial=<planned_episode_id>")`. Consequences:
results do **not** depend on Python iteration order; resume is bit-identical; the test manifest has exactly
one deterministic result; a manifest cannot be regenerated-and-cherry-picked.

### 2.3 Other required manifest fields
```
seed_root, seed_derivation, subseed_derivation   # the frozen rules above
manifest_hash        # sha256 of the fully-resolved manifest (pre-run)
config_hash          # sha256 of confirmatory_v4_config.json
code_commit          # git commit of the analysis code
generator_commit     # git commit of the generator (future, post-GO)
environment_versions # python/torch/numpy/isaac versions, GPU/driver, OS
```

## 3. Construction order (pre-run, deterministic)
1. Instantiate the 15 seeds from the frozen rule (no drawing, no choosing).
2. Draw **one residual per block** per split via `subseed(<split>_residual_seed, "block=<i>")` from
   `TruncatedNormal(0, 0.005, [-0.01,0.01])`; splits use independent domain seeds (no shared residual).
3. Resolve block/session/nominal/candidate order from the corresponding order seeds and per-item `subseed`
   (probe forced first in each session; candidates otherwise ordered by `candidate_order_seed`).
4. Serialize the fully-resolved plan; compute `manifest_hash` and `config_hash`.
5. Record `code_commit`, `generator_commit`, `environment_versions`.
6. Freeze. The manifest is immutable; execution reads from it only. `bootstrap_seed` is already fixed and is
   frozen **before** test unseal.

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
