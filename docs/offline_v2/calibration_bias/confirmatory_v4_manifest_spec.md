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

### 2.2 Identity-addressed sub-randomization (BLOCKER B + FIX2 BLOCKER 2)
Per-item randomness uses `subseed(domain_seed, canonical_identity + "|domain=<label>")` with the **frozen
canonical identity strings** (§2.4) and frozen domain labels (A may not rename them):
```
block residual   : subseed(<split>_residual_seed,      block_identity  + "|domain=residual")
block nuisance    : subseed(master_seed,               block_identity  + "|domain=nuisance")
session order     : subseed(<split>_session_order_seed, session_identity+ "|domain=session_order")
nominal order     : subseed(<split>_nominal_order_seed, session_identity+ "|domain=nominal_order")
candidate order   : subseed(candidate_order_seed,       session_identity+ "|domain=candidate_order")
trial init        : subseed(master_seed,               trial_identity  + "|domain=trial_init")
```
Consequences: results do **not** depend on Python iteration order; resume is bit-identical (keyed on
`planned_episode_id`, attempt excluded); the test manifest has exactly one deterministic result; a manifest
cannot be regenerated-and-cherry-picked. Reference implementation: `confirmatory_v4_identity`.

### 2.4 FIX2 — canonical planned identity (frozen; BLOCKER 2)
```
encoding=UTF-8 ; lowercase ascii literals ; field sep "|" ; kv sep "=" ; no whitespace
block index = zero-based, exactly 2 digits ; float = signed fixed-point, exactly 3 decimals, metre
zero = +0.000 ; negative zero forbidden -> +0.000
block_identity   = v4|split={split}|block={block:02d}
session_identity = v4|split={split}|block={block:02d}|nominal={nominal:+.3f}
trial_identity   = v4|split={split}|block={block:02d}|nominal={nominal:+.3f}|role={role}|offset={offset:+.3f}
planned_episode_id = "v4ep-" + sha256(trial_identity.utf8).hexdigest()[:24]
attempt_id = {planned_episode_id}|attempt={attempt:02d}   (attempt NOT in scientific randomization)
resume_key = planned_episode_id
splits {train,validation,test} ; roles {probe,candidate}
block ranges: train 00..08, validation 00..05, test 00..08
probe offset -0.040 ; candidate offsets {-0.040,+0.000,+0.040}
```
Examples: `v4|split=test|block=08|nominal=+0.035|role=probe|offset=-0.040` →
`planned_episode_id` `v4ep-<24 lowercase hex>`. All **300** planned ids are unique; role distinguishes the
probe from the candidate at −0.04. The manifest stores `canonical_block_identity`,
`canonical_session_identity`, `canonical_trial_identity`, `planned_episode_id` per trial.

### 2.5 Canonical storage order (frozen)
Storage/hash order (distinct from execution order, which follows the order seeds):
`split rank (train=0, validation=1, test=2) → block asc → nominal asc → role (probe=0, candidate=1) →
offset asc`. Canonical JSON: `sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False`,
UTF-8, no trailing newline; SHA256 lowercase hex.

### 2.6 FIX3/FIX4 — hash LAYERS: structure hash ≠ full-manifest integrity hash
The structure-only hash is `confirmatory_v4_identity.canonical_planned_structure_hash()`:
> covers ONLY planned identities + `planned_episode_id`s; **does NOT** cover residual/nuisance/
> execution-order/seeds/environment/generator-commit. It is **not** a fully-resolved manifest anchor.

The fully-resolved manifest integrity lives in `confirmatory_v4_manifest_integrity`, with **phase-specific**
full manifests under seal Scheme 2:
```
train_validation phase : 15 blocks, 57 sessions, 228 trials  (probe + 3 candidates)  -> train_validation_manifest_sha256   (Step 1)
test phase             :  9 blocks, 18 sessions,  72 trials                          -> test_manifest_sha256               (Step 6)
total                  : 300 trials
```
Every runtime record's `science_manifest_sha256` = **its phase's** full-manifest hash (train/val records →
`train_validation_manifest_sha256`; test records → `test_manifest_sha256`).

`fully_resolved_phase_manifest_hash(manifest)` validates then hashes **everything** (deep-copied, canonical
JSON, `allow_nan=False`) except the self field `integrity.full_manifest_sha256`; nothing scientific is
excluded. Covered fields:
- **top-level**: schema_version, phase, protocol/prereg commit, generator commit, runtime commit,
  config_sha256, planned_structure_sha256, frozen seeds, deterministic environment, environment/library
  versions, block/session/trial counts, execution-order definition, manifest algorithm version.
- **block**: split, block index, canonical block identity, exact residual value, all resolved nuisance
  values, residual/nuisance subseeds, block/order keys.
- **session**: canonical session identity, split/block, nominal bias, session/nominal order keys, resolved
  candidate order.
- **trial**: canonical trial identity, planned_episode_id, role, offset, trial-init subseed,
  execution-order index, resume key.

Mutating any of residual / nuisance / session-execution-order / candidate-order / trial-execution-index /
trial-init subseed / frozen seed / config hash / generator commit / runtime commit / environment / planned
identity **changes** the full hash; changing only residual/nuisance/order leaves the **structure** hash
unchanged (they are different layers, verified by tests).

### 2.7 Combined experiment-plan hash (Step 6)
`combined_experiment_plan_sha256 = sha256(canonical{ protocol_commit, config_sha256, generator_commit,
train_validation_manifest_sha256, model_analysis_freeze_sha256, test_manifest_sha256, analysis_code_sha256,
bootstrap_seed })`. Step 8 final analysis verifies: train/val records match the train/val phase hash, test
records match the test phase hash, model hashes match `model_analysis_freeze`, and the combined plan hash
matches — else **`EXPERIMENT_INVALID_MANIFEST_INTEGRITY`**.

### 2.3 Other required manifest fields (FINAL-003: layered hash set; no singular manifest_hash)
```
seed_root, seed_derivation, subseed_derivation    # the frozen rules above
planned_structure_sha256                          # structure-only (identities + episode ids)
train_validation_manifest_sha256                  # full deep-validated train/validation phase manifest (228)
test_manifest_sha256                              # full deep-validated test phase manifest (72)
model_analysis_freeze_sha256                      # 10 model hashes + best-single artifact + code + bootstrap seed
combined_experiment_plan_sha256                   # Step-6 anchor over both phase manifests + freeze + commits
config_sha256                                     # sha256 of the RAW bytes of confirmatory_v4_config.json
protocol_commit                                   # prereg / batch-fix commit (frozen at generator Step 0)
generator_commit                                  # generator commit (future, post-GO; Step 0)
runtime_commit                                    # runtime code commit (Step 0)
analysis_code_sha256                              # analysis code hash (Step 5)
environment_versions                              # python/torch/numpy/isaac versions, GPU/driver, OS
```

### 2.3b Freeze-source table (when each integrity value becomes knowable)
| field | source | knowable |
|---|---|---|
| `config_sha256` | raw bytes of the active `confirmatory_v4_config.json` | **now** (`canonical_config_sha256()`) |
| `schema_version` | `confirmatory_v4_phase_manifest_v1` (frozen) | **now** |
| `planned_structure_sha256` | `confirmatory_v4_identity.canonical_planned_structure_hash()` | **now** |
| `frozen_seeds` | `preregistration_v4.frozen_seeds()` (15 exact ints) | **now** |
| `deterministic_environment` | `deterministic_environment_manifest_contract` (exact) | **now** |
| `protocol_commit` | final prereg / batch-fix commit | generator Step 0 |
| `generator_commit` | future generator commit | Step 0 |
| `runtime_commit` | runtime code commit | Step 0 |
| `model_analysis_freeze_sha256` | after train/val + 10 checkpoints | Step 5 |
| `test_manifest_sha256` | after Step 5 | Step 6 |
| `combined_experiment_plan_sha256` | over both phase manifests + freeze + commits | Step 6 |

Future commit fields keep **40-hex format** validation now; they are pinned to concrete values at generator
Step 0 (never hard-coded to a non-existent commit).

## 3. Construction order (pre-run, deterministic; FIX4 — Scheme-2 phase-layered)
Two phase manifests are resolved and hashed separately (NOT one 300-trial serialization):
1. Instantiate the 15 seeds from the frozen rule (no drawing, no choosing).
2. Draw **one residual per block** per split via the canonical identity, i.e.
   `subseed(<split>_residual_seed, canonical_block_identity + "|domain=residual")` from
   `TruncatedNormal(0, 0.005, [-0.01,0.01])`; splits use independent domain seeds (no shared residual).
3. Resolve block/session/nominal/candidate order from the corresponding order seeds and per-item `subseed`
   keyed on the canonical block/session identity (probe forced first in each session).
4. **Step 1** — resolve+freeze the **train_validation** full manifest (228 trials) → `train_validation_manifest_sha256`.
5. **Step 5** — freeze `model_analysis_freeze` → `model_analysis_freeze_sha256`.
6. **Step 6** — resolve+freeze the **test** full manifest (72 trials) → `test_manifest_sha256`; then compute
   `combined_experiment_plan_sha256`. Each phase full hash is produced by
   `confirmatory_v4_manifest_integrity.fully_resolved_phase_manifest_hash()` **only after** deep validation.
   `bootstrap_seed` is already fixed and is frozen **before** test unseal.

## 4. Per-trial provenance (recorded, not chosen at run time)
Each executed trial records: `episode_id`, `planned_episode_id`, `session_id`, `block_id`,
`nuisance_block_id`, `episode_role` (probe|candidate), `candidate_index`, `probe_index`, `reset_index`,
`order_in_session`, `runtime_commit`, `dirty_worktree`, `full_reset_verified`, `config_sha256`,
`science_manifest_sha256` (= the trial's PHASE full-manifest hash). (These fields already exist in the 306
schema, confirming the runtime can emit them.)

## 5. Integrity checks (any failure → EXPERIMENT_INVALID_MANIFEST_INTEGRITY / technical-invalid; see §2.6, analysis plan §9)
- **Per-record phase anchor (FIX4):** each **train/validation** record `science_manifest_sha256 ==
  train_validation_manifest_sha256`; each **test** record `science_manifest_sha256 == test_manifest_sha256`.
  (There is no single ambiguous `manifest_hash`.)
- `config_sha256` == frozen `config_sha256`.
- `git_commit` == frozen `runtime_commit`; `dirty_worktree == False`.
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
