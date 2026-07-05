# Preregistration v4 FIX4 — summary (offline)

Claude B. Closes the four GO-blockers from Claude C's Fix3 final re-audit (`MODIFY_PREREGISTRATION_V4_FIX3`,
commit `06f5965`) of Fix3 (`7304530`). No Isaac, no confirmatory runtime generator, no formal manifest
instance, no checkpoint, no runtime/confirmatory data, no run authorization. Machine form:
`preregistration_v4_fix4_summary.json`.

**New status: `PREREGISTRATION_V4_FIX4_READY_FOR_FINAL_C_REAUDIT`.** `power_recertification_required = false`.

## Blockers — resolution
| # | blocker | fix |
|---|---|---|
| **a** | `BLOCKER_BEST_SINGLE_OFFSET_DOMAIN_NOT_STRICT` | `_canonical_candidate_offset`: reject bool/non-numeric/numeric-string/non-finite; accept only `isclose(rel_tol=0, abs_tol=1e-12)` of a bank value → exact frozen float; `round(.,3)` never decides legality; fail-fast before any hash. |
| **b** | `BLOCKER_FULL_MANIFEST_VALIDATOR_SHALLOW` | `validate_fully_resolved_phase_manifest` is DEEP (composition, uniqueness, identity↔PID recompute, referential, per-session structure, residual bounds, subseed recompute, exec-order, planned-subset, frozen seeds/config/version, strict keys). |
| **c** | `BLOCKER_COMBINED_HASH_OPTIONAL_FIELDS` | `combined_experiment_plan_hash` requires all 8 fields (no None), sha256 64-hex, commits 40-hex, `bootstrap_seed==9014517173581927929`. |
| **d** | `BLOCKER_ACTIVE_MANIFEST_SPEC_STALE` | purged `block=<i>`, singular `manifest_hash`, `science == frozen manifest_hash` from active docs/config; canonical block-subseed + layered fields + per-phase anchors + Scheme-2 order. |

## a — strict offset domain
`confirmatory_v4_selection._canonical_candidate_offset(value)`:
- reject `bool`; require Python `int`/`float` (numeric strings rejected); require finite;
- accept only if `math.isclose(value, bank, rel_tol=0.0, abs_tol=1e-12)` for some bank offset → return the
  **exact** frozen bank float; otherwise `BestSingleInputError` → `EXPERIMENT_INVALID_BEST_SINGLE_INPUT`.
- rejected: `+0.0004, -0.0004, -0.0396, +0.0404, '0.0', True/False, NaN, ±Inf, None, numpy.bool_`; accepted:
  the three exact bank values and ≤`1e-12` float-representation error (canonicalized).
- runs at projection (fail-fast) so `input_record_set_hash` / `input_projection_hash` use canonical offsets;
  no raw `KeyError`. Bridge production == simulation **4500/4500** unchanged.

## b — deep full-manifest validator
`validate_fully_resolved_phase_manifest(manifest)` enforces (else `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`):
- **strict schema**: allowed top/block/session/trial keys only; unexpected field rejected; nested
  `{"integrity": {...}}` rejected (self-hash is the flat `integrity.full_manifest_sha256`).
- **phase composition**: train_validation 9 train + 6 val blocks / 45 + 12 sessions / 180 + 48 = 228 trials;
  test 9 blocks / 18 sessions / 72 trials; no foreign split.
- **uniqueness**: block identity, `(split,block)`, session identity, `(split,block,nominal)`, trial identity,
  `planned_episode_id`, `execution_order_index`.
- **identity recompute** via `confirmatory_v4_identity`: every canonical block/session/trial string;
  `planned_episode_id == planned_episode_id(canonical_trial_identity)`; `resume_key == planned_episode_id`.
- **referential**: session→block exists, trial→session exists; split/block/nominal/role/offset consistent.
- **per session**: exactly 1 probe(-0.04) + 3 candidates(bank); probe executes before its candidates;
  `resolved_candidate_order` a bank permutation.
- **block-shared residual/nuisance**: one residual per block, finite in `[-0.01,+0.01]`; one nuisance set.
- **subseed recompute**: residual / nuisance / session-order / nominal-order / trial-init subseeds from the
  frozen seeds + identity (manifest-submitted values not trusted).
- **execution order**: `execution_order_index` non-bool int, unique, complete `0..N-1`.
- **planned subset**: `planned_structure_sha256 == canonical_planned_structure_hash()`; trials are exactly
  the phase slice of the 300 planned identities.
- **frozen top-level**: `frozen_seeds` == preregistered 15; `manifest_algorithm_version`; hex formats;
  `deterministic_environment.device == cpu`; recursive finite check.
The full hash is computed **only after** this passes; a 228-duplicate/all-train/PID-inconsistent manifest is
rejected and produces no hash.

## c — combined hash required fields
`combined_experiment_plan_hash(*, train_validation_manifest_sha256, model_analysis_freeze_sha256,
test_manifest_sha256, config_sha256, protocol_commit, generator_commit, analysis_code_sha256, bootstrap_seed)`
— no defaults. sha256 fields must be 64 lowercase hex; commits 40 lowercase hex; `bootstrap_seed` a non-bool
int equal to the frozen `9014517173581927929`. Any None / empty / wrong-length / uppercase / non-hex / bool
/ non-frozen bootstrap → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`; omitting a field raises `TypeError`
(required kw-only). Deterministic hash; any input change changes the hash.

## d — active spec cleanup
Removed from active files: `block=<i>` subseed example (→ `canonical_block_identity + "|domain=residual"`),
singular `"manifest_hash"` field (→ `planned_structure_sha256`, `train_validation_manifest_sha256`,
`test_manifest_sha256`, `model_analysis_freeze_sha256`, `combined_experiment_plan_sha256`), and
`science_manifest_sha256 == frozen manifest_hash` (→ per-phase anchors). Construction order rewritten to
Scheme-2 phases (Step 1 train/val 228; Step 5 freeze; Step 6 test 72 + combined) with no single 300-trial
serialization. `MANIFEST_FIELDS` in config/preregistration updated to the layered set. Historical
summary/audit files are untouched.

## Self-hash convention
Flat top-level key `integrity.full_manifest_sha256` (null before hashing); nested `{"integrity": {...}}`
rejected by the deep validator; only that flat field is excluded from its own payload — every other field is
hashed.

## Unchanged
candidate bank; test geometry ±0.035; DeepSets model + HP + 5 seeds; primary comparator + 0.15 CI; 75
sessions / 300 trials; power result; fix1–fix3 resolutions. Only input guards, the validator, hash required
fields, and active docs changed — legal-input selection and the design are identical → no power recert.

## Tests
- `test_preregistration_v4_fix4.py` — NEW, all PASS (offset domain, deep validator on valid+invalid
  fixtures, combined hash required/format, active-doc cleanliness, bridge 4500/4500).
- `test_preregistration_v4{,_fix1,_fix2,_fix3}.py` — B-owned, status bumped to fix4; all PASS.
- Claude C audit files — **UNMODIFIED**. In the fix3-final-reaudit file: the offset-strict, full-validator,
  and stale-spec strict-xfails now **XPASS(strict)→fail** (blockers a/b/d fixed signals); the combined-hash
  strict-xfail stays xfail because omitting the now-required kwargs raises `TypeError` (a harder failure)
  rather than `ManifestIntegrityError`; and C's fix3-era `test_full_hash_covers_resolved_fields` PASS test
  (which hashed a *bogus* manifest) is superseded — a bogus manifest is now correctly rejected. The genuine
  fixes are proven by the fix4 PASS tests.

## Constraints honored
No Isaac; no generator; no formal manifest instance; no checkpoint; no runtime/confirmatory data; no run
authorization. 306 data, runtime, `models_v2`, capability map, historical power results, C's audit files,
prior preregistration, and the original band-edge verdict are **untouched**. B does **not** authorize A.
