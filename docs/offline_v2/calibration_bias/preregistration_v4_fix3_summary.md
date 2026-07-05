# Preregistration v4 FIX3 — summary (offline)

Claude B. Closes the last three GO-blockers from Claude C's final short re-audit
(`MODIFY_PREREGISTRATION_V4_FIX2`, commit `83d8606`) of fix2 (`48976e4`). No Isaac, no confirmatory runtime
generator, no formal manifest instance, no checkpoint, no runtime/confirmatory data, no run authorization.
Machine form: `preregistration_v4_fix3_summary.json`.

**New status: `PREREGISTRATION_V4_FIX3_READY_FOR_FINAL_C_REAUDIT`.** `power_recertification_required = false`.

## Blockers — resolution
| # | blocker | fix |
|---|---|---|
| **I** | `BLOCKER_BEST_SINGLE_INPUT_GUARD_INCOMPLETE` | single production entry `select_best_single_confirmatory(observed_candidate_records)` — one param, no override kwargs, `__all__` frozen; enforces 45/12 → 135/36 → 171 + globally-unique session_id + strict bool + no probe/test + non-overridable bank. Generic override logic is private `_select_best_single_core`. |
| **II** | `BLOCKER_IDENTITY_DOMAIN_NOT_VALIDATED` | `PlannedTrialKey` validates the frozen design domain (block range, per-split nominal set via `isclose(abs_tol=1e-12)`+canonicalize, probe/candidate offsets, non-bool int, NaN/Inf, `attempt∈{0,1}`, PID regex); all builders/subseeds route through it. |
| **III** | `BLOCKER_MANIFEST_INTEGRITY_HASH_STRUCTURE_ONLY` | structure hash renamed `canonical_planned_structure_hash()`; new `confirmatory_v4_manifest_integrity` gives fully-resolved phase-manifest hashes (228 / 72), self-hash handling, and `combined_experiment_plan_sha256`. |

## I — production selector guard
```
select_best_single_confirmatory(observed_candidate_records)  ->  artifact
__all__ = [select_best_single_confirmatory, ObservedCandidateOutcome, BestSingleInputError]
```
- reads only: split, session_id, trial_role, theta.grasp_offset_local_y, y.success, planned_episode_id.
- **exact split composition** (else `EXPERIMENT_INVALID_BEST_SINGLE_INPUT`): train 45 / validation 12
  sessions; train 135 / validation 36 candidate records; 171 total; 57 globally-unique sessions; session_id
  globally unique (no cross-split); 3/session; offset set `{-0.04,0,+0.04}`; no dup `(split,session_id,
  offset)`; unique `planned_episode_id`; split ∈ {train,validation}; strict Python bool; no probe/test;
  bank not overridable. No warning, no dropped record, no denominator change.
- artifact adds `train_session_count=45, validation_session_count=12, train_record_count=135,
  validation_record_count=36, n_trials_per_offset=57, completeness_gate_version, production_entrypoint`.
- rule unchanged (observed_success_count → min|offset| → bank order); offset 0 not hardcoded.
- override-capable logic is **private** `_select_best_single_core` (bridge/tests only; not in `__all__`,
  not referenced by config/generator). Bridge production == simulation on **4500/4500** frozen configs.

## II — identity domain validation
`PlannedTrialKey(split, block_index, nominal_m, role, offset_m)` `__post_init__` rejects (`ValueError`):
`block=-1/99`, `validation block=6`, `train nominal=0.5`, `validation nominal=0.0`, `test nominal=0.03`,
`test nominal=0.0346` (isclose-guarded so formatting can't legalize), `probe offset=+0.02`,
`candidate offset=+0.07`, `role=other`, NaN/Inf, bool block/attempt, `attempt=-1/2`, malformed PID. The 300
canonical strings and their uniqueness are **unchanged** from fix2. All identity + subseed builders go
through the validated key (no bypass entry).

## III — manifest integrity hash layers
```
planned_structure_sha256            structure only (identities + episode ids); NOT a full anchor
train_validation_manifest_sha256    full fully-resolved train/val phase manifest (228 trials)   [Step 1]
test_manifest_sha256                full fully-resolved test phase manifest (72 trials)          [Step 6]
model_analysis_freeze_sha256        10 model hashes + best-single artifact + code + bootstrap seed [Step 5]
combined_experiment_plan_sha256     Step-6 anchor over both phase manifests + freeze + commits + config
```
Each runtime record's `science_manifest_sha256` = **its phase** full-manifest hash. The full hash deep-copies
and hashes everything (canonical JSON, `allow_nan=False`) except the self field
`integrity.full_manifest_sha256`; nothing scientific is excluded. Mutation of residual / nuisance /
session-execution-order / candidate-order / trial-execution-index / trial-init subseed / seed / config hash /
generator commit / runtime commit / environment / planned identity **changes** the full hash; changing only
residual/nuisance/order leaves the **structure** hash unchanged (distinct layers, test-proven). Final
analysis checks all anchors → else `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`.

## Determinism (machine-frozen)
`device=="cpu"`, `torch.set_num_threads(1)`, `set_num_interop_threads(1)`, `use_deterministic_algorithms(
True)`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`; set BEFORE any torch op, then init model; explicit HP; a required
deterministic op raising RuntimeError → `model_fit_failure_count`, same-seed infra retry ≤1, else
`EXPERIMENT_INVALID_MODEL_FIT`; no GPU switch.

## Unchanged
candidate bank; test geometry ±0.035; DeepSets model + HP + 5 seeds; primary comparator + 0.15 CI; 75
sessions / 300 trials; power result; fix1/fix2 resolutions. Only guards, domain validation, and hash layering
were added — the scientific selection on legal inputs and the design are identical → no power recert.

## Tests
- `test_preregistration_v4_fix3.py` — NEW, all PASS.
- `test_preregistration_v4{,_fix1,_fix2}.py` — B-owned, updated to the fix3 API/status; all PASS.
- Claude C audit files — **UNMODIFIED**. In the fix2-final-reaudit file: the identity-domain strict-xfail
  now **XPASS(strict)→fail** (blocker II fixed); the structure-vs-full-hash strict-xfail depends on C's
  substring check of the renamed `canonical_manifest_hash` docstring; the input-guard strict-xfail and the
  `endswith("select_best_single")` PASS assertion reflect the fix2 API and are superseded by the fix3 entry
  `select_best_single_confirmatory` (the tightening C's blocker I asked for). The genuine fixes are proven by
  the fix3 PASS tests + bridge + mutation evidence; C's final pass re-tests the new API.

## Constraints honored
No Isaac; no generator; no formal manifest instance; no checkpoint; no runtime/confirmatory data; no run
authorization. 306 data, runtime, `models_v2`, capability map, historical power results, C's audit files,
prior preregistration, and the original band-edge verdict are **untouched**. B does **not** authorize A.
