# One-shot batch fix — FINAL-001..005 (offline)

Claude B. Resolves the FROZEN issue list from Claude C's one-shot full-repository audit
(`ONE_SHOT_BATCH_FIX_REQUIRED`, audit commit `01ac19d`) of the final snapshot (`b983b7e`, Fix4 source
`f8fedd45`). **Only** FINAL-001..005 were touched — no scope expansion, no new issue IDs. No Isaac, no
generator, no formal manifest/checkpoint/runtime/confirmatory data, no run authorization. Machine form:
`one_shot_batch_fix_final_001_005.json`.

**New status: `PREREGISTRATION_V4_ONE_SHOT_BATCH_FIX_READY_FOR_FROZEN_ISSUE_REGRESSION`.**
`power_recertification_required = false`.

## FINAL-001 — pre_generator_blocker: frozen seed→order resolution
`confirmatory_v4_identity` now defines the ONLY legal ordering (SHA256-derived integer key + stable
lexicographic sort + canonical-index tie-break; **no** `random.shuffle`, RNG, `hash()`, or dict/set
iteration order):
- `resolve_order(items, *, key_fn, canonical_index_fn)` — items unique; key non-bool int in [0, 2**63-1);
  canonical index unique; input permutation-independent; collisions broken by canonical index.
- `block_order_key(split, bi) = subseed(<split>_block_order_seed, block_identity + "|domain=block_order")`;
  `resolve_block_order(split)`.
- `resolve_session_order(split, bi)` — sort nominals by `(session_order_key, nominal_order_key, canonical
  nominal index)`.
- `candidate_item_order_key(...) = subseed(candidate_order_seed, trial_identity(candidate) +
  "|domain=candidate_order")`; `resolve_candidate_order(split, bi, nom)`; `candidate_order_key_map(...)`.
- `resolve_phase_execution_plan(phase)` → canonical_trial_identity sequence: split order (train,validation /
  test) → `resolve_block_order` → `resolve_session_order` → probe first → `resolve_candidate_order`.

Deep validator now **recomputes and exact-checks** `block_order_key`, `candidate_order_keys`,
`resolved_candidate_order == resolve_candidate_order(...)`, and `execution_order_index ==
resolve_phase_execution_plan(phase)` index (not merely a 0..N-1 permutation).

## FINAL-002 — pre_manifest_blocker: exact frozen integrity values
`confirmatory_v4_manifest_integrity`:
- `canonical_config_sha256()` — sha256 of the **raw bytes** of the active `confirmatory_v4_config.json`
  (Git UTF-8; not re-serialized). Config does NOT store its own hash (no self-reference). Validator:
  `manifest.config_sha256 == canonical_config_sha256()`.
- `PHASE_MANIFEST_SCHEMA_VERSION = "confirmatory_v4_phase_manifest_v1"`; validator exact-checks.
- `deterministic_environment_manifest_contract` (config field, frozen constant `DETERMINISM_MANIFEST_CONTRACT`):
  `{device:"cpu", torch_set_num_threads:1, torch_set_num_interop_threads:1,
  torch_use_deterministic_algorithms:true, OMP_NUM_THREADS:"1", MKL_NUM_THREADS:"1",
  OPENBLAS_NUM_THREADS:"1"}`. Validator requires exact key set + value + **type** (rejects True-for-1, extra
  key, or any GPU). Freeze-source table added to `confirmatory_v4_manifest_spec.md` §2.3b; future commit
  fields stay 40-hex format until generator Step 0.

## FINAL-003 — docs: stale manifest field names
`confirmatory_v4_manifest_spec.md` §2.3 singular `manifest_hash`/`config_hash`/`code_commit` replaced by the
layered set (`planned_structure_sha256, train_validation_manifest_sha256, test_manifest_sha256,
model_analysis_freeze_sha256, combined_experiment_plan_sha256, config_sha256, protocol_commit,
generator_commit, runtime_commit, analysis_code_sha256, environment_versions`). Per-trial provenance
`code_commit` → `runtime_commit`; per-record anchor = the trial's phase full-manifest hash.

## FINAL-004 — science/claim wording
Added verbatim to `confirmatory_v4_claim_scope.md` §7 and `preregistration_v4.md` §16b:
> The task-outcome-error and elapsed-time regression heads are auxiliary multitask training signals.
> Candidate selection and the primary endpoint use predicted/observed success only. This experiment neither
> identifies nor claims that multidimensional error/time prediction outperforms a success-only predictor.

The multi-head model, loss, and training are unchanged (no success-only ablation added); claim boundary
only, no power effect.

## FINAL-005 — snapshot traceability
`final_snapshot_v1/snapshot_manifest.json`: ambiguous `snapshot_commit` (was the phase-1 metadata commit
`1abaa72`) replaced by `metadata_payload_commit = 1abaa72…`, `final_head_commit = b983b7e…`,
`audited_snapshot_commit = b983b7e…`. README updated. These point to the historical frozen snapshot, not to
this batch-fix HEAD; the snapshot hashes describe the audited tree only.

## Unchanged (not touched)
power, candidate bank, test geometry ±0.035, DeepSets model + HP + seeds, primary estimator + 0.15 CI, 75
sessions / 300 trials, 306 data. C's one-shot audit files (`one_shot_full_audit_c.{md,json}`,
`test_one_shot_full_audit_c.py`) and all prior C audits are unmodified.

## Tests
- `test_one_shot_batch_fix_final_001_005.py` — NEW; asserts each FINAL-001..005 fix.
- B-owned suite green; C one-shot audit `--runxfail` → the 5 FINAL assertions PASS (see regression run).
- C audit files unmodified; without `--runxfail` their now-resolved strict-xfails report XPASS(strict) — the
  expected historical signal, not an active failure.
