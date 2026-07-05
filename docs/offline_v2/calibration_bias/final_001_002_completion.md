# FINAL-001 / FINAL-002 completion (offline)

Claude B. Closes the two INCOMPLETE items from Claude C's frozen-issue regression
(`FROZEN_ISSUE_BATCH_FIX_INCOMPLETE`, regression commit `ad2a9ca`) of the batch fix (`673d9fe`). **Only**
FINAL-001 and FINAL-002 remaining parts were touched — FINAL-003/004/005 (RESOLVED) are unchanged; no new
issue IDs. No Isaac, no generator, no formal manifest/checkpoint/runtime/confirmatory data, no run
authorization. Machine form: `final_001_002_completion.json`.

**New status: `PREREGISTRATION_V4_FINAL_001_002_COMPLETION_READY_FOR_C_REGRESSION`.**
`power_recertification_required = false`.

## Frozen-issue status
```
FINAL-001 = fixed
FINAL-002 = fixed
FINAL-003 = unchanged_resolved
FINAL-004 = unchanged_resolved
FINAL-005 = unchanged_resolved
```

## FINAL-001 — the two remaining parts closed
**(§3.5) Reject non-canonical storage order (manifest bytes uniquely determined, not just the hash).**
Reordering the `blocks`/`sessions`/`trials` lists no longer validates. `confirmatory_v4_identity` gains the
frozen canonical **storage** sequences:
- `canonical_phase_block_identities(phase)` — split rank → block_index ascending.
- `canonical_phase_session_identities(phase)` — split rank → block asc → `sorted(NOMINALS[split])` ascending.
- `canonical_phase_trial_identities(phase)` — split → block → nominal → role(probe<candidate) → offset asc
  (reuses the frozen `canonical_manifest_rows` storage rule).

`validate_fully_resolved_phase_manifest` now compares the actual list sequences (by canonical identity) to
these and rejects with distinct messages — `blocks list not in canonical storage order`,
`sessions list not in canonical storage order`, `trials list not in canonical storage order`
(→ `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`). No hash normalization / silent sort — the manifest itself must
be canonical. **Storage order ≠ execution order**: `execution_order_index` still carries the seed-derived
`resolve_phase_execution_plan(phase)` order; a storage list position does not define execution order.

`reference_phase_manifest` now emits `blocks`/`sessions`/`trials` in canonical storage order while keeping
`execution_order_index` from the seed-resolved plan. Two constructions produce byte-identical canonical JSON
and the same full hash.

**(§3.2) Active config description.** `confirmatory_v4_config.json → deep_manifest_validator.checks` now
states `resolved_candidate_order == ID.resolve_candidate_order(...) (seed-recomputed exact order, NOT any
bank permutation)`, `execution_order_index == index in ID.resolve_phase_execution_plan(phase)`, and
`blocks/sessions/trials lists == frozen canonical storage order`. The stale "resolved_candidate_order is a
bank permutation" / "execution_order_index unique, complete 0..N-1" phrasings are gone. `block_order` is now
in the active `domain_labels`; the session full-hash fields (`candidate_order_keys`,
`resolved_candidate_order`) and the storage-vs-execution distinction are documented.

## FINAL-002 — active config description synced (code unchanged)
The exact-value checks were already implemented and passing (`canonical_config_sha256`,
`PHASE_MANIFEST_SCHEMA_VERSION`, the 7-key determinism contract, the freeze-source table). This completion
only synced the **active config description**: `deep_manifest_validator.checks` now states
`config_sha256 == canonical_config_sha256() (raw bytes of active confirmatory_v4_config.json)`,
`schema_version == confirmatory_v4_phase_manifest_v1`, `deterministic_environment == frozen
deterministic_environment_manifest_contract (exact 7-key set, exact values, exact types, no extras, no GPU)`,
and `protocol_commit/generator_commit/runtime_commit: 40-lowercase-hex format at validation; exact values
frozen at generator Step 0`. The stale "config/commit hex formats; …; device==cpu" summary is removed. No
change to `canonical_config_sha256`, `PHASE_MANIFEST_SCHEMA_VERSION`, the determinism validator, or the
freeze-source table.

## Unchanged (not touched)
FINAL-003 layered fields; FINAL-004 disclaimer; FINAL-005 snapshot disambiguation; candidate bank; test
nominal ±0.035; 9/6/9 blocks; residual/probe; DeepSets + HP + seeds; primary comparator; bootstrap + 0.15
threshold; 306 data; power results; C's audit/regression files.

## Tests
- `test_final_001_002_completion.py` — NEW; storage-order identities + rejection of the three list shuffles +
  reference storage/execution separation + reproducible hash + config-description sync.
- C regression `test_frozen_issue_regression_final_001_005_c.py --runxfail` → the FINAL-001/002 assertions
  (incl. the two previously-INCOMPLETE xfails) PASS.
- C one-shot audit `--runxfail` → PASS. B-owned suite green. C audit files unmodified.
