# Validator-description final regression — FINAL-001 / FINAL-002 (Claude C)

**Auditor:** Claude C, independent, **read-only**, **narrow-scope regression** (only the two previously-
failing active-description items: FINAL-001 §2.5 and FINAL-002 §4.5). No re-audit of algorithms, science,
power, runtime, statistics, or FINAL-003/004/005. No new issue IDs. No Isaac, no generator, no manifest/
checkpoint/runtime/confirmatory data.

**Audited consolidation commit:** `390ccaeb69143dd9a11967b02fd9c8c1677b0057`; previous C correction
`6f584f0`; completion `48f28ac`; frozen-issue source `01ac19d`; snapshot `b983b7e`; power-cert `2bf7217`.

## VERDICT: **GO_TO_GENERATOR_IMPLEMENTATION**

Both remaining items are **RESOLVED**; FINAL-003/004/005 remain **UNCHANGED_RESOLVED**. No
`CRITICAL_NEW_EVIDENCE`. **Power re-certification not required.** All five frozen issues are now closed.

| id | status |
|---|---|
| FINAL-001 | **RESOLVED** |
| FINAL-002 | **RESOLVED** |
| FINAL-003 | UNCHANGED_RESOLVED |
| FINAL-004 | UNCHANGED_RESOLVED |
| FINAL-005 | UNCHANGED_RESOLVED |

## Root cause closed
The completion patch had left the originally-flagged `manifest_integrity.deep_validation.checks` (stale)
alongside the corrected `deep_manifest_validator.checks`. This consolidation **removes the stale block
entirely**, leaving exactly **one** authoritative validator description. Verified across **both** active
machine files (`confirmatory_v4_config.json` and `preregistration_v4.json`).

## FINAL-001 — RESOLVED (§2.5 active description)
- **§2.1** exactly one full manifest-validator checks list, at `.deep_manifest_validator.checks`, with
  `deep_manifest_validator.authoritative == true` (both files). (The `best_single.input_completeness.checks`
  list is a distinct validator, correctly excluded.)
- **§2.2** `manifest_integrity.deep_validation` is **absent**; only an explanatory `deep_validation_removed`
  string remains (no `checks` list).
- **§2.3** aggregate scan of both active files finds neither *"resolved_candidate_order is a bank
  permutation"* nor *"execution_order_index … complete 0..N-1"*.
- **§2.4** the authoritative block states `candidate_order_keys == ID.candidate_order_key_map(...)`,
  `resolved_candidate_order == ID.resolve_candidate_order(...) (seed-recomputed exact order)`,
  `execution_order_index == index in ID.resolve_phase_execution_plan(phase)`, and
  `blocks/sessions/trials lists == frozen canonical storage order`.
- The §2.1–2.4 storage-order **code** (`confirmatory_v4_identity.py`, `confirmatory_v4_manifest_integrity.py`)
  is byte-unchanged and was verified resolved in the prior regression.

## FINAL-002 — RESOLVED (§4.5 active description)
- Aggregate scan finds neither *"config/commit hex formats"* nor *"device==cpu"* in either active file.
- The authoritative block states `config_sha256 == canonical_config_sha256()`, `schema_version ==
  confirmatory_v4_phase_manifest_v1`, `deterministic_environment == frozen 7-key contract (exact values +
  types, no extras, no GPU)`, and `protocol/generator/runtime commits 40-hex at validation; exact values
  frozen at generator Step 0`.
- The exact-value validator **code** is byte-unchanged (prior regression verified tampering → INVALID).

## Invariance (unchanged)
`confirmatory_v4_identity.py`, `confirmatory_v4_manifest_integrity.py`, `models_v2`, the power verdict
(`POWER_SUFFICIENT_FOR_PREREG_V4`), `claim_scope`, `manifest_spec`, and `snapshot_manifest` are all
byte-unchanged vs `48f28ac`. Bank `{-0.04,0,+0.04}`, test `±0.035`, blocks 9/6/9, 306 `131750e1…`
unchanged. Only a documentation-string consolidation → **`power_recertification_required = false`**.

## Tests
- **C prior regression `--runxfail`** `test_final_001_002_regression_c.py` → **11 passed**, incl.
  `test_08_no_stale_candidate_order_description_in_any_active_block` and
  `test_09_no_stale_final002_description_in_any_active_block` (both now PASS).
- **B consolidation** `test_validator_description_consolidation.py` → **7 passed**.
- **B completion + frozen-issue `--runxfail`** → **22 passed**.
- **C final regression** `test_validator_description_final_regression_c.py` → **5 passed** (single
  authoritative block, stale removed, no stale semantics, correct semantics, invariance).

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **true** |
| authorizes_generator_smoke | **true** (`smoke_only=true`, isolated informal artifacts only) |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

**Claude A is authorized to implement the confirmatory generator and run `smoke_only=true` isolated tests
only.** Formal phase-manifest instances, model checkpoints, confirmatory records, and the 300-trial run
remain unauthorized and each require their own gate. The ±0.035 power certification carries over unchanged.
