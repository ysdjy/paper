# FINAL-001 / FINAL-002 final regression (Claude C)

**Auditor:** Claude C, independent, **read-only**, **regression-only** (FINAL-001 and FINAL-002 only; no
re-audit of FINAL-003/004/005; no new issue IDs; no scope expansion). No Isaac, no generator, no formal
manifest/checkpoint/runtime/confirmatory data.

**Audited completion commit:** `48f28ac3dc9836680c010ff58e0cccd1c669b01e`; previous regression `ad2a9ca`;
frozen-issue source `01ac19d`; audited snapshot `b983b7e`; power-cert `2bf7217`.

## VERDICT: **GO_TO_GENERATOR_IMPLEMENTATION**

Both remaining issues are **RESOLVED**; FINAL-003/004/005 remain **UNCHANGED_RESOLVED** (source files
byte-unchanged). No `CRITICAL_NEW_EVIDENCE`. **Power re-certification not required.**

| id | status |
|---|---|
| FINAL-001 | **RESOLVED** |
| FINAL-002 | **RESOLVED** |
| FINAL-003 | UNCHANGED_RESOLVED |
| FINAL-004 | UNCHANGED_RESOLVED |
| FINAL-005 | UNCHANGED_RESOLVED |

## FINAL-001 — RESOLVED (verified first-hand)
- **Canonical storage-order functions** (`canonical_phase_block_identities` / `_session_identities` /
  `_trial_identities`): sizes **15/57/228** (train_validation) and **9/18/72** (test), unique; sessions use
  `sorted(NOMINALS[split])`; trials reuse the frozen storage rule (split→block→nominal→role probe<candidate→
  offset asc).
- **Validator rejects non-canonical list order** with distinct messages: shuffling `blocks`/`sessions`/
  `trials` each → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY` (*"blocks/sessions/trials list not in canonical
  storage order"*). **Shuffling `trials` while `execution_order_index` stays a complete 0..227 set is still
  rejected** — storage order is enforced independently of execution index.
- **Manifest bytes uniquely determined:** two independent reference constructions → identical canonical JSON
  **and** identical full hash. No silent sort / hash normalization hides a non-canonical manifest.
- **Storage order ≠ execution order:** the `trials` list order equals the canonical storage order; the
  order sorted by `execution_order_index` equals `resolve_phase_execution_plan(phase)`; the two genuinely
  differ (verified).
- **Active descriptions synced:** config `deep_manifest_validator.checks` now states
  `resolved_candidate_order == ID.resolve_candidate_order(...) (seed-recomputed exact order, NOT any bank
  permutation)`, `execution_order_index == index in ID.resolve_phase_execution_plan(phase)`, and
  `blocks/sessions/trials lists == frozen canonical storage order`; a dedicated `storage_order_vs_execution_
  order` field documents the distinction; `block_order` is in `domain_labels`; the stale "…is a bank
  permutation" phrasing is gone.

## FINAL-002 — RESOLVED (verified)
- **Exact-value checks still valid (no regression):** `canonical_config_sha256() == sha256(raw active
  config bytes)`; a config-hash 1-bit flip, wrong `schema_version`, determinism `True`-for-`1`, GPU device,
  and an extra determinism key each → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`.
- **Active description synced:** the config check list now states `config_sha256 == canonical_config_sha256()
  (raw bytes)`, `schema_version == confirmatory_v4_phase_manifest_v1`, `deterministic_environment == frozen
  7-key contract (exact values + types, no extras, no GPU)`, and `protocol/generator/runtime commits 40-hex
  format at validation; exact values frozen at generator Step 0`. The stale `"config/commit hex formats;
  …; device==cpu"` summary is removed and no future commit is falsely claimed already-fixed.

## Invariants (unchanged) & power
Candidate bank `{-0.04,0,+0.04}`, test `±0.035`, blocks **9/6/9**, `POWER_SUFFICIENT_FOR_PREREG_V4`, 306
data `131750e1…` — all unchanged. FINAL-003/004/005 source files (`manifest_spec`, `claim_scope`,
`snapshot_manifest`) are byte-unchanged vs the previous regression base. Only order-resolution/storage-order/
validator/description work was done → **`power_recertification_required = false`**.

## Tests
- **B completion** `test_final_001_002_completion.py` → **12 passed**.
- **C prior regression `--runxfail`** `test_frozen_issue_regression_final_001_005_c.py` → **10 passed** (the
  3 previously-INCOMPLETE assertions — FINAL-001 §3.5/§3.2, FINAL-002 §4.5 — now PASS).
- **C final regression** `test_final_001_002_regression_c.py` → **10 passed** (the 10 required checks).

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
only.** Formal phase-manifest instances, model checkpoints, and the 300-trial confirmatory run remain
unauthorized and require their own gates. The ±0.035 power certification carries over unchanged.
