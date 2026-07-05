# Validator-description consolidation (FINAL-001/002 root cause) — offline

Claude B. Closes the single root cause from Claude C's corrected FINAL-001/002 regression
(`FROZEN_ISSUE_BATCH_FIX_INCOMPLETE`, correction commit `6f584f0`) of the completion (`48f28ac`): the active
config carried **two** validator description blocks — the corrected `deep_manifest_validator` and the
still-stale `manifest_integrity.deep_validation.checks` — a contradiction. Only this duplication was fixed;
no algorithm/validator/order/hash/power/model/data/claim change; no new issue IDs. No Isaac, no generator,
no formal manifest/checkpoint/runtime/confirmatory data, no run authorization. Machine form:
`validator_description_consolidation.json`.

**New status: `PREREGISTRATION_V4_VALIDATOR_DESCRIPTION_CONSOLIDATED_READY_FOR_C_REGRESSION`.**
`power_recertification_required = false`.

## Root cause
The completion patch added the correct `deep_manifest_validator` block but left the earlier
`manifest_integrity.deep_validation.checks` list (from fix4) in place. That old list still said
`resolved_candidate_order is a bank permutation`, `execution_order_index ... complete 0..N-1`,
`config/commit hex formats`, and `device==cpu` — the exact stale phrasings the regression flags — producing
two contradictory active validator descriptions.

## Fix — Scheme B (remove the duplicate block)
The **single authoritative** validator description is `deep_manifest_validator` (sourced from
`DEEP_MANIFEST_VALIDATOR` in `preregistration_v4.py`), the only active block with a full `checks` list. The
old `manifest_integrity.deep_validation` block is **removed entirely** (a `manifest_integrity.
deep_validation_removed` note is left in its place, and `phase_split_composition` is absorbed into the
authoritative block, so nothing is lost). `DEEP_MANIFEST_VALIDATOR` is marked `"authoritative": true`.

**Why Scheme B, not A:** the frozen regression's aggregator reads `mi["deep_validation"]["checks"]`
*unconditionally* whenever `deep_validation` exists, so a Scheme-A reference object without a `checks` key
would raise `KeyError` and crash the regression. Full removal (the aggregator handles `"deep_validation" not
in mi` gracefully) is the compatible consolidation that yields exactly one checks list.

## Result
- Exactly **one** active validator checks list: `deep_manifest_validator.checks`.
- The old stale phrasings are **gone** from all active config/docs:
  `resolved_candidate_order is a bank permutation`, `config/commit hex formats`, `device==cpu`.
- The authoritative list still carries the correct FINAL-001 semantics (`resolved_candidate_order ==
  ID.resolve_candidate_order(...)` seed-recomputed exact order; `execution_order_index == index in
  ID.resolve_phase_execution_plan(phase)`; `blocks/sessions/trials lists == frozen canonical storage order`;
  storage order ≠ execution order) and FINAL-002 semantics (`config_sha256 == canonical_config_sha256()`;
  `schema_version == confirmatory_v4_phase_manifest_v1`; `deterministic_environment ==` frozen 7-key
  contract; commits 40-hex now / exact at generator Step 0).

## Unchanged (not touched)
Validator code, order resolvers, hash functions, `canonical_config_sha256`, `PHASE_MANIFEST_SCHEMA_VERSION`,
determinism contract; FINAL-003/004/005; candidate bank; test nominal ±0.035; 9/6/9; DeepSets + HP + seeds;
primary comparator; bootstrap + 0.15; 306 data; power results; C's audit/regression files.

## Tests
- `test_validator_description_consolidation.py` — NEW; single authoritative checks list; reference object
  shape; stale phrases absent from all active validator description text; correct semantics present.
- C corrected regression `test_final_001_002_regression_c.py --runxfail` → `test_08` and `test_09` PASS.
- C one-shot / frozen-issue regressions `--runxfail` → PASS. B-owned suite green.
