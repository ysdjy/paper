# Frozen-issue regression FINAL-001..005 (Claude C)

**Auditor:** Claude C, independent, **read-only**, **regression-only** (no new full audit, no scope
expansion, no new issue IDs). No Isaac, no generator, no formal manifest/checkpoint/runtime/confirmatory
data.

**Audited batch-fix commit:** `673d9fe7a7651bc07ec21e758b370ee6747ce618`; frozen-issue source (one-shot
audit) `01ac19d`; audited snapshot `b983b7e`; power-cert `2bf7217`.

## VERDICT: **FROZEN_ISSUE_BATCH_FIX_INCOMPLETE**

Failing IDs (originals only, no new IDs): **FINAL-001, FINAL-002**. `FINAL-003/004/005` are **RESOLVED**.
No `CRITICAL_NEW_EVIDENCE`. **Power re-certification not required** (design/power invariants unchanged).

| id | status |
|---|---|
| FINAL-001 | **INCOMPLETE** |
| FINAL-002 | **INCOMPLETE** |
| FINAL-003 | RESOLVED |
| FINAL-004 | RESOLVED |
| FINAL-005 | RESOLVED |

## FINAL-001 — INCOMPLETE
**Resolved (verified first-hand):** `resolve_order` is a pure SHA256-key + stable-sort + canonical-index
resolver (no RNG/`hash()`/dict-order; input-permutation-independent; rejects bad keys); the block/session/
candidate order functions and `resolve_phase_execution_plan` give **228 / 72** unique, byte-identical plans
in the frozen order (probe first); and the validator now **recomputes** `block_order_key`,
`candidate_order_keys`, `resolved_candidate_order`, and `execution_order_index` from the frozen seeds —
tampering any of them → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`, valid reference passes.

**Still INCOMPLETE — §3.5 canonical storage/hash uniqueness (the explicit "manifest uniquely determined"
criterion):** shuffling the `blocks` / `sessions` / `trials` **lists** (identical content, still valid)
**still validates AND changes the full-manifest hash** (verified: `validates=True, hash_changed=True`).
There is no canonical storage-order requirement and no list normalization before hashing, so two
byte-different but equally-valid manifests produce **different** per-record integrity anchors — the
manifest is not uniquely determined by its content. Also the config `deep_manifest_validator.checks`
description (line ~626) still says "`resolved_candidate_order` is a bank permutation" instead of the
seed-recomputed exact order the code now enforces (§3.2 active-spec sync).

**Minimal fix:** freeze a canonical storage order (blocks by `canonical_block_identity`, sessions by
`canonical_session_identity`, trials by `execution_order_index`) and either reject non-canonical order or
normalize inside `fully_resolved_phase_manifest_hash` before hashing; update the config check description
to the seed-determined order. *No power recert.*

## FINAL-002 — INCOMPLETE
**Resolved (verified):** `canonical_config_sha256()` hashes the raw active `confirmatory_v4_config.json`
bytes and the validator **exact-checks** `config_sha256` (1-bit flip → INVALID); `schema_version ==
"confirmatory_v4_phase_manifest_v1"` exact-checked; `deterministic_environment` exact 7-key contract with
**value + type** strictness (True-for-1 → INVALID, GPU → INVALID, extra key → INVALID); commits remain
40-hex-format-until-seal (legitimate; freeze-source table added).

**Still INCOMPLETE — §4.5 active-config description stale:** `confirmatory_v4_config.json`
`deep_manifest_validator.checks` (line ~631) still states *"frozen_seeds == preregistered 15; config/commit
hex formats; manifest_algorithm_version; **device==cpu**"* — the OLD format-only config check and
device-only determinism check, omitting the now-implemented exact config-hash, exact schema-version, and
full determinism-contract checks. The one-shot audit's §4.5 pre-declared that a **code-correct-but-active-
config-states-old-semantics** state keeps FINAL-002 **INCOMPLETE**.

**Minimal fix:** sync that one description line to the implemented exact checks (config_sha256 ==
canonical_config_sha256; schema_version exact; determinism 7-key value+type contract; commits format-only
until Step 0). Code already correct. *No power recert.*

## FINAL-003 / 004 / 005 — RESOLVED
- **FINAL-003:** `manifest_spec.md §2.3` uses the layered field set; no standalone `manifest_hash`/
  `config_hash`/`code_commit` field entries; per-trial provenance uses `runtime_commit`; per-record anchor
  = phase full-manifest hash.
- **FINAL-004:** `claim_scope §7` + `preregistration_v4 §16b`: error/time heads auxiliary; selection +
  primary use **success only**; no claim that multidimensional error/time prediction outperforms a
  success-only predictor. Model/loss/primary/power unchanged.
- **FINAL-005:** `snapshot_manifest.json` removes ambiguous `snapshot_commit`; adds exact
  `metadata_payload_commit=1abaa72…`, `final_head_commit=b983b7e…`, `audited_snapshot_commit=b983b7e…`;
  README disambiguates (1abaa = phase-1 metadata parent, b983 = audited HEAD, no self-reference).

## Invariants (unchanged) & power
Bank `{-0.04,0,+0.04}`, test `±0.035`, blocks 9/6/9, `POWER_SUFFICIENT_FOR_PREREG_V4`, 306 data
`131750e1…` — all unchanged. Batch fix touched only order-resolution/validator/hash-fields/docs →
`power_recertification_required = false`.

## Tests
- **B batch-fix** `test_one_shot_batch_fix_final_001_005.py` → **17 passed** (does **not** cover §3.5
  storage/hash uniqueness or the stale config descriptions — hence the gaps this regression catches).
- **C regression** `test_frozen_issue_regression_final_001_005_c.py` → **7 passed, 3 xfailed**
  (FINAL-001 §3.5 + §3.2, FINAL-002 §4.5).

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_generator_smoke | **false** |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

**Next step:** Claude B applies the two minimal fixes to **FINAL-001** (canonical storage order / hash
normalization + config description) and **FINAL-002** (config description sync); Claude C then re-regresses
**only** FINAL-001 and FINAL-002. On green, all five are RESOLVED → `GO_TO_GENERATOR_IMPLEMENTATION`
(implementation + `smoke_only=true` only). The ±0.035 power certification carries over unchanged.
