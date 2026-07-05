# Confirmatory v4 generator — GEN-B-001..007 batch fix

Closes the 7 frozen findings from Claude B's generator+smoke consistency audit
(`GENERATOR_SMOKE_BATCH_FIX_REQUIRED`, commit `c0ff67e`, audited generator `b6e6700`). Still smoke-only:
no formal manifest, no combined plan, no checkpoint, no confirmatory data, no 300-trial run, no Isaac.
No Claude B/C file and no frozen active module was modified.

## Per-issue closure
- **GEN-B-001 (validated hash ↔ payload binding)** — `GeneratedPhase` is now an immutable snapshot: it
  stores canonical-JSON of the validated manifest + kat report; `unsealed_manifest`/`kat_report` are
  **copy-on-read** properties (`json.loads` each access). `full_manifest_sha256` is bound to the internal
  snapshot. Mutating a returned copy neither persists nor changes the hash; the dataclass is frozen.
- **GEN-B-002 (envelope injectable after scan)** — `TrialExecutionEnvelope` stores canonical-JSON of the
  public/secret payloads; both are copy-on-read. The scan runs on the mutable payload **before** freezing,
  so post-scan injection into a returned copy cannot persist.
- **GEN-B-003 (unfrozen `target_open_position=0.20`)** — removed. The general envelope's public payload now
  carries only frozen identity/order fields. No synthetic fixture was added (prompt's preferred option).
- **GEN-B-004 (smoke commit context)** — `build_phase_manifest_in_memory` calls `require_smoke_commit_context`
  (raises `SmokeCommitContextError`) after 40-hex validation; under smoke only the `a/b/c*40` placeholder is
  accepted, so a smoke manifest is machine-unmistakable from a future formal freeze. Real-commit freezing is
  a separate future gate, not opened here.
- **GEN-B-005 (smoke environment provenance)** — `EnvironmentVersionContext` is immutable and validates +
  copies at construction (rejects empty mapping / non-str keys / bool / non-str / empty-string values). A
  frozen **10-key** contract (`SMOKE_ENV_VERSION_KEYS`) is enforced by `validate_smoke_environment_provenance`
  at the CLI layer (keeping build permissive so the audit-b reference-equality PASS tests stay green).
  `CLI._env_versions` records `python_implementation`, `python_version`, `numpy_version`, `torch_version`
  (explicit `UNAVAILABLE:<ExcType>` — no silent `except: pass`), `os_system`, `os_release`, `machine`,
  `isaac_status=NOT_IMPORTED_NOT_LAUNCHED`, `gpu_status=NOT_USED_CPU_SMOKE`, `execution_mode=SMOKE_ONLY`. The
  full env enters the manifest and its hash. No Isaac import to fetch versions.
- **GEN-B-006 (strict selection / candidate evidence)** — `canonical_selected_offset` (reject
  bool/str/non-finite/out-of-bank; `abs_tol=1e-12`; returns the exact frozen bank float, using
  `ID.CANDIDATE_BANK`, not a hand-copied bank); `evidence_hash` must be 64 lowercase hex.
  `SessionRunController` holds the frozen `resolved_candidate_order`; `record_candidate_complete` requires the
  next-in-order candidate, no duplicate, `attempt_index ∈ {0,1}` via `ID.attempt_id`; `session_complete`
  requires all 3 completed; candidate **outcomes** never enter the selector.
- **GEN-B-007 (atomic auditable summary)** — `CLI._atomic_write_smoke_summary`: same-dir temp → `flush` →
  `os.fsync` → `os.replace` → dir fsync → **read-back verify**; **idempotent** on identical content;
  `SmokeArtifactCollisionError` on a differing existing file (never silent overwrite); temp cleaned on any
  failure; parent dir created only after KAT/build/rebuild succeed. The returned dict **equals** the on-disk
  JSON (no `_summary_path`); `summary_schema_version` + `write_complete` added.

## Kept unchanged
KAT-first order, `BS.resolved_block_state` as the sole block-state entry (unchecked core never called),
no wrapping of `MI.reference_phase_manifest`, counts 15/57/228 & 9/18/72, canonical storage order,
seed-derived execution order, deep validation before full hash, locked formal writer, public/secret field
definitions, no Isaac, no formal manifest/checkpoint/data. `power_recertification_required=false`.

## Tests
`test_confirmatory_v4_generator` 24 passed (updated for the strict FSM); new
`test_confirmatory_v4_generator_gen_b_fix` 12 passed (reverse-property proofs for all 7);
`test_confirmatory_v4_generator_audit_b -k "not test_GEN_B"` 5 passed (B PASS region intact); the 7
`test_GEN_B_*` demonstrations flip 6/7 (002–007 fail as expected — GEN-B-001's demo asserts only that the
hash is unchanged, true both before and after, so it still passes while the mutability it demonstrates is
fixed and proven by `test_gen_b_001_generated_phase_is_immutable_snapshot`); block-state regressions
12/15/10 passed. Claude B audit files were not modified.

## Status
`GENERATOR_GEN_B_001_007_FIXED_READY_FOR_C_AUDIT` — does NOT authorize a formal manifest, combined plan,
checkpoint, confirmatory data, or the 300-trial run.
