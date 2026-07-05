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
  copies at construction (rejects empty mapping / non-str keys / bool / non-str / empty-string values). The
  exact **10-key** smoke provenance contract (`SMOKE_ENV_VERSION_KEYS` + `SMOKE_ENV_FIXED` markers) is
  enforced by the public `build_phase_manifest_in_memory` entry point itself, not merely by the CLI. A caller
  cannot bypass the CLI and obtain a phase hash with partial provenance: `validate_smoke_environment_provenance`
  runs immediately after the smoke-commit gate and **before** any manifest construction / hashing, so a
  partial env (e.g. `{"numpy_version": "1.26.0"}`, or a missing/extra key, or a wrong fixed marker) raises
  `EnvironmentProvenanceError` and yields no phase and no hash. `CLI._env_versions` records
  `python_implementation`, `python_version`, `numpy_version`, `torch_version` (explicit `UNAVAILABLE:<ExcType>`
  — no silent `except: pass`), `os_system`, `os_release`, `machine`, `isaac_status=NOT_IMPORTED_NOT_LAUNCHED`,
  `gpu_status=NOT_USED_CPU_SMOKE`, `execution_mode=SMOKE_ONLY`. The full env enters the manifest and its hash
  (changing one allowed version string changes the full hash). No Isaac import to fetch versions.
  Because the builder now rejects partial provenance, the two frozen audit-b PASS tests
  (`test_builder_counts_and_orders_and_reference_equal`, `test_authorization_gate_and_formal_writer_locked`)
  that build via the frozen module-level 1-key `_ENV` helper flip to failing — an expected, unavoidable
  consequence of closing GEN-B-005 at the builder (the frozen audit-b file cannot be modified). The
  behaviours those tests covered — reference equality, counts/orders, the authorization gate, and the locked
  formal writer — remain independently green in the generator suite with a full 10-key env
  (`test_independent_build_equals_reference`, `test_authorization_requires_smoke_only`,
  `test_formal_writer_zero_side_effect`).
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
`test_confirmatory_v4_generator` 24 passed (all build sites now use a full 10-key env);
`test_confirmatory_v4_generator_gen_b_fix` 17 passed (12 prior reverse-property proofs + 5 new GEN-B-005 §5
proofs: public builder rejects partial provenance, the gate runs before `_build_unsealed_manifest`, a full env
passes deep validation for 228/72, and provenance is bound into the full hash); block-state regressions
12/15/10 passed. Full `calibration_bias` suite: **34 failed / 435 passed / 2 xfailed**, with **0 new active
failures** — 25 are the documented historical Claude-C reaudit baseline (files that do not import the
generator) and 9 are in the frozen audit-b file: all 7 `test_GEN_B_*` demonstrations now flip (each frozen
finding is demonstrated closed) plus the 2 PASS-region tests that build via the frozen 1-key `_ENV` helper
(`test_builder_counts_and_orders_and_reference_equal`, `test_authorization_gate_and_formal_writer_locked`),
which correctly hit the new provenance gate. Claude B/C audit files were not modified.

## Status
`GENERATOR_GEN_B_001_007_FIXED_READY_FOR_C_AUDIT` — does NOT authorize a formal manifest, combined plan,
checkpoint, confirmatory data, or the 300-trial run. GEN-B-005 is now closed at the public builder itself
(`gen_b_005_public_builder_gate = true`, `partial_environment_provenance_rejected_before_build = true`).
