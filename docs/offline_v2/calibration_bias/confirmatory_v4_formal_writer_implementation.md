# Confirmatory v4 formal manifest writer — implementation (generation LOCKED)

Implements the FORMAL manifest writer + real commit-freeze interface authorized by Claude C's
generator-smoke final gate (`GENERATOR_SMOKE_GATE_PASS`, commit `f41e366`, audited generator fix `4330a8a`).
The gate authorizes **implementing** the writer, not running it: it carries
`authorizes_formal_manifest_generation=false`. This deliverable therefore builds the complete future success
path while remaining **fail-closed** — no file in the current repo can mint a generation authorization, so
every production generation entry fails before any formal phase is constructed and before any filesystem
effect. Nothing under `artifacts/formal/` is created and no Isaac/GPU is used.

New files (module + CLI + tests + this doc + the A2 report). The C-gate-approved generator is left byte-for-byte
unchanged: the formal builder **reuses** `GEN._build_unsealed_manifest` (the independent construction from the
frozen modules) rather than forking or modifying it. No frozen scientific module is touched
(`preregistration_v4`, `confirmatory_v4_identity`, `confirmatory_v4_block_state`,
`confirmatory_v4_manifest_integrity`, `confirmatory_v4_selection`, `models_v2`, power artifacts, 306 data).

## Authorization boundary
- The current C gate authorizes writer *implementation* (`authorizes_formal_manifest_writer_implementation=true`)
  and **locks** generation (`authorizes_formal_manifest_generation=false`).
- A `VerifiedFormalManifestAuthorization` can only be produced by `load_and_verify_formal_manifest_authorization`,
  which requires a future, independently-signed C file with the exact phase verdict
  (`AUTHORIZE_TRAIN_VALIDATION_MANIFEST_GENERATION` / `AUTHORIZE_TEST_MANIFEST_GENERATION`) and
  `authorizes_formal_manifest_generation=true`. The current gate (verdict `GENERATOR_SMOKE_GATE_PASS`) is
  refused (`FormalManifestGenerationNotAuthorized`).
- There is no `--force`, environment variable, `skip_authorization`, or hand-constructed dataclass bypass.

## Two-commit anchor strategy
- **A1** commits the writer code + real commit-freeze interface + tests + this implementation doc. A1's SHA is
  the future `generator_commit` / `writer_implementation_commit` anchor.
- **A2** commits only the machine report (`confirmatory_v4_formal_writer_implementation_report.json`), which
  records A1's SHA. A2 does not touch A1's writer source or tests. HEAD ends at A2; a future authorization
  pins **A1** as the writer implementation commit.

## Phase-specific gate (frozen seal order)
The seal scheme is `STEP0 pre-run freeze → STEP1 train_validation manifest → … → STEP5 freeze models/analysis
→ STEP6 test manifest → …`. The writer distinguishes `TRAIN_VALIDATION` from `TEST` at the machine level: a
single authorization authorizes **exactly one** phase (verdict → phase → `authorized_output_relpath`), and there
is no generic `phase=test` entry that bypasses Step 5. A test authorization must additionally carry the Step-5
prerequisites `model_analysis_freeze_sha256`, `train_validation_manifest_sha256`, `analysis_code_sha256` (all
64-hex); a test request that lacks them, uses the wrong verdict, or points at the wrong output dir raises
`EXPERIMENT_INVALID_EARLY_TEST_ACCESS` **before** any manifest is built.

## Real commit-freeze
`FormalCommitFreeze(protocol, generator, runtime, authorization, authorization_evidence_sha256)` is validated to
be all 40-hex, non-empty, **not** the smoke placeholder `a/b/c*40`, and field-for-field equal to the verified
authorization (with `generator_commit == audited_writer_commit` and the evidence hash bound). Repository
verifiers (`verify_commit_exists`, `verify_clean_worktree`, `verify_head_or_ancestor_policy`) require the commit
to exist (`git cat-file -e <sha>^{commit}`), a clean worktree, and HEAD to equal or descend from the pinned
authorization commit — the current git HEAD is never silently substituted for the pinned commit. All verifiers
accept injectable resolvers so tests never depend on a future commit existing.

## Formal environment provenance
`FormalManifestEnvironmentContext` is an immutable, copy-on-read, canonical-snapshot context with the exact
10-key set and the fixed markers `execution_mode=FORMAL_MANIFEST_GENERATION`,
`isaac_status=NOT_IMPORTED_NOT_LAUNCHED`, `gpu_status=NOT_USED_MANIFEST_GENERATION` (formal manifest generation
is an offline pure step). It rejects partial/extra/empty/bool values and refuses to reuse the `SMOKE_ONLY`
marker; the full env enters the phase-manifest full hash (a changed allowed version string changes the hash; a
changed fixed marker is rejected).

## Stamping and hash layering
`_stamp_manifest` deep-validates (`MI.validate_fully_resolved_phase_manifest`), computes the scientific anchor
`full_manifest_sha256` (`MI.fully_resolved_phase_manifest_hash`, over the **un-stamped** payload), writes the
flat self-hash field `integrity.full_manifest_sha256`, verifies stamping did not alter the payload, canonically
serializes the stamped payload (`allow_nan=False`), and computes `stamped_file_sha256` = sha256 of the stamped
**bytes**. The two hashes are deliberately distinct: `full_manifest_sha256` is the scientific anchor;
`stamped_file_sha256` is on-disk byte integrity.

## Bundle layout (one-shot atomic)
Each phase writes an independent directory `artifacts/formal/confirmatory_v4/<phase>/` containing exactly
`phase_manifest.json` (stamped manifest, carrying `integrity.full_manifest_sha256`), `phase_manifest.sha256`
(`<stamped_file_sha256>  phase_manifest.json\n`), `write_receipt.json` (schema
`confirmatory_v4_formal_manifest_receipt_v1`), and `SEALED`. `write_authorized_formal_manifest_bundle_atomic`
re-verifies the authorization and the prepared↔auth↔commit↔hash binding, enforces path containment, requires the
final directory to not exist, writes into a unique same-parent staging dir (flush+fsync each file, fsync dir,
read-back verify), atomically renames staging→final, fsyncs the parent, and re-reads the final dir. Any failure
removes the staging dir and leaves no final directory and no existing file changed. The write is **one-shot
create**: a second write is refused with `FORMAL_MANIFEST_ALREADY_EXISTS` even if byte-identical (no overwrite,
resume, or force).

## Path safety
`authorized_output_relpath` must be relative, contain no `..`, equal exactly
`artifacts/formal/confirmatory_v4/<phase>`, and have a final directory name equal to the phase; the longest
existing prefix of the resolved target must stay inside the repo root (symlink-escape containment). Absolute
paths, traversal, phase/dir mismatch, and symlinked subroots are rejected.

## Current generation LOCKED
Because no repo file yields a verified generation authorization, `prepare_authorized_formal_phase_in_memory` and
`write_authorized_formal_manifest_bundle_atomic` are unreachable via any real call chain: they fail at the
authorization gate before constructing a phase, creating `artifacts/formal`, creating a temp, or writing a file.
The CLI supports only `--preflight-only`, which confirms writer implementation is authorized and generation is
locked and prints `FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED`; it offers no `--force` or unauthorized write
path.

## Tests
`test_confirmatory_v4_formal_manifest_writer.py` (34 tests) covers §17.1-17.8: the current C gate cannot
authorize generation and the CLI reports generation-locked; the authorization loader accepts well-formed
train_validation/test files and rejects missing fields, wrong verdict, cross-phase, false generation, true
confirmatory data/run, writer-commit mismatch, phase/path mismatch, non-40hex commit, non-64hex hash,
`one_shot=false`; commit-freeze rejects smoke placeholders, auth inconsistency, non-existent commit (mock),
dirty worktree (mock), and auto-HEAD substitution (mock); the formal env enforces the exact key set + fixed
markers + immutability and rejects partial/extra/empty/bool/SMOKE_ONLY; the Step-5 early-test guard fires (with
`_build_unsealed_manifest` monkeypatched to prove the builder is never reached); the stamp pure function is
tested on a `TEST_ONLY_NON_SCIENTIFIC_PAYLOAD` with mock validate/hash (flat self-hash, full-vs-file-hash
separation, deterministic bytes, non-finite rejected); and the atomic bundle helper is exercised in `tmp_path`
(four files, hash agreement, one-shot refusal, unchanged existing bundle, failure leaves no final dir / no
staging residue, path traversal + symlink escape rejected). No test builds a full 228/72 formal payload through
the production success path, and no test writes under the repo `artifacts/formal` tree. All active regression
suites (generator 24, GEN-B fix 17, C generator-smoke final 9, block-state 12/15/10) stay green.
