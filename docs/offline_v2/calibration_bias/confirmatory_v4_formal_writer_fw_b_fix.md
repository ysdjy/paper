# Formal manifest writer — FW-B-001..014 batch fix (git-pinned trust root)

Closes the 14 frozen findings from Claude B's adversarial audit of the confirmatory v4 formal manifest writer +
commit-freeze interface (`FORMAL_WRITER_BATCH_FIX_REQUIRED`, B commit `1075105`, audited A1 `e1a5a67`). Still
implementation-only: no formal manifest is generated, no `artifacts/formal` bundle is created, no combined
plan / checkpoint / confirmatory data, no Isaac. No frozen scientific module and no Claude B/C audit file was
modified; the generator is reused byte-for-byte (`GEN._build_unsealed_manifest`).

## The core refactor — a git commit is the trust root
The production trust root is no longer a local JSON path, a private Python token, or a caller boolean. It is a
real git commit plus the committed bytes of a fixed allowlisted authorization artifact that pins the hardened
writer commit (F1), re-verified against the actual repository on every production entry
(`reverify_authorization_against_git`). New immutable objects: `VerifiedFormalRepositoryContext`,
`VerifiedFormalManifestAuthorization` (now carries `repository_context`, `committed_authorization_canonical_json`,
`committed_authorization_blob_sha256`), and a `GitReader` backend that production builds from the verified repo
root (tests inject a TEST_ONLY reader — a backend, never a skip switch).

## Per-issue closure
- **FW-B-001** — the public local-path loader is permanently locked; the only production loader is
  `load_and_verify_formal_manifest_authorization_from_git`, which reads the artifact from a FIXED allowlisted
  relpath via `git show <authorization_commit>:<relpath>` (committed bytes only, never the working tree), binds
  the evidence hash to those bytes, requires the commit + artifact to exist, and requires the artifact to pin
  F1. Arbitrary/tmp/untracked/dirty/symlink files can no longer mint an authorization.
- **FW-B-002** — all authorization JSON is parsed with a duplicate-key-rejecting `object_pairs_hook`; an exact
  per-phase required-key set (schema version included); unknown/extra fields rejected; strict `type(v) is bool`
  and lowercase-hex enforcement on a duplicate-free object.
- **FW-B-003** — `_AUTHORIZATION_LOADER_TOKEN` is documented and used as an internal misuse guard ONLY; every
  production entry calls `reverify_authorization_against_git`, so a hand-built object (no valid git-pinned
  context) is rejected regardless of the token.
- **FW-B-004** — `reverify_authorization_against_git` re-reads the committed artifact, re-parses it duplicate-
  free, compares `_canon(raw)` to the stored canonical JSON, and cross-checks EVERY critical field against the
  dataclass. `dataclasses.replace` / `object.__setattr__` mutations (raw JSON unchanged) are rejected.
- **FW-B-005** — `prepare_authorized_formal_phase_in_memory` has no `repo_root` (and no skip) argument; the repo
  root comes only from the git-verified authorization context, and commit-existence / clean-tree / HEAD /
  critical-blob verification runs unconditionally.
- **FW-B-006** — HEAD must equal the pinned authorization commit (an arbitrary descendant is rejected; a future
  successor needs a NEW authorization); F1 must be an ancestor of the authorization commit; and the writer /
  generator / every imported science module blob must be byte-identical across F1, the authorization commit,
  and the working tree (`CRITICAL_BLOB_RELPATHS`, hashed into `verified_critical_blob_map_sha256`).
- **FW-B-007** — the test-phase Step-5 gate verifies real artifacts before `GEN._build_unsealed_manifest`: the
  sealed train_validation bundle exists and its scientific `full_manifest_sha256` equals
  `train_validation_manifest_sha256`; `model_analysis_freeze.json` and `analysis_code_freeze.json` exist, their
  raw sha256 match the authorization, their schemas validate, and every listed analysis file's bytes hash to the
  frozen value (paths are safe, no outcome/data paths). Any mismatch → `EXPERIMENT_INVALID_EARLY_TEST_ACCESS`.
- **FW-B-008** — the writer re-parses the un-stamped snapshot, re-runs `MI.validate_fully_resolved_phase_manifest`,
  recomputes the scientific full hash, re-stamps, and byte-compares against the prepared snapshot before any
  filesystem effect; a hand-built garbage `PreparedFormalPhase` fails MI deep validation here.
- **FW-B-009** — the writer takes no `output_root`; the final directory is derived only from the verified repo
  root + `authorized_output_relpath` (path-contained), so a bundle can never be written under `/tmp` or an
  arbitrary root.
- **FW-B-010/011** — the one-shot claim is an atomic `renameat2(RENAME_NOREPLACE)` (never `os.replace`): a new
  target succeeds, an existing (empty OR non-empty) target → `FORMAL_MANIFEST_ALREADY_EXISTS`; the syscall being
  unavailable fail-closes to `FORMAL_ATOMIC_NOREPLACE_UNAVAILABLE` (no `os.replace` fallback). The commit point
  is a successful rename: a pre-commit failure leaves no final dir; a post-commit failure (parent fsync / read-
  back) returns a receipt with `committed=True, write_complete=True, post_commit_verification_passed=False` and a
  warning — it never raises while leaving a look-formal final directory. `verify_formal_manifest_bundle` re-checks
  a bundle afterwards.
- **FW-B-012** — a pre-commit failure removes the staging dir AND any parent directories this call newly created
  (in reverse order, only if empty), leaving no residue.
- **FW-B-013** — every writer exception carries a stable class-level `.verdict`; raw `OSError` at the write
  boundary is wrapped (`FormalBundleWriteError`), and `renameat2` EEXIST maps to `FormalManifestAlreadyExists`.
- **FW-B-014** — the bundle is exactly five fixed files including `bundle_index.json`, which indexes the four
  core files by sha256; its own sha256 (`bundle_index_sha256`) is returned in the receipt for external pinning
  (no self-reference cycle). `verify_formal_manifest_bundle` recomputes the index and the scientific hash and
  cross-checks receipt/SEALED, so post-write tampering of any file is detectable.

## Two-commit anchor
- **F1** (this commit): writer + CLI + tests + implementation/fix docs. Future Claude C authorization must pin
  `writer_implementation_commit = generator_commit = F1`.
- **F2**: the machine report only, recording F1's SHA; it does not modify F1's writer/CLI/tests.

## Tests + expected flips
`test_confirmatory_v4_formal_writer_fw_b_fix.py` (47 tests, §18.1-18.10) proves each finding closed with a
TEST_ONLY in-memory `GitReader` and real `renameat2` claims; `test_confirmatory_v4_formal_manifest_writer.py`
(16 tests) keeps the durable behavioural checks under the safe interface. All 14 frozen
`test_FW_B_00x_*` demonstrations in the (unmodified) audit-b file flip to failing, as designed. One audit-b
PASS-region test flips as an expected consequence of tightening the trust root:
`test_confirmatory_v4_formal_writer_audit_b.py::test_commit_freeze_exact_match_and_smoke_rejected` builds its
authorization via the now-locked local-path loader (`_mint_from_tmp`); the same behaviour (exact commit-freeze
match + smoke-placeholder rejection) is re-proven under the git-pinned loader by
`test_commit_freeze_exact_match_and_smoke_rejected_under_git_loader`. The audit-b file was NOT modified. Active
generator/block-state regressions stay green (24 + 17 + 9 + 12 + 15 + 10 = 87).

## Status
`FORMAL_WRITER_FW_B_001_014_FIXED_READY_FOR_C_AUDIT` — does NOT authorize formal manifest generation,
confirmatory data, or the 300-trial run. Current generation stays locked (no repo file mints an authorization);
the CLI is `--preflight-only`.
