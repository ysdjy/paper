# Confirmatory v4 FORMAL manifest writer + commit-freeze — Claude B one-shot adversarial audit (read-only)

Independent, read-only, adversarial audit of Claude A's formal manifest writer and real commit-freeze
interface. Actual formal generation stays LOCKED; this audit does **not** authorize it.

- audited writer commit (A1): `e1a5a67bc10110b157798b028480116eb0627582`
- audited report commit (A2): `0f28df9499d46efdb1965e7b528c2775e1646644`
- source C generator-smoke gate: `f41e36620b721455263fb8489bc906096a2f7442`
- **VERDICT: `FORMAL_WRITER_BATCH_FIX_REQUIRED`** — 11 BLOCKING + 3 NONBLOCKING. `issue_list_frozen = true`.
- Machine form: `confirmatory_v4_formal_writer_audit_b.json`. Findings demonstrated by
  `test_confirmatory_v4_formal_writer_audit_b.py` (20 passed).
- **No formal manifest / confirmatory data / run is authorized. No `artifacts/formal` was created. No Isaac.**

## Anchor integrity (§2)
`f41e366 → e1a5a67` ahead_by 1 (A1 = writer + preflight CLI + writer tests + implementation doc, additions
only). `e1a5a67 → 0f28df9` ahead_by 1 (A2 = the report JSON only; it does **not** modify A1 writer source or
tests). Future authorizations must pin `writer_implementation_commit = generator_commit = A1 e1a5a67`, never A2.

## What is correct (verified — PASS region)
- **Formal environment context** (`FormalManifestEnvironmentContext`): exact 10-key set, non-empty str, the
  three fixed FORMAL markers, copy-on-read canonical snapshot, immutable, rejects partial/extra/empty/bool and
  the `SMOKE_ONLY` marker; enters the full hash.
- **Commit-freeze exact match** (`validate_formal_commit_freeze`): all 40-hex, non-empty, not the a/b/c*40
  smoke placeholder, field-for-field equal to the verified authorization incl. `generator_commit ==
  audited_writer_commit` and the evidence binding.
- **Path safety** (`_resolve_contained_final_dir`): rejects absolute paths, `..`, phase mismatch, and
  symlink escape of the resolved existing prefix.
- **Current generation lock (the literal current gate)**: the loader refuses the current C gate
  (`GENERATOR_SMOKE_GATE_PASS` is not a generation verdict) and the CLI is `--preflight-only` with no
  `--force/--write/--phase` and no bundle-write call. *(But see the trust-root findings: the fail-closed
  guarantee this rests on is defeated.)*
- **No writer regressions**: full `calibration_bias` suite = **478 passed / 34 failed / 2 xfailed**; the 34
  failures = **25** historical Claude-C reaudit files + **9** Claude-B `generator_audit_b` demonstration flips
  caused by A's earlier GEN-B fix (`4330a8a`); **none imports the formal writer** (targeted 121 passed). A2's
  regression claim is accurate.

## BLOCKING findings (fix all before any formal gate)

The whole module is advertised as fail-closed because "no repository file can mint an authorization." That
premise is false — the authorization trust root, the executed-code pin, the early-test gate, and the payload
integrity are all bypassable.

- **FW-B-001 — authorization trust root not bound to a pinned artifact (§5/§11).** `load_and_verify_...` reads
  **any** local path via `open()`; `authorization_commit` is a decorative caller-supplied 40-hex that is never
  resolved against git; the evidence hash is self-referential over the file's own bytes. An arbitrary tmp /
  untracked / dirty / symlinked / uncommitted JSON with correct fields mints a
  `VerifiedFormalManifestAuthorization`. **Fix:** read the authorization from an allowlisted fixed repo relpath
  via `git show <authorization_commit>:<relpath>`, require the commit to exist and contain the artifact, hash
  the **committed** bytes, reject working-tree/symlink substitutes, and require `audited_writer_commit == A1`.
- **FW-B-002 — non-deterministic authorization schema (§6).** `json.load` has no `object_pairs_hook`; duplicate
  keys resolve last-wins (`generation:false` then `generation:true` mints), and unknown/extra control fields
  are not rejected. **Fix:** duplicate-key-rejecting parse, exact per-phase required-key set, reject extras.
- **FW-B-003 — loader token is a public trust root (§7).** `_AUTHORIZATION_LOADER_TOKEN` is a public module
  attribute; a hand-built authorization object with a self-consistent evidence hash passes
  `_reverify_authorization_integrity`. Production trusts token + fields, not a git-pinned artifact. **Fix:** the
  builder/writer must re-load and re-verify the pinned artifact (FW-B-001); the token stays a misuse guard only.
- **FW-B-004 — incomplete object re-verification (§8).** `_reverify_authorization_integrity` cross-checks only
  `verdict` + three bools + `one_shot` against the raw JSON. A `dataclasses.replace` / `object.__setattr__` that
  mutates `audited_writer_commit` / `protocol_commit` / `generator_commit` / `runtime_commit` /
  `authorized_output_relpath` / `authorized_phase` / test prereqs / `authorization_commit` /
  `power_recertification_required` is undetected (raw unchanged → evidence still matches). **Fix:** re-parse
  and cross-check **every** critical field; reject any divergence.
- **FW-B-005 — commit freeze is optional (§9).** `prepare_authorized_formal_phase_in_memory(..., repo_root=
  None)` defaults to `None`, which skips commit-existence, clean-tree, and HEAD/ancestor verification entirely.
  **Fix:** make `repo_root` mandatory in production and verify all commits (incl. `authorization_commit`),
  clean tree, and HEAD/ancestor unconditionally.
- **FW-B-006 — executed code not pinned to A1 (§10/§11).** `verify_head_or_ancestor_policy` accepts **any**
  descendant of the authorization commit, and nothing compares the imported writer/generator source bytes to
  the A1 blob. A successor commit that modified the writer (clean worktree) still generates. **Fix:** require
  `HEAD == authorization_commit` or an explicit successor allowlist **plus** blob-hash equality of
  writer/generator/protocol/runtime files; verify A1 is an ancestor of / pinned by the authorization commit.
- **FW-B-007 — early-test gate is format-only (§12).** The test-phase Step-5 gate checks only that the three
  prerequisite hashes are 64-hex; it never verifies the train_validation SEALED bundle, `model_analysis_freeze`,
  or analysis code exist and match. Fabricated `a*64/b*64/c*64` with zero Step-5 artifacts build a test formal
  phase (no `EXPERIMENT_INVALID_EARLY_TEST_ACCESS`). **Fix:** require the sealed train_validation bundle +
  model/analysis freeze + analysis-code hash to actually exist and match before `_build_unsealed_manifest`.
- **FW-B-008 — writer never re-validates the scientific payload (§14/§18).**
  `write_authorized_formal_manifest_bundle_atomic` checks only `sha256(stamped bytes)==stamped_file_sha256` and
  `stamped[self]==full_manifest_sha256` — both attacker-controllable. It never re-runs
  `MI.validate_fully_resolved_phase_manifest` or recomputes `MI.fully_resolved_phase_manifest_hash`. A hand-built
  `PreparedFormalPhase` with a **garbage** payload is written as a full SEALED bundle; the final read-back also
  skips deep validation. **Fix:** re-parse the un-stamped snapshot, re-validate, recompute the scientific full
  hash and require equality, re-stamp and byte-compare — or give `PreparedFormalPhase` unforgeable,
  writer-reverifiable process evidence.
- **FW-B-009 — output root unbound (§16).** `output_root` is a free writer argument; prepare (which holds the
  optional repo verification) and write are decoupled, so a bundle writes under any directory (e.g. `/tmp`).
  **Fix:** `output_root` must equal the verified repo root from the commit-freeze context and cannot be a free
  caller argument.
- **FW-B-010 — ambiguous commit on post-rename failure (§17.2).** A failure **after** `os.replace` (parent
  fsync / final read-back) raises to the caller while the fully-populated final bundle dir remains → the caller
  sees FAILURE yet the one-shot dir is occupied (next attempt → `FORMAL_MANIFEST_ALREADY_EXISTS`); the
  post-rename `OSError` also has no stable verdict. **Fix:** define one commit point — pre-rename failure leaves
  no final; post-rename either succeeds with a verifiable receipt or reliably rolls back/quarantines; never
  raise while leaving a look-formal final dir.
- **FW-B-011 — os.replace is not an exclusive one-shot claim (§17.3).** The claim rests on non-atomic `exists()`
  checks + `os.replace`, which **replaces an empty competitor final dir** (verified empirically) and yields a
  raw `OSError(ENOTEMPTY)` with no verdict on a non-empty race. **Fix:** an atomic exclusive claim
  (`os.mkdir(final)` lock / `RENAME_NOREPLACE` / `O_EXCL` sentinel) so an existing bundle can never be replaced;
  map races to `FORMAL_MANIFEST_ALREADY_EXISTS`.

## NONBLOCKING findings
- **FW-B-012 — empty parent-dir residue on pre-rename failure (§17.1).** A pre-rename failure leaves the empty
  `artifacts/formal/confirmatory_v4/` chain behind (no final phase dir, no staging). Contradicts A2's "no
  residue" claim but is empty and does not poison the one-shot lock. **Fix:** remove newly-created parents on
  failure, or document the empty residue.
- **FW-B-013 — writer exceptions lack a `.verdict` (§19).** Unlike the project convention (GEN/MI exceptions
  expose a class-level `verdict`), the writer's exception classes have none, and post-rename/race raise raw
  `OSError`. **Fix:** add a stable class-level `verdict` to each writer exception and wrap raw `OSError` at the
  write boundary.
- **FW-B-014 — receipt/SEALED have no self-hash (§18).** `write_receipt.json` and `SEALED` carry no
  self-verifiable hash/signature and are not indexed by a tamper-evident root, so they can be edited post-write
  undetectably (subsumed once FW-B-008 is fixed). **Fix:** add a receipt/SEALED self-hash and/or a bundle root
  index.

## A2 report truthfulness (§20)
A2's factual claims verified accurate: A1 SHA, "no formal artifact directory created", "34 writer tests
passed", "0 new active failures / no failure comes from the formal writer", the anchoring strategy, and the
current-gate rejection. **However**, three A2 claims are contradicted by the adversarial attacks above and must
not be relied on until fixed: `verified_authorization_only_via_loader` / `no_force_or_env_or_flag_bypass`
(defeated by FW-B-001/002/003 — any local JSON or a hand-built object mints an authorization) and
`failure_leaves_no_final_dir_and_no_staging_residue` (defeated by FW-B-010 post-rename final dir, and FW-B-012
empty parent residue).

## Sub-gate summary
authorization trust root **FAIL (001,002)** · authorization object integrity **FAIL (003,004)** · commit
freeze mandatory **FAIL (005)** · executed-code pinning **FAIL (006)** · test early-access guard **FAIL (007)**
· formal environment PASS · prepared-phase integrity **FAIL (008)** · hash layering **FAIL (008)** · output-root
binding **FAIL (009)** · atomic one-shot bundle **FAIL (011)** · failure recovery **FAIL (010; nonblocking 012)**
· current generation lock PASS *(guarantee defeated by 001–004)* · new active test failures **0**.

`power_recertification_required = false` (no scientific input/design change; all findings are the formal-writer
layer's authorization / commit-pinning / payload-integrity / one-shot atomicity). Post-report new issues:
CRITICAL_NEW_EVIDENCE only. **Claude B does not authorize actual formal manifest generation.**
