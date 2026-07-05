# Confirmatory v4 generator — implementation (KAT-gated, smoke-only)

Implements the confirmatory v4 phase-manifest generator against the re-issued GO. **Smoke-only**: no formal
manifest, no model checkpoint, no confirmatory data, no 300-trial run, no Isaac.

## Authorization
- Re-issued GO: `REISSUE_GO_TO_GENERATOR_IMPLEMENTATION` (Claude C)
- New GO commit: `c8ac0af29a0884c8efb520151729d515613d70bb`
- KAT-fix (block-state sampler) commit audited: `cb92f4ff44faf93f22deb8e49dab79ef61dd1691`
- Prior Claude A blocker (now closed): `33806f0654396c4ef53b5e17beb25166c2db1f13`
- Verified `docs/.../block_state_kat_final_reaudit_c.json`: authorizes generator implementation + smoke;
  formal manifest / confirmatory data / confirmatory run NOT authorized; `power_recertification_required=false`.

## Source-of-truth imports (no scientific constant re-defined)
- `PRE` `preregistration_v4` — `frozen_seeds()`, `SECRET_DENYLIST`, `PRIMARY_SUCCESS.target_tolerance`.
- `ID` `confirmatory_v4_identity` — identities, subseeds, `canonical_phase_{block,session,trial}_identities`,
  `resolve_phase_execution_plan`, `resolve_candidate_order`, `candidate_order_key_map`, order subseeds,
  `planned_episode_id`, `attempt_id`, `canonical_planned_structure_hash`, `canonical_json`, `enumerate_trials`.
- `BS` `confirmatory_v4_block_state` — the UNIQUE frozen block-state sampler: `require_known_answer_compatibility()`
  (mandatory KAT) and `resolved_block_state(split, block_index)`. The private unchecked core is never called.
- `MI` `confirmatory_v4_manifest_integrity` — `PHASE_MANIFEST_SCHEMA_VERSION`, `MANIFEST_ALGORITHM_VERSION`,
  `canonical_config_sha256()`, `deterministic_environment_contract()`, `validate_fully_resolved_phase_manifest()`,
  `fully_resolved_phase_manifest_hash()`, `PHASES`, `PHASE_SPLITS`.

## Mandatory KAT preflight (§6)
`build_phase_manifest_in_memory` runs, in order: (1) `require_generator_authorization`, (2)
`BS.require_known_answer_compatibility()` (the first real work — the mandatory KAT, 6 float.hex vectors,
version `pcg64_normal_floathex_kat_v1`), (3) `validate_commit_context`, (4-7) build top/blocks/sessions/trials,
(8) `MI.validate_fully_resolved_phase_manifest` (which re-runs the KAT and recomputes+exact-checks
`residual_value` / `nuisance_values` from the frozen sampler), (9) `MI.fully_resolved_phase_manifest_hash`.
If the KAT fails, the build raises before any manifest object, hash, or summary exists. The generator source
contains no reference to `_residual_value_from_subseed_unchecked` (test-enforced).

## Block state (§13)
Each block's state is taken ONLY from `BS.resolved_block_state(split, block_index)` →
`{residual_subseed, nuisance_subseed, residual_value (KAT-gated PCG64 rejection), nuisance_values = {}}`. The
generator never samples or recomputes a residual itself. `reference_phase_manifest` is NOT wrapped; the build
is independent and a test asserts byte-equality of the canonical JSON with the reference (equality check only).

## Storage vs execution order (§13–15)
- Blocks/sessions in canonical STORAGE order (split rank → block asc → nominal numeric asc), asserted equal to
  `ID.canonical_phase_{block,session}_identities(phase)`.
- Trials list in `ID.canonical_phase_trial_identities(phase)`; each trial's `execution_order_index` comes from
  `ID.resolve_phase_execution_plan(phase)` (seed-derived; probe-first). Storage ≠ execution (test-checked).
- Phase counts: train_validation 15/57/228, test 9/18/72; never a single 300-trial serialization.

## Public / secret separation (§20)
`build_trial_execution_envelope` splits public (`canonical_trial_identity`, `planned_episode_id`, `resume_key`,
`role`, `offset`, public task fields) from secret (`nominal_bias`, `residual_bias`, `nuisance_values`, block
subseeds). `assert_no_secret_in_public` recursively rejects every ACTIVE `PRE.SECRET_DENYLIST` token; tests
inject `residual_bias`/`actual_bias`/`eff_signed`/`abs_eff`/`oracle_action`/`hidden_state_id` and confirm rejection.

## Session run-state interface (§21, no runtime)
`SessionRunController`: `PLANNED → PROBE_READY → PROBE_COMPLETE → SELECTION_FROZEN → CANDIDATES_READY →
SESSION_COMPLETE`. Probe executes first; `freeze_selection` only from `PROBE_COMPLETE` (probe evidence),
before any candidate outcome; `offer_candidate_outcome_to_selector` always raises (candidate outcomes never
reach the selector); `attempt_id` via `ID.attempt_id` (attempt ∈ {0,1}).

## Formal writer — LOCKED (§22)
`write_formal_phase_manifest_atomic` calls `require_formal_authorization` on its first line and refuses
(`GENERATOR_FORMAL_MODE_NOT_AUTHORIZED`) before any filesystem effect — no dir, no temp, no write, no
overwrite. Test asserts zero side effects.

## Smoke isolation (§18–19, §25)
CLI `confirmatory_v4_generator_cli --smoke-only --phase {train_validation|test}` builds in memory, rebuilds
once, compares canonical JSON + full hash, and writes ONLY a summary to `artifacts/smoke_only/
confirmatory_v4_generator/SMOKE_ONLY_<phase>_summary.json` (dir git-ignored; `SMOKE_ONLY` in every name; no
`manifest.json`/`.pt`/`episode.jsonl`). The summary contains counts + KAT/validation/rebuild flags and a
`smoke_only_in_memory_hash` explicitly marked `not_a_formal_manifest_anchor=true`; it carries NO full
blocks/sessions/trials/residual lists. CLI banner prints `*** SMOKE_ONLY — NOT A FORMAL CONFIRMATORY
ARTIFACT ***` first and last; refuses (exit 2, zero side effects) without `--smoke-only` or with
`--formal`/`--output`.

## Not authorized in this phase
Formal train_validation/test manifests, combined experiment plan, model checkpoints, confirmatory records,
the 300-trial run, and Isaac data collection are all out of scope and gate-locked.

## GEN-B-001..007 batch fix (post Claude B audit)
Claude B's generator+smoke consistency audit (`c0ff67e`, `GENERATOR_SMOKE_BATCH_FIX_REQUIRED`) raised 7
frozen issues; all are closed in this branch — see `confirmatory_v4_generator_gen_b_fix.md` /
`.json`. Summary: immutable copy-on-read `GeneratedPhase` + `TrialExecutionEnvelope` (hash/scan cannot be
bypassed post-validation), unfrozen `target_open_position` removed, smoke build requires the placeholder
commit context, a frozen 10-key smoke environment provenance contract with an atomic auditable summary
(temp+fsync+os.replace, read-back, idempotent, collision-guarded), and a strict selection + candidate-
completion FSM. Still smoke-only; no formal authorization is claimed.

### GEN-B-005 follow-up: provenance enforced at the public builder
The exact 10-key smoke provenance contract is enforced by the public `build_phase_manifest_in_memory` entry
point, not merely by the CLI. A caller cannot bypass the CLI and obtain a phase hash with partial provenance.
`validate_smoke_environment_provenance(environment_versions)` runs immediately after the smoke-commit gate and
before `_build_unsealed_manifest`, so any partial env (single wrong-named key / missing key / extra key /
wrong fixed marker) raises `EnvironmentProvenanceError` and produces no phase and no hash. All generator-test
build sites now use a full 10-key env, and `reference_phase_manifest` is compared with the same full env.
Because the builder now rejects partial provenance, the two frozen audit-b PASS tests that build via the
module-level 1-key `_ENV` helper (`test_builder_counts_and_orders_and_reference_equal`,
`test_authorization_gate_and_formal_writer_locked`) flip — an expected, unavoidable consequence (the frozen
audit-b file is not modified); the covered behaviours stay green in the generator suite under a full env.
