# Confirmatory v4 generator + smoke — final independent gate (Claude C)

**Auditor:** Claude C, independent, **read-only**, **frozen scope** — only `GEN-B-001..007` plus the
kept-invariant generator/smoke behaviour. No re-audit of `FINAL-001..005` or the full preregistration; no new
issue IDs. No Isaac, no formal manifest/checkpoint/confirmatory data.

**Audited final fix:** `4330a8ab596167f59a9308a91600ef9fe253cb97` (B audit `c0ff67e`, first A fix `737aa31`,
initial generator `b6e6700`, generator GO `c8ac0af`, power-cert `2bf7217`).

## VERDICT: **GENERATOR_SMOKE_GATE_PASS**

All seven frozen findings are **RESOLVED**, every kept-invariant area **PASS**, and there are **0 new active
test failures**. **Power re-certification not required.**

| item | status |
|---|---|
| GEN-B-001 validated manifest/hash immutable binding | **RESOLVED** |
| GEN-B-002 public/secret envelope immutable | **RESOLVED** |
| GEN-B-003 unfrozen scientific value removed | **RESOLVED** |
| GEN-B-004 smoke commit context machine-unmistakable | **RESOLVED** |
| GEN-B-005 full 10-key provenance, public-builder gate | **RESOLVED** |
| GEN-B-006 strict selection + candidate completion | **RESOLVED** |
| GEN-B-007 atomic SMOKE_ONLY summary | **RESOLVED** |
| KAT / phase builder / storage+exec order / deep validator / full hash / auth gate / formal-writer zero-side-effect / smoke isolation / power | **PASS** |

## Invariance (§3)
`ahead_by(c0ff67e→4330a8a) = 2`; the diff touches **only** the generator layer (`confirmatory_v4_generator
.py`, `confirmatory_v4_generator_cli.py`, their tests, generator docs). `preregistration_v4.py`,
`confirmatory_v4_identity.py`, `confirmatory_v4_block_state.py`, `confirmatory_v4_manifest_integrity.py`,
`confirmatory_v4_selection.py`, `models_v2`, the power verdict, `config`, and the 306 data (`131750e1…`) are
**byte-unchanged** → `power_recertification_required = false`.

## Per-issue verification (first-hand)
- **GEN-B-001:** `GeneratedPhase` is a frozen dataclass over canonical-JSON; `unsealed_manifest`/`kat_report`
  are copy-on-read. Mutating a returned copy (`residual_value`/`execution_order_index`/`counts`,
  `kat.passed`) does not persist, and `MI.fully_resolved_phase_manifest_hash(g.unsealed_manifest) ==
  g.full_manifest_sha256`.
- **GEN-B-002:** `TrialExecutionEnvelope` copy-on-read — post-return injection (`residual_bias`, nested
  `actual_bias`) does not persist; the public payload holds no denylist field; `assert_no_secret_in_public`
  raises `PublicPayloadLeakageError`.
- **GEN-B-003:** the unfrozen `target_open_position=0.20` is removed (only a docstring note of removal
  remains); the public envelope holds **exactly** `{canonical_trial_identity, planned_episode_id, resume_key,
  role, offset}`.
- **GEN-B-004:** after 40-hex validation, `require_smoke_commit_context` raises `SmokeCommitContextError` for a
  real-looking 40-hex commit (and uppercase/short/non-hex are rejected); no phase/hash/file is produced;
  only the `a/b/c*40` placeholder is accepted under smoke.
- **GEN-B-005:** the exact **10-key** contract (`SMOKE_ENV_VERSION_KEYS`(7) + `SMOKE_ENV_FIXED`(3)) with the
  three fixed markers is enforced by the **public** `build_phase_manifest_in_memory` itself; single-key /
  missing / extra / empty-value / wrong-marker all raise `EnvironmentProvenanceError`. The gate runs **before**
  `_build_unsealed_manifest` — monkeypatching the builder to raise still yields `EnvironmentProvenanceError`
  first for a partial env (no CLI bypass to a partial-provenance hash). A changed allowed version string
  changes the full hash; the CLI records all 10 keys with `torch_version=UNAVAILABLE:<ExcType>` (no silent
  `except`) and never imports Isaac.
- **GEN-B-006:** `canonical_selected_offset` rejects bool/numeric-string/None/NaN/Inf/out-of-bank and accepts
  only `isclose(1e-12)` bank values (exact frozen float via `ID.CANDIDATE_BANK`); `evidence_hash` must be
  64-lowercase-hex. `SessionRunController` enforces probe-first, selection frozen from probe evidence before
  candidates, next-in-frozen-order completion with no duplicate and `attempt ∈ {0,1}` (via `ID.attempt_id`),
  `session_complete` only after all 3, and `offer_candidate_outcome_to_selector` raises.
- **GEN-B-007:** `CLI._atomic_write_smoke_summary` — same-dir temp → flush → fsync → `os.replace` → read-back
  verify; idempotent on identical content; `SmokeArtifactCollisionError` on differing content with the
  original intact; no leftover temp; returned dict == on-disk JSON; the summary carries no full
  blocks/sessions/trials/residual lists; filename carries `SMOKE_ONLY`.

## Real smoke (§12)
`CLI --smoke-only` for **both** phases: `rc=0`, KAT passed (6 vectors), 10-key env, `deep_valid=true`,
`rebuild_match=true`, `orders_distinct=true`, `write_complete=true`, `formal_manifest_written=false`,
`isaac_launched=false`; counts train_validation 15/57/228, test 9/18/72. **Only** `SMOKE_ONLY_{phase}_summary
.json` produced — no formal manifest, checkpoint, episode jsonl, or confirmatory record. The formal writer
refuses with zero filesystem side effect (no `open` in the module).

## Test classification (§13) — 0 new active failures
- **Active** generator/KAT/fix suites → **78 passed**; my final audit suite → **9 passed**.
- **Full** `calibration_bias` suite → **34 failed / 435 passed / 2 xfailed**, all accounted for:
  - **9** in the frozen `test_confirmatory_v4_generator_audit_b.py`: the 7 `test_GEN_B_00X_*` vulnerability
    demonstrations now flip (each finding demonstrated **closed**) + 2 PASS-region tests that build via the
    frozen 1-key `_ENV` helper and now correctly hit the GEN-B-005 gate — documented, expected.
  - **25** historical Claude-C reaudit files (do **not** import the generator): **23** are strict-xfail→XPASS
    flips (issues fixed; `--runxfail` → PASS) and **2** in `test_preregistration_v4_fix3_final_reaudit_c.py`
    are **pre-existing** since fix4 (they exercise the byte-unchanged `manifest_integrity` and were superseded
    by fix4's stricter validator; confirmed pre-existing — that test file and `manifest_integrity` are
    byte-unchanged since `c0ff67e`).
  - **Genuinely new active failures introduced by the generator work: 0.**

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **true** |
| authorizes_generator_smoke | **true** (`smoke_only=true`, isolated artifacts only) |
| authorizes_formal_manifest_writer_implementation | **true** |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

The next phase — implementing the **formal manifest writer** and the real commit-freeze interface — must
start from **this** C GATE_PASS commit. Actual formal `train_validation`/`test` manifest generation, the
combined plan, model checkpoints, confirmatory records, and the 300-trial run remain **unauthorized** and each
require their own gate. The ±0.035 power certification carries over unchanged.
