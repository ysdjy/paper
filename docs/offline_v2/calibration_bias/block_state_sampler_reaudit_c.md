# Block-state sampler blocker re-audit (Claude C)

**Auditor:** Claude C, independent, **read-only**, **narrow scope** — only the closure of
`GENERATOR_IMPLEMENTATION_BLOCKED_MISSING_FROZEN_BLOCK_STATE_SAMPLER` (Claude A blocker `33806f0`). No
re-audit of algorithms/science/power/statistics; **FINAL-001..005 are not reopened**; no new FINAL IDs.
No Isaac, no generator, no formal manifest/checkpoint/confirmatory data.

**Audited fix commit:** `5a515001e43872abf2f2c492845373418288c5b8`; prior GO `45599ca`; power-cert `2bf7217`.

## VERDICT: **BLOCK_STATE_SAMPLER_FREEZE_INCOMPLETE**

Five of the seven sub-areas are **RESOLVED**; two remain **INCOMPLETE** (both inside the sampler blocker,
not new issues). **Power re-certification not required.**

| area | status |
|---|---|
| residual sampler (§3) | **RESOLVED** |
| nuisance none_v1 (§5) | **RESOLVED** |
| validator recomputation (§6) | **RESOLVED** |
| reference placeholder removal (§7) | **RESOLVED** |
| production source uniqueness (§8) | **RESOLVED** |
| **known-answer compatibility gate (§4)** | **INCOMPLETE** |
| **active spec consistency (§9.1)** | **INCOMPLETE** |

## Resolved (verified first-hand)
- **Residual sampler (§3):** law (mean 0.0, σ 0.005, support [−0.01,+0.01] inclusive) is identical across
  `confirmatory_v4_block_state`, `residual_nuisance.DEFAULT_RESIDUAL`, and `config.design.residual`. The
  algorithm is exactly `Generator(PCG64(block_residual_subseed))` → `normal(0,0.005)` → rejection ≤1000, no
  `default_rng`/`hash()`/global-RNG/clip/fallback, exhaustion → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`.
  Subseed domain rejects bool/float/None/negative/≥2⁶³−1 and accepts 0 / 2⁶³−2.
- **Known-answers (§10):** I independently recomputed all 6 vectors (train 0/8, validation 0/5, test 0/8) —
  `residual_subseed`, `nuisance_subseed`, `residual_value.hex()`, `nuisance_values={}` all match exactly; all
  24 blocks are in support, deterministic, and order- and global-RNG-independent.
- **Nuisance none_v1 (§5):** `NUISANCE_POLICY_VERSION="none_v1"`, returns a fresh `{}`, additional block
  nuisance disabled, `{"joint_delta":0.0}` removed; consistent with the certified v3 power (residual-only).
- **Validator recompute (§6):** deep validator recomputes residual/nuisance from the frozen sampler with
  EXACT float equality; rejects 1-ULP tamper, other-in-support, int 0, bool, nuisance extra key, and
  subseed tamper → INVALID; full hash only after deep validation.
- **Reference (§7):** `reference_phase_manifest` fills each block from `BS.resolved_block_state`; placeholders
  `0.001` / `joint_delta` gone; train_validation 228 and test 72 deep-validate.
- **Production uniqueness (§8):** `confirmatory_v4_block_state` is the unique production sampler — the
  validator and reference builder import only it, and `config.block_state_sampler.module` designates only it
  (with `generator_may_not_choose = [RNG/bit generator, truncation method, additional block nuisance]`). The
  exploratory/power samplers are off the production path.
- **Power (§11):** bank/nominals/blocks/residual-law/model/estimator/bootstrap unchanged; `models_v2` and the
  power verdict are byte-unchanged → `power_recertification_required = false`.

## INCOMPLETE (same blocker; must close before GO)

### §4 — no mandatory production known-answer / version gate
The mapping is deterministic **on this environment**, but nothing guarantees it stays fixed across NumPy
versions. `numpy.random.PCG64` raw bits are stable, yet `Generator.normal` (the ziggurat transform) is NumPy
distribution code that can change between versions — a future NumPy could produce a different, equally
in-support residual set. There is **no** production `validate_known_answer_vectors` (or equivalent mandatory
compatibility check) that the generator must run before manifest construction, and **no blocking** NumPy
version freeze (`environment_versions` only *records* numpy; the audit's Scheme-B says recording ≠ freezing).
The deep validator's self-recompute uses the same NumPy, so it **cannot** detect a `Generator.normal` drift.
- **Fix (Scheme A, recommended):** add a pure production `confirmatory_v4_block_state.validate_known_answer_
  vectors()` that recomputes the frozen vectors and asserts `residual_value.hex()` exact match; make the
  active protocol **require** it to pass before any phase manifest is constructed/hashed, else
  `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`. **(Scheme B:** freeze the exact NumPy version with a blocking
  check + still run the KAT.)

### §9.1 — authoritative validator description omits the exact recompute
The CODE exact-recomputes residual/nuisance (verified) and `block_state_sampler.validator_recomputes_exact_
values=true` is recorded, but the **authoritative** `deep_manifest_validator.checks` still reads only
*"block-shared residual finite in [−0.01,+0.01]; one nuisance set per block"* — omitting *"residual_value ==
frozen block-state sampler output"* and *"nuisance_values == none_v1 {}"*. This is the same single-authoritative-
source requirement established in the FINAL-001/002 rounds; §9.1 declares it a remaining gap of this blocker.
- **Fix:** add the two exact-recompute lines to the authoritative `deep_manifest_validator.checks`.

## Tests
- **B block-state suite** `test_confirmatory_v4_block_state.py` → **15 passed** (does not assert a mandatory
  production KAT gate or the authoritative-checks sync — hence the two gaps).
- **C re-audit** `test_block_state_sampler_reaudit_c.py` → **9 passed, 2 xfailed**
  (`known_answer_compatibility_gate`, `active_spec_consistency`).

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_generator_smoke | **false** |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

**Next step:** Claude B closes the two remaining sub-items (mandatory KAT/version gate + authoritative-checks
description); Claude C then re-audits **only** those two. On green, the sampler blocker is closed and the GO
is re-issued (from a fresh C GO commit — not by resuming `45599ca`). The ±0.035 power certification carries
over unchanged.
