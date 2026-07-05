# Preregistration v4 FIX2 — final independent re-audit (Claude C)

**Auditor:** Claude C, independent, **read-only, adversarial, final short pass**. No Isaac, no generator,
no manifest instance, no checkpoint, no runtime/confirmatory data. Three files added only. B's fix2 files,
`models_v2`, historical data/power, and my prior audits are untouched.

**Audited:** `48976e4918a11406daf805f6f6451ea8272b72d5` (fix2) vs fix1 `91a41e5`, prior re-audit `21b676b`,
power-cert `2bf7217`.

## VERDICT: **MODIFY_PREREGISTRATION_V4_FIX2**

Both remaining blockers are resolved **in their core** — the production best-single now reads only
observed outcomes, and the canonical identity format is frozen. But the final adversarial pass finds
**three GO-blocking implementation gaps** (all fixable, none touching the science/power): the best-single
input guard is incomplete, the identity builder does not validate the frozen domain, and the "manifest
integrity hash" is structure-only. **Power re-certification is not required.**

`authorizes_generator_implementation = false` (GO withheld pending the three fixes).

---

## The two closed blockers — cores verified

| blocker | core status | first-hand evidence |
|---|---|---|
| `BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_DEPENDENT` | **RESOLVED (core)** | config `implementation = confirmatory_v4_selection.select_best_single`; `best_single_legal` demoted to `SIMULATION_ONLY_REFERENCE` (“NOT production”). Selector allowlist-projects to the 6 legal fields; **secret perturbation invariant, observed `y.success` decisive** (functional test). Algorithm = max observed-success count → min\|offset\| → bank order, **787/787** vs the frozen rule, **not** hardcoded 0. Rejects test/probe/non-bool. |
| `BLOCKER_PLANNED_IDENTITY_FORMAT_NOT_FROZEN` | **RESOLVED (core)** | `confirmatory_v4_identity` freezes block/session/trial templates, `+.3f`/`+0.000` (neg-zero normalized), 2-digit zero-based block, role distinguishes probe/candidate@−0.04; `planned_episode_id=v4ep-<sha256[:24]>`, **300 planned / 300 unique** (recomputed); domain subseeds via `PRE.subseed` (sha256, not Python `hash()`); attempt excluded from randomization. |

Also verified: fix1 resolutions intact (15 exact seeds, SCHEME_2 seal, model-failure namespace); determinism
flags frozen (`set_num_threads`/`set_num_interop_threads`/`use_deterministic_algorithms`/`OMP,MKL,OPENBLAS`);
power inputs unchanged (`POWER_SUFFICIENT_FOR_PREREG_V4`). Bridge invariance holds → **no power recert**.

---

## New blockers (must fix before generator implementation; no power recert)

### 1. `BLOCKER_BEST_SINGLE_INPUT_GUARD_INCOMPLETE`
- **§3.3** `_check_completeness` verifies **total 57** sessions and 3-per-session, but **not** the **45
  train / 12 validation** composition. A malformed split (e.g. 57 train + 0 val) passes, and could change
  best-single (the baseline is the pooled train+val mean). The config `input_completeness.checks` lists
  “57 unique sessions”, not 45/12.
- **§3.4** `select_best_single` (the config-designated production entry) exposes `validate_completeness=
  False`, `candidate_bank=…`, `expected_candidate_records=…`. The active config **freezes none** of these
  (0 occurrences) and there is **no non-default → INVALID guard**. So A can disable the completeness gate or
  swap the bank on the production path — per the task's §3.4 this alone **withholds GO**.
- **Fix:** (a) assert exactly 45 train + 12 validation sessions; (b) provide a public
  `select_best_single_confirmatory(records)` wrapper that hardcodes `validate_completeness=True`,
  `expected_candidate_records=171`, `candidate_bank=(-0.04,0,+0.04)` and accepts no overrides (or freeze
  them in config and make any non-default production call `EXPERIMENT_INVALID_BEST_SINGLE_INPUT`); keep
  `validate_completeness=False` reachable only inside the sim bridge.

### 2. `BLOCKER_IDENTITY_DOMAIN_NOT_VALIDATED`
`trial_identity`/`block_identity`/`session_identity` validate only `split` and `role`; they format
arbitrary out-of-domain input. Verified **accepted**: `block=99`, `nominal=0.5` (not in split), probe
`offset=+0.02` (must be −0.04), `block=-1` (also breaks 2-digit padding), `offset=0.07` (not in bank).
Despite `enumerate_trials()` being domain-correct, the low-level builders leave A manifest freedom.
- **Fix:** validate the frozen domain in the builders (or a mandatory wrapper): `block∈BLOCKS[split]`,
  `nominal∈NOMINALS[split]`, probe `offset==-0.04`, candidate `offset∈CANDIDATE_BANK`, `block≥0`,
  `attempt∈[0,1]`; reject otherwise.

### 3. `BLOCKER_MANIFEST_INTEGRITY_HASH_STRUCTURE_ONLY`
`canonical_manifest_hash()` hashes only `[canonical_trial_identity, planned_episode_id]` (planned
**structure**). But `confirmatory_v4_manifest_spec.md` calls `manifest_hash` “sha256 of the **fully-resolved
manifest** (pre-run)”, says it is “Reproducible via `confirmatory_v4_identity.canonical_manifest_hash()`”,
and uses it as the per-trial integrity anchor (`science_manifest_sha256 == frozen manifest_hash`). A
structure-only anchor does **not** cover the resolved residual, nuisance, execution order, or environment —
so residual/nuisance/order tampering would **pass** the integrity check. This affects seal/integrity, so
it is a blocker, not a minor.
- **Fix:** rename to `canonical_planned_structure_hash()`, define a **separate full `manifest_hash`** over
  the fully-resolved plan (structure + per-block residual values + nuisance + resolved orders + seeds +
  `config_hash` + code/generator commit + environment) with a frozen serialization, and use **that** as the
  integrity anchor. Correct the manifest_spec wording.

## Minor
- **Determinism (wording):** CPU + thread/interop/deterministic/OMP/MKL/OPENBLAS flags are frozen; also
  assert `device==CPU` at runtime and that a deterministic-op-unavailable `RuntimeError` → model-fit
  INVALID (stated in the fix2 summary; ensure it is in the machine config), with `set_num_interop_threads`
  set before any torch op.

## Tests
- **B fix2 suite** `test_preregistration_v4_fix2.py` → **23 passed**.
- **C final re-audit** `test_preregistration_v4_fix2_final_reaudit_c.py` → **10 passed, 3 xfailed** (the
  three new blockers).
- **C prior audit files** unmodified: the fix1-reaudit canonical-identity strict-xfail now XPASS→fail
  (intended: format frozen); the production-best-single strict-xfail stays xfail because it inspects the
  `SIMULATION_ONLY` `best_single_legal`, which is intentionally secret-dependent — the production path is
  the separate `select_best_single`, covered by the PASS tests above.

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_generator_smoke | **false** |
| authorizes_manifest_instance_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

After B (1) enforces the 45/12 composition and closes the completeness/bank bypass, (2) validates the
frozen identity domain, and (3) separates a true full-manifest integrity hash from the planned-structure
hash, a **very short** Claude C re-audit closes this out. None of the fixes touches the estimator, model,
geometry, or effect, so the ±0.035 power certification carries over unchanged.
