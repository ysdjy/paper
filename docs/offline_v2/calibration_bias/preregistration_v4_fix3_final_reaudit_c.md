# Preregistration v4 FIX3 — final independent re-audit (Claude C)

**Auditor:** Claude C, independent, **read-only, adversarial**. No Isaac, no generator, no formal manifest
instance, no checkpoint, no runtime/confirmatory data. Three files added only; B's fix3 files, `models_v2`,
historical data/power, and my prior audits are untouched.

**Audited:** `7304530eb07d926b96f3e4cbd878ac50fa1d4633` (fix3) vs prior re-audit `83d8606`, power-cert
`2bf7217`.

## VERDICT: **MODIFY_PREREGISTRATION_V4_FIX3**

One of the three prior blockers (identity domain) is **fully resolved**. The other two are resolved in
their **core** (single production entry + 45/12 composition; hash **layering**), but executing the fix3
modules surfaces **four remaining GO-blocking implementation gaps** — none touching the science/power:
a non-strict offset domain in the selector, a shape-only full-manifest validator, an under-constrained
combined hash, and stale active manifest-spec semantics. **Power re-certification is not required.**

`authorizes_generator_implementation = false`.

## Resolved (verified first-hand)
- **`BLOCKER_IDENTITY_DOMAIN_NOT_VALIDATED` — RESOLVED.** `PlannedTrialKey`/`_canon_numeric` reject
  out-of-domain input with `isclose(rel_tol=0, abs_tol=1e-12)` (so `0.0346`≠`0.035`), reject bool/non-finite;
  verified reject block=99, validation block=6, train nominal=0.5, probe offset=+0.02, candidate offset=+0.07,
  block=-1, attempt∉{0,1}. 300 canonical / 300 unique unchanged.
- **Best-single guard (core).** Single entry `select_best_single_confirmatory` (one param, no override
  kwargs, `__all__` frozen, core private); enforces 45/12→135/36→171 + globally-unique session_id + strict
  bool + no probe/test; **order- and secret-invariant** (offset AND artifact hash); offset 0 not hardcoded.
- **Manifest hash layering (core).** `canonical_planned_structure_hash` renamed (structure-only docstring);
  phase hashes 228/72; combined-plan hash; full hash covers all resolved fields (mutating residual changes
  the full hash; structure hash unaffected).
- **Determinism.** Machine-frozen: `device==cpu`, thread/interop/deterministic flags, `OMP/MKL/OPENBLAS=1`,
  set before any torch op, deterministic-op-unavailable → model-fit INVALID, no GPU switch.

## New blockers (fix before generator implementation; no power recert)

### 1. `BLOCKER_BEST_SINGLE_OFFSET_DOMAIN_NOT_STRICT`
The selector canonicalizes offsets with `round(float(x), 3)` and compares to the rounded bank. Verified
**accepted (smuggled into the bank)**: `+0.0004`→0.0, numeric string `'0.0'`→0.0 (via `float()`),
`-0.0396`→−0.04, `+0.0404`→+0.04 (NaN raised a raw `KeyError`, not a clean error). The **identity** module
already does this right (`isclose` `abs_tol=1e-12`, reject bool/str/non-finite) — the selector does not.
- **Fix:** canonicalize each offset like `_canon_numeric`: reject bool/non-numeric/non-finite; require
  `isclose(offset, bank_value, rel_tol=0, abs_tol=1e-12)` (not `round(.,3)`).

### 2. `BLOCKER_FULL_MANIFEST_VALIDATOR_SHALLOW` (subsumes 4.1)
`validate_fully_resolved_phase_manifest` checks only phase, required-key **presence**, list lengths, and
counts — **no** semantic validation. Verified: a `train_validation` manifest with **228 identical/duplicate
trials**, **all-'train' blocks** (wrong 9-train+6-val composition), and `planned_episode_id` **inconsistent**
with `canonical_trial_identity` **validates and produces a "valid" full hash**. A shape-only hash of a
duplicated manifest is not a scientific integrity anchor.
- **Fix (or freeze as required generator-validator rules):** per-phase split composition (train_validation
  = 9 train + 6 val blocks / 45 train + 12 val sessions; test = 9 test blocks / 18 sessions); block/session/
  trial uniqueness; 4 trials/session = probe + 3 candidates (bank); `planned_episode_id ==
  planned_episode_id(canonical_trial_identity)`; referential integrity; residual/nuisance shared within a
  block; `execution_order_index` unique/complete; `resolved_candidate_order` a bank permutation; subseeds
  recomputed from frozen identity+seed; `resume_key == planned_episode_id`; `planned_structure_sha256` ==
  frozen; frozen seeds/config/protocol match; all numerics finite; reject stray fields (incl. a nested
  `integrity` block).

### 3. `BLOCKER_COMBINED_HASH_OPTIONAL_FIELDS`
`combined_experiment_plan_hash` leaves `analysis_code_sha256=None` and `bootstrap_seed=None` as optional;
verified it computes a hash with both omitted, though the preregistration requires the combined anchor to
cover them.
- **Fix:** make both REQUIRED (raise `EXPERIMENT_INVALID_MANIFEST_INTEGRITY` on None/empty) and
  format-validate the hex fields + integer bootstrap seed.

### 4. `BLOCKER_ACTIVE_MANIFEST_SPEC_STALE`
The active `confirmatory_v4_manifest_spec.md` retains pre-fix3 semantics conflicting with the phase-layered
design: §3.2 `subseed(<split>_residual_seed, "block=<i>")` (code keys on canonical `block_identity()` +
`|domain=residual`); §5 `science_manifest_sha256 == frozen manifest_hash` (contradicts lines 93-98 binding it
to the **phase** hash). `config`/`preregistration_v4` `manifest_fields` still list singular `"manifest_hash"`.
- **Fix:** update the block-subseed example to the canonical form; rewrite the §5 integrity anchor to the
  phase hashes; replace the singular `manifest_hash` field with the layered set.

## Minor
- **Flat self-hash key.** `SELF_HASH_FIELD = "integrity.full_manifest_sha256"` (flat dot-key); code and
  manifest_spec agree, so it is consistent, but the shallow validator would hash a stray **nested**
  `{"integrity": {...}}` block as content. Pin the flat convention in the JSON schema + tests and have the
  deepened validator reject any nested `integrity`/unexpected self-hash placement (folds into blocker 2).

## Tests
- **B fix3 suite** `test_preregistration_v4_fix3.py` → **26 passed**.
- **C final re-audit** `test_preregistration_v4_fix3_final_reaudit_c.py` → **7 passed, 4 xfailed** (the four
  new blockers). Prior C audit files unmodified.

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_generator_smoke | **false** |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

After B (1) makes the selector offset domain strict, (2) deepens the full-manifest validator to enforce
composition/uniqueness/identity↔PID/referential/subseed semantics, (3) makes the combined-hash analysis-code
and bootstrap-seed required, and (4) removes the stale manifest-spec semantics, a short re-audit closes this
out — the ±0.035 power certification carries over unchanged.
