# Block-state KAT gate + authoritative description sync (offline)

Claude B. Closes the last two items of the same sampler blocker from Claude C's narrow re-audit
(`BLOCK_STATE_SAMPLER_FREEZE_INCOMPLETE`, commit `0040f28`) of the sampler fix (`5a51500`): (1) a MANDATORY
production known-answer gate (not just test-only drift protection), and (2) the authoritative
`deep_manifest_validator.checks` description of the exact block-state recompute. No re-open of FINAL-001..005,
no sampler redesign, no change to residual law / nuisance policy / power / model / data. No generator, no
Isaac, no formal manifest/checkpoint/confirmatory data, no GO. Machine form:
`block_state_kat_gate_completion.json`.

**New status: `PREREGISTRATION_V4_BLOCK_STATE_KAT_GATE_READY_FOR_C_FINAL_REAUDIT`.**
`power_recertification_required = false`.

## Scheme A: mandatory KAT gate (not a NumPy-version freeze)
`confirmatory_v4_block_state`:
- `KNOWN_ANSWER_VERSION = "pcg64_normal_floathex_kat_v1"`; `KNOWN_ANSWER_VECTORS` — an immutable tuple of 6
  FROZEN literals `(split, block_index, residual_subseed, residual_float_hex, nuisance_subseed)` for
  train 0/8, validation 0/5, test 0/8. Expected values are source-of-truth literals, NOT regenerated from the
  current sampler; byte-consistent with `confirmatory_v4_block_state_known_answers.json` (a test compares).
- Private unchecked core `_residual_value_from_subseed_unchecked` — the (unchanged) `Generator(PCG64(subseed))`
  → `normal(0, 0.005)` → rejection ≤1000 → no clip/fallback algorithm. **Not in `__all__`**; only
  `validate_known_answer_vectors` and the gated public residual API call it; the active spec forbids a
  generator from calling it.
- `validate_known_answer_vectors()` — recomputes each frozen vector via the unchecked core and checks
  `value.hex() == residual_float_hex`, nuisance policy `{}`, and the version constants; any mismatch raises
  `BlockStateCompatibilityError` (message includes vector identity, expected/actual hex, `numpy.__version__`,
  `RESIDUAL_SAMPLER_VERSION`, `KNOWN_ANSWER_VERSION`). Success returns
  `{passed, known_answer_version, sampler_version, numpy_version, vectors_checked=6}` — no phase/manifest.
- `require_known_answer_compatibility()` — the mandatory preflight; runs the KAT **every call** (no cache, so
  a failure can never be masked by a stale success and a monkeypatched version/constant is always re-checked;
  6 short draws — cheap).

## Gate is a mandatory precondition
- Public residual API (`residual_value_from_subseed` → `residual_value` → `resolved_block_state`) calls
  `require_known_answer_compatibility()` before producing any residual; the KAT internally uses only the
  unchecked core (no recursion).
- `validate_fully_resolved_phase_manifest` calls `BS.require_known_answer_compatibility()` at the very top
  (before schema/scientific-field checks); failure → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`, and since
  `fully_resolved_phase_manifest_hash` validates before hashing, **KAT failure → no full hash**.
- Active `block_state_sampler.compatibility_gate` + `deep_manifest_validator.checks` state that recording the
  NumPy version alone is not sufficient to release; the future generator's preflight #1 is
  `require_known_answer_compatibility()`.

## Authoritative validator description (§9.1)
`deep_manifest_validator.checks` replaces the loose "block-shared residual finite; one nuisance set per block"
with: the mandatory KAT gate line, and per-block
`residual_value == BS.residual_value_from_subseed(residual_subseed)` (exact float) +
`nuisance_values == BS.nuisance_values_from_subseed(...) == none_v1 {}`. Still exactly one authoritative
checks list.

## Invariance
residual mean/sigma/support, PCG64 rejection algorithm, `MAX_ATTEMPTS=1000`, `none_v1 {}`, identity/subseeds/
order, candidate bank, nominals, block counts, probe, models_v2, primary estimator, bootstrap, power results,
306 data, and FINAL-001..005 are all unchanged → `power_recertification_required = false`.

## Tests
`test_confirmatory_v4_block_state_kat_gate.py`: normal KAT (6 vectors, report, JSON==constants); hex-mismatch
injection blocks validate / public residual / resolved_block_state / validator / full hash; gate not
bypassable (public source not calling PCG64 directly; resolved_block_state via gated API; validator calls the
gate; unchecked core not in `__all__`); no-cache re-check behaviour. C re-audit
`test_block_state_sampler_reaudit_c.py --runxfail` → the two INCOMPLETE assertions PASS. B-owned suite green.
