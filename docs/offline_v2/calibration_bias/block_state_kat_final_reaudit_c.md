# Block-state mandatory KAT gate — final re-audit (Claude C)

**Auditor:** Claude C, independent, **read-only**, **narrow scope** — only the two previously-INCOMPLETE
items (`known_answer_compatibility_gate`, `active_spec_consistency`). Prior-RESOLVED sampler areas and
FINAL-001..005 are **not** reopened; no new IDs. No Isaac, no generator, no formal manifest/checkpoint/
confirmatory data.

**Audited fix commit:** `cb92f4ff44faf93f22deb8e49dab79ef61dd1691`; prior C re-audit `0040f289`; A blocker
`33806f0`; old GO `45599ca`; power-cert `2bf7217`.

## VERDICT: **REISSUE_GO_TO_GENERATOR_IMPLEMENTATION**

Both remaining items are **RESOLVED**; all prior sampler areas and FINAL-001..005 are **UNCHANGED_RESOLVED**.
No `CRITICAL_NEW_EVIDENCE`. **Power re-certification not required.** The block-state sampler blocker is
closed.

| item | status |
|---|---|
| known_answer_compatibility_gate (§2–§6) | **RESOLVED** |
| active_spec_consistency (§7) | **RESOLVED** |

## known_answer_compatibility_gate — RESOLVED (verified first-hand)
- **Frozen literals (§2):** `KNOWN_ANSWER_VERSION="pcg64_normal_floathex_kat_v1"`,
  `RESIDUAL_SAMPLER_VERSION="pcg64_rejection_v1"`, `KNOWN_ANSWER_VECTORS` = 6 **source-literal** tuples for
  {train 0/8, validation 0/5, test 0/8}; module literals == known-answers JSON field-by-field; expected
  values are not regenerated at import, and the production gate does not depend on reading the JSON.
- **Unchecked core private (§3):** `_residual_value_from_subseed_unchecked` keeps the verified PCG64
  rejection algorithm, is **not** in `__all__`, and the SPEC/config declare generators must never call it;
  the reference builder and validator use only the gated public API.
- **`validate_known_answer_vectors` (§4):** normal path returns `{passed:true, known_answer_version,
  sampler_version, numpy_version==actual, vectors_checked:6}`, recomputing each vector's subseeds, residual
  `float.hex()` (from the unchecked core), and `nuisance=={}`.
- **Drift blocks everything (§4.1–4.4):** hex mismatch, residual/nuisance subseed literal mismatch,
  `RESIDUAL_SAMPLER_VERSION`/`KNOWN_ANSWER_VERSION` drift, and non-empty nuisance each raise
  `BlockStateCompatibilityError` and block **all** of `validate_known_answer_vectors` /
  `require_known_answer_compatibility` / `residual_value(_from_subseed)` / `resolved_block_state` /
  `reference_phase_manifest` / `validate_fully_resolved_phase_manifest` / `fully_resolved_phase_manifest_hash`
  — no full hash, no resolved block state. The hex-mismatch error carries the vector identity, expected +
  actual hex, NumPy version, sampler version, and KAT version.
- **Exhaustion (§4.5):** a monkeypatched unchecked-core `BlockStateSamplerExhausted` propagates with verdict
  `EXPERIMENT_INVALID_MANIFEST_INTEGRITY` (a classified, non-generic manifest-integrity error); the KAT does
  not falsely return success and the downstream hash fails. Acceptable per the audit's error-contract note.
- **No success cache (§5):** `require_known_answer_compatibility` runs the KAT **every** call — pass → inject
  a bad vector → next call blocks → restore → passes again.
- **Gate in validator & hash (§6):** `validate_fully_resolved_phase_manifest` calls the gate **before** any
  scientific-field validation; `fully_resolved_phase_manifest_hash` runs the deep validator first. An
  already-built valid manifest is rejected by both validate and hash when KAT drift is injected — proving the
  gate is a real validator/hash preflight, not a reference-construction artifact. `train_validation` 228 /
  `test` 72 references deep-validate and re-hash identically under a passing KAT.

## active_spec_consistency — RESOLVED (§7)
The single authoritative `deep_manifest_validator.checks` now states the mandatory KAT gate
(`require_known_answer_compatibility` before validation/hash → `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`) and
the exact recompute: `residual_value == BS.residual_value_from_subseed(...)` (Python float, exact) and
`nuisance_values == BS.nuisance_values_from_subseed(...) == none_v1 {}`. The loose old line is no longer the
only rule. `block_state_sampler.compatibility_gate` is complete (function, 6 vectors, `float.hex()` exact,
`numpy_version_alone_is_not_the_gate=true`, `mandatory_before` incl. deep validation + full hash,
`on_mismatch=INVALID`, generator preflight #1, unchecked core forbidden).

## Invariance (unchanged) & power
Residual law `TruncatedNormal(0,0.005,[-0.01,0.01])`, PCG64 rejection, `MAX_ATTEMPTS=1000`, `none_v1 {}`,
`confirmatory_v4_identity` byte-unchanged; bank `{-0.04,0,+0.04}`, ±0.035, blocks 9/6/9, DeepSets/HP/seeds,
primary estimator, bootstrap, 0.15; `models_v2` and the power verdict (`POWER_SUFFICIENT_FOR_PREREG_V4`)
byte-unchanged; FINAL-001..005 unchanged; 306 `131750e1…`. The fixed subseed→draw mapping preserves each
block's marginal law and independence → **`power_recertification_required = false`**.

## Tests
- **C prior re-audit `--runxfail`**: `test_mandatory_known_answer_or_version_gate_exists` and
  `test_authoritative_checks_describe_exact_recompute` → **PASS**.
- **B block-state + KAT-gate suites** → **25 passed**.
- **C final re-audit** `test_block_state_kat_final_reaudit_c.py` → **12 passed**.

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **true** |
| authorizes_generator_smoke | **true** (`smoke_only=true`, isolated informal artifacts only) |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

**Claude A must start from / rebase onto THIS C GO commit** (not the old `45599ca`). Allowed: generator
implementation + `smoke_only=true` isolated smoke. Forbidden: formal `train_validation`/`test` manifest,
combined plan, checkpoints, confirmatory records, the 300-trial run — each requires its own gate. The
±0.035 power certification carries over unchanged.
