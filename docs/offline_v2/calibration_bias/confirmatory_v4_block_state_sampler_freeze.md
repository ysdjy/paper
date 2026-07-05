# Confirmatory v4 block-state sampler freeze (offline)

Claude B. Closes `GENERATOR_IMPLEMENTATION_BLOCKED_MISSING_FROZEN_BLOCK_STATE_SAMPLER` (Claude A blocker
commit `33806f0`): the manifest required each block's `residual_value` + `nuisance_values` (both hash-covered)
but no unique `subseed -> value` function existed and the validator did not recompute them, so different
generators could produce different valid manifests. This is a focused protocol-completeness amendment — no
generator, no Isaac, no formal manifest/checkpoint/confirmatory data, no reopening of FINAL-001..005.
Machine form: `confirmatory_v4_block_state_sampler_freeze.json`; known-answers:
`confirmatory_v4_block_state_known_answers.json`.

**New status: `PREREGISTRATION_V4_BLOCK_STATE_SAMPLER_FROZEN_READY_FOR_C_REAUDIT`.**
`power_recertification_required = false`.

## Unique production module
`deployment_calibration/offline_v2/calibration_bias/confirmatory_v4_block_state.py` — the ONLY production
sampler (no second copy elsewhere). Public API: `RESIDUAL_SAMPLER_VERSION="pcg64_rejection_v1"`,
`NUISANCE_POLICY_VERSION="none_v1"`, `MAX_ATTEMPTS=1000`, `BlockStateSamplerError`,
`BlockStateSamplerExhausted`, `validate_subseed`, `residual_value_from_subseed`, `residual_value`,
`nuisance_values_from_subseed`, `nuisance_values`, `resolved_block_state`.

## Residual sampler (`pcg64_rejection_v1`)
- Law UNCHANGED: `TruncatedNormal(mean=0.0, sigma=0.005 m, support [-0.01,+0.01] m)`, taken from
  `residual_nuisance.DEFAULT_RESIDUAL` (and equal to the active `design.residual`).
- `rng = numpy.random.Generator(numpy.random.PCG64(block_residual_subseed))` — a fresh Generator per block;
  no global RNG, no stream splitting, no `default_rng` future-default, no Python `hash()`; iteration-order
  independent.
- Rejection: up to `1000` draws `float(rng.normal(0.0, 0.005))`, return the first in `[-0.01, +0.01]`; **no
  clip, no fallback-to-mean, no infinite loop**; on 1000 rejections raise `BlockStateSamplerExhausted`
  (→ `EXPERIMENT_INVALID_MANIFEST_INTEGRITY`).
- `validate_subseed` rejects bool, non-int, negative, and `>= 2**63-1`.

## Nuisance policy (`none_v1`)
`nuisance_values_from_subseed(subseed)` validates the subseed and returns exactly `{}`. residual is the only
current block-level hidden variation (v3 removed the artificial grasp perturbation). `nuisance_subseed` stays
a domain-separated, hash-covered reserved field; trial-level init is `trial_init_subseed` + runtime commit,
never block nuisance. The old placeholder `{"joint_delta": 0.0}` is removed. This does not change power (the
certified v3 power used only residual, no additional block nuisance).

## Deep validator recomputation
For every block the validator recomputes from the frozen sampler and requires:
`residual_subseed == ID.block_residual_subseed(...)`, `nuisance_subseed == ID.block_nuisance_subseed(...)`,
`type(residual_value) is float`, `residual_value == residual_value_from_subseed(residual_subseed)` (EXACT, no
tolerance), and `nuisance_values == {} == nuisance_values_from_subseed(nuisance_subseed)` (no extra key).
Distinct errors: `residual_value != frozen sampler output` and `nuisance_values != frozen none_v1 policy`;
sampler exhaustion → manifest-integrity failure.

## Reference fixture
`reference_phase_manifest` now fills each block from `BS.resolved_block_state(split, bi)`; the placeholders
`residual_value = 0.001` and `nuisance_values = {"joint_delta": 0.0}` are gone. Reference remains
test-only, not a production generator.

## Power invariance
Unchanged: residual probability law, candidate bank, train/val/test nominals, block counts, probe,
model/HP/seeds, primary estimator, bootstrap. No new block nuisance. → `power_recertification_required =
false`.
> The target probability law is unchanged. The amendment fixes only the deterministic mapping from each
> preregistered subseed to one residual draw and declares the already-absent additional block nuisance to be
> none_v1.

## Tests
`test_confirmatory_v4_block_state.py` (illegal subseeds; 6 known-answers via `float.hex()`; determinism;
order-independence; all 24 residuals in support; no global-RNG dependence; nuisance exactly `{}` fresh dict;
228/72 reference valid; 1-ULP / other-in-support / int-0 / bool residual → INVALID; nuisance extra key →
INVALID; subseed tamper → INVALID; reproducible canonical JSON + hash; law == DEFAULT_RESIDUAL + config).
B-owned suite green; C historical tests show no new active regression.
