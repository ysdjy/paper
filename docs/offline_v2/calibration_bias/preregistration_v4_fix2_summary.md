# Preregistration v4 FIX2 — summary (offline)

Claude B. Closes the last two implementation blockers from Claude C's second-pass re-audit
(`MODIFY_PREREGISTRATION_V4_FIX1`, commit `21b676b`) of fix1 (`91a41e5`). No Isaac, no confirmatory runtime
generator, no manifest instance, no runtime/confirmatory data, no run authorization. Machine form:
`preregistration_v4_fix2_summary.json`.

**New status: `PREREGISTRATION_V4_FIX2_READY_FOR_FINAL_C_REAUDIT`.** `power_recertification_required = false`.

## Blockers — resolution
| # | blocker | fix |
|---|---|---|
| **1** | `BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_DEPENDENT` | new production selector `confirmatory_v4_selection.select_best_single` reads **only** observed `y.success` (allowlist `split, session_id, trial_role, theta.grasp_offset_local_y, y.success, planned_episode_id`); no tau/nominal/residual/eff/oracle/test/success-model. `best_single_legal` demoted to `SIMULATION_ONLY_REFERENCE`. Active config `best_single.implementation` repointed. Bridge 4500/4500. |
| **2** | `BLOCKER_PLANNED_IDENTITY_FORMAT_NOT_FROZEN` | `confirmatory_v4_identity` freezes block/session/trial templates, `+.3f` padding, `+0.000` zero, `planned_episode_id = v4ep-<sha256[:24]>`, domain subseeds, storage sort, canonical JSON. 300 planned ids, all unique. |

## Production selector interface
```
select_best_single(observed_candidate_records, *, candidate_bank=(-0.04,0.0,0.04)) -> artifact
```
- **reads only**: split, session_id, trial_role, theta.grasp_offset_local_y, y.success, planned_episode_id
  (projected to a frozen `ObservedCandidateOutcome`; secret audit fields projected away).
- **forbidden**: tau, nominal, residual, actual bias, eff, oracle, test records, success model.
- **algorithm**: equal denominators → max integer observed_success_count → tie min|offset| → tie bank order.
- **input completeness** (else `EXPERIMENT_INVALID_BEST_SINGLE_INPUT`): 171 records, 57 sessions,
  3/session, offset set `{-0.04,0,+0.04}`, no dup `(session,offset)`, unique `planned_episode_id`,
  split ∈ {train,validation}, `y.success` bool. No record ignored; denominator fixed.
- **artifact** saves: `n_trials_per_offset=57`, `success_count_per_offset`, `mean_success_per_offset`,
  `tie_set_after_step1/2`, `selected_offset`, `selection_rule_version`, `input_projection_hash`,
  `input_record_set_hash`, `selection_artifact_hash`. Offset 0 is **not** hardcoded.

## Bridge invariance (why no power recert)
`production_best_single_bridge_invariance_v1`: production selector (observed-only) vs
`SIMULATION_ONLY_REFERENCE` on all 9 power-cert configs × 500 seeds = **4500**, **mismatch 0** → the secret
path was non-decisive; best-single, primary contrast, effect, and power all unchanged →
`power_recertification_required = false`.

## Canonical planned identity
```
block   : v4|split={split}|block={block:02d}
session : v4|split={split}|block={block:02d}|nominal={nominal:+.3f}
trial   : v4|split={split}|block={block:02d}|nominal={nominal:+.3f}|role={role}|offset={offset:+.3f}
planned_episode_id = "v4ep-" + sha256(trial_identity.utf8).hexdigest()[:24]
attempt_id = {planned_episode_id}|attempt={attempt:02d}   (attempt NOT in scientific randomization)
resume_key = planned_episode_id
```
zero=+0.000 (negative zero forbidden); block 2-digit zero-based; splits {train,validation,test}; roles
{probe,candidate}; ranges train 00..08 / val 00..05 / test 00..08. **300 planned ids, all unique.**

## Domain subseeds (frozen labels)
`subseed(<seed>, canonical_identity + "|domain=<label>")` for
`residual, nuisance, session_order, nominal_order, candidate_order, trial_init`. Order-independent; resume
keyed on `planned_episode_id`.

## Determinism (minor)
`CPU`, `torch.set_num_threads(1)`, `set_num_interop_threads(1)`, `use_deterministic_algorithms(True)`,
`OMP/MKL/OPENBLAS_NUM_THREADS=1`; explicit HP; deterministic-op-unavailable → model-fit INVALID; no GPU
switch; library versions in `model_analysis_freeze.json`.

## Unchanged
candidate bank; test geometry ±0.035; DeepSets model + HP + 5 seeds; primary comparator + 0.15 CI; 75
sessions / 300 trials; power result. Fix1 resolutions (seeds, seal scheme, model-failure semantics) intact.

## Tests
- `test_preregistration_v4_fix2.py` — NEW, all PASS (production selector, identity, determinism, protocol).
- `test_preregistration_v4.py` + `test_preregistration_v4_fix1.py` — B-owned, status assertions updated to
  fix2; all PASS.
- Claude C audit files (`test_preregistration_v4_audit_c.py`, `test_preregistration_v4_fix1_reaudit_c.py`) —
  **UNMODIFIED**. In the fix1-reaudit file the canonical-identity strict-xfail now **XPASS(strict)→fail**
  (intended signal blocker 2 is fixed); the production-best-single strict-xfail (which inspects
  `best_single_legal` specifically) **stays xfail** because that helper is intentionally the
  secret-dependent `SIMULATION_ONLY_REFERENCE` — the production path is the separate
  `confirmatory_v4_selection.select_best_single`, covered by the fix2 PASS tests.

## Constraints honored
No Isaac; no generator; no manifest instance; no checkpoint; no runtime/confirmatory data; no run
authorization. 306 data, runtime, `models_v2`, capability map, historical power results, C's audit files,
prior preregistration, and the original band-edge verdict are **untouched**. B does **not** authorize A.
