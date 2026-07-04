# Preregistration v4 FIX1 — summary (offline)

Claude B. Resolves the four blockers from Claude C's independent audit (`MODIFY_PREREGISTRATION_V4`, commit
`2ab3906`) of preregistration v4 (`9a02eb5`). No Isaac, no generator, no manifest instance, no confirmatory
data, no run authorization. Machine form: `preregistration_v4_fix1_summary.json`.

**New status: `PREREGISTRATION_V4_FIX1_READY_FOR_C_REAUDIT`** (replaces
`PREREGISTRATION_V4_READY_FOR_INDEPENDENT_AUDIT`). `power_recertification_required = false`.

## Blockers — resolution
| # | blocker | fix |
|---|---|---|
| **A** | `BLOCKER_SECRET_TIEBREAK` | best-single tie-break rewritten to **max mean success → min \|offset\| → fixed bank order**; the `τ−\|eff\|` step (reads secret `actual_bias`) is removed. Impl `best_single_legal`. |
| **B** | `BLOCKER_SEEDS_NOT_FROZEN` | 15 randomization seeds are **exact integers** from a frozen SHA256 rule over `SEED_ROOT="confirmatory-v4\|2bf7217"`; identity-addressed `subseed`; `bootstrap_seed` frozen pre-unseal. |
| **C** | `BLOCKER_SEAL_SCHEME_NOT_UNIQUE` | single scheme `SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION` (8 steps, no either/or); early-test access → `EXPERIMENT_INVALID_EARLY_TEST_ACCESS`. |
| **D** | `BLOCKER_MODEL_FAILURE_SEMANTICS` | model-fit failure is an analysis-stage event in its own `model_fit_failure_count` namespace, **separate** from the 300-trial `runtime_trial_invalid_count`; all 5 seeds must be valid; retry ≤ 1; else `EXPERIMENT_INVALID_MODEL_FIT`. |

## Selection invariance (why no power recert)
`best_single_tiebreak_invariance_v1`: over all 9 power-cert configs × 500 seeds = **4500 replicates**,
**step-1 tie count = 0**, **old↔new mismatch = 0**, all select offset 0. The removed secret tie-break was
**dead code**; best-single, the primary contrast, the effect size, and the power result are unchanged →
`power_recertification_required = false` (matches C's finding).

## Minor issues
- **7.1** stale `band_edge_exit_verdict_v1.json` (`MODIFY_RESIDUAL_OR_DESIGN`, for ±0.03) **superseded, not
  overwritten** by `band_edge_exit_verdict_supersession_v1.{md,json}` (±0.035 validated by 306 + power).
- **7.2** all 8 probe allowlist fields **required**; missing → `TECHNICAL_INVALID_SCHEMA` (no silent
  zero-fill); guard test added.
- **7.3** generator must pass **all** hyperparameters explicitly; `DeepSets` constructor defaults
  (`max_epochs=300, patience=30`) differ from frozen `200/25`; documented, never relied on.
- **7.4** τ=0.0325 stays a **non-primary** sensitivity anchor (not upgraded; no full power rerun).

## Unchanged (fix1 touches none of these)
candidate bank `{-0.04,0,+0.04}`; test geometry ±0.035; DeepSets model + hyperparameters + 5 seeds; primary
comparator `B2_K1 − train/val-only best-single`; 0.15 CI threshold; 75 sessions / 300 trials; power result.

## Tests
- `test_preregistration_v4_fix1.py` — new, all PASS (four fixes + minors + consistency).
- `test_preregistration_v4.py` — B's own suite, updated to fix1 reality, all PASS.
- `test_preregistration_v4_audit_c.py` — **Claude C's audit file, unmodified**. Its 4 `strict-xfail`
  assertions now **XPASS(strict) → reported as failures**; that flip is the *intended, machine-checkable
  signal that the four blockers are fixed* (the xfail expectations were written against the OLD commit
  `9a02eb5`). This file is preserved as immutable audit evidence and is **not** part of B's post-fix passing
  suite (fix1 §6, option 2).

## Constraints honored
No Isaac; no generator; no manifest instance; no confirmatory data; no run authorization. 306 raw data,
runtime, `models_v2` model implementation, capability map, v1/v2/v3 preregistration, historical power
results, and Claude C's audit report/verdict are **untouched**. B does **not** authorize Claude A.
