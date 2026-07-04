# Preregistration v4 FIX1 — independent re-audit (Claude C, second pass)

**Auditor:** Claude C, independent, **read-only, adversarial**. No Isaac, no generator, no manifest
instance, no confirmatory data, no run authorization. Three files added only (this report, its JSON, and
`test_preregistration_v4_fix1_reaudit_c.py`). B's fix1 files, `models_v2`, historical data/power results,
and my first-audit report are untouched.

**Audited:** `91a41e51282f716595bc3553ba500e6e2af95f40` (fix1) vs original v4 `9a02eb5`; first audit
`2ab3906`; power-cert `2bf7217`.

## VERDICT: **MODIFY_PREREGISTRATION_V4_FIX1**

Three of the four original blockers are **fully resolved and independently verified**. The fourth
(secret tie-break) is resolved **at the rule level** but leaves a **new, concrete production-binding
blocker** that the first audit anticipated, plus a determinism-completeness gap in the seed identity
format. Neither open item requires power re-certification.

`authorizes_generator_implementation = false` (GO withheld pending the two open items).

---

## The four original blockers

| # | blocker | status | independent evidence |
|---|---|---|---|
| **A** | `BLOCKER_SECRET_TIEBREAK` | **RESOLVED (rule)** — but see new blocker | rule = max observed success → min\|offset\| → bank order; `τ−\|eff\|` removed; forbidden_inputs complete. **4500/4500** replicates recomputed: 0 step-1 ties, 0 old↔new mismatch, all offset 0 → selection-invariant, no power recert. |
| **B** | `BLOCKER_SEEDS_NOT_FROZEN` | **RESOLVED** | all **15** `frozen_seeds` recompute bit-exactly from the SHA256 rule; `master_seed=6341914557047805261`, `bootstrap_seed=9014517173581927929` match the reference; no collisions; MD==JSON==code; `bootstrap_seed` frozen pre-unseal. |
| **C** | `BLOCKER_SEAL_SCHEME_NOT_UNIQUE` | **RESOLVED** | single `SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION`, Steps 0–8, no either/or; test manifest generated **only after** Step 5 model+analysis freeze; step-order guard on `model_analysis_freeze.json`; early access → `EXPERIMENT_INVALID_EARLY_TEST_ACCESS`; `post_unseal_forbidden` bans reselecting the test seed / generate-many-and-pick. |
| **D** | `BLOCKER_MODEL_FAILURE_SEMANTICS` | **RESOLVED** | `model_fit_failure_count` is a **separate** namespace from `runtime_trial_invalid_count` (not subject to >15); all 5 seeds must be valid for K0 **and** K1 (10 checkpoints, hashes frozen); **no seed replacement/drop**; missing/invalid seed → `EXPERIMENT_INVALID_MODEL_FIT` (no test manifest); retry = initial + **≤1 deterministic, infrastructural-only**; 9 validity conditions; max_epochs≠failure. The ≥4/5 gain gate can no longer mask a missing seed. |

---

## New blockers (must fix before generator implementation; no power recert)

### 1. `BLOCKER_PRODUCTION_BEST_SINGLE_STILL_SECRET_DEPENDENT`
The best-single **rule** is now legal, but `confirmatory_v4_config.json` `best_single.implementation`
points at `learned_selector_power.best_single_legal`, whose body is
`succ(s["nominal"], s["residual"], o, tau)` — it **reconstructs the success label from the secret hidden
state** (`nominal + residual`) and the band parameter `tau`, and never reads an observed `y.success`.
Its docstring calls `succ(...)` “the OBSERVED binary outcome”, but the synthetic session has **no
`y.success` field** — this is a mislabel. **No observed-outcome-only production interface is frozen.**

- **Why it matters:** if A uses/copies `best_single_legal` for the confirmatory best-single, the
  state-agnostic baseline would read secret fields and substitute the fitted `|eff|≤τ` band label for the
  real trial outcome — leakage **and** a correctness error. The preregistration must freeze the *production
  contract*, not a simulation helper.
- **Minimal fix:** freeze `select_best_single(observed_train_val_candidate_records)` reading **only**
  candidate `grasp_offset_local_y`, observed `y.success`, and `split∈{train,validation}`; mean observed
  success per offset → max → min|offset| → bank order; **never** `nominal/residual/actual/eff/tau/secret`.
  Relabel `best_single_legal` explicitly **simulation-only** and stop designating it the implementation.
  Add tests: (a) perturbing secret fields does not change the selection; (b) perturbing observed
  `y.success` changes it per the rule.
- **Power recert:** **not needed** — a correct observed-success selector picks the same offset 0 (offset 0
  has the highest train+val observed success; 4500/4500 invariance). Contract/plumbing fix only.

### 2. `BLOCKER_PLANNED_IDENTITY_FORMAT_NOT_FROZEN`
The `subseed` **function** is frozen, but the canonical `planned_identity` **string format** is only
exemplified: the residual draw uses `subseed(<split>_residual_seed, "block=<i>")` while the examples show
`"block=03"`/`"block=07"` (2-digit zero-pad). `"block=3"` vs `"block=03"` yield **different** subseeds; the
`"trial=<planned_episode_id>"` identity depends on a `planned_episode_id` template that is not frozen.

- **Why it matters:** it leaves A discretion over the identity strings, which changes the actual per-block
  residual/order draws — undermining Blocker B's "one deterministic manifest, no cherry-pick" and Blocker
  C's Step-6 determinism. The manifest is **not uniquely determined by the frozen config+seeds alone.**
- **Minimal fix:** freeze the exact `planned_identity` template (fields, order, separators, integer
  zero-padding) and the `planned_episode_id` format; add a test recomputing a couple of block-residual
  subseeds from the frozen template.
- **Power recert:** not needed.

---

## Minor issues
- **Determinism env flags not fully frozen.** `model.determinism` relies on "CPU full-batch → deterministic"
  and records `environment_versions`, but does not pin `torch.use_deterministic_algorithms(True)`, thread
  count, or CPU device assertion. Robust for this tiny model, but freeze these in `model_analysis_freeze
  .json` for completeness.
- **7.1–7.4 resolved (confirmed):** 7.1 stale band-edge verdict **superseded, not overwritten** (original
  `band_edge_exit_verdict_v1.json` still `MODIFY_RESIDUAL_OR_DESIGN`; separate supersession note, no
  pretend-new-experiment); 7.2 all 8 probe fields **required** → `TECHNICAL_INVALID_SCHEMA` (no silent
  zero-fill); 7.3 HP passed explicitly, `DeepSets` defaults documented as not relied on; 7.4 τ=0.0325
  stays a **non-primary** sensitivity anchor.

## Power evidence still valid
Fix1 changes **none** of: candidate bank `{−0.04,0,+0.04}`, geometry ±0.035, residual, probe, model/HP/5
seeds, primary comparator, bootstrap, 0.15 threshold, effect. Selection is invariant (4500/4500). →
**`power_recertification_required = false`.** `POWER_SUFFICIENT_FOR_PREREG_V4` stands.

## Documentation & consistency
Status is uniformly `PREREGISTRATION_V4_FIX1_READY_FOR_C_REAUDIT`. A stale-semantics sweep
(`τ−|eff|`, `either`, `or generate`, `fix master_seed`, `technical-invalid event`, `4/5 may continue`)
finds hits **only** in historical/pre-power docs (v1/v3 proposals, learned-selector-power-plan), never in
the active v4 confirmatory spec.

## Tests
- **B fix1 suite** `test_preregistration_v4_fix1.py` → **27 passed**.
- **C re-audit suite** `test_preregistration_v4_fix1_reaudit_c.py` → **11 passed, 2 xfailed** (the two new
  blockers).
- **C first-audit file** `test_preregistration_v4_audit_c.py` is **unmodified**; its 4 `strict-xfail`
  assertions now XPASS(strict)→fail — the intended machine-checkable signal that the four **original**
  blockers were fixed against the old commit `9a02eb5`. Preserved as immutable audit history.

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_manifest_instance_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

After B (1) freezes an observed-outcome-only production best-single interface and relabels the sim helper,
and (2) freezes the canonical `planned_identity` format, a **short** Claude C re-audit is required before
GO. Because neither fix touches the estimator, model, geometry, or effect, the existing ±0.035 power
certification will carry over unchanged.
