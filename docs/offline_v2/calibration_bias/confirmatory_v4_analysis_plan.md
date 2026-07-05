# Confirmatory v4 — analysis plan (frozen)

Claude B. The complete, pre-registered analysis. Nothing here is data-dependent; every choice is fixed
before unsealing test. Machine form: `PRIMARY`/`SECONDARY`/`BOOTSTRAP`/`SEED_GATE` in
`preregistration_v4.json`. Estimator matches the power-verified simulation.

## 1. Model (frozen, no confirmatory hyperparameter search)
`deployment_calibration.models_v2.DeepSets`:
| aspect | frozen choice |
|---|---|
| hyperparameters | d_hid=32, d_emb=16, lr=1e-2, max_epochs=200, patience=25, l2=1e-4 |
| model seeds | {1103, 2207, 3301, 4409, 5519} |
| optimizer | Adam(lr=1e-2, weight_decay=1e-4) |
| loss | BCEWithLogits(success) + MSE(z-task_outcome_error) + MSE(z-skill_elapsed_time) |
| batching | full-batch (one Adam step per epoch over the whole train set) |
| feature normalization | standardize static (train mu/sd); standardize probe vectors (train pooled mu/sd); z-normalize the 2 continuous targets (train mu/sd); success not normalized; eps 1e-8 |
| class weighting | none (plain BCE) |
| initialization | default PyTorch init under torch.manual_seed(seed) + np.random.seed(seed) |
| determinism | per-seed; CPU full-batch → deterministic rerun (test-verified) |
| validation metric | the multitask training loss on the validation fold |
| early stopping | patience 25, improvement threshold 1e-5 |
| checkpoint selection | restore state_dict at minimum validation loss |
| p_success | sigmoid(logit), clipped to [0,1] (clip_predictions) |
| candidate argmax | max p_success over the 3-candidate bank |
| tie-break | strict-greater scan over bank order (−0.04, 0.0, +0.04) → earliest bank index on a tie |
| regression clip | error_clip [0.0,0.30], time_clip [0.0,60.0] |
| training failure | **FIX1:** an analysis-stage event in the `model_fit_failure_count` namespace (§9); NOT a runtime trial, NOT part of the 300-trial invalid budget |
| explicit hyperparameters | **FIX1:** generator/training must pass all HP explicitly; constructor defaults `max_epochs=300, patience=30` differ from frozen `200/25` and must not be relied on |

## 2. Best-single (train + validation only) — FIX3: single guarded production entry
Production entry: `confirmatory_v4_selection.select_best_single_confirmatory(observed_candidate_records)` —
**one** business parameter, no override kwargs, `__all__` frozen to the three public names; the generic
override-capable logic is **private** (`_select_best_single_core`, bridge/tests only). Reads **only**
observed candidate outcomes (`split, session_id, trial_role, theta.grasp_offset_local_y, y.success,
planned_episode_id`); **never** tau/nominal/residual/actual/eff/oracle/test/success-model. Rule (equal
denominators, 57/offset): (1) max integer **observed_success_count** → (2) tie: min |offset| → (3) tie:
earliest in bank order. **Completeness gate** (else `EXPERIMENT_INVALID_BEST_SINGLE_INPUT`; no warning, no
drop, no denominator change): exactly 171 records, **train 45 / validation 12 sessions**, **135 / 36
records**, 57 globally-unique sessions, `session_id` globally unique, 3/session, offset set `{-0.04,0,+0.04}`,
no dup `(split,session_id,offset)`, unique `planned_episode_id`, split ∈ {train,validation}, `y.success`
strict bool, no probe/test, bank not overridable. Saves the split counts, per-candidate score, tie-break
trace, final offset, `n_trials_per_offset`, `success_count_per_offset`, hashes.
`learned_selector_power.best_single_legal` is a `SIMULATION_ONLY_REFERENCE`, **not** production. **Test never
participates.** Offset 0 is not hardcoded. Production == simulation on 4500/4500 frozen configs
(`production_best_single_bridge_invariance_v1`) → **no power recertification**.

### 2b. Deterministic model environment (FIX2, frozen)
`device=CPU`; `torch.set_num_threads(1)`; `torch.set_num_interop_threads(1)`;
`torch.use_deterministic_algorithms(True)`; env `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
All hyperparameters passed explicitly (no constructor defaults). If a required deterministic op is
unavailable → model-fit INVALID. Switching to GPU voids the protocol. Library/environment versions are
recorded in `model_analysis_freeze.json`.

## 3. K0 / K1 / Oracle
- **K1 (B2)** = DeepSets with one frozen probe-history entry. **K0 (B1)** = the *same* DeepSets with empty
  history. Same capacity, budget, seeds; only history differs → a clean capacity-matched history ablation.
- **Oracle** reads the secret actual bias; **audit upper bound only** — never in model input, training,
  best-single, or primary selection.

## 4. Primary estimator (statistical unit = test nuisance block)
For each test session and each of the 5 model seeds:
1. B2_K1 predicts p_success for the 3 candidates;
2. select the offset by the frozen argmax/tie-break;
3. read that candidate's already-executed outcome → selected success.

Then, **per test block**:
```
B2_block      = mean over {2 test nominals} of ( mean over {5 seeds} of selected success )
best_single_b = mean over {2 test nominals} of best-single success
gain_b        = B2_block − best_single_b
Δ_primary point estimate = mean over the 9 test blocks of gain_b
```
Candidate trials and model seeds are **not** independent statistical units.

## 5. Confidence interval & primary decision
- **2000 test-block percentile bootstrap**, 95% CI. Each bootstrap resamples whole test blocks (with
  replacement), preserving both test nominals, all 5 seeds, and the block's candidate correspondence.
- **PRIMARY PASS ⇔ `CI_lower(Δ_primary) ≥ 0.15`.** Not point estimate, not CI>0, not episode-level bootstrap.
  The 0.15 threshold is **not** lowered and the comparator is **not** changed.
- Always report: point estimate, 95% CI, per-seed gain, raw block gains.

## 6. Seed stability gate (PRIMARY necessary)
**Precondition (FIX1):** all 5 model seeds exist and are technically valid (§9); a missing/invalid seed makes
the experiment INVALID and is **not** treated as a `gain<0` seed nor dropped from the denominator. Given
that, the gate is `≥ 4/5 seed point gains ≥ 0.15` **and** `no seed point gain < 0`. GO requires this in
addition to the CI event (consistent with the power simulation, where seed-stability ≈ 0.99 at this design).

## 7. Secondary analyses (never replace primary)
- **H2 history:** `Δ_history = B2_K1 − B1_K0`; point estimate + 95% block-bootstrap CI.
- **H3:** B2_K1 vs oracle gap; probe-derived hidden-condition audit accuracy; selected-action accuracy;
  model-seed stability; continuous outcomes; failure mechanism; Gross/Net VOI.
- **VOI:** `λ_time=0.02`, probe_time 16.58 s → Gross VOI_1task, Net VOI_1task, NetVOI(T) for T∈{1,2,5,10}.
  Secondary; T is not chosen post-hoc as a primary endpoint.

## 8. Collision handling in analysis
- **Primary = intention-to-analyze:** every schema-valid task trial counts; no trial dropped for collision.
- **Collision-confounded** trial ≝ schema-valid trial with `max_unintended_contact_force_N > 0.0 N` on
  monitored non-finger links (intended finger–handle contact excluded).
- **Sensitivity:** re-compute the primary contrast excluding pre-registered collision-confounded trials;
  report whether `CI_lower ≥ 0.15` is unchanged. Confirmatory data are **not** used to set the threshold.

## 9. Technical invalidation & model-fit failure (FIX1: two separate namespaces)
**Runtime trials (`runtime_trial_invalid_count`):** only the pre-listed technical faults invalidate a trial
(runtime crash/EPISODE_EXCEPTION, ContactSensor unavailable, incomplete schema incl. a missing probe
allowlist field, manifest mismatch, controller not started, invalid initial state, file corruption).
Legitimate failures are never invalid. Fail-fast then resume at the failed planned trial; retry the same
planned trial without resampling residual/nuisance; no block replacement; **>15 → experiment INVALID**.
Never rerun on a task outcome.

**Model-fit failure (`model_fit_failure_count`) — BLOCKER D:** a model-training failure is an
**analysis-stage** event, **NOT** one of the 300 runtime trials and **NOT** subject to the >15 rule. Each of
the 5 seeds (for both K0 and K1 → 10 checkpoints) must be **valid**: training terminates without exception,
no NaN/Inf, a `best_state` is produced, train & val loss finite, checkpoint reloadable, reloaded inference
finite, state_dict hash saved, schema/leakage checks pass. Reaching `max_epochs` without early-stop is
**valid**. Retry = initial attempt + **at most 1** deterministic retry, only for an infrastructural
exception / abnormal process exit / reproducible I/O failure, with identical data hash, seed, HP, commit,
device/determinism config (first-failure log kept; init/budget unchanged). **No seed replacement, no
dropping a bad seed, no missing seed removed from the denominator.** If any seed is still invalid after the
retry → **`EXPERIMENT_INVALID_MODEL_FIT`** (do not generate the test manifest, do not unseal test). The seed
gate (§6) requires **all 5 valid first**, so `≥4/5` can never mask a missing seed.

## 10. GO / FAIL / INVALID
```
PRIMARY PASS       : CI_lower(Δ_primary) ≥ 0.15  AND seed gate
PRIMARY FAIL       : any primary gate unmet
EXPERIMENT INVALID : only pre-registered integrity/leakage/manifest/technical-failure conditions
```
No exploratory rewrite of a non-ideal confirmatory result.
