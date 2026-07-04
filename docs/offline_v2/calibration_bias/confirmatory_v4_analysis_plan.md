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
| training failure | a training exception is TECHNICAL-INVALID (§17); scored conservatively as failure in power sim |

## 2. Best-single (train + validation only)
Rule (frozen): (1) max mean success → (2) tie: max mean frozen continuous margin `τ − |eff|` → (3) tie:
min |offset| → (4) tie: fixed numeric bank order. Save per-candidate train/val score, tie-break trace, final
offset, selection hash. **Test never participates.** Exploratory expectation offset 0, but **computed in
confirmatory, not hardcoded**.

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
`≥ 4/5 seed point gains ≥ 0.15` **and** `no seed point gain < 0`. GO requires this in addition to the CI
event (consistent with the power simulation, where seed-stability ≈ 0.99 at this design).

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

## 9. Technical invalidation
Only the pre-listed technical faults invalidate a trial (runtime crash/EPISODE_EXCEPTION, ContactSensor
unavailable, incomplete schema, manifest mismatch, controller not started, invalid initial state, file
corruption). Legitimate failures are never invalid. Fail-fast then resume at the failed planned trial;
retry the same planned trial without resampling residual/nuisance; no block replacement; **>15 technical-
invalid trials → experiment INVALID**. Never rerun on a task outcome.

## 10. GO / FAIL / INVALID
```
PRIMARY PASS       : CI_lower(Δ_primary) ≥ 0.15  AND seed gate
PRIMARY FAIL       : any primary gate unmet
EXPERIMENT INVALID : only pre-registered integrity/leakage/manifest/technical-failure conditions
```
No exploratory rewrite of a non-ideal confirmatory result.
