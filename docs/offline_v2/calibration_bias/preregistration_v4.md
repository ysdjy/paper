# Confirmatory Preregistration v4 (offline; NO Isaac, NO generator, NO confirmatory data)

Claude B. Branch `experiment/offline-calibration-preregistration-v4` from
`2bf7217907f24ae54a08db71bbdcf624b110ccf0`. This document is the complete, auditable, unambiguous
preregistration of the confirmatory experiment. All frozen quantities are emitted from
`deployment_calibration/offline_v2/calibration_bias/preregistration_v4.py`; the machine forms are
`preregistration_v4.json`, `confirmatory_v4_config.json`, `confirmatory_v4_audit_checklist.json`. Markdown
and JSON are kept consistent (guarded by tests).

**Status: `PREREGISTRATION_V4_READY_FOR_INDEPENDENT_AUDIT`.** This phase authorizes nothing — Claude C must
return GO before Claude A implements the generator. No confirmatory generator, runtime code, manifest
instance, confirmatory data, or run authorization is produced here.

## 1. Scientific positioning (§2)
> This confirmatory experiment tests whether, **when the hidden deployment calibration state is stable
> across repeated tasks**, the history formed by **one fixed full-task-level diagnostic trial** lets a
> capacity-matched history-conditioned selector choose a more appropriate discrete action parameter on
> **unseen nominal deployment conditions** and raise subsequent task success.

- The probe is always named **"repeated-task deployment diagnostic full-task trial."** It is NOT one-shot
  online adaptation, a low-cost internal query, a non-destructive probe, or unseen-hidden-state generalization.
- Generalization is limited to: **unseen nominal deployment conditions, while the actual hidden-state
  support remains inside the training support.**
  ```
  test actual support  = [-0.045,-0.025] ∪ [+0.025,+0.045]
  train actual support = [-0.05, +0.05]
  ```

## 2. Frozen design (§5)
```
candidate bank            = {-0.04, 0.00, +0.04}
train nominal biases      = {-0.04,-0.02,0.00,+0.02,+0.04}
validation nominal biases = {-0.01,+0.01}
test nominal biases       = {-0.035,+0.035}
train/val/test blocks     = 9 / 6 / 9
residual ~ TruncatedNormal(0, σ=0.005 m, support [-0.01,+0.01] m)   (one draw per block; splits independent)
K=1 fixed diagnostic probe offset = -0.04
```
Forbidden: changing test bias, candidate bank, probe, or residual; running K=2; adding a second seed dataset.

## 3. Confirmatory hypotheses (§3)
**Primary H1** — `Δ_primary = seed-averaged B2_K1 selected success − train/val-only best-single success`,
evaluated on the 9 test blocks:
```
PRIMARY PASS  ⇔  95% test-block-bootstrap CI lower bound of Δ_primary ≥ 0.15   (threshold NOT lowered)
```
**Secondary H2** — `Δ_history = B2_K1 − capacity-matched B1_K0`; report point estimate + 95% block-bootstrap
CI; does **not** replace the primary comparator.
**Secondary H3** — report: B2_K1 vs state-aware oracle gap; probe-derived hidden-condition audit accuracy;
selected-action accuracy; model-seed stability; continuous outcomes; failure mechanism; Gross/Net VOI.
**Primary claim (fixed):** *diagnostic history improves subsequent action-selection success.* Net deployment
value is secondary.

## 4. Data structure & statistical unit (§4)
- **Independent statistical unit = nuisance/deployment block.** Counts: train 9, val 6, test 9. The three
  splits share no residual, seed, or nuisance.
- Within one block, ALL trials share: residual bias, initial joint perturbation, target jitter, mechanism,
  task target, controller parameters, deployment calibration state. The residual is **constant** across the
  block's nominal, probe, and candidate trials.
- **Sessions:** train 9×5=45, val 6×2=12, test 9×2=18 → **75 total** (independently recomputed in
  `preregistration_v4.py:trial_counts`).
- **Each session executes 1 separate diagnostic probe trial + 3 separate candidate trials** →
  **75×4 = 300 full-task trials** (frozen count).
- Probe and the candidate at `offset=-0.04` are **two different full-task trials**; the probe outcome is
  **never** reused as a candidate label. Task-level reset between trials; hidden deployment state and block
  nuisance held fixed; drawer/task dynamic state reset. Probe always precedes candidates; candidate order
  randomized by a frozen seed; session order randomized by a frozen master seed.

## 5. Outcome definitions (§6)
- **Primary binary success** = the frozen runtime `y.success` (`episode_schema_v2.py`); tolerance 0.02;
  reach/pull timeouts 16.0 s. Retained failure reasons (never deleted): `REACH_TIMEOUT` (APPROACH),
  `PULL_TIMEOUT`, `POSITION_TIMEOUT`, `HANDLE_DETACHED`. All **legitimate task failures are kept** as
  failures; only pre-listed technical faults (§9) are invalid.
- **Continuous secondary:** task_outcome_error, final_joint_position, skill_elapsed_time,
  true_handle_error_at_close_3d, true_handle_error_at_close_local_y, gripper_width_at_close, phase_durations.
- Full schema/history/feature detail in `confirmatory_v4_schema.md`.

## 6. Model freeze (§8)
`deployment_calibration.models_v2.DeepSets` (permutation-invariant set encoder). Frozen HP
`d_hid=32, d_emb=16, lr=1e-2, max_epochs=200, patience=25, l2=1e-4`; seeds `{1103,2207,3301,4409,5519}` —
**exactly** the power-verified values. Optimizer Adam(lr=1e-2, weight_decay=1e-4); loss = BCEWithLogits(success)
+ MSE(z-task_outcome_error) + MSE(z-skill_elapsed_time); **full-batch** (one step/epoch); standardized
static + probe features and z-normalized continuous targets (success not normalized); early stopping on the
validation multitask loss (patience 25, threshold 1e-5) with **best-val checkpoint restore**; `p_success =
sigmoid(logit)` clipped to [0,1]; candidate selection = argmax p_success with strict-greater scan over bank
order (tie → earliest bank index). No hyperparameter search on confirmatory data. Full list in
`confirmatory_v4_analysis_plan.md` / `MODEL` in the JSON.

## 7. Best-single (§9)
Train+validation **only** (test never participates). Rule: (1) max mean success; (2) tie → max mean frozen
continuous margin `τ−|eff|`; (3) tie → min |offset|; (4) tie → fixed numeric bank order. Save per-candidate
train/val score, tie-break trace, final offset, selection hash. Exploratory expectation is offset 0 but it
is **computed in confirmatory, not hardcoded**.

## 8. History allowlist / secret denylist (§7)
Model reads history ONLY via `features.probe_vector` — **8 fields**: `theta.grasp_offset_local_y`,
`theta.max_pos_step`, `theta.pull_lead`, `success`, `task_outcome_error`, `skill_elapsed_time`,
`pull_phase_duration`, `final_joint_position`. Candidate static inputs via `features.static_features` — 8
dims (theta×3, g.target_open_position, g.target_tolerance, x.initial_mechanism_joint_pos, x.gripper_width,
member==sektion_cabinet flag). **Denied** to the model: nominal/residual/actual bias, eff_signed, abs_eff,
nuisance_block_id, block_seed, residual_seed, split identity, future candidate outcomes, oracle action,
hidden-state class label, secret_deployment_state, hidden_state_id, damping. K0 vs K1 differ **only** by
empty history vs one frozen probe entry. A test asserts the allowlist equals the real code.

## 9. Collision & technical invalidation (§16, §17)
- **Collision (frozen from existing evidence):** clean smoke 0 N; intentional positive control ~240 N;
  explore-306 max unintended contact 0.0 N and 0 frames with the ContactSensor available for all 306
  (backend monitors **non-finger** links → intended finger–handle contact excluded). **Primary =
  intention-to-analyze** (include all schema-valid task trials, drop none for collision). Collision-confounded
  trial ≝ schema-valid trial with `max_unintended_contact_force_N > 0.0 N` on non-finger links (nonzero rule).
  **Sensitivity** re-runs the primary excluding pre-registered collision-confounded trials. Confirmatory data
  are **not** used to set this threshold.
- **Technical-invalid (frozen):** only runtime crash/EPISODE_EXCEPTION, ContactSensor unavailable, incomplete
  schema, manifest mismatch, controller not started, invalid initial state, file corruption. Legitimate
  failures (timeouts, HANDLE_DETACHED, approach/pull/offset failures) are **never** invalid. Policy:
  fail-fast then resume at the failed planned trial; retry the **same** planned trial without resampling
  residual/nuisance; no block replacement; **max 15** technical-invalid trials (>15 → experiment INVALID).
  Never rerun on a task outcome.

## 10. Test sealing (§10)
Executable order (full protocol in `confirmatory_v4_seal_unseal_protocol.md`): (1) build+validate train/val;
(2) keep test sealed (or generate only after models frozen); (3) select best-single from train/val only;
(4) train K0/K1 from train/val only; (5) freeze 5 seed-model hashes; (6) freeze analysis-code hash;
(7) unseal test; (8) one-shot final analysis. After unsealing: no retrain, no history/bank/threshold/seed/
bootstrap/exclusion changes.

## 11. Primary estimator & CI (§11, §12)
Per test session × model seed: B2_K1 predicts p_success for the 3 candidates → frozen argmax → selected
success from the already-executed candidate. **Per test block, average over the 2 test nominals, then over
the 5 model seeds** (best-single averaged over the 2 nominals in the block). `gain_b = seed-averaged B2
selected success − best-single success`; point estimate = `mean_b(gain_b)`. Candidate trials and model seeds
are **not** independent statistical units. CI: **2000 test-block percentile bootstrap**, resampling whole
test blocks (keeping both nominals, all 5 seeds, block candidate correspondence). Also report point estimate,
CI, each-seed gain, raw block gains.

## 12. Seed stability gate (§13)
`≥4/5 seed point gains ≥ 0.15` and `no seed point gain < 0`. This is a **PRIMARY necessary quality gate**
(GO requires it), consistent with the power simulation.

## 13. K0/K1/Oracle (§14)
K0 and K1 are the **same** DeepSets (same capacity, budget, seeds); only the history differs. Oracle reads the
secret actual bias — **audit upper bound only**, never in model input, training, best-single, or primary
selection. Report K1 vs best-single, K1 vs K0, K1 vs oracle, oracle vs best-single.

## 14. Probe cost & VOI (§15)
Probe = one independent full-task trial. `λ_time=0.02`, probe_time 16.58 s. Report Gross VOI_1task,
Net VOI_1task, NetVOI(T) for T∈{1,2,5,10}. **Secondary**; T is not chosen post-hoc as a primary endpoint.

## 15. Data isolation (§19)
306 exploration data, capability map, and synthetic power data do **not** enter confirmatory train/val/test
training. Test inaccessible until models frozen. Secret fields audit-only; feature cache excludes secret.
Split-isolation and leakage tests included.

## 16. Constructed positive-control boundary (§20)
The 3-point candidate bank is a **constructed positive-control skill library** frozen by the independent
exploration phase, to validate the history-conditioned action-selection mechanism. It is **not** a claim over
arbitrary continuous action spaces, arbitrary tasks, or long-horizon autonomous adaptation.

## 17. GO / FAIL / INVALID (§21)
```
PRIMARY PASS       : CI_lower(Δ_primary) ≥ 0.15  AND seed gate (≥4/5 ≥0.15, none <0)
PRIMARY FAIL       : any primary gate unmet
EXPERIMENT INVALID : only the pre-registered integrity/leakage/manifest/technical-failure conditions
```
A non-ideal result may **not** be rewritten as an exploratory PASS.

## 18. Hard constraints
No Isaac; no confirmatory generator; no runtime code; no confirmatory manifest instance; no confirmatory
data; no run authorization. Writes only under the three allowed calibration_bias dirs. 306 raw data, runtime,
capability map, and all prior preregistration/analysis/audit artifacts are untouched.
