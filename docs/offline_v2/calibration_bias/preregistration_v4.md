# Confirmatory Preregistration v4 — FIX1 (offline; NO Isaac, NO generator, NO confirmatory data)

Claude B. Branch `experiment/offline-calibration-preregistration-v4-fix1` from Claude C's audit commit
`2ab39063c42f545b35beb53015c6a16402660c37` (design frozen at power-cert `2bf7217`). This document is the
complete, auditable, unambiguous preregistration of the confirmatory experiment, **revised to resolve the
four blockers** from Claude C's independent audit (`MODIFY_PREREGISTRATION_V4`). All frozen quantities are
emitted from `deployment_calibration/offline_v2/calibration_bias/preregistration_v4.py`; the machine forms
are `preregistration_v4.json`, `confirmatory_v4_config.json`, `confirmatory_v4_audit_checklist.json`.
Markdown and JSON are kept consistent (guarded by tests).

**Status: `PREREGISTRATION_V4_FIX1_READY_FOR_C_REAUDIT`.** This phase authorizes nothing — Claude C must
return GO before Claude A implements the generator. No confirmatory generator, runtime code, manifest
instance, confirmatory data, or run authorization is produced here.

## 0. Fix1 changes (resolving Claude C's four blockers)
- **A `BLOCKER_SECRET_TIEBREAK`** — best-single tie-break no longer reads the secret; the illegal
  `τ−|eff|` step is removed (§7). Proven selection-invariant on all 4500 frozen replicates
  (`best_single_tiebreak_invariance_v1`), so **no power recertification** is required.
- **B `BLOCKER_SEEDS_NOT_FROZEN`** — all 15 randomization seeds are now exact integers from a frozen SHA256
  rule; run-time seed choice is forbidden (§Manifest).
- **C `BLOCKER_SEAL_SCHEME_NOT_UNIQUE`** — a single seal scheme (`SCHEME_2_MODEL_FREEZE_BEFORE_TEST_
  GENERATION`) is frozen; the either/or is deleted (§10).
- **D `BLOCKER_MODEL_FAILURE_SEMANTICS`** — model-fit failure is an analysis-stage event in its own
  namespace, separate from the 300-trial runtime-invalid budget; all 5 seeds must be valid (§6, §9).
- Minor: probe 8 allowlist fields are **required** (no silent zero-fill); the generator must pass all
  hyperparameters explicitly (constructor defaults differ); the stale band-edge exit verdict is superseded,
  not overwritten; τ=0.0325 stays a non-primary sensitivity anchor.
- `power_recertification_required = false` (bank, geometry, model, estimator, effect all unchanged).

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
order (tie → earliest bank index). No hyperparameter search on confirmatory data. **FIX1:** the
generator/training must pass **all** frozen hyperparameters explicitly — the `DeepSets` constructor defaults
(`max_epochs=300, patience=30`) differ from the frozen `200/25` and must never be relied on. **Model-fit
validity & failure semantics** (all 5 seeds must be valid, retry ≤ 1, separate `model_fit_failure_count`
namespace, `EXPERIMENT_INVALID_MODEL_FIT`) are in the analysis plan §1/§9. Full list in
`confirmatory_v4_analysis_plan.md` / `MODEL` in the JSON.

## 7. Best-single (§9) — FIX1: no secret tie-break
Train+validation **only** (test never participates). **Legal rule (fix1):** (1) maximize observed mean
binary success over train+val candidate trials; (2) tie → min |offset|; (3) tie → first in the frozen bank
order `{-0.04, 0.00, +0.04}`. **No** `τ−|eff|` / secret continuous margin, no nominal/residual/actual bias,
no eff, no oracle, no test outcomes; no new observable continuous tie-break is introduced. Implementation:
`learned_selector_power.best_single_legal`. Save per-candidate train/val score, tie-break trace, final
offset, selection hash. Exploratory expectation is offset 0 but it is **computed in confirmatory, not
hardcoded**. The removed secret tie-break was **dead code** on every frozen config (4500/4500 replicates,
0 step-1 ties, 0 old↔new mismatches → `best_single_tiebreak_invariance_v1`), so **no power recertification**.

## 8. History allowlist / secret denylist (§7)
Model reads history ONLY via `features.probe_vector` — **8 fields**: `theta.grasp_offset_local_y`,
`theta.max_pos_step`, `theta.pull_lead`, `success`, `task_outcome_error`, `skill_elapsed_time`,
`pull_phase_duration`, `final_joint_position`. Candidate static inputs via `features.static_features` — 8
dims (theta×3, g.target_open_position, g.target_tolerance, x.initial_mechanism_joint_pos, x.gripper_width,
member==sektion_cabinet flag). **Denied** to the model: nominal/residual/actual bias, eff_signed, abs_eff,
nuisance_block_id, block_seed, residual_seed, split identity, future candidate outcomes, oracle action,
hidden-state class label, secret_deployment_state, hidden_state_id, damping. K0 vs K1 differ **only** by
empty history vs one frozen probe entry. A test asserts the allowlist equals the real code.

## 8b. Frozen seeds (§Manifest) — FIX1: no run-time seed choice
All randomization is fixed **now**, not when the generator runs. Frozen derivation:
```
SEED_ROOT = "confirmatory-v4|2bf7217907f24ae54a08db71bbdcf624b110ccf0"
seed(label)    = int(sha256(f"{SEED_ROOT}|{label}").hexdigest()[:16], 16) % (2**63-1)
subseed(ds,id) = int(sha256(f"{ds}|{id}").hexdigest()[:16], 16) % (2**63-1)
```
15 exact integers are stored in `confirmatory_v4_config.json → frozen_seeds` (master_seed
6341914557047805261, bootstrap_seed 9014517173581927929, plus the residual/block-order/session-order/
nominal-order/candidate-order seeds for each split). Identity-addressed `subseed` fixes per-block residual,
per-block nuisance, per-session candidate order, and per-trial init so results do not depend on Python
iteration order, resume is bit-identical, and no manifest can be regenerated-and-picked. `bootstrap_seed`
is frozen **before** unseal. Full spec in `confirmatory_v4_manifest_spec.md`.

## 9. Collision & technical invalidation (§16, §17)
- **Collision (frozen from existing evidence):** clean smoke 0 N; intentional positive control ~240 N;
  explore-306 max unintended contact 0.0 N and 0 frames with the ContactSensor available for all 306
  (backend monitors **non-finger** links → intended finger–handle contact excluded). **Primary =
  intention-to-analyze** (include all schema-valid task trials, drop none for collision). Collision-confounded
  trial ≝ schema-valid trial with `max_unintended_contact_force_N > 0.0 N` on non-finger links (nonzero rule).
  **Sensitivity** re-runs the primary excluding pre-registered collision-confounded trials. Confirmatory data
  are **not** used to set this threshold.
- **Technical-invalid (frozen, RUNTIME trials only):** only runtime crash/EPISODE_EXCEPTION, ContactSensor
  unavailable, incomplete schema (incl. any missing probe allowlist field), manifest mismatch, controller
  not started, invalid initial state, file corruption. Legitimate failures (timeouts, HANDLE_DETACHED,
  approach/pull/offset failures) are **never** invalid. Policy: fail-fast then resume at the failed planned
  trial; retry the **same** planned trial without resampling residual/nuisance; no block replacement;
  **max 15** `runtime_trial_invalid_count` (>15 → experiment INVALID). Never rerun on a task outcome.
  **FIX1 — two separate namespaces:** `runtime_trial_invalid_count` (the 300 runtime trials; subject to the
  >15 rule) vs `model_fit_failure_count` (model-training failures; governed by §6/§9 of the analysis plan;
  **not** part of the 300-trial budget). A model-fit failure never consumes the runtime-invalid budget.

## 10. Test sealing (§10) — FIX1: one unique scheme
Exactly one scheme, `SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION` (no either/or). Full protocol in
`confirmatory_v4_seal_unseal_protocol.md`: **Step 0** pre-run freeze+hash (commit, config, all seeds,
manifest algorithm, schema, generator commit, planned structure, test identities) — test outcomes do not
exist; **Step 1** generate train/val manifest; **Step 2** run train/val; **Step 3** freeze train/val data
(integrity+leakage+hashes+lock); **Step 4** select best-single + train K0/K1 (5 seeds each, all 10 valid);
**Step 5** freeze analysis (10 model hashes, best-single artifact, feature+analysis code hashes, bootstrap
seed, exclusion/invalidation rules) → `model_analysis_freeze.json`; **Step 6** generate the test manifest
**once, deterministically**, only after Step 5; **Step 7** run test; **Step 8** one-shot final analysis.
Generating or reading any test outcome before Step 5 → **`EXPERIMENT_INVALID_EARLY_TEST_ACCESS`**. After
Step 6/unseal: no retrain, no history/bank/threshold/seed/bootstrap/exclusion change, no test-seed reselect,
no generate-many-and-pick.

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

## 17. GO / FAIL / INVALID (§21) — FIX1
```
PRIMARY PASS       : CI_lower(Δ_primary) ≥ 0.15  AND seed gate (all 5 seeds valid, ≥4/5 ≥0.15, none <0)
PRIMARY FAIL       : any primary gate unmet
EXPERIMENT INVALID : pre-registered integrity/leakage/manifest/RUNTIME-technical conditions,
                     OR EXPERIMENT_INVALID_MODEL_FIT (a model seed invalid after ≤1 retry),
                     OR EXPERIMENT_INVALID_EARLY_TEST_ACCESS (test outcome touched before Step 5)
```
The seed gate's `≥4/5` can **never** hide a missing/invalid seed: all 5 must first be valid, else the
experiment is INVALID (not a `gain<0` seed, not dropped from the denominator). A non-ideal result may **not**
be rewritten as an exploratory PASS.

## 18. Hard constraints
No Isaac; no confirmatory generator; no runtime code; no confirmatory manifest instance; no confirmatory
data; no run authorization. Writes only under the three allowed calibration_bias dirs. 306 raw data, runtime,
capability map, and all prior preregistration/analysis/audit artifacts are untouched.
