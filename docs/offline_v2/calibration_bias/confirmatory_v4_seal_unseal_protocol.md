# Confirmatory v4 — seal / unseal protocol (frozen, executable) — FIX1: ONE unique scheme

Claude B. The single, unambiguous procedure that guarantees the test split cannot influence any modelling or
selection choice. Machine form: `SEAL_SCHEME` / `SEAL_UNSEAL` / `ACCESS_CONTROL` / `POST_UNSEAL_FORBIDDEN`
in `preregistration_v4.json`.

## 1. The unique scheme (BLOCKER C)
`SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION`. **All either/or wording is deleted**; the previous "test may
be generated early and sealed" alternative is removed. The test manifest is generated **only after** every
model and the analysis code are frozen and hashed — so test outcomes do not even exist until Step 6.

## 2. Executable steps (single path)
```
STEP 0  PRE-RUN FREEZE
        freeze + hash: the fix1 v4 commit, confirmatory config, all 15 exact seeds, the manifest-generation
        algorithm, the schema, the generator commit (future, post-GO), the train/val/test PLANNED structure,
        and the test split identities + nominal sets. No test outcomes exist.

STEP 1  GENERATE TRAIN/VAL MANIFEST  (compute train_validation_manifest_sha256; 228 trials)
        deterministically from the frozen config + seeds; save its hash. No resample.

STEP 2  RUN TRAIN/VAL TRIALS
        generate ONLY train (45 sessions) + validation (12 sessions) runtime data.

STEP 3  FREEZE TRAIN/VAL DATA
        integrity validation, leakage validation, dataset hashes, train/val lock file.

STEP 4  SELECT BEST-SINGLE + TRAIN MODELS   (train + validation only)
        apply the PRODUCTION selector confirmatory_v4_selection.select_best_single_confirmatory (single entry, observed y.success only,
        no secret; input-completeness gate -> EXPERIMENT_INVALID_BEST_SINGLE_INPUT); train DeepSets K0 and K1
        for each of the 5 model seeds (10 checkpoints) under the frozen deterministic env (CPU, 1 thread,
        use_deterministic_algorithms, OMP/MKL/OPENBLAS=1, explicit HP); ALL 10 must be valid (analysis-plan
        §9); save state_dict hashes.

STEP 5  FREEZE ANALYSIS  (before ANY test trial)
        freeze: 5 K0 hashes, 5 K1 hashes, best-single artifact/hash, feature-extraction code hash,
        final-analysis code hash, bootstrap seed, collision sensitivity rule, exclusion/invalidation rules
        -> write model_analysis_freeze.json.

STEP 6  GENERATE TEST MANIFEST  (compute test_manifest_sha256; 72 trials; then combined_experiment_plan_sha256)   (ONLY after Step 5 passes)
        one-shot, deterministic, from the frozen test seeds + manifest algorithm + generator commit + config;
        save the test manifest hash. No reselecting seeds, no generate-many-and-pick, no change to test
        geometry / block count / nominal / order / residual rules.

STEP 7  RUN TEST TRIALS   per the test manifest (18 sessions).

STEP 8  ONE-SHOT FINAL ANALYSIS
        run the frozen estimator + 2000 test-block bootstrap + seed gate exactly once; emit the verdict.
```

## 3. Access control & early-test access (BLOCKER C)
- Test outcomes are **non-existent before Step 6**; the train/val feature cache excludes test fields; all
  test access is written to an audit log.
- Generating or reading **any** test outcome **before Step 5 completes** →
  **`EXPERIMENT_INVALID_EARLY_TEST_ACCESS`** (whole experiment invalid).

## 4. Forbidden after Step 6 / unseal
`retrain`, `change history`, `change bank`, `change threshold`, `change seeds`, `change bootstrap`,
`change exclusion rules`, `reselect test seed`, `generate multiple test manifests and pick`,
`change test geometry/block count`. Any such action voids the confirmatory status.

## 5. Enforcement hooks (for the generator/analysis, implemented by A after GO)
- **Leakage assertion**: the modelling code (Steps 3–5) has no read path to test outcomes or any secret field
  (matches `SECRET_DENYLIST`).
- **Hash pinning**: Step 8 refuses to run unless the 10 model hashes and the analysis-code hash equal the
  values frozen at Step 5.
- **One-shot guard**: Step 8 records that the final analysis has executed; a second execution is refused.
- **Step-order guard**: Step 6 refuses unless `model_analysis_freeze.json` from Step 5 exists and validates.

## 6. Recovery (technical faults only)
A runtime technical-invalid fault (analysis-plan §9) resumes from the frozen manifest at the failed planned
trial (no residual/nuisance resampling, no block replacement) and does **not** permit revisiting Step 3–5
choices; >15 runtime technical-invalid trials → experiment INVALID. A **model-fit** failure is handled by the
separate `model_fit_failure_count` path (retry ≤ 1; else `EXPERIMENT_INVALID_MODEL_FIT`), and blocks Step 6.

## 7. Not produced in this phase
Protocol specification only. No sealing code, no models, no test manifest, no test data, and no execution are
produced here; A implements the hooks only after C returns GO.
