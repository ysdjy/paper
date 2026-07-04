# Confirmatory v4 — seal / unseal protocol (frozen, executable)

Claude B. The executable procedure that guarantees the test split cannot influence any modelling or
selection choice. Machine form: `SEAL_UNSEAL` / `POST_UNSEAL_FORBIDDEN` in `preregistration_v4.json`.

## 1. Principle
best-single selection and K0/K1 training use **train + validation only**. Test outcomes are revealed
**once**, after every model and the analysis code are frozen and hashed. The primary contrast is then
computed a single time.

## 2. Executable steps
```
STEP 1  BUILD_TRAINVAL
        generate + schema-validate train (45 sessions) and validation (12 sessions).
        Assert: no test session materialized yet (or test kept in a sealed store).

STEP 2  SEAL_TEST
        test (18 sessions) is either (a) not yet generated, or (b) generated into a sealed store whose
        outcomes are inaccessible to the modelling code (enforced by a leakage test).

STEP 3  SELECT_BEST_SINGLE   (train + validation only)
        apply the frozen best-single rule; write per-candidate scores, tie-break trace, final offset,
        selection hash.

STEP 4  TRAIN_MODELS         (train + validation only)
        train DeepSets K0 and K1 for each of the 5 model seeds; early stop on validation loss.

STEP 5  FREEZE_MODEL_HASHES
        compute + record a state_dict hash for each of the 5×{K0,K1} models. Immutable hereafter.

STEP 6  FREEZE_ANALYSIS_HASH
        compute + record the sha256 of the analysis code (estimator + bootstrap + gates).

STEP 7  UNSEAL_TEST
        reveal test candidate + probe outcomes. Record unseal timestamp + operator.

STEP 8  FINAL_ANALYSIS       (one shot)
        run the frozen estimator + 2000 test-block bootstrap + seed gate exactly once; emit the verdict.
```

## 3. Forbidden after UNSEAL (STEP 7)
`retrain`, `change history`, `change bank`, `change threshold`, `change seeds`, `change bootstrap`,
`change exclusion rules`. Any such action voids the confirmatory status.

## 4. Enforcement hooks (for the generator/analysis, implemented by A after GO)
- A **leakage assertion** that the modelling code (STEPs 3–6) has no read path to test outcomes or any secret
  field (matches the `SECRET_DENYLIST`).
- **Hash pinning**: STEP 8 refuses to run unless the 5 model hashes and the analysis-code hash equal the
  values frozen at STEPs 5–6.
- **One-shot guard**: STEP 8 records that the final analysis has executed; a second execution is refused
  (re-running requires an explicit, logged, pre-registered technical-invalid recovery, not a re-analysis).

## 5. Recovery (technical faults only)
If a technical-invalid fault (analysis-plan §9) occurs before STEP 7, resume from the frozen manifest at the
failed planned trial (no residual/nuisance resampling, no block replacement). A technical fault does **not**
permit revisiting STEPs 3–6 choices. Exceeding 15 technical-invalid trials → experiment INVALID.

## 6. Not produced in this phase
This is the protocol specification only. No sealing code, no models, no test data, and no execution are
produced here; A implements the hooks only after C returns GO.
