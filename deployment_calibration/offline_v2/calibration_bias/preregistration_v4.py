"""Confirmatory preregistration v4 -- FIX1 (offline; NO Isaac, NO generator, NO manifest instance, NO
confirmatory data, NO run authorization).

Single source of truth for every frozen quantity in preregistration v4 after Claude C's audit
(MODIFY_PREREGISTRATION_V4, commit 2ab3906). Fix1 resolves the four blockers WITHOUT changing the
candidate bank, test geometry, model, primary estimator, or effect (C: power_recertification_required =
false; proven by best_single_tiebreak_invariance_v1):

  A BLOCKER_SECRET_TIEBREAK      -> best-single tie-break no longer reads the secret (tau-|eff| removed).
  B BLOCKER_SEEDS_NOT_FROZEN     -> all randomization seeds derived by a frozen SHA256 rule with exact ints.
  C BLOCKER_SEAL_SCHEME_NOT_UNIQUE -> a single seal scheme (model-freeze-before-test-generation), no either/or.
  D BLOCKER_MODEL_FAILURE_SEMANTICS -> model-fit failure is an analysis-stage event, separate namespace from
                                     the 300-trial runtime-invalid budget; all 5 seeds must be valid.

The Markdown docs quote this module; the JSON artifacts are emitted from it; tests validate consistency and
that the allowlist/denylist/hyperparameters match the REAL power-verified code (`models_v2`).
"""

from __future__ import annotations

import hashlib
import json
import os

from deployment_calibration.models_v2 import features as FEAT

# ============================ FROZEN DESIGN (unchanged by fix1) ============================
BASE_COMMIT = "2bf7217907f24ae54a08db71bbdcf624b110ccf0"
AUDIT_COMMIT = "2ab39063c42f545b35beb53015c6a16402660c37"
V4_COMMIT = "9a02eb5ae320564bc8710380b919101d6537c02a"
CANDIDATE_BANK = (-0.04, 0.0, 0.04)
TRAIN_NOMINALS = (-0.04, -0.02, 0.0, 0.02, 0.04)
VAL_NOMINALS = (-0.01, 0.01)
TEST_NOMINALS = (-0.035, 0.035)
PROBE_OFFSET = -0.04
K = 1
TRAIN_BLOCKS, VAL_BLOCKS, TEST_BLOCKS = 9, 6, 9
RESIDUAL = {"dist": "TruncatedNormal", "mean": 0.0, "sigma": 0.005, "support": [-0.01, 0.01], "unit": "m"}
TRAIN_ACTUAL_SUPPORT = [min(TRAIN_NOMINALS) + RESIDUAL["support"][0], max(TRAIN_NOMINALS) + RESIDUAL["support"][1]]
TEST_ACTUAL_SUPPORT = {str(tn): [tn + RESIDUAL["support"][0], tn + RESIDUAL["support"][1]] for tn in TEST_NOMINALS}


def trial_counts():
    sessions_train = TRAIN_BLOCKS * len(TRAIN_NOMINALS)      # 45
    sessions_val = VAL_BLOCKS * len(VAL_NOMINALS)            # 12
    sessions_test = TEST_BLOCKS * len(TEST_NOMINALS)         # 18
    sessions_total = sessions_train + sessions_val + sessions_test
    trials_per_session = 1 + len(CANDIDATE_BANK)             # 4
    return {"sessions_train": sessions_train, "sessions_val": sessions_val, "sessions_test": sessions_test,
            "sessions_total": sessions_total, "trials_per_session": trials_per_session,
            "probe_trials_per_session": 1, "candidate_trials_per_session": len(CANDIDATE_BANK),
            "full_task_trials_total": sessions_total * trials_per_session}      # 300


# ============================ BLOCKER B: FROZEN SEEDS (deterministic, not run-time chosen) ============================
SEED_ROOT = "confirmatory-v4|2bf7217907f24ae54a08db71bbdcf624b110ccf0"
SEED_DERIVATION = ("seed(label) = int(sha256(f'{SEED_ROOT}|{label}'.encode()).hexdigest()[:16], 16) "
                   "% (2**63 - 1); SEED_ROOT = 'confirmatory-v4|<power_cert_commit>'; UTF-8 bytes; first 16 "
                   "hex chars; base 16; modulus 2**63-1")
SUBSEED_DERIVATION = ("subseed(domain_seed, planned_identity) = int(sha256("
                      "f'{domain_seed}|{planned_identity}'.encode()).hexdigest()[:16], 16) % (2**63 - 1)")
SEED_LABELS = ("master_seed", "train_residual_seed", "validation_residual_seed", "test_residual_seed",
               "train_block_order_seed", "validation_block_order_seed", "test_block_order_seed",
               "train_session_order_seed", "validation_session_order_seed", "test_session_order_seed",
               "train_nominal_order_seed", "validation_nominal_order_seed", "test_nominal_order_seed",
               "candidate_order_seed", "bootstrap_seed")


def seed(label: str) -> int:
    return int(hashlib.sha256(f"{SEED_ROOT}|{label}".encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)


def subseed(domain_seed: int, planned_identity: str) -> int:
    return int(hashlib.sha256(f"{domain_seed}|{planned_identity}".encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)


def frozen_seeds() -> dict:
    return {label: seed(label) for label in SEED_LABELS}


# ============================ FROZEN MODEL ============================
DEEPSETS_HP = {"d_hid": 32, "d_emb": 16, "lr": 1e-2, "max_epochs": 200, "patience": 25, "l2": 1e-4}
MODEL_SEEDS = (1103, 2207, 3301, 4409, 5519)
# BLOCKER D: model validity conditions (analysis-stage; NOT a runtime trial)
MODEL_VALIDITY_CONDITIONS = [
    "training process terminates normally (no exception)",
    "no NaN/Inf during training",
    "best_state checkpoint produced",
    "train and validation loss both finite",
    "checkpoint reloadable",
    "inference from the reloaded checkpoint is finite",
    "state_dict hash saved",
    "input schema / leakage checks pass",
]
MODEL_FIT = {
    "all_seeds_must_converge": True,
    "required": "5/5 fixed seeds valid for BOTH K0 and K1 (10 checkpoints); all hashes frozen",
    "reaching_max_epochs_is_not_failure": "hitting max_epochs without early-stop is VALID if the validity "
                                          "conditions hold",
    "retry": "initial attempt + at most 1 deterministic retry, ONLY for an infrastructural exception / "
             "abnormal process exit / reproducible I/O failure",
    "retry_constraints": ["same data hash", "same model seed", "same hyperparameters", "same code commit",
                          "same device/determinism config", "keep first-failure log",
                          "do NOT change initialization or training budget"],
    "no_seed_replacement": True,
    "no_dropping_bad_seed": True,
    "missing_or_invalid_seed_makes_experiment_invalid": True,
    "invalid_verdict": "EXPERIMENT_INVALID_MODEL_FIT",
    "on_invalid": "do NOT generate the test manifest; do NOT unseal test",
    "namespace": "model_fit_failure_count  (SEPARATE from runtime_trial_invalid_count; NOT subject to the "
                 ">15 runtime rule)",
}
MODEL = {
    "class": "deployment_calibration.models_v2.DeepSets",
    "name": "B2_deepsets",
    "encoder": "deepsets (permutation-invariant: shared phi MLP per probe, mean+sum pool, rho)",
    "hyperparameters": DEEPSETS_HP,
    "model_seeds": list(MODEL_SEEDS),
    "optimizer": "Adam(lr=1e-2, weight_decay=1e-4)",
    "loss": "BCEWithLogits(success) + MSE(z-task_outcome_error) + MSE(z-skill_elapsed_time)  [multitask]",
    "batching": "FULL-BATCH gradient descent (one Adam step per epoch over the entire train set)",
    "feature_normalization": "standardize static (train mu/sd), standardize probe vectors (train pooled "
                             "mu/sd), z-normalize the two continuous targets (train mu/sd); success target "
                             "NOT normalized; eps 1e-8",
    "class_weighting": "none (plain BCE)",
    "initialization": "default PyTorch init under torch.manual_seed(seed) and np.random.seed(seed)",
    "determinism": "seeds set per model_seed; CPU full-batch -> deterministic rerun (verified by test)",
    "validation_metric": "the multitask training loss evaluated on the validation fold",
    "early_stopping": "patience=25 epochs, improvement threshold 1e-5 on val loss",
    "checkpoint_selection": "restore the state_dict at minimum validation loss (best_state)",
    "p_success_output": "sigmoid(success logit), then clipped to [0,1] (clip_predictions)",
    "candidate_selection": "argmax_o p_success over the 3-candidate bank",
    "prediction_tie_break": "strict-greater argmax scanning the bank in fixed order (-0.04, 0.0, +0.04); "
                            "on an exact tie the earliest bank index is kept",
    "regression_clip": {"error_clip": [0.0, 0.30], "time_clip": [0.0, 60.0]},
    # BLOCKER D fix: a model-training failure is an ANALYSIS-STAGE event, NOT one of the 300 runtime trials.
    "training_failure_handling": "a model-fit failure is an analysis-stage event governed by MODEL_FIT "
                                 "(namespace model_fit_failure_count); it is NOT a runtime trial and does "
                                 "NOT consume the 300-trial runtime-invalid budget",
    "all_seeds_must_converge": True,
    "model_validity_conditions": MODEL_VALIDITY_CONDITIONS,
    "model_fit": MODEL_FIT,
    # 7.3: generator/training must pass every frozen hyperparameter EXPLICITLY (defaults differ: the
    # DeepSets constructor defaults are max_epochs=300, patience=30, NOT the frozen 200/25).
    "must_pass_all_hyperparameters_explicitly": True,
    "constructor_defaults_differ": {"max_epochs_default": 300, "patience_default": 30,
                                    "note": "must override with 200/25; never rely on defaults"},
    "no_hyperparameter_search_on_confirmatory": True,
}

# ============================ HISTORY ALLOWLIST / SECRET DENYLIST ============================
HISTORY_ALLOWLIST = [
    "theta.grasp_offset_local_y", "theta.max_pos_step", "theta.pull_lead",
    "success", "task_outcome_error", "skill_elapsed_time",
    "pull_phase_duration", "final_joint_position",
]
# 7.2: every allowlist field is REQUIRED on each probe HistoryEntry; a missing field is
# TECHNICAL_INVALID_SCHEMA -- never silently zero-filled.
HISTORY_FIELDS_REQUIRED = True
MISSING_PROBE_FIELD_VERDICT = "TECHNICAL_INVALID_SCHEMA"
STATIC_ALLOWLIST = [
    "theta.grasp_offset_local_y", "theta.max_pos_step", "theta.pull_lead",
    "g.target_open_position", "g.target_tolerance",
    "x.initial_mechanism_joint_pos", "x.gripper_width", "x.member==sektion_cabinet (bool)",
]
SECRET_DENYLIST = [
    "nominal_bias", "residual_bias", "actual_bias", "eff_signed", "abs_eff",
    "nuisance_block_id", "block_seed", "residual_seed", "split_identity",
    "future_candidate_outcomes", "oracle_action", "hidden_state_class_label",
    "secret_deployment_state", "hidden_state_id", "damping_eff", "damping_post",
    "tau_minus_abs_eff (secret continuous margin)",
]

# ============================ OUTCOME SCHEMA ============================
FAILURE_REASONS = ["NONE", "REACH_TIMEOUT", "PULL_TIMEOUT", "POSITION_TIMEOUT",
                   "HANDLE_DETACHED", "INVALID_PARAM", "EPISODE_EXCEPTION", "OTHER"]
PRIMARY_SUCCESS = {
    "definition": "y.success as produced by the frozen runtime open_drawer success condition",
    "reference_schema": "deployment_calibration/contracts/episode_schema_v2.py",
    "target_tolerance": 0.02, "reach_timeout_s": 16.0, "pull_timeout_s": 16.0,
    "retained_failure_reasons": ["NONE", "REACH_TIMEOUT (APPROACH)", "PULL_TIMEOUT",
                                 "POSITION_TIMEOUT", "HANDLE_DETACHED"],
    "legitimate_task_failures_kept": True,
}
CONTINUOUS_SECONDARY = [
    "task_outcome_error", "final_joint_position", "skill_elapsed_time",
    "true_handle_error_at_close_3d", "true_handle_error_at_close_local_y",
    "gripper_width_at_close", "phase_durations",
]

# ============================ COLLISION RULE ============================
COLLISION_RULE = {
    "evidence": {"clean_smoke_force_N": 0.0, "intentional_positive_control_force_N": "~240",
                 "explore_306_max_unintended_contact_force_N": 0.0,
                 "explore_306_unintended_contact_frame_count": 0,
                 "contact_sensor_available_frac": 1.0,
                 "sensor_backend": "isaaclab.sensors.ContactSensor(robot_contact.net_forces_w, non-finger links)"},
    "monitored_scope": "robot non-finger links (arm/hand) vs cabinet; intended finger-handle contact EXCLUDED",
    "primary": "intention-to-analyze: primary success includes ALL schema-valid task trials regardless of "
               "unintended-contact force (no trial dropped for collision)",
    "collision_confounded_definition": "a schema-valid trial with max_unintended_contact_force_N > 0.0 N "
                                       "on monitored (non-finger) links",
    "numeric_threshold_N": 0.0,
    "sensitivity": "re-run the primary contrast excluding pre-registered collision-confounded trials; "
                   "report whether the CI_lower>=0.15 verdict is unchanged",
    "confirmatory_data_not_used_to_set_threshold": True,
}

# ============================ TECHNICAL INVALIDATION (RUNTIME TRIALS ONLY) ============================
TECHNICAL_INVALID = {
    "scope": "RUNTIME full-task trials ONLY (namespace runtime_trial_invalid_count); model-fit failures are "
             "governed separately by MODEL_FIT and do NOT count here",
    "invalid_conditions": ["runtime_crash / EPISODE_EXCEPTION", "ContactSensor unavailable",
                           "incomplete schema (missing required field, incl. any probe allowlist field)",
                           "manifest hash mismatch", "controller not started",
                           "invalid initial state / full_reset not verified", "file corruption"],
    "legitimate_failures_never_invalid": ["timeout (REACH/PULL/POSITION)", "HANDLE_DETACHED",
                                          "approach/pull failure", "offset-induced failure"],
    "policy": "fail-fast then resume from the frozen manifest at the failed planned trial",
    "retry_same_planned_trial_allowed": True,
    "resample_residual_or_nuisance_on_retry": False,
    "replace_block_allowed": False,
    "max_technical_invalid_trials": 15,      # applies to runtime_trial_invalid_count ONLY
    "experiment_invalid_if": ">15 runtime technical-invalid trials, OR any unresolved manifest/leakage/"
                             "integrity breach",
    "never_rerun_on_task_outcome": True,
}
INVALID_COUNTERS = {
    "runtime_trial_invalid_count": "counts the 300 runtime full-task trials only; subject to the >15 rule",
    "model_fit_failure_count": "counts model-training failures; SEPARATE; governed by MODEL_FIT; a single "
                               "invalid/missing seed after retry -> EXPERIMENT_INVALID_MODEL_FIT",
}

# ============================ RANDOMIZATION / PROVENANCE ============================
MANIFEST_FIELDS = list(SEED_LABELS) + ["seed_root", "seed_derivation", "subseed_derivation",
                                       "manifest_hash", "config_hash", "code_commit", "generator_commit",
                                       "environment_versions"]

# ============================ ANALYSIS / PRIMARY ============================
GAIN_THRESHOLD = 0.15
BOOTSTRAP = {"n_boot": 2000, "ci": 0.95, "method": "test-block percentile bootstrap",
             "resample_unit": "test nuisance block (keeps both test nominals, all 5 seeds, block candidate "
                              "correspondence)",
             "bootstrap_seed_frozen_before_unseal": True, "bootstrap_seed": seed("bootstrap_seed")}
SEED_GATE = {">=4/5 seed point gains >= 0.15": True, "no seed point gain < 0": True,
             "precondition": "all 5 model seeds exist and are technically valid (MODEL_FIT); a missing/"
                             "invalid seed makes the experiment INVALID and is NOT treated as a gain<0 seed "
                             "nor dropped from the denominator",
             "role": "PRIMARY necessary quality gate (GO requires it)"}
PRIMARY = {
    "comparator": "Delta_primary = seed-averaged B2_K1 selected success - train/val-only best-single success",
    "statistical_unit": "test nuisance block",
    "aggregation": "per test block: average over the 2 test nominals, then average over the 5 model seeds "
                   "(B2); best-single averaged over the 2 nominals in the same block",
    "point_estimate": "mean over test blocks of gain_b",
    "primary_pass": "CI_lower(Delta_primary) >= 0.15  (95% test-block bootstrap, n_boot=2000)",
    "plus_gate": SEED_GATE,
    "not_allowed": ["point estimate as primary", "CI>0 as primary", "episode-level bootstrap",
                    "lowering 0.15", "changing the comparator"],
}
SECONDARY = {
    "H2_history": "Delta_history = B2_K1 - capacity-matched B1_K0 (same DeepSets, empty history); "
                  "report point estimate + 95% block bootstrap CI (does NOT replace primary)",
    "H3_reports": ["B2_K1 vs state-aware oracle gap", "probe-derived hidden-condition audit accuracy",
                   "selected-action accuracy", "model-seed stability", "continuous outcomes",
                   "failure mechanism", "Gross/Net VOI"],
    "k0_k1_capacity": "K0 and K1 are the SAME DeepSets (same d_hid/d_emb/budget/seeds); the ONLY difference "
                      "is empty history (K0) vs one frozen probe-history entry (K1)",
    "oracle": "OracleZ / state-aware oracle reads the secret actual bias; AUDIT UPPER BOUND ONLY -- never "
              "in model input, training, best-single, or primary selection",
}
VOI = {"lambda_time": 0.02, "probe_time_s": 16.58, "horizons": [1, 2, 5, 10],
       "priority": "SECONDARY", "no_post_hoc_T_as_primary_endpoint": True}

# ============================ BLOCKER A: LEGAL BEST-SINGLE (no secret) ============================
BEST_SINGLE = {
    "data": "train + validation candidate trials ONLY (test never participates)",
    "rule": ["1. maximize observed mean binary success over train+val candidate trials",
             "2. tie -> minimize abs(offset)",
             "3. tie -> first in the frozen candidate-bank numeric order {-0.04, 0.00, +0.04}"],
    "forbidden_inputs": ["nominal bias", "residual bias", "actual bias", "eff_signed", "abs_eff",
                         "tau-|eff| / secret continuous margin", "oracle action", "test outcomes"],
    "no_new_observable_continuous_tiebreak": True,
    "implementation": "deployment_calibration.offline_v2.calibration_bias.learned_selector_power.best_single_legal",
    "must_save": ["per-candidate train/val success score", "tie-break trace", "final offset", "selection hash"],
    "exploratory_expectation": 0.0,
    "hardcoded_in_confirmatory": False,
    "invariance_proof": "best_single_tiebreak_invariance_v1.json (4500 replicates: 0 step-1 ties, 0 old/new "
                        "mismatches -> the removed secret tie-break was dead code; no power recert)",
}

CLAIM_SCOPE = {
    "confirmatory_question": ("whether, when the hidden deployment calibration state is stable across "
                              "repeated tasks, the history from ONE fixed full-task-level diagnostic trial "
                              "lets a capacity-matched history-conditioned selector pick a better discrete "
                              "action parameter on UNSEEN nominal deployment conditions and raise subsequent "
                              "task success"),
    "probe_term": "repeated-task deployment diagnostic full-task trial",
    "forbidden_terms": ["one-shot online adaptation", "low-cost internal query", "non-destructive probe",
                        "unseen hidden-state generalization"],
    "generalization_claim": "unseen NOMINAL deployment conditions, actual hidden-state support inside training support",
    "test_actual_support": TEST_ACTUAL_SUPPORT, "train_actual_support": TRAIN_ACTUAL_SUPPORT,
    "positive_control_note": ("the 3-point candidate bank is a constructed positive-control skill library "
                              "frozen by the independent exploration phase, to validate the history-"
                              "conditioned action-selection mechanism; NOT a claim over arbitrary continuous "
                              "action spaces, arbitrary tasks, or long-horizon autonomous adaptation"),
    "primary_claim": "diagnostic history improves subsequent action-selection success",
    "net_value_is_secondary": True,
}

DATA_ISOLATION = {
    "306_explore_not_in_confirmatory": True, "capability_map_not_in_training": True,
    "synthetic_power_data_not_in_training": True, "test_inaccessible_until_model_frozen": True,
    "secret_fields_audit_only": True, "feature_cache_excludes_secret": True,
}

# ============================ BLOCKER C: UNIQUE SEAL SCHEME ============================
SEAL_SCHEME = "SCHEME_2_MODEL_FREEZE_BEFORE_TEST_GENERATION"
SEAL_UNSEAL = [
    "Step 0 PRE-RUN FREEZE: freeze+hash the fix1 v4 commit, confirmatory config, all exact seeds, the "
    "manifest-generation algorithm, schema, generator commit (future, post-GO), train/val/test planned "
    "structure, and the test split identities + nominal sets. No test outcomes exist yet.",
    "Step 1 GENERATE TRAIN/VAL MANIFEST deterministically from the frozen config+seeds; save its hash; no resample.",
    "Step 2 RUN TRAIN/VAL TRIALS (only train/validation runtime data generated).",
    "Step 3 FREEZE TRAIN/VAL DATA: integrity + leakage validation, dataset hashes, train/val lock file.",
    "Step 4 SELECT BEST-SINGLE and TRAIN MODELS on train/validation ONLY (5 fixed seeds for K0 and K1; all "
    "10 must be valid); save state_dict hashes.",
    "Step 5 FREEZE ANALYSIS (before ANY test trial): 5 K0 hashes, 5 K1 hashes, best-single artifact/hash, "
    "feature-extraction code hash, final-analysis code hash, bootstrap seed, collision sensitivity rule, "
    "exclusion/invalidation rules -> write model_analysis_freeze.json.",
    "Step 6 GENERATE TEST MANIFEST once, deterministically, from the frozen test seeds + algorithm + "
    "generator commit + config (only after Step 5 passes); save the test manifest hash. No reselecting "
    "seeds, no generate-many-and-pick, no changing test geometry/block count/nominal/order/residual rules.",
    "Step 7 RUN TEST TRIALS per the test manifest.",
    "Step 8 ONE-SHOT FINAL ANALYSIS with the frozen models and analysis code.",
]
ACCESS_CONTROL = {
    "test_outcomes_nonexistent_before_step6": True,
    "trainval_feature_cache_excludes_test_fields": True,
    "test_manifest_only_after_step5": True,
    "all_test_access_logged": True,
    "early_test_access_verdict": "EXPERIMENT_INVALID_EARLY_TEST_ACCESS",
    "early_test_access_condition": "generating or reading any test outcome before Step 5 completes",
}
POST_UNSEAL_FORBIDDEN = ["retrain", "change history", "change bank", "change threshold", "change seeds",
                         "change bootstrap", "change exclusion rules", "reselect test seed",
                         "generate multiple test manifests and pick", "change test geometry/block count"]

STATUS = "PREREGISTRATION_V4_FIX1_READY_FOR_C_REAUDIT"


def preregistration_dict():
    return {
        "title": "Confirmatory Preregistration v4 FIX1 -- calibration-bias diagnostic history & action selection",
        "status": STATUS,
        "base_commit": BASE_COMMIT, "audit_commit": AUDIT_COMMIT, "v4_original_commit": V4_COMMIT,
        "power_certification": "test_geometry_power_verdict_v1.json = POWER_SUFFICIENT_FOR_PREREG_V4",
        "power_recertification_required": False,
        "audit_verdict_addressed": "MODIFY_PREREGISTRATION_V4 (Claude C, 2ab3906)",
        "blockers_fixed": ["A BLOCKER_SECRET_TIEBREAK", "B BLOCKER_SEEDS_NOT_FROZEN",
                           "C BLOCKER_SEAL_SCHEME_NOT_UNIQUE", "D BLOCKER_MODEL_FAILURE_SEMANTICS"],
        "claim_scope": CLAIM_SCOPE,
        "design": {"candidate_bank": list(CANDIDATE_BANK), "train_nominals": list(TRAIN_NOMINALS),
                   "val_nominals": list(VAL_NOMINALS), "test_nominals": list(TEST_NOMINALS),
                   "probe_offset": PROBE_OFFSET, "K": K,
                   "blocks": {"train": TRAIN_BLOCKS, "val": VAL_BLOCKS, "test": TEST_BLOCKS},
                   "residual": RESIDUAL, "train_actual_support": TRAIN_ACTUAL_SUPPORT,
                   "test_actual_support": TEST_ACTUAL_SUPPORT},
        "trial_counts": trial_counts(),
        "seed_root": SEED_ROOT, "seed_derivation": SEED_DERIVATION, "subseed_derivation": SUBSEED_DERIVATION,
        "frozen_seeds": frozen_seeds(),
        "model": MODEL, "model_fit": MODEL_FIT, "invalid_counters": INVALID_COUNTERS,
        "history_allowlist": HISTORY_ALLOWLIST, "history_fields_required": HISTORY_FIELDS_REQUIRED,
        "missing_probe_field_verdict": MISSING_PROBE_FIELD_VERDICT,
        "static_allowlist": STATIC_ALLOWLIST, "secret_denylist": SECRET_DENYLIST,
        "outcome": {"primary_success": PRIMARY_SUCCESS, "failure_reasons": FAILURE_REASONS,
                    "continuous_secondary": CONTINUOUS_SECONDARY},
        "best_single": BEST_SINGLE,
        "primary": PRIMARY, "secondary": SECONDARY, "voi": VOI, "bootstrap": BOOTSTRAP,
        "collision_rule": COLLISION_RULE, "technical_invalid": TECHNICAL_INVALID,
        "manifest_fields": MANIFEST_FIELDS, "data_isolation": DATA_ISOLATION,
        "seal_scheme": SEAL_SCHEME, "seal_unseal": SEAL_UNSEAL, "access_control": ACCESS_CONTROL,
        "post_unseal_forbidden": POST_UNSEAL_FORBIDDEN,
        "hard_constraints": ["no Isaac", "no confirmatory generator", "no manifest instance",
                             "no confirmatory data", "no run authorization",
                             "306/runtime/models_v2/capability-map/prior-prereg/C-audit untouched"],
        "go_fail_invalid": {
            "PRIMARY_PASS": "CI_lower(Delta_primary) >= 0.15 AND seed gate (all 5 valid, >=4/5>=0.15, none<0)",
            "PRIMARY_FAIL": "any primary gate unmet",
            "EXPERIMENT_INVALID": "pre-registered integrity/leakage/manifest/runtime-technical conditions, "
                                  "OR EXPERIMENT_INVALID_MODEL_FIT, OR EXPERIMENT_INVALID_EARLY_TEST_ACCESS",
            "no_exploratory_rewrite": "a non-ideal result may NOT be rewritten as an exploratory PASS",
        },
    }


def audit_checklist():
    return {
        "for": "Claude C re-audit of preregistration v4 FIX1",
        "must_verify": [
            "BLOCKER A: best_single rule has NO tau/eff/secret; implementation=best_single_legal; invariance proven",
            "BLOCKER B: all 15 randomization seeds are exact ints from the frozen SHA256 rule; subseed rule frozen",
            "BLOCKER C: exactly ONE seal scheme (model-freeze-before-test-generation); no either/or; early-access INVALID",
            "BLOCKER D: model-fit failure separate namespace; all-5-seeds-valid required; retry<=1; no seed drop",
            "trial count 300 / 75 sessions unchanged; bank/geometry/model/estimator unchanged",
            "history allowlist == features.probe_vector (8); 8 fields REQUIRED (no silent zero-fill)",
            "generator must pass all hyperparameters explicitly (constructor defaults differ 300/30)",
            "bootstrap seed frozen before unseal",
            "band_edge exit verdict superseded (not overwritten)",
            "power_recertification_required == false, justified by invariance proof",
            "markdown and JSON consistent; no generator/manifest/data produced",
        ],
        "deliverable_gate": "C must return GO before A implements the generator; this phase authorizes nothing",
    }


def emit(docs_dir=None):
    docs_dir = docs_dir or _docs_dir()
    os.makedirs(docs_dir, exist_ok=True)
    pre = preregistration_dict()
    with open(os.path.join(docs_dir, "preregistration_v4.json"), "w") as f:
        json.dump(pre, f, indent=2)
    cfg = {"base_commit": BASE_COMMIT, "audit_commit": AUDIT_COMMIT, "design": pre["design"],
           "trial_counts": pre["trial_counts"], "seed_root": SEED_ROOT, "seed_derivation": SEED_DERIVATION,
           "subseed_derivation": SUBSEED_DERIVATION, "frozen_seeds": frozen_seeds(),
           # top-level convenience ints too (a re-audit reads master_seed/bootstrap_seed directly)
           "master_seed": seed("master_seed"), "residual_seed": seed("train_residual_seed"),
           "bootstrap_seed": seed("bootstrap_seed"),
           "model": MODEL, "model_fit": MODEL_FIT, "invalid_counters": INVALID_COUNTERS,
           "history_allowlist": HISTORY_ALLOWLIST, "history_fields_required": HISTORY_FIELDS_REQUIRED,
           "missing_probe_field_verdict": MISSING_PROBE_FIELD_VERDICT,
           "static_allowlist": STATIC_ALLOWLIST, "secret_denylist": SECRET_DENYLIST,
           "best_single": BEST_SINGLE, "primary": PRIMARY, "bootstrap": BOOTSTRAP,
           "collision_rule": COLLISION_RULE, "technical_invalid": TECHNICAL_INVALID,
           "seal_scheme": SEAL_SCHEME, "access_control": ACCESS_CONTROL,
           "manifest_fields": MANIFEST_FIELDS, "voi": VOI, "status": STATUS}
    with open(os.path.join(docs_dir, "confirmatory_v4_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(docs_dir, "confirmatory_v4_audit_checklist.json"), "w") as f:
        json.dump(audit_checklist(), f, indent=2)
    return pre


def _docs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "docs", "offline_v2", "calibration_bias")


def verify_allowlist_matches_code():
    assert list(FEAT.PROBE_KEYS) == HISTORY_ALLOWLIST, (list(FEAT.PROBE_KEYS), HISTORY_ALLOWLIST)
    assert FEAT.PROBE_DIM == len(HISTORY_ALLOWLIST) == 8
    assert FEAT.STATIC_DIM == 8
    return True


if __name__ == "__main__":
    verify_allowlist_matches_code()
    p = emit()
    print("emitted fix1 JSON artifacts. status:", p["status"])
    print("seeds:", {k: p["frozen_seeds"][k] for k in ("master_seed", "bootstrap_seed")})
    print("trials:", p["trial_counts"]["full_task_trials_total"], "power_recert:",
          p["power_recertification_required"])
