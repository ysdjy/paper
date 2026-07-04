"""Confirmatory preregistration v4 -- frozen constants + JSON emitters (offline; NO Isaac, NO generator,
NO confirmatory data).

This module is the single source of truth for every frozen quantity in preregistration v4. The Markdown
docs quote it, the JSON artifacts are emitted from it, and the tests validate that the model allowlist /
denylist / hyperparameters match the REAL power-verified code (`models_v2`). Nothing here runs Isaac,
generates data, or authorizes a run.

Design was fixed by the independent exploration phases and certified by
`test_geometry_power_verdict_v1.json` = POWER_SUFFICIENT_FOR_PREREG_V4 at commit 2bf7217.
"""

from __future__ import annotations

import json
import os

from deployment_calibration.models_v2 import features as FEAT

# ============================ FROZEN DESIGN ============================
BASE_COMMIT = "2bf7217907f24ae54a08db71bbdcf624b110ccf0"
CANDIDATE_BANK = (-0.04, 0.0, 0.04)
TRAIN_NOMINALS = (-0.04, -0.02, 0.0, 0.02, 0.04)
VAL_NOMINALS = (-0.01, 0.01)
TEST_NOMINALS = (-0.035, 0.035)
PROBE_OFFSET = -0.04                    # K=1 fixed diagnostic probe
K = 1

TRAIN_BLOCKS, VAL_BLOCKS, TEST_BLOCKS = 9, 6, 9
RESIDUAL = {"dist": "TruncatedNormal", "mean": 0.0, "sigma": 0.005, "support": [-0.01, 0.01], "unit": "m"}

# actual (hidden) support -- claim scope
TRAIN_ACTUAL_SUPPORT = [min(TRAIN_NOMINALS) + RESIDUAL["support"][0], max(TRAIN_NOMINALS) + RESIDUAL["support"][1]]
TEST_ACTUAL_SUPPORT = {str(tn): [tn + RESIDUAL["support"][0], tn + RESIDUAL["support"][1]] for tn in TEST_NOMINALS}

# ============================ TRIAL COUNT (independently recomputed) ============================
def trial_counts():
    sessions_train = TRAIN_BLOCKS * len(TRAIN_NOMINALS)      # 9*5 = 45
    sessions_val = VAL_BLOCKS * len(VAL_NOMINALS)            # 6*2 = 12
    sessions_test = TEST_BLOCKS * len(TEST_NOMINALS)         # 9*2 = 18
    sessions_total = sessions_train + sessions_val + sessions_test
    trials_per_session = 1 + len(CANDIDATE_BANK)            # 1 probe + 3 candidates = 4
    return {
        "sessions_train": sessions_train, "sessions_val": sessions_val, "sessions_test": sessions_test,
        "sessions_total": sessions_total,
        "trials_per_session": trials_per_session,
        "probe_trials_per_session": 1, "candidate_trials_per_session": len(CANDIDATE_BANK),
        "full_task_trials_total": sessions_total * trials_per_session,      # 75*4 = 300
    }

# ============================ FROZEN MODEL ============================
# EXACTLY the hyperparameters used in the power-verified simulation (learned_selector_power.DEEPSETS_HP).
DEEPSETS_HP = {"d_hid": 32, "d_emb": 16, "lr": 1e-2, "max_epochs": 200, "patience": 25, "l2": 1e-4}
MODEL_SEEDS = (1103, 2207, 3301, 4409, 5519)
MODEL = {
    "class": "deployment_calibration.models_v2.DeepSets",
    "name": "B2_deepsets",
    "encoder": "deepsets (permutation-invariant: shared phi MLP per probe, mean+sum pool, rho)",
    "hyperparameters": DEEPSETS_HP,
    "model_seeds": list(MODEL_SEEDS),
    "optimizer": "Adam(lr=1e-2, weight_decay=1e-4)",           # torch_models._TorchHistoryModel.fit
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
    "training_failure_handling": "a training exception is a TECHNICAL-INVALID event (section 17); in the "
                                 "power simulation it was scored conservatively as a power failure",
    "no_hyperparameter_search_on_confirmatory": True,
}

# ============================ HISTORY ALLOWLIST / SECRET DENYLIST ============================
# The model reads history ONLY through features.probe_vector -> these 8 fields, in this order.
HISTORY_ALLOWLIST = [
    "theta.grasp_offset_local_y", "theta.max_pos_step", "theta.pull_lead",
    "success", "task_outcome_error", "skill_elapsed_time",
    "pull_phase_duration", "final_joint_position",
]
# Candidate decision inputs the model reads (features.static_features -> 8 dims).
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
]

# ============================ OUTCOME SCHEMA (frozen, referenced) ============================
FAILURE_REASONS = ["NONE", "REACH_TIMEOUT", "PULL_TIMEOUT", "POSITION_TIMEOUT",
                   "HANDLE_DETACHED", "INVALID_PARAM", "EPISODE_EXCEPTION", "OTHER"]
PRIMARY_SUCCESS = {
    "definition": "y.success as produced by the frozen runtime open_drawer success condition",
    "reference_schema": "deployment_calibration/contracts/episode_schema_v2.py",
    "target_tolerance": 0.02,
    "reach_timeout_s": 16.0, "pull_timeout_s": 16.0,
    "retained_failure_reasons": ["NONE", "REACH_TIMEOUT (APPROACH)", "PULL_TIMEOUT",
                                 "POSITION_TIMEOUT", "HANDLE_DETACHED"],
    "legitimate_task_failures_kept": True,
}
CONTINUOUS_SECONDARY = [
    "task_outcome_error", "final_joint_position", "skill_elapsed_time",
    "true_handle_error_at_close_3d", "true_handle_error_at_close_local_y",
    "gripper_width_at_close", "phase_durations",
]

# ============================ COLLISION RULE (frozen from existing evidence) ============================
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
    "numeric_threshold_N": 0.0,      # > 0.0 N flags a collision-confounded trial (nonzero rule)
    "sensitivity": "re-run the primary contrast excluding pre-registered collision-confounded trials; "
                   "report whether the CI_lower>=0.15 verdict is unchanged",
    "confirmatory_data_not_used_to_set_threshold": True,
}

# ============================ TECHNICAL INVALIDATION (frozen) ============================
TECHNICAL_INVALID = {
    "invalid_conditions": ["runtime_crash / EPISODE_EXCEPTION", "ContactSensor unavailable",
                           "incomplete schema (missing required field)", "manifest hash mismatch",
                           "controller not started", "invalid initial state / full_reset not verified",
                           "file corruption"],
    "legitimate_failures_never_invalid": ["timeout (REACH/PULL/POSITION)", "HANDLE_DETACHED",
                                          "approach/pull failure", "offset-induced failure"],
    "policy": "fail-fast then resume from the frozen manifest at the failed planned trial",
    "retry_same_planned_trial_allowed": True,
    "resample_residual_or_nuisance_on_retry": False,
    "replace_block_allowed": False,
    "max_technical_invalid_trials": 15,      # 5% of 300; exceeding -> whole experiment INVALID
    "experiment_invalid_if": ">15 technical-invalid trials, OR any unresolved manifest/leakage/integrity breach",
    "never_rerun_on_task_outcome": True,
}

# ============================ RANDOMIZATION / PROVENANCE ============================
MANIFEST_FIELDS = ["master_seed", "residual_seed", "block_order", "session_order", "nominal_order",
                   "candidate_order", "model_seeds", "bootstrap_seed", "manifest_hash", "config_hash",
                   "code_commit", "environment_versions"]

# ============================ ANALYSIS / PRIMARY ============================
GAIN_THRESHOLD = 0.15
BOOTSTRAP = {"n_boot": 2000, "ci": 0.95, "method": "test-block percentile bootstrap",
             "resample_unit": "test nuisance block (keeps both test nominals, all 5 seeds, block candidate "
                              "correspondence)"}
SEED_GATE = {">=4/5 seed point gains >= 0.15": True, "no seed point gain < 0": True,
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

BEST_SINGLE = {
    "data": "train + validation ONLY (test never participates)",
    "rule": ["1. max mean success",
             "2. tie -> max mean frozen continuous margin/utility (tau - |eff|)",
             "3. tie -> min |offset|",
             "4. tie -> fixed numeric order of the bank"],
    "must_save": ["per-candidate train/val score", "tie-break trace", "final offset", "selection hash"],
    "exploratory_expectation": 0.0,
    "hardcoded_in_confirmatory": False,
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
    "test_actual_support": TEST_ACTUAL_SUPPORT,
    "train_actual_support": TRAIN_ACTUAL_SUPPORT,
    "positive_control_note": ("the 3-point candidate bank is a constructed positive-control skill library "
                              "frozen by the independent exploration phase, to validate the history-"
                              "conditioned action-selection mechanism; NOT a claim over arbitrary continuous "
                              "action spaces, arbitrary tasks, or long-horizon autonomous adaptation"),
    "primary_claim": "diagnostic history improves subsequent action-selection success",
    "net_value_is_secondary": True,
}

DATA_ISOLATION = {
    "306_explore_not_in_confirmatory": True,
    "capability_map_not_in_training": True,
    "synthetic_power_data_not_in_training": True,
    "test_inaccessible_until_model_frozen": True,
    "secret_fields_audit_only": True,
    "feature_cache_excludes_secret": True,
}

SEAL_UNSEAL = [
    "1. generate + validate train/validation",
    "2. keep test SEALED (or generate only after models frozen)",
    "3. select best-single from train/validation ONLY",
    "4. train K0/K1 from train/validation ONLY",
    "5. freeze the 5 seed-model hashes",
    "6. freeze the analysis-code hash",
    "7. UNSEAL test",
    "8. one-shot final analysis",
]
POST_UNSEAL_FORBIDDEN = ["retrain", "change history", "change bank", "change threshold", "change seeds",
                         "change bootstrap", "change exclusion rules"]

STATUS = "PREREGISTRATION_V4_READY_FOR_INDEPENDENT_AUDIT"


def preregistration_dict():
    return {
        "title": "Confirmatory Preregistration v4 -- calibration-bias diagnostic history & action selection",
        "status": STATUS,
        "base_commit": BASE_COMMIT,
        "power_certification": "test_geometry_power_verdict_v1.json = POWER_SUFFICIENT_FOR_PREREG_V4",
        "claim_scope": CLAIM_SCOPE,
        "design": {
            "candidate_bank": list(CANDIDATE_BANK), "train_nominals": list(TRAIN_NOMINALS),
            "val_nominals": list(VAL_NOMINALS), "test_nominals": list(TEST_NOMINALS),
            "probe_offset": PROBE_OFFSET, "K": K,
            "blocks": {"train": TRAIN_BLOCKS, "val": VAL_BLOCKS, "test": TEST_BLOCKS},
            "residual": RESIDUAL,
            "train_actual_support": TRAIN_ACTUAL_SUPPORT, "test_actual_support": TEST_ACTUAL_SUPPORT,
        },
        "trial_counts": trial_counts(),
        "model": MODEL,
        "history_allowlist": HISTORY_ALLOWLIST, "static_allowlist": STATIC_ALLOWLIST,
        "secret_denylist": SECRET_DENYLIST,
        "outcome": {"primary_success": PRIMARY_SUCCESS, "failure_reasons": FAILURE_REASONS,
                    "continuous_secondary": CONTINUOUS_SECONDARY},
        "best_single": BEST_SINGLE,
        "primary": PRIMARY, "secondary": SECONDARY, "voi": VOI, "bootstrap": BOOTSTRAP,
        "collision_rule": COLLISION_RULE, "technical_invalid": TECHNICAL_INVALID,
        "manifest_fields": MANIFEST_FIELDS, "data_isolation": DATA_ISOLATION,
        "seal_unseal": SEAL_UNSEAL, "post_unseal_forbidden": POST_UNSEAL_FORBIDDEN,
        "hard_constraints": ["no Isaac", "no confirmatory generator", "no confirmatory data",
                             "no run authorization", "306/runtime/capability-map/prior-prereg untouched"],
        "go_fail_invalid": {
            "PRIMARY_PASS": "CI_lower(Delta_primary) >= 0.15 AND seed gate",
            "PRIMARY_FAIL": "any primary gate unmet",
            "EXPERIMENT_INVALID": "only pre-registered integrity/leakage/manifest/technical-failure conditions",
            "no_exploratory_rewrite": "a non-ideal result may NOT be rewritten as an exploratory PASS",
        },
    }


def audit_checklist():
    return {
        "for": "Claude C independent audit",
        "must_verify": [
            "train/val/test nominals exactly {-0.04..+0.04}/{-0.01,+0.01}/{-0.035,+0.035}",
            "candidate bank {-0.04,0,+0.04}; probe fixed -0.04; K=1",
            "blocks 9/6/9; sessions 45/12/18=75; full-task trials 300",
            "probe and candidate(-0.04) are DISTINCT full-task trials; probe outcome not reused as candidate label",
            "history allowlist == features.probe_vector fields (8); static allowlist == features.static_features (8)",
            "secret denylist enforced; feature cache excludes secret",
            "DeepSets hyperparameters/seeds == power-verified values; no confirmatory hp search",
            "K0 and K1 capacity-matched (same model, empty vs 1-probe history)",
            "best-single train/val only, tie-break frozen, not hardcoded",
            "test sealed until models frozen; seal/unseal executable; post-unseal changes forbidden",
            "primary estimator: per-block avg over 2 nominals then 5 seeds; unit=test block",
            "primary event CI_lower>=0.15 (2000 test-block bootstrap); threshold not lowered; comparator unchanged",
            "seed gate >=4/5>=0.15 and none<0 is a PRIMARY necessary gate",
            "collision rule frozen (nonzero-N confounded def + intention-to-analyze primary + sensitivity)",
            "technical-invalidation list frozen; legitimate failures never invalid; no rerun on task outcome",
            "no 306/capability/synthetic reuse in confirmatory training",
            "claim scope wording matches; probe term correct; positive-control boundary stated",
            "markdown and JSON consistent",
        ],
        "deliverable_gate": "C must return GO before A implements the generator; this phase authorizes nothing",
    }


def emit(docs_dir=None):
    docs_dir = docs_dir or _docs_dir()
    os.makedirs(docs_dir, exist_ok=True)
    pre = preregistration_dict()
    with open(os.path.join(docs_dir, "preregistration_v4.json"), "w") as f:
        json.dump(pre, f, indent=2)
    # confirmatory config = the operational subset a generator would consume (still NOT a generator)
    cfg = {"base_commit": BASE_COMMIT, "design": pre["design"], "trial_counts": pre["trial_counts"],
           "model": MODEL, "history_allowlist": HISTORY_ALLOWLIST, "static_allowlist": STATIC_ALLOWLIST,
           "secret_denylist": SECRET_DENYLIST, "best_single": BEST_SINGLE, "primary": PRIMARY,
           "bootstrap": BOOTSTRAP, "collision_rule": COLLISION_RULE, "technical_invalid": TECHNICAL_INVALID,
           "manifest_fields": MANIFEST_FIELDS, "voi": VOI}
    with open(os.path.join(docs_dir, "confirmatory_v4_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(docs_dir, "confirmatory_v4_audit_checklist.json"), "w") as f:
        json.dump(audit_checklist(), f, indent=2)
    return pre


def _docs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "docs", "offline_v2", "calibration_bias")


# consistency guard: the frozen allowlists must equal the REAL feature extractor
def verify_allowlist_matches_code():
    probe_from_code = [k for k in FEAT.PROBE_KEYS]
    assert probe_from_code == HISTORY_ALLOWLIST, (probe_from_code, HISTORY_ALLOWLIST)
    assert FEAT.PROBE_DIM == len(HISTORY_ALLOWLIST) == 8
    assert FEAT.STATIC_DIM == 8
    return True


if __name__ == "__main__":
    verify_allowlist_matches_code()
    p = emit()
    print("emitted preregistration_v4.json / confirmatory_v4_config.json / confirmatory_v4_audit_checklist.json")
    print("status:", p["status"], "| trials:", p["trial_counts"]["full_task_trials_total"],
          "| sessions:", p["trial_counts"]["sessions_total"])
