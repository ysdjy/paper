# Global reference scan (Fix4 snapshot)

Search across the tracked calibration_bias dirs (`docs/offline_v2/calibration_bias`,
`deployment_calibration/offline_v2/calibration_bias`,
`deployment_calibration/tests/offline_v2/calibration_bias`). Counts are total matching lines (includes
historical/summary/audit files). Classification distinguishes **active** (the 10 active docs + 4 modules)
from historical/audit-only. **Nothing was modified in this task.**

| term | total lines | active-file hits | classification |
|---|---|---|---|
| `select_best_single` | 112 | prod entry `select_best_single_confirmatory` + private `_select_best_single_core` | active-valid |
| `best_single_legal` | 45 | `learned_selector_power.py` (SIMULATION_ONLY_REFERENCE) + docs describing it | active-valid (not production) |
| `validate_completeness` | 25 | private core kwarg only; production entry has none | active-valid |
| `canonical_manifest_hash` | 15 | `confirmatory_v4_identity.py:246` deprecated **alias** to `canonical_planned_structure_hash` (+ `:278` __main__ print) | active-valid (documented alias) |
| `manifest_hash` | 68 | `manifest_spec.md §2.3 lines 127` (**stale**), plus meta-refs ("purged of singular manifest_hash", "no single ambiguous manifest_hash") | **active-potential-conflict** (see known_issues §A) |
| `config_hash` | 5 | `manifest_spec.md:128` (**stale** field list) | **active-potential-conflict** (known_issues §A) |
| `code_commit` | 7 | `manifest_spec.md:129, :153` (**stale**; layered set uses `runtime_commit`/`protocol_commit`/`generator_commit`) | **active-potential-conflict** (known_issues §A) |
| `science_manifest_sha256` | 33 | active spec/config use per-phase anchor (train_validation_/test_manifest_sha256) | active-valid |
| `block=<i>` | 18 | active hits are only fix4 **purge-descriptions** in preregistration_v4.md / audit_checklist / preregistration_v4.py; NO active usage as a subseed key | active-valid (meta only) |
| `candidate_order_key` | 0 | absent from active schema (candidate order = `resolved_candidate_order`) | historical-only / n/a (see known_issues §B) |
| `resolved_candidate_order` | 17 | validator checks bank-permutation only | active-valid (depth question: known_issues §B) |
| `block_order_key` | 10 | validator checks non-bool-int + unique only | active-valid (depth question: known_issues §B) |
| `actual_bias` | 91 | SECRET_DENYLIST entries + audit/analysis prose (forbidden model input) | active-valid (denylist) |
| `residual_bias` | 34 | denylist / analysis prose | active-valid (denylist) |
| `eff_signed` | 41 | denylist / 306 schema / analysis prose | active-valid (denylist / data field) |
| `tau_minus_abs_eff` | 3 | SECRET_DENYLIST label ("tau_minus_abs_eff (secret continuous margin)") | active-valid (denylist) |
| `confirmatory run` | 19 | hard-constraint / claim-scope prose ("no confirmatory run authorized") | active-valid |

## Active-potential-conflicts (for the one-shot audit)
Only one active-file class of conflict: the **stale singular field names** (`manifest_hash`, `config_hash`,
`code_commit`) in `confirmatory_v4_manifest_spec.md` §2.3 (lines 127–129) and line 153. The machine config
(`confirmatory_v4_config.json → manifest_fields`) and all code are already on the fix4 layered set. See
`known_issues_at_freeze.md` §A. **Not** modified here (snapshot rule: record, do not fix).

## Everything else
All other hits are active-valid (production selector, deprecated-alias, denylist labels, per-phase anchors,
purge-description meta-text) or historical/audit-only (in fix1–fix4 summaries and C audit reports, which are
retained unmodified).
