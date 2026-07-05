# Known issues at freeze — for the one-shot full audit (NOT fixed in this snapshot task)

These are observations recorded at the Fix4 freeze for Claude C's one-shot full audit to adjudicate. **None
was modified in this snapshot task.** They do **not** change any legal-input scientific selection, the
design, or the power certification. Line numbers are against the snapshot tree (Fix4 `f8fedd45`).

---

## A. Active manifest-spec residual stale field names (CONFIRMED)
`docs/offline_v2/calibration_bias/confirmatory_v4_manifest_spec.md`, section **§2.3 "Other required manifest
fields"**, still lists the pre-fix4 singular field names:
- line **127**: `manifest_hash        # sha256 of the fully-resolved manifest (pre-run)`
- line **128**: `config_hash          # sha256 of confirmatory_v4_config.json`
- line **129**: `code_commit          # git commit of the analysis code`
- line **153** (per-trial provenance prose): `` `science_manifest_sha256`, `code_commit` `` list.

The fix4 layered system (and `MANIFEST_FIELDS` in `preregistration_v4.py` / `confirmatory_v4_config.json`)
uses instead: `planned_structure_sha256, train_validation_manifest_sha256, test_manifest_sha256,
model_analysis_freeze_sha256, combined_experiment_plan_sha256, config_sha256, protocol_commit,
generator_commit, runtime_commit, analysis_code_sha256`.

Why it survived: C's fix3-reaudit blocker-d test only checked for `block=<i>`, the literal `== frozen
manifest_hash`, and a config-JSON `"manifest_hash"` key — it did **not** scan this prose field list.
- **Scope:** documentation-only inconsistency in one active file. The machine config
  (`confirmatory_v4_config.json → manifest_fields`) and code are already the layered set (verified: no
  singular `manifest_hash` in config/preregistration JSON). Recommend the full audit direct a one-line doc
  fix (§2.3 field list + line 153) in a later pass; **not** fixed here per the snapshot no-code-change rule.

---

## B. Deep manifest order-validation depth (design question for audit)
`confirmatory_v4_manifest_integrity.validate_fully_resolved_phase_manifest` currently:
- `resolved_candidate_order`: checked to be a **length-3 bank permutation** only. It is **not** recomputed
  from a candidate-order subseed, and it is **not** cross-checked against the per-session trial execution
  order.
- `block_order_key`: checked to be a **non-bool int and globally unique** only; **not** recomputed from a
  frozen seed.
- `execution_order_index`: checked **unique + complete 0..N-1 + probe-before-candidates**; **not** checked
  for consistency with the block/session/nominal/candidate order keys.
- active schema: there is **no `candidate_order_key`** field (global scan: `candidate_order_key` = 0 hits);
  candidate ordering is represented only by `resolved_candidate_order`.

Recomputed-from-seed today (subseeds trusted only if they match `confirmatory_v4_identity`): block residual
subseed, block nuisance subseed, session-order subseed, nominal-order subseed, trial-init subseed. The three
"order key / candidate order" items above are the gap between "permutation/uniqueness checked" and "uniquely
recomputed from the frozen seed." For the audit to decide: is permutation+uniqueness sufficient, or must the
resolved candidate order + block/session order be deterministically re-derived from `candidate_order_seed`
etc. (which would require freezing a canonical order-resolution function, not currently defined)?

---

## C. Frozen-value vs format-only validation (design question for audit)
In the same deep validator, the following are checked as noted:
- **config SHA256** (`config_sha256`): **format only** (64-lowercase-hex regex). Not compared to the actual
  hash of `confirmatory_v4_config.json`.
- **protocol / generator / runtime commit**: **format only** (40-hex regex). Not compared to a frozen
  expected commit value (the true commits are unknown until the post-GO generator/run exists).
- **schema_version**: **presence + hashed only**; the value is not validated against a frozen expected.
- **deterministic_environment**: only `device == "cpu"` is checked; the full field set
  (`set_num_threads`/`set_num_interop_threads`/`use_deterministic_algorithms`/`OMP/MKL/OPENBLAS`) is **not**
  required/matched by the validator (though it is frozen in `determinism_env` of the config).
- **frozen_seeds**: **exact-value** check (`== preregistration_v4.frozen_seeds()`).
- **manifest_algorithm_version**: **exact-value** check.
- **planned_structure_sha256**: **exact-value** check (`== canonical_planned_structure_hash()`).

For the audit to decide whether config SHA / commits / schema_version / full determinism env should be
exact-frozen-value checks (some, like commits, are only knowable at generation time and may legitimately
remain format-only-until-freeze; the generator freeze step, seal protocol Step 0/5, is where the concrete
values are pinned).

---

## D. C historical audit test files show expected xfail-flips / superseded PASS
The four `test_preregistration_v4_*_c.py` files are **audit-only** and, when run against the fixed tree, show
strict-xfail→XPASS ("failures") and a few superseded PASS assertions. These are the **intended** machine
signals that each round's blockers were fixed and are **not** active failures (see `test_report.md`).

---

## Non-issues (verified clean)
- 306 raw data unchanged (episodes.jsonl sha256 `131750e1…9524c20e`; dir sha256 `718e0a70…c68844f`).
- No singular `"manifest_hash"` in `confirmatory_v4_config.json` / `preregistration_v4.json` (only the fix4
  purge-description meta-references remain, which are correct).
- `canonical_manifest_hash` in `confirmatory_v4_identity.py` is the intentional **deprecated alias** to
  `canonical_planned_structure_hash` (structure-only), documented as such.
- `best_single_legal` references are the `SIMULATION_ONLY_REFERENCE`; the production entry is
  `select_best_single_confirmatory`.
