# One-shot full-repository final audit — calibration-bias preregistration v4 (Claude C)

**Auditor:** Claude C, independent, **read-only, adversarial, one-shot**. No Isaac, no generator, no
formal manifest instance, no checkpoint, no runtime/confirmatory data. Three files added only; B's code,
docs, historical audits, power results, data, and snapshot metadata are untouched.

**Audited snapshot HEAD:** `b983b7e9e638e2ed03589e2bd061139b073ce53f` (fix4 source `f8fedd45`; power-cert
`2bf7217`). **Status:** `FINAL_SNAPSHOT_READY_FOR_ONE_SHOT_FULL_AUDIT`.

## VERDICT: **ONE_SHOT_BATCH_FIX_REQUIRED**

The scientific core, power certification, leakage control, statistics, and all fix1–fix4 guard machinery
are **sound and verified first-hand**. Five issues remain and are **frozen** below (one pre-generator, one
pre-manifest, three nonblocking). **No fatal issue. Power re-certification not required.** After a single
batch fix by Claude B and a regression restricted to `FINAL-001..005`, GO can be granted.

**The issue list is frozen (FINAL-001..005).** Per the charter, only `CRITICAL_NEW_EVIDENCE` (changes a
main scientific conclusion, causes test leakage, makes the primary estimator wrong, makes formal data
non-traceable, or enables safety/data fabrication) may add an item later; ordinary naming/format/style/
non-production-bypass/unused-helper findings may **not** be escalated to new blockers.

## Conclusions by area
- **Scientific claim.** Core proposition sound and powered: history-conditioned selection beats train/val
  best-single on **unseen nominal** test biases (±0.035), with the actual hidden state **in training
  support** (`test [-0.045,-0.025]∪[+0.025,+0.045] ⊂ train [-0.05,+0.05]`). Selection = `argmax p_success`
  and the primary endpoint = **selected success only**; the DeepSets error/time heads are **auxiliary**
  multitask signals and do **not** drive selection or the primary — so the experiment does **not** identify
  a benefit of *multidimensional* prediction over success-only prediction. The frozen claim scope does not
  over-claim this, but it should say so explicitly → **FINAL-004**.
- **Power.** **VALID.** Fix4 touched only guards/validator/hash-fields/docs; bank, ±0.035, residual, probe,
  DeepSets+HP+5 seeds, comparator, block bootstrap, 0.15 CI, seed gate, 300 trials unchanged. Learned power
  0.984–0.986 (separate from structural 1.000) at the test-block unit stands. **No recert.**
- **Leakage.** **Clean.** Production `select_best_single_confirmatory` reads only `{split, session_id,
  trial_role, theta.grasp_offset_local_y, y.success, planned_episode_id}` (secret-perturbation invariant);
  nominal/residual/actual/eff/tau/oracle/test/success-model forbidden + in `SECRET_DENYLIST`; K0 empty
  history / K1 one frozen probe; the simulation-only `best_single_legal` is not the production entry.
- **Manifest/order.** Deep validator now enforces composition, uniqueness, identity↔PID recompute,
  referential integrity, per-session structure, residual bounds, subseed recompute, exec-order, planned
  subset, and frozen seeds — it **rejects** the bogus 228-duplicate manifest and **accepts** a valid 228/72
  fixture (not over-strict). **Remaining:** execution order is not seed-determined (**FINAL-001**); some
  knowable-now integrity values are format-only (**FINAL-002**); one active doc retains stale field names
  (**FINAL-003**).
- **Statistics.** **Correct.** Unit = test block (9); per-block avg over 2 nominals then 5 seeds; gain vs
  same-block train/val-only best-single; 2000 test-block percentile bootstrap; PRIMARY = `CI_lower≥0.15`
  **and** seed gate; intention-to-analyze; collision/VOI/tau strictly secondary.

## Frozen issue list

| id | category | severity | summary |
|---|---|---|---|
| **FINAL-001** | manifest | **pre_generator_blocker** | No canonical subseed→permutation order resolver; candidate/session/block execution order is not uniquely determined by the frozen seed (validator accepts any bank permutation / unique int). |
| **FINAL-002** | manifest | **pre_manifest_blocker** | `config_sha256`, `schema_version`, and the full `deterministic_environment` set are format/presence-only, though knowable at manifest time; should be exact-checked (commits legitimately format-only until seal). |
| **FINAL-003** | docs | nonblocking | `manifest_spec.md` §2.3 (lines 127-129/153) retains stale singular `manifest_hash`/`config_hash`/`code_commit` (config+code already layered). |
| **FINAL-004** | science | nonblocking | Prereg does not explicitly state the error/time heads are auxiliary and that multidimensional-prediction superiority is not claimed/identified (primary = success only). |
| **FINAL-005** | docs | nonblocking | `snapshot_manifest.snapshot_commit` = `1abaa72` (metadata-payload commit) ≠ final audited HEAD `b983b7e`; ambiguous field, no `final_head_commit` recorded. |

Full evidence, affected files, minimal fixes, and machine acceptance tests are in
`one_shot_full_audit_c.json` and encoded in `test_one_shot_full_audit_c.py`.

### FINAL-001 (pre_generator_blocker) — the recurring "manifest uniqueness" gap, settled
Order **subseeds** exist (`session_order_subseed`, `nominal_order_subseed`, `candidate_order_subseed`) and
`manifest_spec §3` says order is "resolved from the order seeds", but **no function maps a subseed to a
permutation**, and the validator checks `resolved_candidate_order` only as *a* bank permutation and
`block_order_key` only as a unique int — never re-derived from the seed (there is no `candidate_order_key`
field at all). So A would have to **invent** the order algorithm when implementing the generator, breaking
the "manifest uniquely determined by the frozen seed" guarantee. **Fix:** freeze one canonical
`resolve_order(order_seed, items) -> tuple` (stable-argsort of items by per-item `subseed(order_seed,
item_identity+'|domain=<order>')`), apply it to candidate/session/block order, and have the deep validator
**re-derive** those orders from the frozen seeds and reject any mismatch (cross-check `execution_order_index`
against them). *Alternative:* if order is scientifically irrelevant (full reset per trial, block-constant
state), fix a single canonical order, drop the order subseeds, and state that order is fixed. This is
classified pre-generator because it is the missing piece of the generator spec, but it does **not** affect
the science, leakage, or power.

## Verified-pass (batch)
Scientific core & support relation; power certification valid; leakage clean (observed-only selector,
denylist, K0/K1 isolation); single production entry + 45/12 composition + strict offset domain; identity
domain strict + 300 unique; deep validator rejects bogus / accepts valid; combined hash requires all 8
fields; correct block-level statistics + seed gate + intention-to-analyze; model/determinism frozen;
seal Scheme 2 unique; 306 data immutable; 300 trials; no confirmatory data/manifest/checkpoint.

## Tests
- **B fix4 suite** `test_preregistration_v4_fix4.py` → **21 passed**; snapshot `test_report.md` records
  244 B-owned passing.
- **C one-shot audit** `test_one_shot_full_audit_c.py` → **6 passed, 5 xfailed** (FINAL-001..005 frozen).

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_generator_smoke | **false** |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

**Next step:** Claude B performs ONE batch fix of FINAL-001..005; Claude C then regresses **only** those
five IDs (via the acceptance tests). On green, the verdict becomes `GO_TO_GENERATOR_IMPLEMENTATION`
(generator implementation + `smoke_only=true` isolated artifacts) — formal manifest, checkpoints, and the
300-trial confirmatory run remain unauthorized and require their own gates. The ±0.035 power certification
carries over unchanged.
