# Split & statistics audit (Audit F)

**Auditor:** Claude C. From first-hand reading of `splits.py`, `oracle.py`, `voi.py`, `history.py`,
`validator.py`, and the confirmatory design artifacts.

## 1. Bootstrap unit = the nuisance block — **correct**
`block_bootstrap_ci_scheme.resample_unit = "independent nuisance block within the split"`, 2000 reps over
the **9 test blocks**. A test block = one residual applied to both test biases {−0.03,+0.03} × 7 offsets +
probes. Resampling **whole blocks** preserves the within-block cross-bias and cross-offset correlation
induced by the **shared residual** — this is the right independent unit. Resampling sessions or rows would
break that correlation and understate CIs; the design does not do that. ✓

## 2. Split isolation — **enforced** (`audit_split`)
Pairwise-disjoint **bias levels**, plus disjoint **sessions**, **nuisance seeds**, and **nuisance blocks**
across train/val/test; partition (no drop/dupe); each test bias **bracketed by train** on both sides
(interpolation). Blocks may pair *within* a split (matched context) but never *cross* a split. Since each
block has a distinct seed→distinct residual, residual draws are train/val/test-disjoint. ✓

**Caveat (semantic, from Audit A):** the split holds out the **nominal** label, not the **actual** hidden
state (test actual ⊂ train actual support). The split is a valid *nominal-condition* interpolation split;
it is **not** a held-out-physical-state split. Report accordingly.

## 3. Best-single decided by train only — **primitive supports it; ingest must enforce it**
`oracle.best_single_offset(train_groups, test_groups, …)` correctly selects the offset on **train** utility
and evaluates on **test**. **But** `decision_value(...)` (used for the exploratory capability-map analysis)
calls it **in-sample** (`groups, groups`). That is acceptable for exploration, but the **confirmatory
ingest pipeline (not yet built) MUST pass `train_groups` for selection and never let test influence the
best-single choice, model fitting, thresholds, or λ.** This is a required check when the ingest is
implemented; it cannot be certified now because no confirmatory pipeline processes data yet.

## 4. Test does not touch model/threshold/hyperparameter selection — **design intent, to be enforced**
The preregistration states λ frozen, best-single on train, GO thresholds fixed. The **code path that will
consume confirmatory data does not exist**, so this is a spec to enforce, not a fact to certify. Flag for
the ingest smoke test: assert no test session id appears in any training/selection call.

## 5. History leakage-safety — **enforced** (reused, audited primitive)
`validator` builds each candidate's history via `build_history` + `assert_history_legal`: same-session,
strictly-earlier **probes only**; candidate outcomes never enter history; probes carrying candidate fields
are rejected at source. Poison unit tests pass. Candidate-outcome-in-history is structurally blocked. ✓

## 6. K semantics — **correct**
- **K=1** uses **only `probe_m040`** (first probe, idx 0); **K=2** adds `probe_p040` (idx 1) in **fixed
  order**; K=0/1/2 reported; **no third probe**; stop rule K=1. Verified in `run_confirmatory_design_v3`
  (`PROBES=[("probe_m040",−0.04),("probe_p040",+0.04)]`, sliced `PROBES[:K]`). ✓
- Note (from Audit C): the K-curve classifier uses the **hard band 0.02**, consistent with
  design-validation but inconsistent with the power sim's soft 0.03 — see the power audit.

## 7. Seeds reporting plan — **adequate in spec, verify at run**
The preregistration requires **≥5 model seeds** and **stability across split seeds** (GO §4(5)), with
block-bootstrap CIs. That is an adequate plan. Two items to enforce at analysis time:
- report the full seed grid (model × split) and the **per-seed** gain/VSI, not just the mean;
- the independence blocker (`independence.audit`) must be **re-run on the confirmatory data** and pass
  (`blocker==False` **with genuine outcome variance**) — this ties directly to the Audit-C variance risk:
  if the residual produces no operating-point variance, the blocker's spirit (though maybe not its literal
  trigger) is violated and CIs are degenerate.

## Verdict
The split and bootstrap **design** is statistically sound and correctly implemented in the primitives:
block-level resampling, full split isolation, leakage-safe history, correct K semantics. The **open
enforcement items** are all in the **not-yet-built confirmatory ingest**: (a) best-single/model/threshold
selection strictly train/val-only; (b) re-run leakage + independence validators on the real data;
(c) surface per-seed results. No statistical-methodology defect found; the binding risk is the
**variance-degeneracy** carried over from Audit C, not the split logic itself.
