# Calibration-bias stage — preregistration DRAFT v1

Claude B, branch `experiment/offline-calibration-bias-v1`. This is a **draft** protocol prepared
BEFORE any capability-map or confirmatory data exists. It fixes the exploration/confirmation
boundary, the hidden variable, the model-legal inputs, the frozen main metric, the exploration gate,
the confirmatory split, and the candidate/probe/independence rules. Nothing here has seen real
outcomes. The `preregistration_v1` (final) will be committed only after Claude A's exploratory
capability map is read and a single candidate/bias/probe freeze proposal is issued.

Rationale for this stage: the damping stage was a clean MODIFY — history identified the state and the
landscape re-ranked, but a **robust generalist** (`c1_steady`) made the state's decision value ≈ 0
(VSI ≈ +0.0006). Calibration bias is chosen because the hidden `bias_y` interacts **directly** with a
controllable (`grasp_offset_local_y`): the best offset should move with the bias and no single offset
can compensate the whole range, so state information should have genuine decision value **by physics,
not by pruning the candidate bank**.

---

## 1. Exploration vs confirmation — a hard wall

**Exploratory capability map (Claude A, then B reads read-only).**
- Purpose ONLY: choose bias range, offset grid, probe set, and sanity-check that the compensation
  structure exists. May be re-run and inspected freely.
- MUST NOT be used as confirmatory hypothesis-test data and MUST NOT be pooled with the confirmatory
  pilot when computing any reported result.

**Confirmatory pilot (frozen before generation).**
- bias levels, candidate (offset) bank, probe set, utility, split, and the GO/MODIFY/STOP thresholds
  are frozen in `preregistration_v1` BEFORE the confirmatory data is generated.
- After the confirmatory test data exists, the main metric, thresholds, and split are immutable.

B never asks A to launch the confirmatory run; that waits for explicit user confirmation.

## 2. Hidden variable and legal inputs (enforced by `validator.py`)

Hidden state: **`bias_y`** = the handle local-Y calibration bias the controller actually uses (m).

A deployable model MAY read: observable `x` (existing contract keys), `g`, candidate `theta`
(especially `grasp_offset_local_y`), the leakage-safe same-session earlier-probe history `H`, and
recorded **observable** nuisance features.

A deployable model MUST NOT read: `bias_y`, `bias_id`/`bias_level_id`, `secret_deployment_state`, any
effective / ground-truth calibration error, future probes, candidate outcomes, other-session history,
or the **raw nuisance seed** (provenance, not a feature).

Validator + poison tests enforce: bias/secret/calib keys never in `x`; raw seed never in `x`; a probe
never carries candidate-only fields; every candidate's history is same-session, strictly earlier,
whitelisted-field probes only; and the nuisance seed is not a one-to-one function of bias.

## 3. Preregistered scientific questions (report the four levels separately)

- **H1 Prediction** — does history improve candidate-outcome prediction under a **capacity control
  that can represent the task**? (see §8 caveat: the honest control is DeepSets K=0 vs K>0, not
  linear B1 vs B2-mean, because compensation success is a non-monotonic band in offset).
- **H2 Ranking** — does a different bias change the ranking of candidate offsets (switch / reversal)?
- **H3 Decision value** — does a state-aware selector beat the train-selected **best-single offset**
  and any robust-generalist baseline, moving toward the state-aware oracle? Reported under the
  **frozen utility AND success-only** (co-primary).
- **H4 Probe value** — is Gross VOI(K) > 0, and after charging real probe time at the frozen λ_time,
  is **Net VOI(K)** still > 0?

Prediction, ranking, selection, and decision value are reported as distinct claims and never
conflated.

## 4. Main metric and frozen utility

Primary utility (unchanged from the damping stage, frozen):

    U = success − 1.0 · task_error − 0.02 · time

**Co-primary: success-only.** Because calibration bias should change grasp SUCCESS (not merely time),
`U_success = success` is reported alongside frozen `U` as a co-primary. A decision-value claim must
hold on success-only, not only via the time term.

### Frozen exploration gate (GO-to-preregistration) — thresholds fixed now
These decide whether the design is worth taking to a confirmatory pilot; they are **design gates, not
final significance**. Implemented in `pipeline.exploration_gate` / `GATE`.
1. **≥ 3 bias levels have a different best offset** (`n_distinct_best_offsets_across_bias ≥ 3`).
2. **No robust generalist**: no single offset is within `0.02` utility of the per-bias best at every
   bias (`robust_offset.robust_generalist_exists == False`).
3. **State-aware vs best-single**: selected-success absolute gain **≥ 0.15** (success-only) OR
   frozen-utility **VSI ≥ 0.05**.
4. **Replicate-independence audit passes** (`independence.blocker == False`).
5. **Pairing / leakage / provenance all pass** (`validator.ok == True`).

All five required. These thresholds are frozen in this draft and will not be relaxed after seeing data.

## 5. Confirmatory split (avoids discrete-state memorisation) — `splits.py`

- **≥ 5–7 continuous bias levels.**
- Some bias levels **train-only**; some intermediate levels **validation**; **≥ 1–2 unseen INTERIOR
  levels test-only** — chosen so each test level is **bracketed by train levels on both sides**
  (measures interpolation across the bias range, not extrapolation past its edges); extremes stay in
  train to anchor the range.
- **Double isolation**: split by bias level AND by session; a level's sessions never straddle splits.
- **Nuisance seeds are disjoint across splits** and independent of bias; the same deterministic seed
  never crosses a split.
- Output `split_manifest.json` (level ids + values + sessions + seeds per split) with a **pairwise-
  disjoint + partition + interpolation** audit; unit-tested.

## 6. Candidate and probe preregistration principles

**Candidates (offsets).**
- A **matched offset bank**: the same offset-id set with identical θ across ALL bias levels,
  sessions, and targets (validator-checked).
- The offset grid must **cover the full compensation range** implied by the bias range (so the best
  offset for every bias is representable).
- **No offset may be removed after the confirmatory test** because it is "inconvenient".
- The **best-single-offset** baseline is chosen ONLY on train/validation.

**Probes.**
- Fixed **2–3 offsets**, identical across all sessions, executed before any candidate.
- Report the **K = 0,1,2,3 adaptation curve**; record the **real** probe elapsed time.
- **Net VOI** charges probe time at the frozen λ_time.

## 7. Independent-sample audit rules — `independence.py`

Reports: session-level nuisance provenance (distinct seeds? varied within a bias level?), per-cell
(bias × target × offset) replicate variance, near-duplicate outcome fingerprints, and effective
sample counts. **Blocker rule:** if same-bias sessions are near-deterministic clones AND nuisance
seeds are absent or not varied within a level, `blocker = True` → formal CIs may **not** be narrowed
by session count and **GO is disallowed**. (This is exactly the damping-stage failure mode we refuse
to repeat.) Formal CIs bootstrap over independent nuisance seeds / bias-level blocks, not raw sessions.

## 8. Models and baselines — `pipeline.py` (reuses the frozen framework)

B0, B1 static, B2 mean, B2 DeepSets, B2 GRU, Oracle-Z, best-single offset, robust-generalist (if one
exists), Oracle-Candidate, state-aware oracle, state-agnostic oracle.

Report: B1 vs B2-mean pure-history gain, DeepSets/GRU **K=0 vs K>0**, held-out **bias-level**
generalization, per-bias / per-offset / per-target metrics, **success-only AND frozen VSI**,
switch / rank-reversal / regret / VOI.

> **Capacity-control caveat (discovered on synthetic).** Compensation success is a **band**
> (`|offset + bias| ≤ tol`), which is **non-monotonic** in the offset. A linear model (B1, B2-mean)
> *cannot represent it regardless of history*, so the linear B1→B2-mean comparison **understates**
> the history effect (on synthetic, both sit near chance while nonlinear DeepSets reaches AUROC 0.93).
> The honest, preregistered capacity control for H1 is therefore **DeepSets K=0 vs K>0** (identical
> capacity, isolating history). We will additionally report a **feature-enriched linear** baseline
> (adding `|offset|`, `offset²`, and offset×history terms) so a capacity-matched *linear* comparison
> is still available, but the primary H1 evidence is the DeepSets K-sweep.

## 9. Synthetic validation (SYNTHETIC ONLY — never a paper result)

`offline_v2/calibration_bias/synthetic.py` builds a controllable compensation world
(`effective_error = offset + bias`, best offset = −bias, no robust offset). The pipeline recovers, on
this synthetic (`synthetic_only: true` stamped on every episode and artifact):
- validation PASS, split interpolation + seed-disjoint PASS, independence non-blocking;
- **frozen VSI ≈ 0.59**, **success-only VSI ≈ 0.57**, switch 1.0, rank-reversal 0.42, no robust offset;
- on **unseen interior biases**: DeepSets/GRU **select the compensating offset (selSucc ≈ 1.0)** while
  B0/B1/B2-mean/OracleZ/best-single all **fail (selSucc 0.0, regret ≈ 1.07)** — the target decision
  value;
- DeepSets **K=0 AUROC 0.39 → K≥1 ≈ 0.93** (history reveals the bias / band centre);
- **Net VOI** positive at K=1 (+0.43) and **decaying to ≈ 0 by K=3** as probe time is charged — a
  clear Net-VOI boundary.
A negative control (wide tolerance + flat error) yields `robust_generalist_exists = True` and VSI ≈ 0,
and a deterministic-replicate control trips the independence **blocker** — so the gate discriminates
positive from negative worlds. See `tests/offline_v2/calibration_bias/` and the synthetic artifact set.

## 10. Ingesting Claude A's data (when it arrives)

1. Read A's exploratory capability map **read-only** by absolute path; record `source_run_path`,
   `source_git_commit`, `candidate_bank_sha256`, `dirty_worktree`, `damping_verified`/reset flags.
2. Run, in order: **validator → independence → decision-value / exploration gate**.
3. Issue **one** freeze proposal (bias levels, offset bank, probe set, split) — no iterative tuning.
4. Commit `preregistration_v1` (final) BEFORE any confirmatory data is generated.
5. Do **not** request the confirmatory run; wait for user confirmation.

If the capability map fails the gate (e.g. a robust offset exists, or VSI/success-gain below
threshold), report a blocker and recommend a design change (wider bias range so no offset covers it,
or tighter grasp tolerance) rather than proceeding.
