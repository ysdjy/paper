# Selection-interaction pilot — offline analysis v1

Claude B, branch `experiment/offline-eval-v2`. Independent offline analysis of Claude A's
matched-candidate selection-interaction pilot, under the **frozen** utility
`U = success − 1.0·task_error − 0.02·time`. Every number is reproducible from `episodes.jsonl`
via `deployment_calibration/evaluation/offline_v2/run_selection_interaction.py`; no Isaac, no
runtime edits, no writes to A's tree.

Conclusions are separated into four levels throughout: **prediction**, **ranking**, **selection**,
and **decision value**.

## Provenance & integrity
- source (read-only): `.../data/selection_interaction_pilot_v1_20260703_110727/`
- source git commit / runtime_design_commit: `1921641acec5a7e153f45f909c69613ed6036a12`
- candidate_bank_sha256: `d1d80d9646aae536f72dd3a7edb0bec764c1304b9ced6df5bd81f093256d7516`
- dirty_worktree: `false`; damping_verified: `true`; full_reset_all_verified: `true`.
- 162 episodes = 27 probes + 135 candidates; 9 sessions (3 damping × 3 replicates);
  27 selection groups (session × target); **9 matched groups** (target × replicate, spanning 3 damping).

### Pairing / leakage validation — ALL PASS (`validation_report.json`)
- episode_count 162 ✓; matched bank ✓; **9 matched groups** ✓
- every matched group has the identical candidate-id set at all 3 damping levels, with identical
  θ per candidate-id, single target ✓
- no candidate outcome enters any candidate's decision-legal history ✓
- no `secret`/`hidden`/`damping` key inside any `x` ✓
- 27 selection groups present ✓

### Replicate independence — technical repeats (see `replicate_independence_audit_v1.md`)
Discrete outcomes identical across all 3 replicates in 45/45 cells; only **3 unique hidden states**.
CIs over sessions are therefore not i.i.d.; decision-value numbers are descriptive, and any CI is a
coarse **target-block** bootstrap flagged EXPLORATORY.

---

## 1. Prediction (does history improve outcome prediction?) — YES, capacity-controlled

Test fold, split seed 0, 5 model seeds (from `metrics.json` / `metrics_ci.json`):

| model | AUROC | AUPRC | Brier | ECE | fail-macroF1 | errMAE | timeMAE |
|---|---|---|---|---|---|---|---|
| B0 baserate | 0.500 | 0.622 | 0.235 | ~0 | 0.384 | 0.060 | 0.470 |
| **B1 static** | **0.872** | 0.939 | 0.152 | 0.137 | 0.669 | 0.038 | 0.436 |
| **B2 mean (capacity-matched)** | **1.000** | 1.000 | 0.022 | 0.088 | 0.977 | 0.032 | 0.378 |
| B2 deepsets | 1.000 | 1.000 | ~0 | ~0 | 1.000 | 0.002 | 0.030 |
| B2 gru | 1.000 | 1.000 | ~0 | ~0 | 1.000 | 0.002 | 0.020 |
| OracleZ (sees damping) | 0.968 | 0.983 | 0.072 | 0.113 | 0.855 | 0.031 | 0.435 |

(Exact values in `main_results.csv`, 5 model seeds, session-bootstrap CIs in `metrics_ci.json`.)

**Capacity control (the honest history signal).** B1 and B2-mean share the SAME linear head; the
only difference is the mean-pooled probe history. AUROC **0.872 → 1.000** and Brier **0.152 → 0.022**
is therefore a genuine history effect, not model capacity. The **adaptation curve** (`adaptation_curve.csv`)
is textbook: B2-mean AUROC **K=0: 0.872 (= B1) → K=1: 0.947 → K=2: 1.000 → K=3: 1.000** — a single
probe already lifts prediction and a second saturates it. DeepSets/GRU reach the same ceiling with more
capacity but do not exceed the capacity-matched signal in a way that matters (the task is
near-deterministic once damping is known).

**Caveat (determinism).** Because damping → outcome is deterministic and replicates are technical
repeats, "AUROC 1.00" partly reflects that once history reveals the damping level, success is fixed.
The meaningful, defensible claim is the **B1→B2 gap under matched capacity**, not the absolute 1.00.

**Level reached: prediction ✓.**

## 2. Ranking (does the hidden state re-order the candidates?) — YES

From the matched bank (`decision_value.json`):
- **Optimal-candidate switch rate = 0.667** (6 / 9 matched groups change their argmax-true-utility
  candidate across damping). By target: T012 never switches; T020 and T028 switch c2_balanced (low
  damping) → c1_steady (mid/high), × 3 replicates = 6.
- **Pairwise rank-reversal rate = 0.170** (46 / 270 candidate-pair × damping-pair orderings flip).

The landscape genuinely re-ranks: aggressive candidates (c2/c3/c4) are competitive at low damping
but **collapse to failure** at mid/high damping (utility ≈ −0.3), while c0/c1 stay reliable.

**Level reached: ranking ✓.**

## 3. Selection (does a history model pick better than baselines?) — YES vs naive, NO vs robust generalist

Held-out test (9 groups, split seed 0, `selection_heldout.json`); state-aware oracle U = 0.7958:

| policy | mean regret | top-1 | selSucc | gap to state-aware oracle |
|---|---|---|---|---|
| B0 baserate | 0.0263 | 0.00 | 1.00 | 0.0266 |
| B1 static | 0.0263 | 0.00 | 1.00 | 0.0266 |
| B2 mean | 0.0211 | 0.11 | 1.00 | 0.0214 |
| B2 deepsets | **0.0000** | 0.89 | 1.00 | 0.0003 |
| B2 gru | 0.0000 | 0.89 | 1.00 | 0.0003 |
| OracleZ | 0.0211 | 0.11 | 1.00 | 0.0214 |
| **robust generalist (always c1_steady)** | **0.0007** | — | 1.00 | **0.0009** |
| best single archetype (train-selected = c1_steady) | 0.0007 | — | 1.00 | 0.0009 |

Two facts dominate:
- **Every policy, including B0, has selSucc = 1.00.** c0/c1 succeed at every damping, so no selector
  ever fails; utility differences are pure speed/error and are tiny.
- **The trivial robust generalist "always c1_steady" has regret 0.0007 ≈ the oracle**, and the
  train-selected best-single archetype IS c1_steady. So B2's headline "regret 0.000 vs B1 0.026" is
  real but measures B2 beating a *poorly-chosen* learned baseline; against the *robust generalist*
  B2's advantage is ≈ 0.0006 utility.

**Level reached: selection ✓ vs naive baselines, ✗ vs a robust generalist.**

## 4. Decision value (is knowing the hidden state worth anything?) — NO (VSI ≈ 0)

Frozen-utility decision-value table (`decision_value.json`, full 27 groups / 9 matched groups):

| quantity | value |
|---|---|
| Oracle-Candidate mean utility | 0.7958 |
| State-agnostic Oracle mean utility | 0.7952 |
| State-aware Oracle mean utility | 0.7958 |
| **VSI = state-aware − state-agnostic** | **+0.00059** |
| robust-generalist gap to state-aware oracle | +0.00059 |
| optimal-candidate switch rate | 0.667 |
| pairwise rank-reversal rate | 0.170 |

Independently recomputed VSI **+0.00059** matches A's reported +0.0006. The robust-generalist gap
equals the VSI exactly, confirming **c1_steady is the state-agnostic optimal policy** and knowing
damping adds ~0.0006 utility on a ~0.8 scale (< 0.1%).

**Argmax switch ≠ meaningful gain.** The switches (c2→c1) occur where c2 and c1 differ by ~0.005
utility at low damping; committing to c1 everywhere loses essentially nothing. So a high switch rate
coexists with VSI ≈ 0.

Target-block bootstrap of VSI (resample 3 targets, EXPLORATORY): CI reported in
`decision_value.json` — near zero, as expected with 3 blocks.

**Level reached: decision value ✗ — state information has no meaningful selection value here.**

---

## 5. Value of probing (exploratory) — `voi.json`

Gross VOI(K) = selected true utility with K probes − robust-generalist utility. Net VOI(K) subtracts
the real cumulative probe time at λ_time = 0.02. Because the robust generalist already ≈ oracle,
**Gross VOI is ≈ VSI (~0.0006) and Net VOI is strongly negative** once the seconds spent probing are
charged at the frozen time weight — i.e. under the frozen utility, running probes to identify damping
costs far more than the selection benefit. (Exploratory; does not define a new utility.)

## 6. Utility sensitivity (supplementary — main result stays frozen) — `utility_sensitivity.json`

| utility | VSI | robust-gen gap | robust succ |
|---|---|---|---|
| success only | 0.0000 | 0.0000 | 1.00 |
| success + error | 0.0000 | 0.0000 | 1.00 |
| **frozen main (λ_time 0.02)** | **0.0006** | 0.0006 | 1.00 |
| λ_time 0.05 | 0.0046 | 0.0046 | 1.00 |
| λ_time 0.10 | 0.0115 | 0.0115 | 1.00 |
| λ_time 0.20 | 0.0254 | 0.0254 | 1.00 |
| λ_time 0.50 | 0.0670 | 0.0670 | 1.00 |

Diagnosis: VSI is ~0 under success and success+error because **c1_steady never fails and is nearly
the most accurate everywhere** — the low decision value is a **candidate-bank structural property**
(a robust generalist exists that is both reliable and fast), not merely light speed weighting. VSI
grows with λ_time but even at λ_time = 0.5 (25× the frozen value) it is only 0.067 and the robust
generalist still succeeds 100%. **We do NOT adopt any larger λ_time as the main metric**; this grid
is purely diagnostic per the plan.

---

## Bottom line
- **prediction**: history clearly improves outcome prediction (capacity-matched B1 0.87 → B2 1.00).
- **ranking**: the hidden state re-orders candidates (switch 0.67, reversal 0.17).
- **selection**: a history model beats naive learned baselines but only ties a trivial robust generalist.
- **decision value**: knowing the hidden state is worth **VSI ≈ +0.0006** — negligible — because the
  candidate bank contains a robust generalist (c1_steady) that is near-optimal at every damping.

This is a clean, honest **MODIFY**: *state information improves prediction, but has no decision value
against a candidate set that already contains a robust generalist.* See `final_go_modify_stop_v1.md`.
