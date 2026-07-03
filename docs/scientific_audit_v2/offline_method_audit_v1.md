# Offline Method Audit — Claude B Phase-1 framework + reanalysis (v1)

**Auditor:** Claude C (independent). **Scope:** the `offline_v2` / `models_v2` framework and the
72-episode reanalysis (`docs/offline_v2/pilot_reanalysis_v1.md`) on branch
`experiment/offline-eval-v2` (HEAD `98c418c96a`). No code/data/utility modified. I read the code,
re-ran the unit tests, and **ran B's unmodified pairing/oracle/utility on Claude A's matched pilot**
(the Phase-2 computation B has not yet pushed).

**Framing:** B's Phase-1 numbers were computed on the **old, unmatched** `damping_pilot_v2` data — not
A's matched pilot. So this audit grades (a) the *method/code*, and (b) the old-data reanalysis; the
matched-data decision-value numbers come from my independent run of B's code (§4).

---

## 1. Method correctness — the definitions are right

| component | check | verdict |
|---|---|---|
| **AUROC** (`metrics.auroc`) | tie-aware via `sklearn.roc_auc_score`; constant→0.5; single-class→NaN+warn; numpy fallback = Mann-Whitney avg-rank | **correct** |
| **Split** (`splits.audit_split`) | pairwise-disjoint + partition (no drop/dup) + candidate-group-within-one-split; session-level, stratified by (drawer, hidden_state_id) | **correct** |
| **Leakage builder** (`history.build/assert`) | whitelist fields only; rejects future/cross-session/candidate-field/privileged-key/over-cutoff; validates each entry vs canonical legal probe | **correct** (poison tests 13/13 pass, re-run) |
| **Utility** (`utility.py`) | `U = p − λ_err·err − λ_time·time`, λ frozen (1.0/0.02), clip bounds fixed, `selected_on="validation"` | **correct**, one label caveat (§5) |
| **VSI / switch / reversal** (`oracle.py`) | definitions below | **correct** |
| **Pairing** (`pairing.build_matched_bank`) | data-driven matched-group construction; returns not-matched with reason on unmatched data | **correct** |
| **Capacity control** (`models_v2`) | B1 and B2_mean share one logistic+ridge head; B2_mean only adds `history_mean(H)` | **genuine capacity match** |

**VSI is a sound, model-independent decision-value ceiling.**
`VSI = mean_true_U(state-aware oracle) − mean_true_U(state-agnostic oracle)`, both using **realized**
(true) utility. State-aware picks argmax-U per damping; state-agnostic commits to one candidate
(argmax mean-over-damping U) scored at every damping. Properties I verified:
- VSI ≥ 0 by construction (per-cell max ≥ any fixed choice).
- The in-sample state-agnostic choice is the *strongest possible* single candidate, so VSI is
  **conservative** (as small as the data allow) — it cannot over-state state value. Good.
- Switch-rate and rank-reversal use strict orderings with a 1e-9 tie tolerance (ties are skipped, not
  counted as reversals). Correct.

**Switch/reversal vs VSI — a definitional caution the paper must honor.** Switch-rate and rank-reversal
measure whether the candidate *ordering* moves with state; **only VSI measures whether moving with it is
worth anything.** They can diverge arbitrarily (a switch between two ε-apart candidates inflates
switch-rate while VSI≈0). The code keeps them separate and correct; the *reporting* must not conflate
them (see §4, and A's data where switch=67 % but VSI=+0.0007).

---

## 2. The three fixed bugs are real, and the fixes are right

B independently found and corrected three defects in the **original** eval:
1. **AUROC not tie-aware** — argsort ranking returned 0.179 for a constant predictor; correct value 0.5.
2. **"Disjoint split" was a non-check** — original tested the 3-way intersection `train∩val∩test==∅`,
   which is empty even under full train/val overlap. Fixed to pairwise + partition + group-within-split.
3. **D3 evaluated in-sample** — original fit on all 9 sessions and scored the same 9 (acc=1.0 by
   construction). Fixed to leave-one-session-out.

All three are legitimate methodology bugs; the fixes are standard and I concur. B also correctly
demotes a **mislabel** (the old `B2_seq_deepsets` was a mean+max feature vector, not a network) and
ships a real DeepSets + GRU.

---

## 3. B2's gain is **capacity, not history** — B says so, and the data prove it

From `adaptation_curve.csv` (test fold, K = number of probes in history):

| model | K=0 AUROC | K=0 regret | K=0 top1 |
|---|---|---|---|
| B2_deepsets | 0.946 | 0.381 | 0.667 |
| **B2_gru** | **0.982** | **0.000** | **1.000** |

The torch models reach near-perfect prediction **and zero selection regret with ZERO probes** — before
seeing any history. Their level is a flexible nonlinear trunk fitting 30 training candidates' θ→outcome
map, **not** state inference. The **capacity-matched** history effect is the linear `B2_mean`
(same head as B1): AUROC **0.536 → 0.714 → 0.696** (K=0→1→2/3), Brier 0.276→0.210 — real but modest and
non-monotonic (K=1 best; N-noise). Crucially, **B2_mean's regret/top1 are identical to B0/B1 at every K**
(0.580 / 0.333): history helps *prediction*, not *selection*, in the capacity-matched model. Under `full`
utility the ablation shows the torch models at regret 0.0 while **OracleZ — which knows the true damping —
sits at 0.58**, which is only possible as 3-group overfitting (a learned model cannot truly beat the
state oracle). **Answer to "B2's gain = history or capacity?": capacity.** B flags this explicitly and
recommends reporting the capacity-matched effect; I concur fully.

Nuance to record: `OracleZ` as implemented is a **capacity-matched** state-aware *prediction* baseline
(linear head + true z), **not** an absolute prediction ceiling — a nonlinear no-state model can exceed it
on capacity alone. The correct absolute **decision** ceiling is the VSI state-aware oracle (true
utility), which the framework has and uses. This distinction should be stated so "DeepSets > OracleZ" is
never read as "history beats state knowledge."

---

## 4. Independent Phase-2 preview — B's code on A's matched data

Phase 2 (B running the framework on A's matched pilot) is **not pushed** (offline `status.md`:
"await Claude A's matched bank"). I ran B's **unmodified** `pairing` + `oracle` + `utility` on A's
162-episode matched pilot. Results:

- `validate_bank_report` → **`is_matched_bank: true`, 3 matched groups** (one per target), each spanning
  damping {3,20,40} with 5 candidate keys, **0 excluded**. B's validator accepts A's bank.
- Decision value under B's **frozen** utility variants (no weights invented — these are B's own
  `utility_variants`):

| utility | state-aware U | state-agnostic U | **VSI** | switch | reversal | agnostic pick |
|---|---|---|---|---|---|---|
| success_only | +1.00000 | +1.00000 | **0.00000** | 0/3 | 0/16 | c0_conservative |
| success_error | +0.99062 | +0.99062 | **0.00000** | 0/3 | 6/90 | **c1_steady** |
| full (λ_t=0.02) | +0.79613 | +0.79546 | **+0.00067** | 2/3 | 16/90 (0.178) | **c1_steady** |

**This is the crux result.** VSI is ~0 under every frozen utility. The switch (2/3) and reversal (18 %)
appear only once the small time term is added, and the switch is between candidates within ~0.001 U at
D=3, with the state-agnostic optimum being the **robust generalist `c1_steady`** at every damping.
**Under the frozen utility, knowing the hidden state buys essentially nothing for selection.** This
matches A's self-reported gap (+0.0006) and, more importantly, is computed by B's audited estimators —
so it will be Phase 2's headline unless the bank/utility are changed (which would raise the §7 hazard).

---

## 5. Residual method concerns (small, but must be logged)

1. **Utility provenance label.** λ=(1.0, 0.02) are hardcoded design defaults; with only 3 val sessions
   (all 3 states), there is no evidence a λ-search was actually run on validation. λ is really an
   **a-priori design constant** (fine, and safer than tuning) — but `selected_on="validation"` overstates
   provenance. It is **not tuned on test** (verified), so no violation; relabel as "a-priori frozen."
2. **Session bootstrap treats replicates as independent.** The bootstrap resamples 9 sessions, but A's
   9 sessions are 3 states × 3 near-identical replicates (runtime audit §4). Effective independent n over
   the state is ~3; at scale-up this yields falsely narrow CIs. At current N the CIs are already huge, so
   the practical harm is limited — but the formal run must bootstrap over **states/replicate-blocks**, not
   pretend 9 independent sessions.
3. **No held-out state.** With 3 states in every fold, the pipeline never tests generalization to an
   unseen damping (runtime audit §5). "Held-out D3" is leave-replicate-out. This is a property of the
   data, not a code bug, but the framework should surface it (e.g. a leave-one-**state**-out mode that
   here is simply infeasible with 3 states).
4. **Probe cost absent from utility** (runtime audit §8). With VSI≈0, any probe cost makes state-aware
   selection net-negative; the utility should optionally net out probe execution time.

None of these are fabrication or leakage; they are precision/interpretation issues for the formal claim.

---

## 6. Concordance with B's own verdict

B graded the old pilot: **D1 supported (descriptive), D2 supported, D3 supported (held-out, CI
degenerate), D4 supported-but-exploratory-and-capacity-controlled, D5 not supported / not testable
(unmatched bank)** → **MODIFY**. Every one of these matches my independent reading. B did **not**
fabricate VSI/switch/reversal on unmatched data (returned `available: false` with a reason), did **not**
tune λ on test, and correctly separated capacity from history. **The Phase-1 method is sound and
honest.** My additions are the matched-data VSI preview (§4) and the four precision caveats (§5).

**Offline-method verdict: framework PASS (correct, leakage-safe, honest); the matched-data decision
value it will report is VSI≈0.**
