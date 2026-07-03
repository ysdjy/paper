# Runtime & Data Audit — Claude A selection-interaction pilot (v1)

**Auditor:** Claude C (independent). **Scope:** runtime + raw data of Claude A's matched-candidate
pilot. No A/B code, data, model, or utility was modified; Isaac was not launched; nothing was tuned.
All numbers below are **re-derived first-hand** from the committed artifacts (not copied from A's
self-check), unless explicitly attributed.

## Sources audited (paper remote, `experiment/runtime-selection-v2`)
| item | commit | verified |
|---|---|---|
| design freeze | `1921641acec5…` | present, readable |
| data (`selection_interaction_pilot_v1_20260703_110727`) | `b061cdf0ce04…` | present, readable |
| runtime self-check report | `af4a27077db9…` (branch HEAD) | present, readable |
| candidate bank | `configs/selection_interaction_candidate_bank_v1.json` | sha256 `d1d80d96…` |

Design: `docs/selection_interaction_pilot_design_v1.md`. Report: `docs/selection_interaction_pilot_runtime_v1.md`.

---

## 1. What the pilot is

162 episodes = **9 sessions × 18**. Each session = 3 fixed probes (P0/P1/P2) + 3 targets
(open positions 0.12 / 0.20 / 0.28 m) × 5 archetype candidates. The 9 sessions =
**3 damping levels {3, 20, 40} × 3 replicates**. Hidden state = drawer-joint DAMPING, fixed per
session. Utility (frozen, stated a priori): `U = p_success − 1.0·err − 0.02·time`.

---

## 2. Integrity checks — **PASS** (re-derived first-hand)

| check | method | result |
|---|---|---|
| Completeness | count roles | 162 eps = 135 candidate + 27 probe ✓ |
| Bank hash pinned | sha256 of bank file vs every episode's `candidate_bank_sha256` | single value, matches file ✓ |
| **θ is a pure function of `candidate_id`** | group θ by candidate_id across all damping | 15 ids, **0 damping-dependent θ** ✓ |
| Matched coverage | count each candidate_id per damping | every id appears **exactly 3× at each of {3,20,40}** ✓ |
| Damping set/persist | metadata `damping_set_maxerr`, `damping_post` drift | 0.00 / 0.00 ✓ |
| Reset invariants | `full_reset_all_verified`, n fail | true, 0 ✓ |
| Provenance | `dirty_worktree` on the data commit | false ✓ |

**Verdict on Q "matched candidate — truly paired across damping?": YES, confirmed.** The bank writes
θ literally per `candidate_id`; the same archetype has identical θ across all 3 targets and is read
from the same frozen file for every session, so θ is provably independent of damping, replicate, and
session seed. This is a genuine improvement over the previous (unmatched-Sobol) damping pilot. The
matched-bank test (`tests/test_matched_candidate_bank_v1.py`) enforces this, and Claude B's independent
`pairing.validate_bank_report` run (see offline audit) returns `is_matched_bank: true, n_matched_groups=3`.

---

## 3. Hidden-state leakage — **no leakage into model-legal fields; oracle fields correctly quarantined**

Every candidate episode carries fields that **encode the hidden state directly**:
`hidden_state_id` (1:1 with damping: L1_low=3, L2_mid=20, L3_high=40), `secret_deployment_state.damping`,
`damping_eff ∈ {3,20,40}`, `damping_post`. These are the leakage surface.

- The **model-legal** channels are (a) the candidate's own `x` (identical resets → carries no damping;
  verified `assert_no_privileged_in_x`) and (b) same-session prior probes via B's whitelist builder,
  which drops any key containing `damping/secret/hidden` and validates each entry against a canonical
  legal probe. B's poison-injection tests (future probe, cross-session probe, candidate-outcome field,
  privileged key, cutoff) all reject correctly (13/13 pass, re-run first-hand).
- **Conclusion:** the four leakage fields are oracle-only and are not reachable by the legal feature
  path. **No hidden-state leakage.** The *design intent* — probes reveal damping — is not leakage; it
  is the causal signal the experiment is built to measure.

**One caveat, not a leak:** because there are only **3 distinct damping values** with near-deterministic
probe signatures, "recovering the state from probe history" is a 3-way classification of perfectly
separated clusters. It is trivially solvable and does **not** demonstrate generalization to unseen
damping (see §5).

---

## 4. The 3 replicates are **not independent samples** — they are near-deterministic re-runs

This is the most important data-structure finding.

- The generator applies **no per-replicate seed, no randomized reset pose, no sensor/actuator noise,
  no domain randomization**. `H.reset(spec)` and `H.run(spec, g, θ)` take no seed. Replicate `r=0,1,2`
  at a fixed damping differ **only** by the `session_id`/`replicate_id` label.
- Empirically the replicates are **not byte-identical** (small PhysX/GPU floating-point nondeterminism):
  median within-replicate SD of `task_outcome_error` ≈ **6·10⁻⁴**, and some replicate pairs are exactly
  equal while others differ by <10⁻³. This variation is **1–2 orders of magnitude below** the
  cross-damping signal (e.g. c4_aggressive T028 cross-damping mean-range 0.185).
- **Replicate noise never flips a success label** in the cells that matter. Example: the report's
  "D=20 c2_balanced = 3/9" is entirely a **target** effect, not replicate variability —
  T012 succeeds (err 0.0151) on all 3 replicates; T020 (0.0947) and T028 (0.1747) fail on all 3.
  The "3/9" is really **1/3 targets counted 3×**.

**Implications the formal run must respect:**
1. The effective independent sample size over the hidden state is **3 (damping levels)**, not 9 sessions.
   The "9 matched groups" are 3 conditions × 3 near-duplicate replicates.
2. A **session-level bootstrap over 9 sessions treats replicates as independent draws** and will
   understate CI width. The only modeled between-session variance source is solver floating-point
   noise, which is not a deployment-relevant variability. (At current scale B's CIs are already huge,
   so this doesn't rescue precision — but at scale-up it would produce falsely narrow CIs.)
3. Any claim of "n=9 sessions" of statistical evidence about damping is **≈3× overstated**.

---

## 5. Only 3 hidden states, and the split cannot hold one out

All 9 sessions occupy exactly **3 damping values**. A session-level split therefore puts the **same 3
states in train, val, and test** — there is **no held-out state**. "Leave-one-session-out" leaves 2
same-state replicates in train, so it is **leave-replicate-out, not leave-state-out**. Consequences:

- D3 "identify the hidden state" is classification among 3 *already-seen*, cleanly separated clusters →
  trivially near-perfect, and says nothing about identifying a **novel** damping.
- D4/prediction gains are measured only at these 3 exact damping values; interpolation/extrapolation to
  other damping is untested.

This is a **structural ceiling of the pilot**, independent of sample size.

---

## 6. Physical result is real, but "landscape reorder ≠ selection value"

The physics is genuine and matches the pre-registered prior: at high damping the aggressive long-carrot
pull rips the handle off (`HANDLE_DETACHED`), while conservative/steady candidates keep it attached.
Candidate success collapses with damping for c2/c3/c4; c0/c1 stay 9/9. The candidate **landscape
reorders violently** (aggressive candidates swing >1.1 in utility across damping).

**But reordering is not decision value.** Re-derived with B's frozen utility (see offline audit for the
independent run):

| frozen utility | VSI (state-aware − state-agnostic, true U) | switch rate | rank-reversal |
|---|---|---|---|
| success_only | **0.00000** | 0/3 | 0/16 |
| success_error | **0.00000** | 0/3 | 6/90 |
| full (λ_time=0.02) | **+0.00067** | 2/3 | 16/90 (0.178) |

The 67 % "optimal-candidate switch" and 18 % rank-reversal exist **only under the time term** and even
then move utility by **~0.0007**, because a **robust generalist `c1_steady` (27/27, best or within
~0.001 at every damping)** is the state-agnostic pick. **Reporting switch-rate/rank-reversal as evidence
that state-aware selection helps would be a misrepresentation** of a VSI≈0 result. To A's credit, the
runtime report says exactly this ("the raw selection payoff is negligible").

---

## 7. Manufacturing risk flagged in A's own §5 (NOT yet acted on)

A's report §5 ("Recommendations for the formal run") proposes, to make selection pay:
1. **Remove the robust generalist** (design the high-damping-safe candidate to be *worse* at low damping
   so no single θ is near-optimal), and
2. **Raise λ_time** so the fast/aggressive candidate genuinely beats the robust-but-slow one.

Both are direct routes to **manufacturing an adaptation benefit after observing that the honest result
is null.** A behaved correctly *here* — the bank/utility are frozen and A labels §5 "design notes — NOT
applied here." The risk is **downstream**: if the formal bank is engineered to lack a generalist and/or
λ_time is raised *because* the frozen result was VSI≈0, the resulting "selection helps" would be a
post-hoc construction, and the paper could only claim "in banks deliberately built without a generalist"
— not a general deployment result. This is the single biggest guardrail for Phase 2 (see
`final_recommendation_v1.md`). Note the divergence: **B's §6 says keep the frozen utility and freeze λ
on validation only — the correct stance.**

---

## 8. Probe cost is not accounted for

`true_utility` charges only the *candidate's* success/error/time. The **3 probes per session** (~9–10 s
skill time + reset each, ≈30 s of execution before any candidate is chosen) are **free** in this
accounting. Since VSI≈0, paying any probe cost to enable state-aware *selection* is **strictly negative**
here; even for *prediction*, the value of history must be weighed against acquisition cost. The formal
run should report utility **net of probe cost**, or at minimum state that probes are treated as free.

---

## 9. What the sim can and cannot support

**Supported (simulation, this asset):** damping changes outcome probability (D1); damping is hidden from
`x` (D2); damping is recoverable from probe outcomes at these 3 values (D3); the candidate landscape
reorders with damping (physical). These are legitimate, and the matched-bank machinery works.

**NOT supported:** (a) that state-aware *selection* beats a robust generalist under a defensible frozen
utility (VSI≈0); (b) any **real-robot deployment** claim — the hidden variable is a single scalar sim
damping at 3 hand-picked values with no domain randomization, no sensor noise, deterministic-ish physics,
and no held-out state. Sim damping ≠ the joint distribution of friction/backlash/payload/perception a
real deployment calibrates against. Real-robot generalization is out of evidence.

---

## 10. Runtime/data verdict

**Runtime and raw data quality: PASS.** Matched bank genuine, integrity clean, no leakage, honest
self-report. **Scientific reservations (carry into the claims/verdict docs):** replicates are not
independent (effective n≈3 states); no held-out state; VSI≈0 under frozen utility; switch/reversal must
not be sold as selection value; probe cost omitted; A's §5 is a manufacturing hazard for Phase 2.
