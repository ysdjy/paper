# Claims vs Evidence — selection-interaction pilot (v1)

**Auditor:** Claude C (independent). Each candidate paper claim is graded against first-hand evidence.
Grades: **SUPPORTED** (evidence holds at pilot scale), **SUPPORTED-EXPLORATORY** (holds but N-limited /
sim-only), **NOT SUPPORTED** (evidence is null or absent), **MISREPRESENTATION-RISK** (a true statistic
that would mislead if framed as a different claim).

Frozen utility throughout: `U = p − 1.0·err − 0.02·time`. Decision-value numbers are from B's unmodified
oracle on A's matched data (offline audit §4).

| # | Claim (as could appear in the paper) | Grade | Evidence |
|---|---|---|---|
| C1 | Hidden per-session damping changes outcome **probabilities** | **SUPPORTED** | candidate success collapses 9/9→3/9→0/9 (c2) and 9/9→0/9→0/9 (c3/c4) across D=3/20/40; `HANDLE_DETACHED` is the mechanism. Descriptive over full data. |
| C2 | Damping is **hidden from the observable `x`** | **SUPPORTED** | identical resets → `x` carries no damping; `assert_no_privileged_in_x` guard; B1 (static) > 0.5 comes from θ→success, not x-leakage. |
| C3 | Damping is **recoverable from probe history** | **SUPPORTED-EXPLORATORY** | LOSO nearest-centroid acc 1.00 (chance 0.33); probe P1 final-pos spread 0.266/0.080/0.039 monotone. **But only 3 states, cleanly separated, all present in every fold** → trivial, no generalization to unseen damping; CI degenerate. |
| C4 | Probe history **improves candidate outcome prediction** | **SUPPORTED-EXPLORATORY** | capacity-matched B2_mean AUROC 0.536→0.714, Brier 0.276→0.210, errMAE 0.084→0.073, across 3 split seeds. Modest, N=30-train, sim-only. **Torch AUROC (0.89) is capacity, not history** (K=0 already 0.95). |
| C5 | The candidate **landscape reorders** with damping (interaction is real) | **SUPPORTED** | aggressive candidates swing >1.1 U across damping; rank-reversal 16/90 (0.178) under full utility. Matches the pre-registered physics prior. |
| C6 | Optimal candidate **switches** with damping (67 % of groups) | **SUPPORTED but MISREPRESENTATION-RISK** | switch 2/3 under full utility — **true**. But it is 0/3 under success-only/success-error, and the switch is between candidates within ~0.001 U. Presenting it as evidence of *selection value* misreads a VSI≈0 result. |
| C7 | Knowing the state **improves plan selection** (state-aware selector beats state-agnostic) | **NOT SUPPORTED** | **VSI = +0.00067** (full), 0.00000 (success-only/error). A robust generalist `c1_steady` (27/27) is the state-agnostic optimum at every damping. Even the *oracle* upper bound on selection value is ~0. |
| C8 | Calibration has **decision value** (not just predictive value) | **NOT SUPPORTED at pilot scale** | prediction (C3/C4) ≠ decision (C7). Under a defensible frozen utility, decision value ≈ 0 because a state-agnostic generalist exists. This is the paper's core claim and it currently fails. |
| C9 | The **matched candidate bank** enables clean cross-damping comparison | **SUPPORTED** | θ pure function of `candidate_id` (0 violations), bank sha pinned, each id appears 9×; B's independent validator: `is_matched_bank: true`, 3 groups. Genuine improvement over the unmatched pilot. |
| C10 | 9 sessions give **9 independent matched groups** of evidence | **NOT SUPPORTED** | 9 sessions = 3 damping × 3 **near-identical replicates** (within-rep SD ~6·10⁻⁴, labels never flip). Effective independent n over the state ≈ **3**. Replicates are not independent draws. |
| C11 | Results transfer to **real-robot deployment** | **NOT SUPPORTED** | single scalar sim damping, 3 hand-picked values, no domain randomization / sensor noise, deterministic-ish physics, no held-out state. Sim damping ≠ real friction/backlash/payload/perception distribution. Out of evidence. |

## Honest headline the evidence *does* support

> In a matched-candidate drawer-opening pilot, a hidden per-session damping is (i) real (changes
> outcomes), (ii) hidden from observation, (iii) recoverable from a few probes, and (iv) it reorders the
> candidate landscape — **yet under a frozen, defensible utility it carries essentially no decision value
> for plan selection, because a single robust generalist policy is near-optimal at every state.**
> Calibration here buys *prediction*, not *selection*.

That is a precise, publishable, **negative-leaning** contribution. It is stronger and more honest than a
"selection helps" claim the data cannot support.

## Claims that would require changing the experiment (and why that is a hazard)

To flip C7/C8 to SUPPORTED, one must **remove the robust generalist** and/or **raise λ_time** (A's §5).
Doing so *after* seeing VSI≈0, and reporting only the engineered result, converts the finding from
"we measured decision value" to "we constructed a scenario with decision value." That is legitimate
**only** if (a) the bank/utility change is pre-registered and motivated independently of the null, and
(b) the paper explicitly scopes the claim to "banks without a robust generalist." See
`final_recommendation_v1.md` §Guardrails.
