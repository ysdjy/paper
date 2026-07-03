# GO / MODIFY / STOP — selection_interaction_pilot_v1

Claude B, branch `experiment/offline-eval-v2`. Verdict on Claude A's matched-candidate
selection-interaction pilot under the **frozen** utility `U = success − 1.0·err − 0.02·time`.
Backed by `decision_value.json`, `selection_heldout.json`, `stability.json`, `validation_report.json`,
`replicate_audit.json`, `metrics.json`, `go_modify_stop.json`.

## Verdict: **MODIFY**

> **State information improves prediction, but has no decision value against a candidate set that
> already contains a robust generalist.**

## GO criteria — scorecard

| # | GO requirement | met? | evidence |
|---|---|---|---|
| 1 | pairing + leakage pass | **YES** | 162 eps, 9 matched groups, candset/θ/target consistent, no candidate-outcome in history, no secret in x |
| 2 | B2 beats capacity-matched B1 on prediction, stable | **YES** | B1 AUROC 0.87 → B2-mean 1.00 (same linear head); stable across seeds |
| 3 | frozen-utility VSI is meaningful | **NO** | VSI = **+0.00059** on a ~0.80 utility scale (< 0.1%) |
| 4 | B2 selection beats B1 **and** robust generalist, toward oracle | **PARTIAL** | beats B1 (regret 0.026→0.000) but only ties robust generalist (0.0007); gap to it ≈ 0.0006 |
| 5 | not dependent on a single split / seed | **YES (but see 6)** | regret stable over 3 split seeds: B2 0.000/0.0002/0.0002, B1 0.026/0.027/0.026 |
| 6 | statistical independence acceptable | **NO** | replicates are technical repeats (45/45 cells deterministic); only 3 unique hidden states |

GO needs **all** of 1–6. Criteria 3, 4, 6 fail ⇒ not GO. Criterion 1 holds and D1–D2-style checks
pass ⇒ not STOP. **MODIFY.**

## Why this is MODIFY, not STOP or GO

Everything the mechanism needs is present and correct:
- the hidden state **is identifiable** from probes (D3 LOSO acc = 1.00),
- the candidate **landscape re-ranks** with the hidden state (switch 0.667, reversal 0.170),
- a history model **selects** the truly-best candidate (B2 regret ≈ 0).

But the candidate bank contains **c1_steady**, a *robust generalist* that succeeds at every damping
and is nearly the fastest/most-accurate everywhere. So:
- state-agnostic oracle (0.7952) ≈ state-aware oracle (0.7958) ⇒ **VSI ≈ 0**,
- "always c1_steady" gets regret 0.0007 ≈ oracle,
- every policy (incl. B0) has **selSucc = 1.00** — no selector ever fails.

Hence knowing the deployment state changes *which candidate is argmax* (ranking) without changing
*how well you do* (decision value). Under the frozen utility, Net VOI of probing is **negative** (the
seconds spent probing cost more than the ~0.0006 selection gain).

This is a scientifically clean **negative decision-value result**, not a pipeline failure — which is
exactly the MODIFY case the plan anticipated ("history identifies damping, landscape reranks, switch
rate > 0, but VSI ≈ 0 and robust c1 ≈ state-aware oracle").

## Recommendation for the next experiment

I recommend **Route B (introduce a hidden state that directly moves the optimal candidate), starting
with a single-variable handle-pose / calibration-bias pilot**, over Route A.

### Why not Route A (keep damping, re-pre-register candidate bank / utility)
Route A can manufacture a positive VSI only by (a) **removing the robust generalist** from the bank or
(b) **up-weighting time** until speed differences dominate. Both are reviewer-fragile:
- deleting a legitimately robust candidate to create an effect is exactly the "don't hand-pick the
  bank to produce a result" failure the plan forbids;
- a large λ_time needs an application-level justification (a real throughput cost) and must be
  pre-registered; our sensitivity grid shows even λ_time = 0.5 yields VSI only 0.067 while c1 still
  succeeds 100%, so the payoff is small and the optics are poor.
The core physics is the problem: damping mostly changes *whether aggressive candidates fail*, and a
gentle candidate avoids failure across the whole damping band — so state rarely has decision value.

### Why Route B (handle-pose / calibration bias)
A hidden **grasp/handle calibration bias** interacts *directly* with the controllable
`grasp_offset_local_y`:
- the bias is hidden (never in `x`), and a candidate's `grasp_offset` can **compensate** it;
- under different biases the **best grasp offset genuinely changes**, so the optimal candidate
  switches **and** the achievable utility of the wrong choice drops (no single offset is robust to
  all biases) ⇒ **VSI > 0 by construction of the physics, not by pruning the bank**;
- damping can be retained as a secondary dynamics hidden state for a later 2-D study.

Concrete proposal to Claude A (B does not implement runtime):
1. **Single-variable pilot first**: hidden handle-Y bias ∈ {−δ, 0, +δ} (δ chosen so a fixed offset
   cannot cover all three); candidate bank spans grasp_offset across the full reliable range; keep the
   matched-bank design and 3-probe history.
2. Add **nuisance variation independent of the hidden state** (small init/friction/sensor jitter per
   session, not in `x`) so replicates become independent samples and CIs mean something.
3. Freeze the candidate bank and utility **before** generating data; keep the current frozen U as the
   main metric; pre-register any throughput utility with an application-level time cost.
4. Only after single-variable VSI is demonstrated, consider the bias × damping 2-D combination.

### Risk note
Route B changes the hidden variable, so it needs a fresh capability map for grasp-offset × bias and a
new matched bank — more engineering than Route A. But it targets the actual scientific claim (state
information has decision value) instead of engineering an artifact, so it is the lower reviewer-risk
path. Recommended.
