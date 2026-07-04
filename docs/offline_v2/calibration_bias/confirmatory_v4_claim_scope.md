# Confirmatory v4 — claim scope & wording (frozen)

Claude B. Fixes the exact scientific wording so the confirmatory result cannot be over-claimed. Machine
form: `CLAIM_SCOPE` in `preregistration_v4.json`.

## 1. The confirmatory question (verbatim)
> Whether, when the hidden deployment calibration state is stable across repeated tasks, the history formed
> by **one fixed full-task-level diagnostic trial** lets a **capacity-matched history-conditioned selector**
> choose a more appropriate **discrete action parameter** on **unseen nominal deployment conditions** and
> raise **subsequent task success**.

## 2. Probe terminology (fixed)
Always: **"repeated-task deployment diagnostic full-task trial."**
Never: "one-shot online adaptation", "low-cost internal query", "non-destructive probe", or "unseen
hidden-state generalization." The probe consumes a full task execution and its cost is charged (VOI, §15).

## 3. Generalization claim (bounded)
> Generalization is to **unseen nominal deployment conditions**, while the **actual hidden-state support
> remains inside the training support**.
```
test actual support  = [-0.045,-0.025] ∪ [+0.025,+0.045]
train actual support = [-0.05, +0.05]           →  test ⊂ train  (verified)
```
So the test **nominal** biases ±0.035 are unseen, but every realized hidden state falls within a region the
models saw in training. We do **not** claim generalization to unseen hidden states.

## 4. Primary claim (fixed)
> **Diagnostic history improves subsequent action-selection success.**
Evidenced by `CI_lower(B2_K1 − train/val-only best-single) ≥ 0.15` and the history ablation `B2_K1 − B1_K0`.
Net deployment value (VOI) is **secondary**; even a positive one-shot Net VOI does not become the headline,
and the deployment horizon T is not chosen post-hoc.

## 5. Constructed positive-control boundary (honest)
> The 3-point candidate bank `{-0.04, 0.00, +0.04}` is a **constructed positive-control skill library**,
> frozen by the independent exploration phase, used to validate the history-conditioned action-selection
> **mechanism**.
It is **not** a claim over:
- arbitrary continuous action spaces,
- arbitrary tasks or mechanisms,
- long-horizon or autonomous adaptation.
The test geometry (±0.035) was chosen — before any confirmatory data exist — precisely so the state-agnostic
best-single straddles the empirical success edge and a state-aware action does not; this is a designed,
disclosed positive control for a mechanism, not a benchmark of general capability.

## 6. What a PASS licenses / does not license
- **Licenses:** the claim that, in this drawer-calibration setting with a stable hidden state and this frozen
  3-action library, one full-task diagnostic trial yields history that a capacity-matched selector uses to
  pick better actions and raise success, generalizing across unseen nominal conditions within the trained
  hidden-state support.
- **Does not license:** online/low-cost/non-destructive framing; unseen-hidden-state claims; continuous or
  open-ended action selection; transfer to other tasks; or treating Net VOI as the primary contribution.
