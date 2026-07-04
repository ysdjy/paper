# Confirmatory PRE-POWER freeze proposal v1 (offline)

Claude B, branch `experiment/offline-calibration-confirmatory-redesign-v1`. A **proposal** for the selected
redesign — **NOT preregistration v4**, no confirmatory generator, no confirmatory data. It freezes the
inputs the next (learned-selector power) phase needs. Nothing here is run this phase.

## Selected design: `SELECT_CANDIDATE_BANK_REDESIGN` → **Design A (3-point bank)**
Non-degenerate primary contrast with the **current residual** (no enlargement); robust best-single;
mechanism (history-conditioned action selection improves grasp success) demonstrated as a **constructed
positive control**. See `confirmatory_redesign_verdict_v1.json`.

## 1. Nominal-bias split (kept from v3)
- train {−0.04, −0.02, 0.00, +0.02, +0.04}; validation {−0.01, +0.01}; **test {−0.03, +0.03}** (unseen
  interior nominal, bracketed by train). Actual-state support ⊂ train actual-state support.

## 2. Candidate bank (CHANGED)
**{−0.04, 0.00, +0.04}** (matched: every offset executed in every block; only offset varies within a
block). A pre-discretized skill-parameter library frozen before confirmatory data. No offset leaks the
hidden state.

## 3. Residual distribution (kept)
`actual_bias = nominal + residual`, `residual ~ TruncNormal(0, σ=0.005, [−0.01, +0.01])`; one per block,
shared across offsets; blocks independent; residual/actual/nominal secret-only.

## 4. Candidate execution (kept)
Matched design per block; probes before candidates; leakage guards + record validator + hard self-check as
in the frozen runtime.

## 5. Probe order (kept, re-validated for Design A)
first = offset −0.04, second = +0.04; K=0/1/2, K=1 = first probe only; probe = repeated-task deployment
diagnostic **full-task trial**. K=1 gives a clean 1-bit test-bias classification → selected success 1.00.

## 6. Best-single selection rule (FROZEN, train/val only)
Chosen on **train + validation only** (test never participates; model seeds do not affect it):
1. maximize mean success; 2. tie → maximize mean margin (τ − |eff|); 3. tie → smallest |offset|.
For Design A this yields **offset 0** by a **0.333** success margin (robust, no thin tie). A design whose
best-single is a thin tie (< 0.15 margin) is **rejected** (Design C).

## 7. History schema (kept)
Same leakage-safe probe-history schema; model reads x, g, θ (offset), H; never residual/actual/nominal.

## 8. Endpoints
- **Primary:** success-only decision value — state-aware selected success vs train/val best-single, on the
  unseen test biases, under the hard-edge model.
- **Secondary:** continuous outcomes (task_error, time, true-handle-error-at-close).

## 9. Outcome model (FROZEN)
Empirical **hard threshold** `success = 1[|actual + offset| ≤ τ]`, τ = 0.03425, sensitivity
{0.0342, 0.03425, 0.0343, 0.0325}. No fabricated soft scale (`confirmatory_hard_edge_model_v1.md`).

## 10. Block counts (TENTATIVE — to be fixed by the learned-selector power sim)
Start from v3's train 9 / val 6 / test 9; the learned-selector power simulation (next phase) sets the final
counts. Bootstrap unit = nuisance **block**.

## 11. GO / NO-GO criteria (proposal)
- pairing/leakage/provenance; capacity-matched history gain (DeepSets K=0 vs K>0); **success-only
  selected-success gain ≥ 0.15 with a non-degenerate block-bootstrap 95% CI** (Design A gives ~0.19–0.25,
  non-degenerate); stability across split + model seeds; replicate independence acceptable (genuine
  residual-driven variance at the operating cells — now satisfied by Design A);
- **Net VOI(K=1):** flagged. Under a full-task diagnostic probe at λ_time = 0.02, Design A's **Net VOI is
  NEGATIVE (−0.082)** while Gross/success VOI is +0.25. v4 must either scope the powered claim to
  **success decision value** (drop Net-VOI-positive as a hard gate) or pre-register an explicit
  application-level probe time cost. (`confirmatory_probe_voi_revalidation_v1.md`.)

## 12. Paper claim scope (honest)
"A hidden calibration state determines grasp success through a sharp compensation edge; a history-conditioned
selector that infers the state picks the compensating action and **succeeds where the best state-agnostic
fixed action fails** (+~0.2 success on constructed test conditions)." This is a **constructed
positive-control** demonstration of the mechanism, **not** a claim of large gains in an arbitrary continuous
action space.

## 13. Hidden-state & probe wording (frozen)
outcome-determining hidden state = `actual_bias = nominal + residual`; test = "generalization to unseen
**nominal** deployment conditions with in-support actual state" (NOT "unseen hidden state"); probe =
"repeated-task deployment diagnostic full-task trial" (never low-cost/non-destructive/online).

## 14. Runtime validation need
**No new large runtime run needed for Design A**: its offsets {−0.04, 0, +0.04} are a subset of those
already executed in the 135-ep capability map and the 306-ep band-edge run. The next phase is **offline**
(learned-selector power sim on the empirical hard-edge model). At most a tiny schema/safety smoke may be
warranted before any eventual confirmatory data — but **no second 306 and no larger residual**
(`needs_larger_residual = false`, `needs_new_runtime_validation = false`).

## 15. Learned-selector power inputs (FROZEN for the next phase — not run here)
selected design = A; empirical hard-edge model + τ sensitivity; threshold-uncertainty handling = report over
the whole interval; train/val/test blocks (from the power sim); **actual DeepSets K=0/K=1 pipeline**; **≥ 5
model seeds**; simulate full synthetic datasets from the empirical model with probe-outcome misreads;
best-single train/val-only; test used only for final evaluation; **block-bootstrap over test nuisance
blocks**; power event:
`P( 95% CI lower bound of [learned B2 selected-success gain over best-single] ≥ 0.15 )`.
Report **structural/oracle power** and **learned-selector power** separately; if learned-selector power < 0.8,
adjust block counts or narrow the claim — do not cite structural power in its place.

_This is a proposal. Preregistration v4 is NOT issued here._
