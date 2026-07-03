# Exploration gate result — calibration-bias capability map v1

Claude B, using the **latest frozen AND gate** (branch `17cc4a9`, `pipeline.GATE` /
`run_capability_map.exploration_gate`). NOT A's old OR gate. Source: `exploration_gate.json`.

## Verdict: **PASS**

| # | frozen criterion | result | value |
|---|---|---|---|
| 1 | ≥ 3 bias levels have a different best offset | **PASS** | 5 distinct best offsets |
| 2 | no robust generalist (within 0.02 of per-bias best at every bias) | **PASS** | most-robust worst-case gap 1.42 |
| 3 | **success-only gain ≥ 0.15 AND frozen VSI ≥ 0.05** | **PASS** | 3a gain 0.40 (block-wise min) ✓ · 3b VSI 0.524 ✓ |
| 4 | replicate-independence — no frozen blocker | **PASS** | blocker = False |
| 5 | integrity / pairing / leakage / provenance | **PASS** | validator all-pass |

All five required → **gate PASS**. Criterion 3 uses the **conjunction** (both success-only gain and
frozen VSI), and the success gain is taken from the **worst** leave-one-block-out fold (0.40) — a
conservative reading. This is a **design gate**, not final significance.

## Difference from Claude A's verdict
A reported `gate_pass = false`, driven entirely by A's `replicates_independent = false` acting as a hard
criterion. B's frozen gate uses criterion 4 = "no blocker" (a technical-clone bug guard), which is
`False` here because the blocks are correctly applied and vary continuously; and B does not add a
success-label-flip requirement. All other criteria agree. The disagreement is **only** about whether the
near-deterministic success label blocks the exploration gate — under the frozen protocol it does not.
See `independence_adjudication_v1.md`.

## Mandatory caveat carried forward
The gate PASSES, but the data is **EXPLORATORY-ONLY** (success label deterministic across n=3 blocks).
`preregistration_v1.md` therefore requires the confirmatory pilot to deliver genuine outcome variance
(more independent blocks partitioned by split; level grid touching the success-band edge; primary
inference on continuous outcomes + selection regret) and the failure-mechanism instrumentation.

## Why this is not the damping outcome
Damping stage: history identified the state and the landscape re-ranked, but a robust generalist made
VSI ≈ +0.0006 → MODIFY. Calibration bias: **no robust generalist exists**, the optimal offset moves
monotonically with the hidden bias, and knowing the bias is worth a **0.40 success gain** and **0.524
frozen VSI** — genuine decision value, by physics, not by pruning the bank.

## Consequence
Because the gate passes, the single freeze proposal and final preregistration are produced:
`confirmatory_freeze_proposal_v1.md` and `preregistration_v1.md`. **No confirmatory run is requested**;
that awaits explicit user confirmation.
