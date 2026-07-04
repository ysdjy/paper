# Confirmatory redesign — problem diagnosis v1 (offline)

Claude B, branch `experiment/offline-calibration-confirmatory-redesign-v1`. Offline design fix driven by
the 306-episode empirical hard edge. No Isaac, no new data, no confirmatory generator, no preregistration
v4. Read-only over the 306 residual realizations.

## What is already established (keep)
- Calibration bias is a **real outcome-determining hidden state** (135-ep capability map + 306-ep edge run).
- Probe history carries information that discriminates the hidden state.
- The controllable grasp offset **systematically interacts** with the hidden state.
- **No robust generalist** across the bias range.
- The residual **can flip success labels** near the empirical edge (306 per-offset mixed labels).
- The edge is **not** a collision / IK / joint-limit confound (all 0 N contact, 0 IK failures, 0 real
  joint-limit involvement).

## What actually failed in v3 (precise attribution)
> **The v3 problem is a mismatch between the DECISION DESIGN and the empirical edge — NOT a residual
> sampling failure, and NOT the absence of a success edge.**

Two things compounded:
1. **The empirical edge is at |eff| ≈ 0.0343 m, not the assumed ~0.02.** v3's `design_validation` /
   power used a compensation band of 0.02, so it selected the state-agnostic **best-single = −0.02**. Under
   the *measured* edge (0.0343) with a success-maximizing rule, the best-single is actually **offset 0**.
2. **The fine 7-point bank makes the best-single a near-tie.** In the current bank
   {−0.06, −0.04, −0.02, 0, +0.02, +0.04, +0.06}, on train+val the top offsets are: **0 (mean success
   0.754)**, +0.02 (0.706), −0.02 (0.691) — a **top-2 margin of only 0.048**. Offset 0 straddles the edge at
   the test biases (→ non-degenerate), but ±0.02 does **not** (−0.02: +0.03 always succeeds / −0.03 always
   fails → **degenerate**). So the current bank's best-single sits on a **knife-edge** between a
   non-degenerate and a degenerate operating point; a small change in the tie-break, the threshold, or a
   continuous-utility term flips it.

So the real defect: **the confirmatory candidate bank does not ROBUSTLY place the state-agnostic best-single
operating point at the empirical edge.** The residual is fine (σ=0.005, |·|≤0.01 already flips labels near
the edge); it is the *decision geometry* that must change.

## The fix (this phase's finding)
**Coarsen the candidate bank** so the state-agnostic best-single is *robustly* offset 0 (at the edge for the
test biases) while the state-aware selector uses ±0.04 to reach deep-in-band success. In the 3-point bank
**{−0.04, 0, +0.04}** the best-single wins by a **0.333 margin** (0 vs ±0.04) — no near-tie — and its test
operating cells straddle the edge (|eff| ∈ [0.021, 0.037] and [0.023, 0.039], both cross 0.0343), so the
residual flips labels in the **primary** contrast → **non-degenerate** block bootstrap, with the **current
residual** (no enlargement). Full comparison: `confirmatory_candidate_bank_comparison_v1.md`.

## Why NOT enlarge the residual first
To make the residual straddle the edge while keeping the *state-aware compensated* |eff| ≈ 0.01 would need
an effective residual of ≈ 0.0343 − 0.01 ≈ **0.024 m** (support ≈ ±0.025 m). That would: enlarge the actual
hidden-state support ~2.5×, weaken the nominal-deployment-condition interpretation, change train/test
actual-state support relations and probe identifiability, require re-checking compensability, demand new
runtime capability validation, and make a ~2.5 cm calibration residual physically implausible. The
candidate-bank fix achieves a non-degenerate contrast **without** any of this, so residual enlargement is a
**fallback**, not the default (see `confirmatory_redesign_verdict_v1.json`: `needs_larger_residual = false`).
