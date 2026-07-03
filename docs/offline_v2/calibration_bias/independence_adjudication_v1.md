# Independence adjudication — calibration-bias capability map v1

Claude B, frozen rules @ `17cc4a9`. Adjudicates whether the 3 replicate nuisance blocks are
independent statistical samples, using `offline_v2/calibration_bias/independence.py` (frozen). Does
**not** add any "success label must flip" post-hoc condition. Source: `independence_adjudication.json`.

## The four questions, answered separately

**1. Independent random sampling? — YES (in design, correctly applied).**
Three distinct nuisance blocks (block 0/1/2), each with a distinct `block_seed` drawn from a master RNG
that is independent of bias, and distinct `robot_joint_delta` / `target_jitter`. The SAME block gives an
**identical** `x`/`g` across all 5 bias levels (paired matched context; within-session and matched-block
`x` spread = 0). So the blocks are genuinely independent draws of the (small) nuisance distribution and
are **correctly applied** — this is not an implementation bug.

**2. Label flips? — NONE (success is deterministic across replicates).**
0 of 35 (bias × offset) cells show a success-label flip across the 3 replicates; 1 cell shows a
`failure_reason` difference. The discrete SUCCESS outcome does not vary block-to-block.

**3. Continuous outcomes change? — YES, modestly.**
Within-cell across-replicate max std: `final_joint_position` 0.0072, `task_outcome_error` 0.0052,
`skill_elapsed_time` **1.20 s**. The blocks are **not byte-identical clones** — continuous outcomes
genuinely vary (the time spread is well above float noise).

**4. Technical repeats? — near-deterministic in the LABEL, not clones.**
Independent draws whose **discrete** response is robust to the injected nuisance (≤0.03 rad joint
perturbation, ≤0.005 m target jitter), so success/failure does not change; the continuous response does.

## Frozen blocker rule: **NOT triggered (`blocker = False`)**
The rule fires only when the data is near-deterministic AND the blocks are absent / not varied within a
level (a technical-clone implementation bug). Here:
- `discrete_deterministic = False` (34/35 cells share a failure_reason, not all 35),
- continuous `final_pos` max std 0.0072 > the 0.005 near-determinism threshold,
- nuisance seeds are present and **varied within every bias level**.
→ none of the blocker conditions hold, so `blocker = False`.

## A (runtime) borderline-NOT-MET vs B (frozen) not-blocked — reconciled
| | Claude A | Claude B (frozen) |
|---|---|---|
| criterion | success-cell median SD < 0.003 **and** 0 flips | blocker = near-det **and** blocks unapplied/unvaried |
| reading | "replicates not statistically independent enough" (CI-oriented bar) | "no technical-clone bug" (bug guard) |
| verdict | `replicates_independent = false` → A gate_pass = false | `blocker = false` → criterion 4 passes |

Both agree the **discrete success label is near-deterministic**. They differ only on whether that blocks
the exploration gate. A applies a stricter CI-readiness bar as a hard gate; B's frozen protocol uses a
bug-guard that does not fire (the blocks are correctly applied and vary continuously), and B does **not**
add a label-flip requirement (as instructed). Per the frozen protocol, the exploration gate is **not
blocked** — but the data is unambiguously **exploratory-only**.

## Can the 3 blocks serve as independent exploratory samples?
**Yes, for descriptive statistics** of the compensation landscape — they are 3 correctly-applied,
independent nuisance draws. **No, for formal CIs** — n=3 with ~0 success-label variance cannot support
a meaningful confidence interval on success rate. The compensation STRUCTURE (which the gate evaluates:
best offset moves with bias, no robust offset, VSI, distinct best offsets) is clear and robust to the
nuisance; the SAMPLING PRECISION is not.

## Sufficiency verdict
- Exploratory description + gate evaluation: **SUFFICIENT.**
- Formal success-rate CIs: **NOT supported** — deferred to the confirmatory pilot, which must provide
  genuine outcome variance and a proper split.

## What is (and is NOT) recommended
- **No nuisance redesign is mandated as a blocker** — the blocks are correctly applied; this is not the
  damping-stage clone bug. We do **NOT** recommend adding noise to manufacture success-label flips.
- For the CONFIRMATORY pilot (a POWER, not bug, consideration): use **more independent blocks partitioned
  by split** (blocks disjoint across train/val/test), and design the level grid so some cells sit near the
  **success-band edge** (|bias+offset| ≈ 0.02) where the nuisance naturally induces genuine outcome
  variance; frame primary inference on the CONTINUOUS outcomes (task_error, time, handle error) and on
  selection regret, since mid-band success is deterministic. Details in `preregistration_v1.md`.
