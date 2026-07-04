# Confirmatory probe / VOI revalidation v1 (offline)

Claude B. Re-checks probe identifiability and value-of-information for the recommended **Design A**
({−0.04, 0, +0.04}) under the empirical hard edge and the frozen residual. Probes are **repeated-task
deployment diagnostic full-task trials** — NOT "low-cost", "non-destructive", or "one-shot in-task"
probes. Machine form: comparison JSON `detail.probe`.

## Probe design (Design A)
Fixed probe offsets **{−0.04 (first), +0.04 (second)}** (both in the bank). K=0 = no history; K=1 = the
**fixed first probe** only; K=2 = both probes.

## Identifiability & selected success
| K | probes | selected success (test biases) |
|---|---|---|
| 0 | — | **0.75** (= best-single offset 0) |
| 1 | −0.04 | **1.00** |
| 2 | −0.04, +0.04 | 1.00 |

- The **first probe (−0.04) alone deterministically distinguishes the two test nominals**: at +0.03 →
  |−0.01+residual| ≤ 0.02 → **succeeds** for all residuals; at −0.03 → |−0.07+residual| ≈ 0.07 → **fails**
  for all residuals. So K=1 gives a clean **1-bit** classification of {−0.03, +0.03} → the selector applies
  the compensating offset (∓0.04) → deep-in-band success 1.00. The candidate-bank change does **not** harm
  probe identifiability (it improves it: the coarse bank makes the probe outcomes cleanly separated).
- K=2 adds the +0.04 probe; on the two test biases it is redundant (K=1 already saturates), but it adds
  robustness for finer future bias grids.

## Gross / Net VOI (success; frozen λ_time = 0.02)
| K | gross VOI (success) | probe time (s, full-task proxy) | time cost | **net VOI** |
|---|---|---|---|---|
| 0 | 0.00 | 0.0 | 0.000 | 0.00 |
| 1 | **+0.25** | 16.58 | 0.332 | **−0.082** |
| 2 | +0.25 | 33.17 | 0.663 | −0.413 |

- **Gross VOI(K=1) = +0.25** on success — history knowledge raises selected success from 0.75 to 1.00.
- **Net VOI(K=1) = −0.082 (NEGATIVE)** under the frozen utility: a probe is a **full-task diagnostic
  trial**, and its time (≈16.6 s, using the 306 success/failure times 9.4 s / 23.8 s as a proxy) charged at
  λ_time = 0.02 costs ≈ 0.33, exceeding the 0.25 success gain.

## Honest implication for the final GO
The current frozen final-GO condition requires **Net VOI(K=1) > 0**. Under Design A with full-task-trial
probe timing, **Net VOI is negative** — probing to identify the state does not pay for itself in the frozen
combined utility, even though the **success** decision value (+0.25) is real and non-degenerate. This must
be surfaced before v4:
- the **success-only** decision-value claim ("knowing the state lets the selector succeed where the
  best-single fails, +0.25") is supported and is the paper's core mechanism claim;
- the **Net-VOI-positive** claim is **not** supported for a full-task diagnostic probe at λ_time = 0.02.
Options for v4 (flagged, NOT decided here): (a) scope the primary claim to success decision value / Gross
VOI and drop Net-VOI-positive as a GO gate; (b) pre-register an application-level time cost that makes the
probe economics explicit; (c) use a cheaper diagnostic if one is physically justified — but it must still
be a full-task trial (no "cheap probe" wording). The probe-time proxy here is preliminary (from 306 times);
a confirmatory run would measure it directly.
