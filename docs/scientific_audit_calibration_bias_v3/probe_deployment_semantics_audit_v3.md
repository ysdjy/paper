# Probe deployment-semantics audit (Audit D)

**Auditor:** Claude C. Based on first-hand reading of the generator, the probe records, and `voi.py`.

## What a probe actually is (verified)
- A probe is a **full `open_drawer` skill attempt** — the identical skill the "real" candidate runs, at a
  fixed diagnostic offset (`probe_m040`=−0.04, `probe_p040`=+0.04). Probe skill-elapsed time ≈ **11.78 s**
  (K=1) / 23.22 s (K=2) — i.e. a full task execution, not a light touch.
- The generator calls `H.reset(bspec)` **before every episode** (probe and candidate). So each probe is
  followed by a **full environment reset** (robot → default joints, drawer → closed layout) before the
  next episode.
- A probe therefore **changes the drawer state** (it opens, or attempts to open, the drawer) and requires
  a reset to restore the initial condition.

## What deployment scenario this supports — and does not
| scenario | supported? | why |
|---|---|---|
| **Same-deployment, repeated-task calibration with a full reset between attempts** | **YES** | the protocol literally runs a diagnostic drawer-open, resets, then runs the task drawer-open under the same hidden bias |
| **One-shot / in-task online calibration** (calibrate *during* the single real execution) | **NO** | the probe consumes a full task attempt and needs a reset; it is not a low-cost pre-action within one execution |
| **Non-destructive quick probe** (a brief motion that does not disturb the task object) | **NO** | the probe opens the drawer and must be undone by a reset |

**Probes are diagnostic trials, not cheap non-destructive actions.** The "value of probing" here is the
value of spending one (or two) **complete, state-changing, reset-requiring task rehearsals** to infer the
calibration bias before the scored attempt.

## Reset semantics in a real deployment
The simulator reset (robot re-homed, drawer re-closed to a deterministic layout, zero velocities) maps in
the real world to **a human or fixture re-closing the drawer and re-homing the arm between every probe and
the real task**. This is a strong, often-unrealistic assumption for "deployment calibration." It is
reasonable for a **benchtop / instrumented-cell** setting; it is not a claim about autonomous field
deployment.

## Net-VOI accounting is incomplete (verified in `voi.py`)
`Net VOI(K) = Gross VOI(K) − λ_time × cumulative_probe_time`. It subtracts **only probe execution time**
(0.02 × 11.78 ≈ 0.24 utility for K=1). It does **not** subtract:
- **failure risk of the probe itself** — a probe at offset ±0.04 can `HANDLE_DETACH` / `POSITION_TIMEOUT`
  (the failure-mechanism table shows exactly these at large `|bias±0.04|`), which in the real world may
  mean a dropped/damaged handle needing recovery;
- **energy / wear** of a full extra manipulation;
- **reset / recovery cost** (re-closing the drawer, re-homing) — which is the dominant real cost of the
  probe-then-task protocol.

So the reported Net VOI is an **upper bound** on deployment value; the true net value is lower.

## K=1 stop rule
Reasonable **only under the repeated-task-with-reset semantics above**: run one diagnostic drawer-open,
reset, then the scored open. The 1-bit `m040` discrimination (succeeds at +0.03, fails at −0.03) justifies
K=1 for *these two test biases*; note it does **not** resolve the full 9-bias grid (predicted K=1 grid
success 0.889), so K=1 is a test-set convenience, not a general stopping rule.

## Permitted vs forbidden statements
**Allowed:** "diagnostic-probe calibration for repeated tasks under a fixed deployment bias, with a full
reset between the probe and the scored attempt"; "Net VOI net of probe *time*"; "K=1 suffices to
discriminate the two held-out test biases."

**Forbidden:** "online / in-task calibration"; "low-cost / non-destructive probe"; "one-shot deployment
calibration"; any Net-VOI claim presented as the true deployment cost without acknowledging that probe
failure risk, energy, and reset/recovery cost are excluded.

**Recommended enhancement (not a blocker):** report Net VOI additionally with a probe-failure/recovery
penalty (e.g. charge a fixed recovery cost when a probe fails), and state the reset assumption explicitly.
