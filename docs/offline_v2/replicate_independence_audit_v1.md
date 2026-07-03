# Replicate independence audit — selection_interaction_pilot_v1

Claude B, branch `experiment/offline-eval-v2`. Read-only audit of whether the 3 replicate
sessions per damping level are **independent statistical samples** or **technical repeats** of a
near-deterministic simulator. This determines how CIs may legitimately be computed.

- source (read-only): `.../data/selection_interaction_pilot_v1_20260703_110727/`
- source git commit: `1921641acec5a7e153f45f909c69613ed6036a12`
- candidate_bank_sha256: `d1d80d9646aae536…`
- 9 sessions = 3 damping (3.0 / 20.0 / 40.0) × 3 replicates; 45 physical cells
  (damping × target × candidate_id), 3 replicates each.
- Regenerate: `deployment_calibration/evaluation/offline_v2/.../replicate_audit.json`
  (`offline_v2.replicate_audit.audit_replicates`).

## Findings

**Discrete outcomes are fully deterministic across replicates.**
- `success` identical across all 3 replicates in **45 / 45** cells.
- `failure_reason` identical across all 3 replicates in **45 / 45** cells.

**Continuous outputs vary only at sub-centimetre / sub-0.1 s scale.**
| output | max within-cell std | max within-cell range | mean within-cell std |
|---|---|---|---|
| final_joint_position (m) | 0.0047 | 0.0100 | 0.0006 |
| task_outcome_error (m) | 0.0047 | 0.0100 | 0.0006 |
| skill_elapsed_time (s) | 0.085 | 0.18 | — |
| pull_phase_duration (s) | small | — | — |

Probe episodes show slightly more session-to-session variation (final-pos std up to ~0.013 m for
probe #1), i.e. the replicate sessions are **not byte-identical** — there is minor numerical
non-determinism — but it never flips a discrete outcome and barely moves continuous ones.

**Verdict: replicates are technical repeats, not independent samples** (`near_deterministic: true`).

## Effective sample sizes

| quantity | value |
|---|---|
| unique hidden states (damping draws) | **3** |
| unique physical conditions (damping × target × candidate) | 45 |
| unique (target × damping) decision contexts | 9 |
| technical repeats per cell | 3 |
| sessions total | 9 |
| **independent-session upper bound** | **3** (only damping is the random draw) |

The hidden state takes **3** distinct values and each is re-run 3×; there are **not** 9
independent deployment sessions.

## Consequences for statistics (enforced downstream)

1. The 9 sessions are **not** treated as 9 i.i.d. samples; nothing bootstraps over them as if
   independent.
2. Main decision-value numbers (VSI, switch, reversal, regret) are reported as **descriptive
   statistics over unique conditions / matched groups**, not as sampling estimates with tight CIs.
3. Where a CI is shown, it is a **target-block bootstrap** (resample the 3 targets, each carrying
   its replicates) and is explicitly labelled **EXPLORATORY** — with 3 target blocks it is coarse.
4. Because replicates are near-deterministic, a session-split model can see a test cell that is
   near-identical to a training cell; **held-out "generalization" is partly memorization of
   deterministic cells** and must not be read as generalization to unseen physics. This is flagged
   in `selection_interaction_analysis_v1.md`.

## Recommendation to the formal run (informs Claude A; B does not edit runtime)

Introduce **nuisance variation that is independent of the hidden state** so replicates become real
samples: e.g. small randomized initial drawer offset / robot-home jitter / friction seed / sensor
noise, drawn per session and **not** entering `x`. Without it, adding more replicates only adds
technical repeats and cannot narrow any genuine confidence interval.
