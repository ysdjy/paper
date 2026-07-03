# Failure-mechanism audit — calibration-bias capability map v1

Claude B, read-only. Source: `failure_mechanism.json`. Purpose: confirm the success band is driven by
the compensation quantity |bias + offset|, and state exactly which competing failure mechanisms the
current fields can and cannot exclude.

## Success band is determined by |bias + offset|
| \|bias + offset\| | n | success rate | failure reasons |
|---|---|---|---|
| 0.000 | 15 | **1.00** | — |
| 0.020 | 30 | **1.00** | — |
| 0.040 | 24 | 0.00 | POSITION_TIMEOUT ×24 |
| 0.060 | 18 | 0.00 | HANDLE_DETACHED ×18 |
| 0.080 | 12 | 0.00 | HANDLE_DETACHED ×12 |
| 0.100 | 6 | 0.00 | HANDLE_DETACHED ×5, POSITION_TIMEOUT ×1 |

**Success ⇔ |bias + offset| ≤ 0.02**, cleanly and monotonically. The grasp offset compensates the hidden
handle bias; the compensation is one-dimensional and matches the design intent (best offset ≈ −bias).

## Two ordered failure regimes
| failure_reason | n | last phase | handle_rel_err | time (s) | final_pos | detached |
|---|---|---|---|---|---|---|
| POSITION_TIMEOUT | 25 | **APPROACH** (25/25) | 0.000 | 20.8 | 0.000 | 0/25 |
| HANDLE_DETACHED | 35 | **PULL** (35/35) | 0.085 | 10.0 | 0.000 | 35/35 |

- **|eff| = 0.04 → POSITION_TIMEOUT in APPROACH**: the run never grasps (handle_rel_err 0) and times out
  in the APPROACH phase (~20.8 s). Consistent with the offset making the pre-grasp/grasp pose hard to
  reach.
- **|eff| ≥ 0.06 → HANDLE_DETACHED in PULL**: the run grasps ~8.5 cm off-centre and the handle slips out
  during PULL (gripper slip due to offset error).

## What can and cannot be excluded
| candidate mechanism | excluded? | reason |
|---|---|---|
| success driven by \|bias+offset\| | **confirmed** | monotone table above |
| gripper slip (detach regime) | **partial** | handle_detached + handle_rel_err 0.085 ⇒ slip is the likely cause, but no contact field to rule out a collision-induced detach |
| joint-limit failure | **cannot exclude** | no joint-limit-margin field; APPROACH timeout could be a limit hit |
| IK local unreachability | **cannot exclude** | no IK failure/clamp count; APPROACH timeout could be IK non-convergence |
| collision | **cannot exclude** | no collision/contact field or no-collision verification |

Available `y` fields: success, failure_reason, final_joint_position, task_outcome_error, overshoot,
skill_elapsed_time, phase_durations, handle_detached, handle_relative_error, command_tracking_error,
phase_goal_error, wall_clock_time. These support the |bias+offset| conclusion but **do not** let us
attribute the APPROACH-timeout regime to a specific low-level cause.

Per protocol, we do **not** guess. The following are written into `preregistration_v1.md` as **required
confirmatory instrumentation** so the confirmatory data can attribute failures unambiguously:
1. minimum joint-limit margin (per episode),
2. IK failure / clamp count,
3. an explicit `failure_phase` field,
4. true handle error at the **end of CLOSE_GRIPPER** (not only final),
5. gripper width at close,
6. collision / contact availability, or an explicit no-collision verification.

## Bearing on the science
The compensation claim (hidden bias moves the optimal offset; a history model that infers bias can pick
the compensating offset) rests on the |bias+offset| band, which is unambiguous. The **decision-value**
conclusion does not depend on resolving the APPROACH-timeout micro-mechanism. But the confirmatory paper
claim ("failure is grasp-offset compensation, not a kinematic artifact") **does** require the instrumentation
above to exclude joint-limit / IK / collision confounds.
