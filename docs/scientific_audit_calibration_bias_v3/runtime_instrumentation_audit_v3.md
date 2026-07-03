# Runtime instrumentation audit (Audit E)

**Auditor:** Claude C. Field-by-field feasibility, from first-hand reading of `ik_joint_adapter.py`,
`isaac_open_drawer.py`, the skill state machine, and the 135-ep `y` schema. **None of the six fields is
present in the current `y`** (current `y` = success, failure_reason, final_joint_position,
task_outcome_error, overshoot, skill_elapsed_time, phase_durations, handle_detached,
handle_relative_error, command_tracking_error, phase_goal_error, wall_clock_time). So all six are
**required new instrumentation**; the question is whether each is implementable.

## Field-by-field
| # | field | precise definition it must have | raw signal exists? | verdict |
|---|---|---|---|---|
| 1 | `minimum_joint_limit_margin` | **per-episode min over all IK steps and all 7 arm joints** of `min(q−q_lower, q_upper−q)` against **`soft_joint_pos_limits`**; report in **rad (absolute)** and normalized by range | **yes** — `ik_joint_adapter` holds `_joint_lower/_joint_upper` (from `soft_joint_pos_limits`) and `joint_pos` every `solve()` | **implementable**; `solve()` must expose the margin, then aggregate the episode-min |
| 2 | `ik_failure_count` | count of `solve()` returns with `success=False` (non-finite target, degenerate ee-quat, non-finite `q_des`) over the episode | **yes** — `solve()` already returns `IKResult.success` | **implementable**; add a per-episode counter |
| 3 | `joint_clamp_count` | **two distinct counters:** (a) **step clamp** — `|q_des−q_curr|` hit `max_joint_step` (line 211); (b) **limit clamp** — `q_des` clipped to `_joint_upper/_joint_lower` (line 213). Must report the two **separately** | **yes** — both clamps happen on separate lines in `solve()`, so they are **distinguishable** | **implementable**; `solve()` must emit both flags, aggregate per episode |
| 4 | `true_handle_error_at_close` | TCP-vs-**true**-handle relative error recorded at the **last step of `CLOSE_GRIPPER`, before `PULL`** | **yes** — per-step telemetry logs `handle_relative_position_error` + `skill_state`; the true handle is the sim link (privileged) | **implementable**; capture at the CLOSE_GRIPPER→PULL transition |
| 5 | `gripper_width_at_close` | actual gripper width at the **same instant** (end of CLOSE_GRIPPER) | **yes** — `state.robot.gripper_width` is logged per step with `skill_state` | **implementable**; capture at the same transition |
| 6 | `failure_phase` | the **last active state-machine phase before termination** (ARC_TO_FACE / MOVE_TO_PRE_GRASP / APPROACH / CLOSE_GRIPPER / PULL / SETTLE / RELEASE) | **yes** — the skill `runtime.history` records phase transitions (`_phase_durations` already reads it) | **implementable**; extract the last phase from the transition log |
| 7 | `collision / contact` | a **contact/collision signal** on the gripper/arm vs the cabinet, OR a **reproducible no-collision verification** | **NO** — `isaac_open_drawer.py` hardcodes `contact_available=False`, `collision_available=False`; the harness **collision monitor is off** | **BLOCKER as specified** |

(The task lists six; #6/#7 above are the "collision/contact" item plus the explicit `failure_phase` — the
count matches the six preregistered fields.)

## The collision field is the instrumentation blocker
The prereg phrasing — *"collision/contact **availability** or a no-collision **verification**"* — is **too
vague** to exclude cabinet collision. "Availability = False" (the current state) excludes nothing. To
attribute the **APPROACH-timeout** failure regime (the band-edge failure at `|eff|≈0.03–0.04`, exactly the
confirmatory operating region) to grasp-offset compensation rather than a collision, the confirmatory run
**must** either:
- (a) enable a **`ContactSensor`** on the gripper + forearm against the cabinet/drawer bodies, with a
  frozen force threshold and per-episode max-force + contact-phase logging; **or**
- (b) provide a **reproducible geometric no-collision check** (signed distance between gripper mesh and
  cabinet colliders below a frozen margin every step), logged per episode.

Until one is specified and enabled, the paper **cannot** claim "failure is grasp-offset compensation, not
a collision artifact."

## Can the six fields exclude each competing mechanism?
| competing mechanism | excluded by | status |
|---|---|---|
| joint-limit failure | `minimum_joint_limit_margin`→0 + limit-clamp count>0 | **yes, once #1/#3 implemented** |
| IK local unreachability | `ik_failure_count`>0, or step-clamp saturating with no progress | **yes, once #2/#3 implemented** |
| cabinet **collision** | contact force / signed-distance | **NO until #7 concretized+enabled** |
| initial empty-grasp (grasped air) | `gripper_width_at_close` ≈ open + `true_handle_error_at_close` large | **yes, once #4/#5 implemented** |
| grasp-offset miss (off-centre grasp) | `true_handle_error_at_close` large but bounded | **yes, once #4 implemented** |
| PULL-phase slip | `handle_detached` + `handle_relative_error` (already in `y`) + `failure_phase==PULL` | **yes** (already largely present) |

## Verdict
Five of the six are **implementable from signals that already exist** (IK `solve()`, per-step telemetry,
skill history) but are **not yet wired into `y`** — this is the instrumentation work a runtime-implementation
phase would build (with a schema/instrumentation smoke test). The **collision field is a preregistration
blocker**: its criterion is under-specified and the monitor is off; it must be made concrete and enabled
**before** the confirmatory run, because the failure regime it must exclude coincides with the confirmatory
operating point.
