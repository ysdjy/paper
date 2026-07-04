# Band-edge collision & instrumentation v1 (read-only)

Claude B. Independent verification of the contact/collision fields and the six instrumentation groups, and
whether the boundary is a compensation effect rather than a collision / joint-limit / IK artifact. Machine
form in `band_edge_data_integrity_v1.json` + `_band_edge_analysis_bundle_v1.json`.

## Collision — `EDGE_NOT_COLLISION_DRIVEN = true`
Recomputed from all 306 records (not A's summary):
- `contact_sensor_available = true` on every episode.
- **max unintended contact force = 0.0 N**, min 0.0 N; **0 episodes** with any unintended contact frame;
  `first_unintended_contact_phase = NONE`, `contact_force_by_phase = {}` throughout.
- Backend: `isaaclab.sensors.ContactSensor(robot_contact.net_forces_w, non-finger links)`.

**Collision-threshold proposal:** `THRESHOLD_NOT_IDENTIFIABLE_FROM_FROZEN_RULE`. The frozen rule places the
threshold above the clean-baseline noise floor of the band-edge force distribution; that distribution is
**identically 0.0 N**, so there is no noise floor and any ε > 0 separates — no unique numeric threshold is
implied. B does **not** pick a value. Since there are **0** collision-confounded episodes, the full-data and
collision-excluded sensitivity results are **identical**.

## Failure mechanism vs |eff| (compensation-consistent, not collision)
| mechanism | n | |eff| min | max | mean |
|---|---|---|---|---|
| HANDLE_DETACHED / PULL | 33 | 0.0343 | 0.0592 | 0.0385 |
| POSITION_TIMEOUT / APPROACH | 62 | 0.0371 | 0.0566 | 0.0463 |

Near the edge (just past 0.034) the grasp is slightly off-centre and the handle **detaches during PULL**;
further out the grasp cannot complete the **APPROACH** and times out. Both are grasp-offset compensation
failures; neither involves contact force (all 0 N).

## Six instrumentation groups — `INSTRUMENTATION_COMPLETE = true`
Every record carries all six groups (0 missing fields across 306 episodes).
- **Joint-limit:** `joint_limit_margin_negative_flag` is set on records with a ~−1e-5 rad soft-limit
  numerical excursion (per the schema v1.1 clarification). **Real** joint-limit involvement — a *meaningful*
  negative signed margin AND `joint_limit_clamp_count > 0` — occurs in **0** episodes. Min signed margin is
  ~−1e-5 for both success and failure, i.e. the excursion is unrelated to success.
- **IK:** **0** episodes with any `ik_failure_count > 0`; max consecutive IK failures 0. The APPROACH
  timeouts are not IK solver failures.
- **Clamps:** step-clamp and limit-clamp counted separately; limit-clamp count 0 across the run.
- **Failure phase / close snapshot / gripper:** present; failures show larger close-time handle error
  (see continuous outcomes).

## `PRIMARY_EDGE_MECHANISM_NOT_EXPLAINED_BY_JOINT_OR_IK_CONFOUND = true`
With 0 real joint-limit involvements, 0 IK failures, and 0 N contact, the success boundary at |eff| ≈ 0.034
is a **grasp-offset compensation** effect, not a joint-limit / IK-unreachability / collision artifact. Do
**not** read the ~−1e-5 rad signed margin as a joint-limit failure — it is the documented soft-limit
numerical artifact.
