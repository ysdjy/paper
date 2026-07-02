# PROJECT_MAP — franka_v1_skill_lab

Old module → new V1 location, and migration type.

## Legend
- **migrate**: logic re-implemented/adapted in V1 (new file is the real thing)
- **wrapper**: V1 file re-exports / delegates to the legacy implementation
- **stub**: V1 file is interface + TODO only
- **new**: did not exist before V1

## Mapping

| V1 path | Source (legacy) | type |
|---------|-----------------|------|
| `scene/scene_contract.py` | (scattered constants in entries/env cfg) | **new** |
| `scene/scene_registry.py` | `saved_scenes/`, `SceneLayoutModule/saved_scenes/` | **new** |
| `scene/v1_task_ids.py` | `TASK_ID` literal in `skill_test_ui_joint.py` | **new** |
| `scene/saved_scenes/v1_active/scene_v1_registry.json` | — | **new** |
| `scene/tools/check_scene_v1.py` | — | **new** |
| `scene/tools/print_scene_objects.py` | `SceneLayoutModule/inspect_saved_scene.py` | **wrapper** |
| `layout_editor/layout_v1_ui.py` | `SceneLayoutModule/scene_layout_ui.py` | **migrate** (adapted + registry) |
| `skill_runtime/entries/skill_test_ui_joint_v1.py` | `franka_skill_state_machine/entries/skill_test_ui_joint.py` | **wrapper** (+`--scene_registry`) |
| `skill_runtime/state_machine/skill_executor.py` | `franka_skill_state_machine/state_machine/skill_executor.py` | **wrapper** |
| `skill_runtime/skills/grasp_joint_skill.py` | `…/skills/grasp_skill.py` | **wrapper** |
| `skill_runtime/skills/place_joint_skill.py` | `…/skills/place_skill.py` | **wrapper** |
| `skill_runtime/skills/open_drawer_ik_skill.py` | `…/skills/open_drawer_skill.py` | **wrapper** |
| `skill_runtime/skills/close_drawer_ik_skill.py` | `…/skills/close_drawer_skill.py` | **wrapper** |
| `skill_runtime/skills/microwave_door_ik_skill.py` | `…/skills/microwave_door_skill.py` | **wrapper** |
| `skill_runtime/runtime/*.py` | `franka_skill_state_machine/runtime/*.py` | **wrapper** |
| `teleop_collection/gello/` (README/configs) | `projects/gello_franka_teleop/` | **wrapper** |
| `teleop_collection/gello/joint_teleop_safety.py` | plan in `…/docs/next_stage_isaac_control_plan.md` | **migrate** (new impl of the plan) |
| `teleop_collection/gello/mock_gello_source.py` | — | **new** |
| `teleop_collection/entries/gello_to_isaac_joint_test.py` | (stage-2 plan) | **migrate** (mock-runnable) |
| `teleop_collection/entries/collect_teleop_demos_joint_v1.py` | `pi05_isaacsim_baseline/scripts/collect_demos.sh` | **stub** |
| `pi05_training/WORKFLOW/*.sh` | `pi05_isaacsim_baseline/WORKFLOW/*.sh` | **wrapper** (V1 task override) |
| `pi05_training/adapters/isaaclab_obs_adapter_v1.py` | `…/scripts/isaaclab/isaac_obs_utils.py` | **migrate** (joint obs) |
| `pi05_training/adapters/joint_action_adapter_v1.py` | `…/adapters/action_adapters/` (IK-Rel) | **migrate** (joint action) |
| `pi05_training/adapters/skill_demo_to_lerobot.py` | `…/adapters/data_conversion/` | **wrapper** + validator |
| `perception_foundationpose/sim/sim_gt_pose_as_foundationpose.py` | — | **new** |
| `perception_foundationpose/sim/pose_entry.py` | `pi05_…/adapters/real_robot/foundationpose_object_stub.py` | **migrate** (unified schema) |
| `perception_foundationpose/foundationpose/foundationpose_runner.py` | `franka_d435_foundationpose/…/foundationpose/` | **stub** (façade) |
| `perception_foundationpose/foundationpose/foundationpose_object_adapter.py` | — | **new** |
| `perception_foundationpose/d435/d435_observation_adapter.py` | re-exports `sensors/d435/` | **wrapper** (moved to sensors/) |
| `perception_foundationpose/foundationpose/foundationpose_input_builder.py` | — | **new** (D435→FP bundle) |
| `perception_foundationpose/sim/test_d435_foundationpose_input.py` | — | **new** (offline FP-input test) |
| `sensors/d435/d435_config.py` | (camera consts in `stack_ik_rel_visuomotor_env_cfg.py`) | **new** (single source of truth) |
| `sensors/d435/d435_scene_cfg.py` | `…/visuomotor` `CameraCfg` | **migrate** (`attach_d435` + visible body) |
| `sensors/d435/d435_observation_adapter.py` | `franka_d435_foundationpose/…/camera/` | **migrate** (canonical RGB-D adapter) |
| `sensors/d435/d435_visual_debug.py` | — | **new** (`--show_camera_debug`) |
| `sensors/d435/test_d435_observation.py` | — | **new** (offline contract test) |
| `asset_pipeline/` | `SapienAssetPipeline/` | **wrapper** (notes only) |

## Where the real (heavy) code still lives

These are intentionally NOT moved (isolated heavy deps / proven runtime):

- Skill state machine runtime → `projects/franka_skill_state_machine/`
- Scene layout original → `SceneLayoutModule/`
- pi0.5 / OpenPI pipeline → `pi05_isaacsim_baseline/` (`.venv_openpi`)
- FoundationPose / D435 bridge → `franka_d435_foundationpose/` (own conda env)
- GELLO hardware reader → `projects/gello_franka_teleop/` (`.venv-gello`)
- SAPIEN → USD assets → `SapienAssetPipeline/`
