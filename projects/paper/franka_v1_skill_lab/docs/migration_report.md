# Migration report — franka_v1_skill_lab

Honest status of the V1 reorg. **Nothing old was deleted. The mainline stays
joint-action.**

## 1. New directories created
```
projects/franka_v1_skill_lab/
  scene/  layout_editor/  skill_runtime/  teleop_collection/
  pi05_training/  perception_foundationpose/  asset_pipeline/
  configs/  scripts/  docs/
```
(full tree in `PROJECT_MAP.md`).

## 2. Migrated (new real implementation)
- `scene/` contract + registry + tools — **new**, pure-python, runs offline.
- `layout_editor/layout_v1_ui.py` — adapted from `SceneLayoutModule/scene_layout_ui.py`
  with canonical `scene_v1_latest.*` save + registry update.
- `teleop_collection/gello/joint_teleop_safety.py` + `mock_gello_source.py` +
  `entries/gello_to_isaac_joint_test.py` — implements the stage-2 plan; mock-runnable.
- `pi05_training/adapters/{isaaclab_obs_adapter_v1,joint_action_adapter_v1}.py`
  — joint obs/action (replaces IK-Rel logic).
- `perception_foundationpose/sim/{pose_entry,sim_gt_pose_as_foundationpose}.py`
  — unified PoseEntry + sim-GT source.

## 3. Wrappers (delegate to proven legacy code)
- `skill_runtime/` (entry + `runtime/`+`state_machine/`+`skills/` shims) →
  `projects/franka_skill_state_machine/`. Adds `--scene_registry`.
- `pi05_training/WORKFLOW/*.sh` → `pi05_isaacsim_baseline/WORKFLOW/` with V1 task.
- `pi05_training/adapters/skill_demo_to_lerobot.py` → legacy converter.
- `scene/tools/print_scene_objects.py` → `SceneLayoutModule/inspect_saved_scene.py`.
- `asset_pipeline/` → `SapienAssetPipeline/` (notes only).
- `teleop_collection/gello` real read → `projects/gello_franka_teleop/`.

## 4. Stubs (interface + TODO)
- `perception_foundationpose/foundationpose/foundationpose_runner.py` (mock identity pose).
- `perception_foundationpose/foundationpose/foundationpose_object_adapter.py` (`foundationpose` mode raises).
- `perception_foundationpose/d435/d435_observation_adapter.py`.
- `teleop_collection/entries/collect_teleop_demos_joint_v1.py` (`--dry_run` only).
- `teleop_collection/gello/mock_gello_source.py::RealGelloSource`.

## 5. Run commands per module
See `docs/commands.md`. Acceptance set:
| command | needs GPU? | status |
|---------|-----------|--------|
| `check_project_layout.py` | no | **passes** |
| `smoke_foundationpose_sim_v1.sh` | no | **passes** |
| `smoke_pi05_v1.sh` | no | **passes** |
| `smoke_gello_readonly.sh` | no | **passes** |
| `smoke_skill_ui_v1.sh` / `smoke_layout_v1.sh` | no (compile-check) | **passes** |
| `layout_v1_ui.py` | yes | byte-compiles; not launched here |
| `skill_test_ui_joint_v1.py` | yes | byte-compiles; not launched here |

## 6. What is NOT done yet (blockers + reasons)
1. **Scene manifest pose restore.** `--scene_registry` logs provenance and
   validates the contract but does not re-apply per-object xforms from
   `scene_v1_latest.json` onto the base env cfg. *Reason:* the base task cfg
   already matches the V0 fixed scene, so the live scene is correct; applying
   the manifest on top is a clean follow-up, not a fix.
   *Next:* in `skill_test_ui_joint_v1.py`, after `gym.make`, read the manifest
   and `provider.set_object_pose(name, xform)` for each contract object.
2. **GPU-side runs not executed here.** Layout editor, skill UI, real eval need
   a display/GPU; they only byte-compile in this environment. *Next:* run the
   two `./isaaclab.sh` commands on the workstation and confirm Save V1 writes the
   registry.
3. **Real teleop drive + collection.** `--drive_isaac` and the HDF5 recorder are
   stubs. *Reason:* need live Isaac + (mock or real) GELLO loop + HDF5 writer.
   *Next:* wire `MockGelloSource` → `make_joint_action_from_q_des` in a live env,
   then add HDF5 writing per `data_format/demo_hdf5_schema.md`.
4. **Real pi0.5 train/serve with OpenPI.** Not run (no GPU time; policy must not
   download models). *Next:* register the joint-action OpenPI config, run a short
   LoRA in `.venv_openpi`, serve, eval.
5. **Real FoundationPose.** Runner/D435 are façades. *Next:* delegate to
   `franka_d435_foundationpose/` in its conda env via the existing ZMQ server.

## 6b. Wrist cameras (TWO cameras in the shared scene)

The shared V1 scene fixes **two** wrist cameras (under
`panda_hand/` (flattened — spawner needs an existing parent)), so skill runtime, teleop, pi0.5/VLA and FoundationPose
all load the same instrumented scene:

- `foundationpose_d435_rgbd` — RGB **+ depth**, 640×480 (FoundationPose). Depth
  is REQUIRED: `capture()` raises `Depth output is not available for
  foundationpose_d435_rgbd.` rather than returning `None`.
- `vla_libero_eye_in_hand` — RGB (opt. depth), 128×128 / 224×224 (pi0.5/VLA).
  Pose from robosuite/LIBERO `robot0_eye_in_hand`; the `right_hand → panda_hand`
  alignment is **APPROXIMATE** (identity default). Looks along the wrist approach
  (eye-in-hand), verified not straight down by the axis check.

The D435 housing mesh is **hidden by default** (`--show_camera_body` reveals it;
visual only, no collider). `attach_wrist_cameras(env_cfg)` is the single attach
point; `attach_d435` remains as a back-compat shim. The registry `sensors` block
now carries both cameras.

Earlier single-camera notes (kept for history):

- **Source of truth:** `sensors/d435/d435_config.py` (pure python: mount prim,
  optical offset, intrinsics, image keys). The Isaac-side attach lives in
  `sensors/d435/d435_scene_cfg.py::attach_d435(env_cfg)` — the ONE place the
  camera is added.
- **Mount:** `{ENV_REGEX_NS}/Robot/panda_hand` (Franka wrist). Camera prim
  `wrist_d435_color`; a visible, collision-free `D435_Body` cuboid lets you see
  it in the layout editor even without camera rendering. **No collider / no rigid
  body → Franka physics unchanged.**
- **Reads:** RGB = `rgb`; depth = `distance_to_image_plane` (float32 m);
  intrinsics from the live camera (`data.intrinsic_matrices`) with a nominal
  D435 fallback (`fx=fy=610, cx=320, cy=240`); camera-to-world from
  `data.pos_w` / `data.quat_w_world`. All via
  `sensors/d435/d435_observation_adapter.py`.
- **Registry:** `scene_v1_registry.json` now has a `sensors.wrist_d435` block;
  `check_scene_v1.py --expect_sensor wrist_d435` validates it. The layout editor
  writes it on Save V1.
- **Consumers wired:** layout editor (`--enable_wrist_d435`), skill runtime
  (`--enable_wrist_d435 / --show_camera_debug / --camera_name`, via a
  `gym.make` patch so the legacy entry is untouched), teleop collect schema,
  pi0.5 `wrist_images_from_d435` → `images.wrist_rgb/.wrist_depth`, FoundationPose
  `foundationpose_input_builder`.

### Ready vs stub (D435)
- **ready (no GPU):** config, observation contract + `mock_frame`,
  `test_d435_observation.py`, registry integration, FoundationPose input builder
  + `test_d435_foundationpose_input.py`, both new smoke scripts.
- **ready (needs GPU + `--enable_cameras`):** live RGB + depth render and capture.
- **still missing / TODO:**
  1. **Depth only renders with `--enable_cameras`.** Without it, `depth` is
     `None`; the adapter/builder emit an explicit WARNING (never silent). The
     GPU render path is not exercised in this environment.
  2. **Left/right IR streams not simulated.** D435 is used as RGB-D (what
     FoundationPose needs); `wrist_d435_left_ir/right_ir` are reserved names only.
  3. **Masks are placeholders.** `mask_path` is `null`; the sim
     `semantic_segmentation` stream is available as a cheap placeholder but a
     real segmenter/FoundationPose mask is not wired.
  4. **Real FoundationPose** still runs in its own env; the builder only produces
     the input bundle (object poses come from sim-GT until the real runner lands).

## 7. Next minimal verifiable task
Run on the workstation:
```bash
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --enable_wrist_d435
# confirm the D435 body is visible on the Franka wrist, arrange objects, click
# "Save V1", then:
python projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py   # active_usd exists=True
./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
  --num_envs 1 --show_affordance_debug --grasp_backend joint_ik \
  --place_backend joint_ik --drawer_backend ik_pull \
  --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json --seed 1
```
That closes the loop: edit V1 scene → registry → skill runtime reads it.
