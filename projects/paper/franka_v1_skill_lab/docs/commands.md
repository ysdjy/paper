# Commands — franka_v1_skill_lab

All paths relative to the IsaacLab repo root. `python3` = system interpreter
(pure-python checks); `./isaaclab.sh -p` = IsaacLab interpreter (GPU/sim).

## Layout / contract checks (no GPU)
```bash
python projects/franka_v1_skill_lab/scripts/check_project_layout.py
python projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py
python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py
```

## Offline smoke tests (no GPU, no hardware)
```bash
bash projects/franka_v1_skill_lab/scripts/smoke_foundationpose_sim_v1.sh
bash projects/franka_v1_skill_lab/scripts/smoke_pi05_v1.sh
bash projects/franka_v1_skill_lab/scripts/smoke_gello_readonly.sh
bash projects/franka_v1_skill_lab/scripts/smoke_skill_ui_v1.sh    # compile-check + prints run cmd
bash projects/franka_v1_skill_lab/scripts/smoke_layout_v1.sh      # compile-check + prints run cmd
```

## Layout editor (GPU + display)
```bash
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0
# re-open the active V1 scene:
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --load_latest_v1
```

## Skill runtime (GPU + display)
```bash
./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
  --num_envs 1 --show_affordance_debug \
  --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
  --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
  --seed 1
```

## Teleop (mock = no hardware)
```bash
python projects/franka_v1_skill_lab/teleop_collection/entries/gello_to_isaac_joint_test.py \
  --mock_gello --steps 30
python projects/franka_v1_skill_lab/teleop_collection/entries/collect_teleop_demos_joint_v1.py --dry_run
# real GELLO read-only (legacy, isolated venv):
cd projects/gello_franka_teleop && source .venv-gello/bin/activate
python scripts/read_gello_joints.py --config configs/gello_franka.yaml --hz 30
```

## pi0.5 pipeline
```bash
source projects/franka_v1_skill_lab/pi05_training/WORKFLOW/pipeline.env
bash projects/franka_v1_skill_lab/pi05_training/WORKFLOW/2_convert.sh --dry_run
bash projects/franka_v1_skill_lab/pi05_training/WORKFLOW/4_serve.sh        # mock server
bash projects/franka_v1_skill_lab/pi05_training/WORKFLOW/5_eval.sh         # prints real eval cmd
# real eval (needs running policy server):
./isaaclab.sh -p pi05_isaacsim_baseline/scripts/isaaclab/run_policy_in_isaaclab.py \
  --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --num_rollouts 5 --policy_port 8008
```

## Perception
```bash
python projects/franka_v1_skill_lab/perception_foundationpose/sim/sim_gt_pose_as_foundationpose.py --demo
```
