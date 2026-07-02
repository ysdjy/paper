# Runtime commands (platform snapshot pre-drawer-v2)

Environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd <IsaacLab-root>      # scripts run from IsaacLab root; paper files live in projects/paper
```

## Full GUI platform (all panels)
```bash
./isaaclab.sh -p projects/paper/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
  --controller state_machine.skill_test_controller:SkillTestController \
  --ui all --no_cameras --no_stream
```

## Headless drawer regression (desktop + Sektion)
```bash
SKILL_TEST_AUTORUN="open_drawer:top_drawer,close_drawer:top_drawer,open_drawer:sektion_top_drawer,close_drawer:sektion_top_drawer" \
./isaaclab.sh -p projects/paper/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
  --controller state_machine.skill_test_controller:SkillTestController \
  --headless --no_cameras
```

## Coffee lever (world-Z sweep)
```bash
SKILL_TEST_COFFEE=1 SKILL_TEST_COFFEE_TARGET=0.55 \
./isaaclab.sh -p projects/paper/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
  --controller state_machine.skill_test_controller:SkillTestController \
  --headless --no_cameras
```

## Cleanup stuck instances (safe; only test_mode_ui)
```bash
ps -eo pid,cmd | grep '[t]est_mode_ui' | awk '{print $1}' | xargs -r kill -9
```

## Legacy v1 paper data generator (frozen; do NOT use for v2 formal data)
```bash
./isaaclab.sh -p projects/paper/deployment_calibration/data_generation/generate_open_drawer.py \
  --n_conditions 12 --candidates 3 --seed 7 --drawers top_drawer \
  --run_id <id> --output_dir projects/paper/deployment_calibration/data --headless
```
