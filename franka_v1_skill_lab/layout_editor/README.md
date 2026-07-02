# layout_editor/ — edit & save the V1 scene

## 1. What this module does
A standalone Isaac Sim GUI to author the V1 scene: load the base task (or the
last saved V1 scene), move/add USD appliances, then save. "Save V1" writes the
canonical `scene_v1_latest.usd` + `scene_v1_latest.json` (plus a timestamped
backup) into `scene/saved_scenes/v1_active/` and updates
`scene_v1_registry.json`, so every consumer immediately sees the new scene.
Adapted from `SceneLayoutModule/scene_layout_ui.py`.

## 2. Upstream dependencies
- Isaac Sim / IsaacLab (`./isaaclab.sh`), `env_isaaclab`.
- `scene/` (registry write API), `configs/scene_v1.yaml`.
- USD appliance assets from `SapienAssetPipeline/usd_assets/`.

## 3. Downstream consumers
- `scene/saved_scenes/v1_active/` (the saved scene + registry) → everyone.

## 4. Common commands
```bash
# author from the base task
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0

# re-open the active V1 scene to keep editing
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --load_latest_v1
```

## 5. Current status
**ready** (needs GPU + display to actually launch). The save/registry logic is
implemented; `smoke_layout_v1.sh` byte-compiles it offline.

## V0 → V1 workflow
1. The legacy `SceneLayoutModule` saves `scene_v0_*.usd` snapshots — these are
   the **V0 seed**. Copy/keep your favourite under
   `scene/saved_scenes/v0_seed/` for reference.
2. Launch `layout_v1_ui.py` on the base task. The base task cfg already places
   cabinet/microwave/coffee_machine to match the V0 fixed scene.
3. Arrange objects in the viewport. Click **Save V1**. This produces
   `scene_v1_latest.usd/.json` and updates the registry.
4. The skill runtime, teleop, pi0.5 and perception modules now point at this V1
   scene via the registry.

## 6. Troubleshooting
- *Nothing saved* — you must click "Save V1"; saving on launch is intentional-off.
- *Registry not updated* — check the printed `[layout_v1] registry updated:` path;
  pass `--registry <path>` to target a non-default registry.
- *Re-open fails (`No active V1 USD yet`)* — you haven't saved a V1 scene; run
  without `--load_latest_v1` first.
- *Object missing after reload* — only objects under `/World/envs/env_0` and
  stage-root appliances are captured in the manifest; confirm placement.
