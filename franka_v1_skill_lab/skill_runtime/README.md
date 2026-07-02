# skill_runtime/ — joint-action state-machine skill tester

## 1. What this module does
The V1 entry point for testing Franka skills in the joint-action env. It wraps
the proven legacy state machine in `projects/franka_skill_state_machine/` and
adds a `--scene_registry` flag that resolves and logs the active V1 scene.

Skills and their backends (V1 mainline):
- **grasp / place** → `joint_ik` (internal DLS IK → `q_des` → joint-position action)
- **open_drawer / close_drawer** → `ik_pull` (physical grasp + pull via IK, no policy)
- **open_door / close_door** → `ik` (OpenDoorIKSkill / CloseDoorIKSkill; **WIP**, not fully stable)

## 2. Upstream dependencies
- Isaac Sim / IsaacLab (`./isaaclab.sh`), `env_isaaclab`.
- `projects/franka_skill_state_machine/` — the real runtime (`runtime/`,
  `state_machine/`, `skills/`, `learned_drawer/`). `_legacy.py` puts it on path.
- `scene/` — registry resolution.

## 3. Downstream consumers
- Humans (interactive UI). Also a reference control loop for `pi05_training`
  (how a joint action is built from `q_des`).

## 4. Common commands
```bash
# V1 joint skill UI (the mainline command)
./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
  --num_envs 1 --show_affordance_debug \
  --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
  --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
  --seed 1

# offline compile + registry check (no GPU)
bash projects/franka_v1_skill_lab/scripts/smoke_skill_ui_v1.sh
```
The legacy command still works unchanged:
```bash
./isaaclab.sh -p projects/franka_skill_state_machine/entries/skill_test_ui_joint.py \
  --num_envs 1 --show_affordance_debug --grasp_backend joint_ik \
  --place_backend joint_ik --drawer_backend ik_pull --seed 1
```

## 5. Current status
**wrapper.** The V1 entry delegates to the legacy entry via `runpy`; the
`runtime/`, `state_machine/`, `skills/` files here are re-export shims (real
code in `franka_skill_state_machine/`).
- `--scene_registry` logs provenance and validates the object contract.
- **TODO:** the registry does not yet re-apply per-object poses from
  `scene_v1_latest.json` into the live env (the base cfg already matches the V0
  fixed scene, so the running scene is correct today). See `docs/migration_report.md`.

## 6. Troubleshooting
- *`Legacy joint skill entry not found`* — `projects/franka_skill_state_machine/`
  was moved/renamed; fix the path in `_legacy.py` / the entry.
- *`unrecognized arguments: --scene_registry`* — you ran the **legacy** entry,
  not the V1 wrapper; use `skill_test_ui_joint_v1.py`.
- *Open/Close Door unstable* — known WIP; grasp/place/drawer are the stable set.
- *Drawer won't move* — use `--drawer_backend ik_pull` and the top/middle drawer
  (bottom is locked in the asset).
