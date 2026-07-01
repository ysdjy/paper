# Scene V0 → V1

## What V0 was
`SceneLayoutModule/scene_layout_ui.py` loads the base task, lets you move/add
appliances, and saves timestamped `scene_v0_*.usd` + `*.json` snapshots into
`SceneLayoutModule/saved_scenes/`. There was no single "active" pointer and no
shared registry — downstream code re-derived object poses ad hoc (e.g.
`inspect_saved_scene.py` → hand-edit the env cfg).

## What V1 adds
1. **A registry.** `scene/saved_scenes/v1_active/scene_v1_registry.json` names
   the one active scene (`active_usd`, `active_manifest`) for all consumers.
2. **Canonical names.** The layout editor saves `scene_v1_latest.usd/.json`
   (plus timestamped backups) instead of only timestamped files.
3. **A typed contract.** `scene/scene_contract.py` fixes the task id, control
   mode, object list, skills, consumers — imported, not copy-pasted.
4. **One save action** (`Save V1`) that writes USD + JSON + updates the registry.

## How to migrate a V0 scene to V1
```bash
# (a) keep a V0 snapshot for reference
cp SceneLayoutModule/saved_scenes/scene_v0_<ts>.usd \
   projects/franka_v1_skill_lab/scene/saved_scenes/v0_seed/

# (b) open the V1 editor on the base task and arrange objects
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0
#   → click "Save V1 (USD+JSON+registry)"

# (c) verify
python projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py
python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py
```

## V0 kept on purpose
`SceneLayoutModule/` is **not** deleted. It remains the original V0 editor and
the source of the V0 seed scenes. V1 only adds the registry-aware editor and the
contract on top.

## Known gap
The skill runtime / pi0.5 still load the **base task cfg**, which already encodes
the V0 fixed-scene poses for cabinet/microwave/coffee_machine. Re-applying the
saved V1 manifest's per-object xforms on top of the base cfg at load time is the
remaining migration step (see `migration_report.md`). Until then, "active V1
scene" = base task cfg + (provenance from) the registry.
