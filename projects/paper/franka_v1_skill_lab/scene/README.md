# scene/ — Franka Skill Scene V1 contract + registry

## 1. What this module does
The single source of truth for the V1 scene. Defines the task id, control mode,
object list, skills, and consumers (`scene_contract.py`), the canonical task ids
(`v1_task_ids.py`), and the active-scene registry read/write API
(`scene_registry.py`). Pure python — no Isaac/torch — so any venv can import it.

## 2. Upstream dependencies
- None at runtime. Mirrors the base task `Isaac-Stack-Cube-Franka-JointPolicy-v0`
  (`source/isaaclab_tasks/.../stack/config/franka/`) and `configs/scene_v1.yaml`.

## 3. Downstream consumers
- `skill_runtime` — reads the registry via `--scene_registry`.
- `teleop_collection`, `pi05_training` — import `V1_BASE_TASK_ID`, object list.
- `perception_foundationpose` — imports the tracked-object list.
- `layout_editor` — calls `update_active_scene()` after saving.

## 3b. Sensors in the contract
The registry carries a `sensors` block (`SceneRegistry.sensors`). V1 ships one
sensor — the wrist **D435** (`sensors.wrist_d435`, mounted on `panda_hand`).
The mount geometry / intrinsics live in `sensors/d435/d435_config.py`; the scene
package only names it (`V1_SENSORS`) and stores the descriptor
(`default_sensors()`). Validate with:
```bash
python projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py --expect_sensor wrist_d435
```
See `sensors/README.md` for the full camera contract.

## 4. Common commands
```bash
# validate the registry against the contract (no GPU)
python projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py

# print the active V1 scene objects (from manifest, no GPU)
python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py
# or from the saved USD (needs isaacsim pxr)
python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py --from usd
```
```python
from franka_v1_skill_lab.scene import (
    V1_BASE_TASK_ID, V1_OBJECTS, load_registry, resolve_active_scene, update_active_scene,
)
```

## 5. Current status
**ready.** Contract + registry + tools all run with the system python. The
`v1_active/scene_v1_latest.usd|json` do not exist until the layout editor saves
one; until then consumers fall back to the base task cfg (this is expected and
`check_scene_v1.py` reports it as OK).

## 6. Troubleshooting
- *`check_scene_v1` says no saved V1 USD* — normal on a fresh checkout. Open the
  layout editor and click "Save V1".
- *`print_scene_objects --from usd` fails to import pxr* — run it under
  `./isaaclab.sh -p ...` instead of the system python (it needs the isaacsim USD libs).
- *registry object mismatch* — someone edited `scene_v1_registry.json` by hand;
  it must match `V1_OBJECTS` in `scene_contract.py`.
