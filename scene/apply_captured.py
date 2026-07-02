"""Reproduce the user's LATEST scene in the paper env from a frozen ground-truth capture.

`captured_scene_v1.json` was dumped from the running `test_mode_ui` scene (every object's runtime
world pose + scale + drawer joints). env_origin was (0,0,0), so root_pos_w == env-local pos. We bake
those exact values into the paper env cfg (no dependency on the other project's scene_sync manifest
conventions). This is the "ground-truth capture -> bake" the user approved.

Mapped articulations/rigids (cabinet/robot/appliances/cubes/knife/sektion) get their init_state
pos/rot + spawn.scale overwritten. Editor-added props (Prop_*) are spawned as static visual assets at
their captured poses (they are decorative for open_drawer; make rigid later if a grasp experiment needs
them). Cameras are added separately (need enable_cameras) via ``include_cameras``.
"""

from __future__ import annotations

import json
from pathlib import Path

MAPPED = ["cabinet", "robot", "coffee_machine", "knife", "sektion_cabinet",
          "cube_1", "cube_2", "cube_3", "dishwasher", "microwave"]

DEFAULT_CAPTURE = str(Path(__file__).with_name("captured_scene_v1.json"))


def apply_captured_scene(env_cfg, path: str = DEFAULT_CAPTURE, include_props: bool = True,
                         verbose: bool = True) -> dict:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    objs = json.loads(Path(path).read_text(encoding="utf-8"))["objects"]
    scene = env_cfg.scene
    applied, spawned_props, missing = [], [], []

    # 1) reposition + rescale existing cfg members to the captured ground truth
    for name in MAPPED:
        e = objs.get(name)
        member = getattr(scene, name, None)
        if e is None:
            continue
        if member is None:
            missing.append(name)
            continue
        ist = getattr(member, "init_state", None)
        changed = []
        if ist is not None and e.get("root_pos_w"):
            ist.pos = tuple(float(v) for v in e["root_pos_w"][:3])
            changed.append("pos")
        if ist is not None and e.get("root_quat_w"):
            ist.rot = tuple(float(v) for v in e["root_quat_w"][:4])
            changed.append("rot")
        spawn = getattr(member, "spawn", None)
        if spawn is not None and hasattr(spawn, "scale") and e.get("scale"):
            spawn.scale = tuple(float(v) for v in e["scale"][:3])
            changed.append("scale")
        applied.append(f"{name}({'+'.join(changed)})")

    # 2) spawn editor-added props as STATIC visual assets at captured poses
    if include_props:
        for name, e in objs.items():
            if not name.startswith("Prop_"):
                continue
            usd = e.get("usd_path")
            if not usd or getattr(scene, name, None) is not None:
                continue
            pos = tuple(float(v) for v in (e.get("root_pos_w") or (0.0, 0.0, 0.0))[:3])
            rot = tuple(float(v) for v in (e.get("root_quat_w") or (1.0, 0.0, 0.0, 0.0))[:4])
            scale = tuple(float(v) for v in (e.get("scale") or (1.0, 1.0, 1.0))[:3])
            setattr(scene, name, AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=UsdFileCfg(usd_path=usd, scale=scale),
            ))
            spawned_props.append(name)

    if verbose:
        print(f"[apply_captured] repositioned: {applied}", flush=True)
        print(f"[apply_captured] spawned {len(spawned_props)} props: {spawned_props}", flush=True)
        if missing:
            print(f"[apply_captured] cfg missing members (not repositioned): {missing}", flush=True)
    return {"applied": applied, "spawned_props": spawned_props, "missing": missing}
