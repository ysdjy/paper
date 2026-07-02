#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""无头端到端:仿真抓帧 + GT -> :5601 Qwen -> xiaoyu 记忆 pipeline -> 打印 PlanningContext。

一个 env_isaaclab 进程跑仿真 + 记忆;Qwen 留在 :5601(qwen3vl 进程)不抢显存。
记忆 pipeline 用 RemoteQwenPerceiver 注入,所以 Qwen 只加载一次。

Run(需 :5601 Qwen server 已起):
    conda activate env_isaaclab
    ./isaaclab.sh -p projects/franka_v1_skill_lab/perception_qwen/memory_integration_test.py --headless
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))
# xiaoyu 记忆模块包根上 path
_MEM_PKG = _PROJECTS_DIR / "xiaoyu" / "brainary_memory_pkg"
sys.path.insert(0, str(_MEM_PKG))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode  # noqa: E402

DEFAULT_PROP_URLS = [
    "SapienAssetPipeline/usd_assets/SimReadyProps/lemon_01/lemon_01_base.usd",
    "SapienAssetPipeline/usd_assets/SimReadyProps/orange_01/orange_01_base.usd",
    "SapienAssetPipeline/usd_assets/SimReadyProps/pomegranate01/pomegranate01_base.usd",
]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Headless perception+memory integration test.")
    ap.add_argument("--qwen_addr", default="http://127.0.0.1:5601")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--out_dir", default="logs/memory_integration_test")
    ap.add_argument("--store_dir", default="logs/memory_integration_test/store")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--settle", type=int, default=15)
    return ap


def main() -> int:
    ap = build_arg_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, app_launcher, simulation_app)
    simulation_app.close()
    return rc


def _pose7(d: dict) -> list:
    """{pos[3], quat_wxyz[4]} -> [x,y,z, qw,qx,qy,qz]。"""
    p = d.get("pos", [0, 0, 0])
    q = d.get("quat_wxyz", [1, 0, 0, 0])
    return [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _scene_state_from_gt(gt: dict) -> dict:
    """把 collect_state() 的 GT 映射成记忆模块的 scene_state schema。"""
    robot = gt.get("robot", {})
    objs = gt.get("objects", {})
    nan7 = [float("nan")] * 7
    ee = robot.get("ee_pose_world")
    return {
        "cube_pose": _pose7(objs["cube_1"]) if "cube_1" in objs else nan7,
        "target_pose": _pose7(objs["cube_2"]) if "cube_2" in objs else nan7,
        "ee_pose": _pose7(ee) if ee else nan7,
        "gripper_width": float(robot.get("gripper_width_m", float("nan"))),
        "robot_joint_pos": list(robot.get("arm_joints_rad", [])),
        "step_index": 0,
    }


def _run(args, app_launcher, simulation_app) -> int:
    import numpy as np
    from PIL import Image

    from franka_v1_skill_lab.scene_interface import SceneSession

    items = [
        {"name": f"Prop_fruit_{i}", "usd_path": url, "pos": (0.45, -0.20 + 0.12 * i, 0.0),
         "scale": (1.0, 1.0, 1.0), "rigid": True, "ground": True}
        for i, url in enumerate(DEFAULT_PROP_URLS)
    ]
    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=args.task, device=args.device, headless=True,
        enable_cameras=True, enable_fp=True, free_microwave_door=False, load_latest_scene=True,
        add_microwave_stand=False, replace_microwave_with_fridge=True, lock_knife=True,
        enable_collision_monitor=False, spawn_init_markers=False, refine_handle_collisions=True,
        apply_saved_camera_offsets=True, control_hz=50.0, reset_mode=ResetMode.STATIC,
        seed=args.seed, extra_assets=tuple(items), add_robot_stand=True,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    try:
        from franka_v1_skill_lab.scene_interface import camera_offsets
        camera_offsets.apply_saved_offsets_runtime(session.env)
    except Exception as exc:  # pragma: no cover
        print(f"[mem-int] WARNING: camera offsets failed: {exc}", flush=True)

    session.reset(seed=args.seed)
    for _ in range(int(args.settle)):
        session.hold()

    from franka_v1_skill_lab.perception_qwen.remote_perceiver import RemoteQwenPerceiver
    from franka_v1_skill_lab.scene_describer.describer_client import _cap_rgb, collect_state

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    front = _cap_rgb(session.front_cam)
    wrist = _cap_rgb(session.wrist_cam)
    if front is None or wrist is None:
        print("[mem-int] ERROR: no camera RGB", flush=True)
        session.close()
        return 1
    fpaths = []
    for name, rgb in (("front", front), ("wrist", wrist)):
        fp = out / f"{name}.png"
        Image.fromarray(np.asarray(rgb)[..., :3].astype("uint8")).save(fp)
        fpaths.append(str(fp))

    gt = collect_state(session)
    scene_state = _scene_state_from_gt(gt)
    print(f"[mem-int] GT objects: {list(gt.get('objects', {}).keys())}", flush=True)
    print(f"[mem-int] scene_state: {json.dumps(scene_state, ensure_ascii=False)}", flush=True)

    # ---- 记忆 pipeline:RemoteQwenPerceiver(:5601) + 三层记忆,Qwen 不在本进程加载 ----
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig
    from memory_module import PerceptionMemoryPipeline

    perceiver = RemoteQwenPerceiver(addr=args.qwen_addr)
    memory = EmbodiedManipulationMemorySystem(
        config=MemorySystemConfig(store_dir=args.store_dir, embodiedltm_base_url=None,
                                  rig_metadata={"robot": "franka", "dof": 7, "env": "isaac_sim"})
    )
    pipe = PerceptionMemoryPipeline.create_from_existing(perceiver, memory)

    task = "pick up the cube and place it on the target"
    pipe.session_start()
    pipe.begin_episode(scene_id="isaac_ep001", task_instruction=task)

    print("[mem-int] process_perception via :5601 Qwen ...", flush=True)
    results = pipe.process_perception(
        image_paths=fpaths,
        candidate_labels=["cube", "fridge", "fruit", "dishwasher", "target", "table", "knife"],
        current_location="table",
        scene_state=scene_state,
    )
    print("\n=== RAW PERCEPTION (from :5601) ===", flush=True)
    for r in results:
        print(f"  [{Path(r.image_path).stem}] label={r.primary_label} conf={r.confidence}", flush=True)

    ctx = pipe.get_planning_context()
    print("\n=== PLANNING CONTEXT (to_prompt_text) ===", flush=True)
    print(ctx.to_prompt_text(), flush=True)
    print("\n=== fields ===", flush=True)
    print("visible_objects   :", ctx.visible_objects, flush=True)
    print("scene_description :", getattr(ctx, "scene_description", None), flush=True)
    print("recommended_skills:", ctx.recommended_skills, flush=True)
    print("similar_episodes  :", len(getattr(ctx, "similar_episodes", []) or []), flush=True)

    pipe.record_action("grasp", success=True, feedback="gripper closed")
    pipe.end_episode(success=True, blueprint_skills=["move_above", "descend", "grasp", "lift", "place", "retreat"])
    pipe.session_end()

    (out / "planning_context.json").write_text(
        json.dumps({"visible_objects": ctx.visible_objects,
                    "scene_description": getattr(ctx, "scene_description", None),
                    "task_type": getattr(ctx, "task_type", None),
                    "recommended_skills": ctx.recommended_skills}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[mem-int] DONE. store -> {args.store_dir}", flush=True)
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
