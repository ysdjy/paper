#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""无头 4 相机感知 + 记忆集成测试(生成报告数据)。

流程:启动场景(4 相机 front/wrist/left/right, 桌面 3 cube) -> 抓 4 视角 RGB + GT ->
:5601 Qwen 识别 4 视角(对 GT 评感知) -> 真实感知喂记忆 pipeline 跑 5 回合(触发归纳,看记忆积累)
-> 落盘 report_sim.json + 4 张 PNG。

Run(需 :5601 Qwen 已起 + conda activate env_isaaclab):
    ./isaaclab.sh -p projects/franka_v1_skill_lab/perception_qwen/memory_perception_report_sim.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECTS_DIR))
sys.path.insert(0, str(_PROJECTS_DIR / "xiaoyu" / "brainary_memory_pkg"))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID  # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode  # noqa: E402

_CUBE_WORDS = ("cube", "方块", "立方", "block", "方块体", "正方体")
_LABELS = ["cube", "block", "knife", "fridge", "table", "gripper", "robot arm", "fruit"]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Headless 4-cam perception + memory report.")
    ap.add_argument("--qwen_addr", default="http://127.0.0.1:5601")
    ap.add_argument("--task", default=V1_BASE_TASK_ID)
    ap.add_argument("--out_dir", default="logs/mem_perc_report")
    ap.add_argument("--store_dir", default="logs/mem_perc_report/store")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=5)
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


def _has_cube(rec) -> bool:
    hay = (rec.primary_label + " " + rec.scene + " "
           + " ".join(str(o.get("name", "")) for o in rec.objects)).lower()
    return any(w in hay for w in _CUBE_WORDS)


def _run(args, app_launcher, simulation_app) -> int:
    import numpy as np
    from PIL import Image

    from franka_v1_skill_lab.scene_interface import SceneSession

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=args.task, device=args.device, headless=True,
        enable_cameras=True, enable_fp=True, free_microwave_door=False, load_latest_scene=True,
        add_microwave_stand=False, replace_microwave_with_fridge=True, lock_knife=True,
        enable_collision_monitor=False, spawn_init_markers=False, refine_handle_collisions=True,
        apply_saved_camera_offsets=True, control_hz=50.0, reset_mode=ResetMode.STATIC,
        seed=args.seed, add_robot_stand=True,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    try:
        from franka_v1_skill_lab.scene_interface import camera_offsets
        camera_offsets.apply_saved_offsets_runtime(session.env)
    except Exception as exc:  # pragma: no cover
        print(f"[report] WARNING: camera offsets: {exc}", flush=True)

    session.reset(seed=args.seed)
    for _ in range(int(args.settle)):
        session.hold()

    from franka_v1_skill_lab.perception_qwen.remote_perceiver import RemoteQwenPerceiver
    from franka_v1_skill_lab.scene_describer.describer_client import _cap_rgb, collect_state

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 抓 4 视角 ----
    cams = {"front": session.front_cam, "wrist": session.wrist_cam,
            "left": getattr(session, "left_cam", None), "right": getattr(session, "right_cam", None)}
    view_paths = {}
    for name, cam in cams.items():
        if cam is None or not cam.is_available():
            print(f"[report] view {name} unavailable", flush=True)
            continue
        rgb = _cap_rgb(cam)
        if rgb is None:
            continue
        fp = out / f"{name}.png"
        Image.fromarray(np.asarray(rgb)[..., :3].astype("uint8")).save(fp)
        view_paths[name] = str(fp)
    print(f"[report] captured views: {list(view_paths)}", flush=True)

    # ---- GT ----
    gt = collect_state(session)
    objs = gt.get("objects", {})
    cubes = {k: v for k, v in objs.items() if "cube" in k.lower()}
    def _on_table(p):
        z = p["pos"][2]
        return -0.1 < z < 0.5 and abs(p["pos"][0]) < 2 and abs(p["pos"][1]) < 2
    cubes_on_table = {k: v for k, v in cubes.items() if _on_table(v)}
    print(f"[report] GT objects={list(objs)} cubes={list(cubes)} on_table={list(cubes_on_table)}", flush=True)

    # ---- 感知:Qwen 识别每个视角 ----
    perceiver = RemoteQwenPerceiver(addr=args.qwen_addr)
    names = list(view_paths.keys())
    results = perceiver.recognize_batch([view_paths[n] for n in names], candidate_labels=_LABELS)
    per_view = []
    for n, r in zip(names, results):
        per_view.append({"view": n, "primary_label": r.primary_label, "confidence": r.confidence,
                         "scene": r.scene, "objects": r.objects, "cube_detected": _has_cube(r)})
        print(f"[report] {n}: label={r.primary_label} cube_detected={_has_cube(r)}", flush=True)
    n_cube_views = sum(1 for v in per_view if v["cube_detected"])

    # ---- 集成记忆:真实感知喂记忆,跑 N 回合看积累 ----
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig
    from memory_module import PerceptionMemoryPipeline

    memory = EmbodiedManipulationMemorySystem(
        config=MemorySystemConfig(store_dir=args.store_dir, embodiedltm_base_url=None,
                                  rig_metadata={"robot": "franka", "dof": 7, "env": "isaac_sim"}))
    pipe = PerceptionMemoryPipeline.create_from_existing(perceiver, memory)
    good = ["move_above", "descend", "grasp", "lift", "place", "retreat"]
    task = "pick up the cube and place it on the target"
    # scene_state:用第一个在桌 cube 作 cube_pose,第二个作 target
    ck = list(cubes_on_table) or list(cubes)
    def _pose7(k):
        p = objs[k]; return [*[float(x) for x in p["pos"]], *[float(x) for x in p["quat_wxyz"]]]
    scene_state = {
        "cube_pose": _pose7(ck[0]) if ck else [float("nan")] * 7,
        "target_pose": _pose7(ck[1]) if len(ck) > 1 else [float("nan")] * 7,
        "ee_pose": [*gt["robot"]["ee_pose_world"]["pos"], *gt["robot"]["ee_pose_world"]["quat_wxyz"]],
        "gripper_width": gt["robot"]["gripper_width_m"], "robot_joint_pos": gt["robot"]["arm_joints_rad"],
        "step_index": 0,
    }

    pipe.session_start()
    for ep in range(int(args.episodes)):
        pipe.begin_episode(scene_id=f"isaac_ep{ep}", task_instruction=task)
        if ep == 0:
            # 第 1 回合:真实感知(front+wrist 两图,省时间)喂记忆
            pipe.process_perception(
                image_paths=[view_paths[n] for n in names if n in ("front", "wrist")] or [view_paths[names[0]]],
                candidate_labels=_LABELS, current_location="table", scene_state=scene_state)
        else:
            memory.update_observation(visible_objects=["cube", "target"], current_location="table")
            memory.ingest_scene_state(scene_state)
        pipe.record_action("grasp cube", success=True, feedback="gripper closed")
        pipe.end_episode(success=True, blueprint_skills=good)
    ctx = pipe.get_planning_context()
    schema = memory.query_task_schema(task)
    cube_kb = memory.query_object("cube")
    pipe.session_end()

    # ---- 持久化校验 ----
    mem2 = EmbodiedManipulationMemorySystem(
        config=MemorySystemConfig(store_dir=args.store_dir, embodiedltm_base_url=None))
    schema2 = mem2.query_task_schema(task)

    report = {
        "scene": {
            "views_captured": list(view_paths),
            "n_views": len(view_paths),
            "gt_objects": list(objs),
            "gt_cubes": {k: objs[k]["pos"] for k in cubes},
            "cubes_on_table": list(cubes_on_table),
        },
        "perception": {
            "per_view": per_view,
            "n_views_detected_cube": n_cube_views,
            "cube_detection_rate": round(n_cube_views / max(len(per_view), 1), 3),
        },
        "memory": {
            "episodes_run": int(args.episodes),
            "task_type": schema["task_type"],
            "success_rate_after": schema["success_rate"],
            "recommended_skills_after": schema["blueprint_skills"],
            "cube_affordances": cube_kb["affordances"],
            "cube_likely_locations": cube_kb["likely_locations"],
            "planning_context_visible": ctx.visible_objects,
            "planning_context_text": ctx.to_prompt_text(),
            "persistence_reload_success_rate": schema2["success_rate"],
            "persistence_ok": schema["success_rate"] == schema2["success_rate"],
            "snapshot": memory.snapshot(),                       # 三层记忆全量快照
            "planning_context_full": ctx.to_dict(),             # 完整 PlanningContext
        },
    }
    # 落盘的记忆文件内容(给用户直接看)
    try:
        store = Path(args.store_dir)
        report["memory"]["store_files"] = {
            "semantic_kb": json.loads((store / "semantic_kb.json").read_text()) if (store / "semantic_kb.json").exists() else None,
            "episodes_meta": json.loads((store / "episodes_meta.json").read_text()) if (store / "episodes_meta.json").exists() else None,
            "n_episode_lines": sum(1 for _ in open(store / "episodes.jsonl")) if (store / "episodes.jsonl").exists() else 0,
        }
    except Exception as exc:  # pragma: no cover
        report["memory"]["store_files"] = {"error": str(exc)}
    (out / "report_sim.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== REPORT (sim) ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\n[report] saved -> {out/'report_sim.json'}", flush=True)
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
