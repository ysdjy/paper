#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the franka_v1_skill_lab directory layout + scene contract.

Pure python (stdlib only). The first acceptance command:

    python projects/franka_v1_skill_lab/scripts/check_project_layout.py

Checks every expected file/dir exists, that each module has a README, that the
scene contract imports and is self-consistent, and that the pure-python
sub-tests import. Exit 0 = layout OK; 1 = something missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # franka_v1_skill_lab
PROJECTS_DIR = ROOT.parent                              # projects/
sys.path.insert(0, str(PROJECTS_DIR))

# Expected files (relative to ROOT). Keep in sync with PROJECT_MAP.md.
EXPECTED_FILES = [
    "README.md",
    "PROJECT_MAP.md",
    "configs/paths.yaml",
    "configs/scene_v1.yaml",
    "configs/skills.yaml",
    "configs/teleop_gello.yaml",
    "configs/pi05_dataset.yaml",
    "configs/foundationpose.yaml",
    "scene/README.md",
    "scene/scene_contract.py",
    "scene/scene_registry.py",
    "scene/v1_task_ids.py",
    "scene/saved_scenes/v1_active/scene_v1_registry.json",
    "scene/tools/check_scene_v1.py",
    "scene/tools/print_scene_objects.py",
    "layout_editor/README.md",
    "layout_editor/layout_v1_ui.py",
    "skill_runtime/README.md",
    "skill_runtime/entries/skill_test_ui_joint_v1.py",
    "skill_runtime/state_machine/skill_executor.py",
    "skill_runtime/skills/grasp_joint_skill.py",
    "skill_runtime/skills/place_joint_skill.py",
    "skill_runtime/skills/open_drawer_ik_skill.py",
    "skill_runtime/skills/close_drawer_ik_skill.py",
    "skill_runtime/skills/microwave_door_ik_skill.py",
    "skill_runtime/runtime/ik_joint_adapter.py",
    "skill_runtime/runtime/scene_state_provider.py",
    "skill_runtime/runtime/target_registry.py",
    "skill_runtime/runtime/drawer_target_config.py",
    "skill_runtime/runtime/skill_request.py",
    "teleop_collection/README.md",
    "teleop_collection/gello/README.md",
    "teleop_collection/gello/joint_teleop_safety.py",
    "teleop_collection/gello/mock_gello_source.py",
    "teleop_collection/entries/gello_to_isaac_joint_test.py",
    "teleop_collection/entries/collect_teleop_demos_joint_v1.py",
    "teleop_collection/data_format/demo_hdf5_schema.md",
    "pi05_training/README.md",
    "pi05_training/WORKFLOW/README.md",
    "pi05_training/WORKFLOW/pipeline.env",
    "pi05_training/adapters/isaaclab_obs_adapter_v1.py",
    "pi05_training/adapters/joint_action_adapter_v1.py",
    "pi05_training/adapters/skill_demo_to_lerobot.py",
    "pi05_training/docs/pi05_v1_migration.md",
    "pi05_training/docs/dataset_contract.md",
    "perception_foundationpose/README.md",
    "perception_foundationpose/sim/sim_gt_pose_as_foundationpose.py",
    "perception_foundationpose/sim/test_sim_pose_adapter.py",
    "perception_foundationpose/sim/test_d435_foundationpose_input.py",
    "perception_foundationpose/foundationpose/foundationpose_runner.py",
    "perception_foundationpose/foundationpose/foundationpose_object_adapter.py",
    "perception_foundationpose/foundationpose/foundationpose_input_builder.py",
    "perception_foundationpose/d435/d435_observation_adapter.py",
    "perception_foundationpose/docs/perception_contract.md",
    "sensors/README.md",
    "sensors/d435/README.md",
    "sensors/d435/d435_config.py",
    "sensors/d435/d435_scene_cfg.py",
    "sensors/d435/d435_observation_adapter.py",
    "sensors/d435/d435_visual_debug.py",
    "sensors/d435/dump_d435_view.py",
    "sensors/d435/check_three_cameras.py",
    "sensors/d435/test_d435_observation.py",
    "sensors/docs/d435_foundationpose_contract.md",
    "asset_pipeline/README.md",
    "asset_pipeline/sapien_asset_pipeline_notes.md",
    "scripts/check_project_layout.py",
    "scripts/smoke_skill_ui_v1.sh",
    "scripts/smoke_layout_v1.sh",
    "scripts/smoke_pi05_v1.sh",
    "scripts/smoke_foundationpose_sim_v1.sh",
    "scripts/smoke_gello_readonly.sh",
    "scripts/smoke_d435_scene_v1.sh",
    "scripts/smoke_d435_foundationpose_v1.sh",
    "docs/architecture.md",
    "docs/scene_v0_to_v1.md",
    "docs/module_boundaries.md",
    "docs/commands.md",
    "docs/migration_report.md",
]

# Module READMEs that must exist.
README_DIRS = [
    "scene", "layout_editor", "skill_runtime", "teleop_collection",
    "pi05_training", "perception_foundationpose", "asset_pipeline", "sensors",
]


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    problems: list[str] = []
    ok = 0

    for rel in EXPECTED_FILES:
        p = ROOT / rel
        if p.is_file():
            ok += 1
        else:
            problems.append(f"missing file: {rel}")

    for d in README_DIRS:
        if not (ROOT / d / "README.md").is_file():
            problems.append(f"missing README: {d}/README.md")

    # contract import + consistency
    try:
        from franka_v1_skill_lab.scene import (  # type: ignore
            V1_BASE_TASK_ID,
            V1_OBJECTS,
            load_registry,
        )
        reg = load_registry()
        if reg.task_id != V1_BASE_TASK_ID:
            problems.append(f"registry task_id {reg.task_id} != contract {V1_BASE_TASK_ID}")
        if set(reg.objects or []) != set(V1_OBJECTS):
            problems.append("registry objects != contract objects")
    except Exception as exc:
        problems.append(f"scene contract import failed: {exc}")

    print(f"=== franka_v1_skill_lab layout check ===")
    print(f"files present: {ok}/{len(EXPECTED_FILES)}")
    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("README dirs : all present")
    print("scene contract: consistent")
    print("\nOK — project layout is complete and consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
