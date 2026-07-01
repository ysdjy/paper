#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the active V1 scene registry against the V1 contract.

Pure python (no Isaac / torch). Runnable from the system interpreter:

    python projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py

Checks:
  * registry file exists and is valid JSON;
  * schema_version / task_id / control_mode match the contract;
  * registry object list == contract object list;
  * active USD / manifest existence (reported, not fatal — a fresh checkout has
    no saved scene yet and falls back to the base task cfg).

Exit code 0 = contract consistent (even if no saved USD yet); 1 = contradiction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the scene package importable (projects/ on path).
_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import (  # noqa: E402
    V1_BASE_TASK_ID,
    V1_CONTROL_MODE,
    V1_OBJECTS,
    load_registry,
    resolve_active_scene,
)
from franka_v1_skill_lab.scene.scene_contract import SCHEMA_VERSION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None, help="Path to scene_v1_registry.json")
    ap.add_argument(
        "--expect_sensor",
        default=None,
        help="Fail unless this sensor is present AND enabled in the registry (e.g. wrist_d435).",
    )
    args = ap.parse_args()

    problems: list[str] = []
    reg = load_registry(args.registry)
    info = resolve_active_scene(args.registry)
    sensors = reg.sensors or {}

    print("=== Franka V1 scene check ===")
    print(f"schema_version : {reg.schema_version}")
    print(f"task_id        : {reg.task_id}")
    print(f"control_mode   : {reg.control_mode}")
    print(f"active_usd     : {info['usd']}  (exists={info['usd_exists']})")
    print(f"active_manifest: {info['manifest']}  (exists={info['manifest_exists']})")
    print(f"objects        : {reg.objects}")
    print(f"sensors        : {sorted(sensors.keys())}")
    for name, s in sensors.items():
        en = s.get("enabled", False)
        res = s.get("resolution")
        parent = s.get("parent")
        print(f"  - {name}: enabled={en} parent={parent} rgb={s.get('rgb')} depth={s.get('depth')} res={res}")

    if reg.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {reg.schema_version!r} != contract {SCHEMA_VERSION!r}")
    if reg.task_id != V1_BASE_TASK_ID:
        problems.append(f"task_id {reg.task_id!r} != contract {V1_BASE_TASK_ID!r}")
    if reg.control_mode != V1_CONTROL_MODE:
        problems.append(f"control_mode {reg.control_mode!r} != contract {V1_CONTROL_MODE!r}")
    if set(reg.objects or []) != set(V1_OBJECTS):
        missing = set(V1_OBJECTS) - set(reg.objects or [])
        extra = set(reg.objects or []) - set(V1_OBJECTS)
        problems.append(f"object set mismatch (missing={sorted(missing)}, extra={sorted(extra)})")

    if args.expect_sensor:
        s = sensors.get(args.expect_sensor)
        if s is None:
            problems.append(f"expected sensor {args.expect_sensor!r} not in registry")
        elif not s.get("enabled", False):
            problems.append(f"expected sensor {args.expect_sensor!r} is present but enabled=False")
        else:
            print(f"\nexpected sensor : {args.expect_sensor} present and enabled — OK")

    if not info["usd_exists"]:
        print("\nNOTE: no saved V1 USD yet -> consumers fall back to the base task cfg. This is OK.")

    if problems:
        print("\nFAIL — contract contradictions:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK — registry is consistent with the V1 contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
