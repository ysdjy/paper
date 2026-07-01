#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print the objects in the active V1 scene.

Two modes:
  * --from manifest (default): read scene_v1_latest.json (pure python, fast,
    no Isaac). Prints name / path / translate / orient / scale / asset refs.
  * --from usd: delegate to the legacy USD inspector
    (SceneLayoutModule/inspect_saved_scene.py) on the active V1 USD. Needs the
    isaacsim pxr libs (run under ./isaaclab.sh if pxr is not importable).

    python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py
    python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py --from usd
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.scene import load_registry, resolve_active_scene  # noqa: E402


def _print_from_manifest(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        print(f"No active V1 manifest yet: {manifest_path}")
        print("Save a scene in the layout editor first (layout_editor/layout_v1_ui.py).")
        return 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[manifest] {manifest_path}")
    print(f"[manifest] task={data.get('task')} control_mode={data.get('control_mode')}")
    print(f"[manifest] contract_objects={data.get('contract_objects')}")
    for obj in data.get("objects", []):
        xf = obj.get("xform") or {}
        print("=" * 72)
        print(f"PRIM {obj.get('path')}  ({obj.get('type')})  name={obj.get('name')}")
        if xf:
            print(f"  translate   = {xf.get('translate')}")
            print(f"  orient_wxyz = {xf.get('orient_wxyz')}")
            print(f"  scale       = {xf.get('scale')}")
        for ref in obj.get("asset_refs", []):
            print(f"  ref: {ref}")
    return 0


def _print_from_usd(usd_path: Path) -> int:
    inspector = _REPO_ROOT / "SceneLayoutModule" / "inspect_saved_scene.py"
    if not usd_path.is_file():
        print(f"No active V1 USD yet: {usd_path}")
        return 0
    if not inspector.is_file():
        print(f"Legacy USD inspector not found: {inspector}")
        return 1
    print(f"[usd] delegating to {inspector} --usd {usd_path}")
    return subprocess.call([sys.executable, str(inspector), "--usd", str(usd_path)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None)
    ap.add_argument("--from", dest="source", choices=["manifest", "usd"], default="manifest")
    args = ap.parse_args()

    info = resolve_active_scene(args.registry)
    if args.source == "manifest":
        return _print_from_manifest(Path(info["manifest"]))
    return _print_from_usd(Path(info["usd"]))


if __name__ == "__main__":
    raise SystemExit(main())
