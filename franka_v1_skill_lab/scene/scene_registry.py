# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Read / write / resolve the V1 scene registry.

The registry is a small JSON file that tells every consumer which saved scene
is *active*. The layout editor writes it after a save; the skill runtime,
teleop collection, pi0.5 training and perception modules read it to discover
the active USD + manifest.

Pure python (json + pathlib only). Safe to import from any venv.

Registry path (canonical):
    scene/saved_scenes/v1_active/scene_v1_registry.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .scene_contract import (
    SCHEMA_VERSION,
    V1_CONSUMERS,
    V1_OBJECTS,
)
from .v1_task_ids import V1_BASE_TASK_ID, V1_CONTROL_MODE


def default_sensors() -> dict:
    """The default ``sensors`` block for a fresh V1 registry (both wrist cameras).

    Pulls the two camera descriptors from the sensors package (the source of
    truth). Falls back to a minimal inline block if the sensors package is not
    importable, so the registry never fails to build.
    """
    try:
        from franka_v1_skill_lab.sensors.d435.d435_config import default_sensors_block

        return default_sensors_block(show_body=False)
    except Exception:
        return {
            "foundationpose_d435_rgbd": {
                "enabled": True, "type": "rgbd", "consumer": "foundationpose",
                "parent": "panda_hand", "rgb": True, "depth": True,
                "resolution": [640, 480], "visible_body": False,
                "frame_id": "foundationpose_d435_rgbd",
            },
            "vla_libero_eye_in_hand": {
                "enabled": True, "type": "rgb", "consumer": "vla_pi05",
                "parent": "panda_hand", "rgb": True, "depth": False,
                "resolution": [128, 128], "visible_body": False,
                "frame_id": "vla_libero_eye_in_hand", "libero_compatible": True,
            },
        }

# Resolve key locations relative to this file (…/scene/scene_registry.py).
_SCENE_DIR = Path(__file__).resolve().parent
V1_ACTIVE_DIR = _SCENE_DIR / "saved_scenes" / "v1_active"
V0_SEED_DIR = _SCENE_DIR / "saved_scenes" / "v0_seed"
DEFAULT_REGISTRY_PATH = V1_ACTIVE_DIR / "scene_v1_registry.json"

DEFAULT_ACTIVE_USD = "scene_v1_latest.usd"
DEFAULT_ACTIVE_MANIFEST = "scene_v1_latest.json"


# ---------------------------------------------------------------------------
# 写入授权闸门
# ---------------------------------------------------------------------------
# 约定：最新场景 (scene_v1_latest.usd + scene_v1_registry.json) 是“单一可信源”，
# 只能由布局编辑器 layout_editor/layout_v1_ui.py 在用户点击 **Save V1** 时写入。
# 其他任何模块（skill_runtime / teleop / pi05 / perception 等，含其它 AI 跑的脚本）
# 只允许通过 resolve_active_scene() / load_registry() 只读加载，不得修改最新场景。
#
# 为把这条约定变成硬约束：写函数（save_registry / update_active_scene）默认拒绝，
# 除非调用方在本进程内先显式调用 authorize_scene_write()。布局 UI 的保存回调会调它；
# 下游模块不会，于是任何误写都会以清晰的 PermissionError 失败而不是悄悄覆盖场景。
_SCENE_WRITE_AUTHORIZED = False


def authorize_scene_write(reason: str = "") -> None:
    """授权当前进程写入最新场景 / registry。仅供布局编辑器在 Save 时调用。"""
    global _SCENE_WRITE_AUTHORIZED
    _SCENE_WRITE_AUTHORIZED = True
    if reason:
        print(f"[scene_registry] 已授权写入场景：{reason}", flush=True)


def revoke_scene_write() -> None:
    """撤销写入授权（保存完成后可调用，恢复只读保护）。"""
    global _SCENE_WRITE_AUTHORIZED
    _SCENE_WRITE_AUTHORIZED = False


def _require_write_authorization() -> None:
    if not _SCENE_WRITE_AUTHORIZED:
        raise PermissionError(
            "最新场景 (scene_v1_latest.usd / scene_v1_registry.json) 为受保护资源，"
            "只能通过 layout_editor/layout_v1_ui.py 的 Save V1 按钮修改。"
            "下游模块请使用 resolve_active_scene() / load_registry() 只读加载；"
            "如确有正当理由需要写入，请先显式调用 authorize_scene_write(reason=...)。"
        )


@dataclass
class SceneRegistry:
    """In-memory view of ``scene_v1_registry.json``."""

    schema_version: str = SCHEMA_VERSION
    task_id: str = V1_BASE_TASK_ID
    control_mode: str = V1_CONTROL_MODE
    active_usd: str = DEFAULT_ACTIVE_USD
    active_manifest: str = DEFAULT_ACTIVE_MANIFEST
    objects: list[str] | None = None
    consumers: list[str] | None = None
    sensors: dict | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.objects is None:
            self.objects = list(V1_OBJECTS)
        if self.consumers is None:
            self.consumers = list(V1_CONSUMERS)
        if self.sensors is None:
            self.sensors = default_sensors()

    # --- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "active_usd": self.active_usd,
            "active_manifest": self.active_manifest,
            "control_mode": self.control_mode,
            "objects": list(self.objects or []),
            "consumers": list(self.consumers or []),
            "sensors": dict(self.sensors or {}),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneRegistry":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            task_id=data.get("task_id", V1_BASE_TASK_ID),
            control_mode=data.get("control_mode", V1_CONTROL_MODE),
            active_usd=data.get("active_usd", DEFAULT_ACTIVE_USD),
            active_manifest=data.get("active_manifest", DEFAULT_ACTIVE_MANIFEST),
            objects=data.get("objects"),
            consumers=data.get("consumers"),
            sensors=data.get("sensors"),
            note=data.get("note", ""),
        )

    # --- absolute path helpers ---------------------------------------------
    def usd_path(self, registry_path: Path | str | None = None) -> Path:
        base = _registry_dir(registry_path)
        return (base / self.active_usd).resolve()

    def manifest_path(self, registry_path: Path | str | None = None) -> Path:
        base = _registry_dir(registry_path)
        return (base / self.active_manifest).resolve()


def _registry_dir(registry_path: Path | str | None) -> Path:
    if registry_path is None:
        return V1_ACTIVE_DIR
    p = Path(registry_path)
    return p.parent if p.suffix == ".json" else p


def load_registry(registry_path: Path | str | None = None) -> SceneRegistry:
    """Load the registry. Returns defaults (and does NOT write) if absent."""
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    if not path.is_file():
        return SceneRegistry()
    data = json.loads(path.read_text(encoding="utf-8"))
    return SceneRegistry.from_dict(data)


def save_registry(registry: SceneRegistry, registry_path: Path | str | None = None) -> Path:
    """Write the registry JSON, creating parent dirs as needed.

    受写入授权闸门保护：未经 authorize_scene_write() 会抛 PermissionError。
    """
    _require_write_authorization()
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def update_active_scene(
    usd_name: str,
    manifest_name: str,
    registry_path: Path | str | None = None,
    note: str = "",
    sensors: dict | None = None,
) -> Path:
    """Point the registry at a new active USD + manifest and persist it.

    Called by the layout editor right after it saves ``scene_v1_latest.usd``
    and ``scene_v1_latest.json``. If ``sensors`` is given it overwrites the
    sensors block (so a save records exactly which sensors are in the scene);
    otherwise the existing sensors block is preserved.

    受写入授权闸门保护：未经 authorize_scene_write() 会抛 PermissionError。
    """
    _require_write_authorization()
    registry = load_registry(registry_path)
    registry.active_usd = usd_name
    registry.active_manifest = manifest_name
    if sensors is not None:
        registry.sensors = sensors
    if note:
        registry.note = note
    return save_registry(registry, registry_path)


def resolve_active_scene(registry_path: Path | str | None = None) -> dict:
    """Resolve the active scene to absolute paths + existence flags.

    Returns a dict the runtime can log:
        {task_id, control_mode, usd, manifest, usd_exists, manifest_exists}
    """
    registry = load_registry(registry_path)
    usd = registry.usd_path(registry_path)
    manifest = registry.manifest_path(registry_path)
    return {
        "task_id": registry.task_id,
        "control_mode": registry.control_mode,
        "objects": list(registry.objects or []),
        "sensors": dict(registry.sensors or {}),
        "usd": str(usd),
        "manifest": str(manifest),
        "usd_exists": usd.is_file(),
        "manifest_exists": manifest.is_file(),
    }
