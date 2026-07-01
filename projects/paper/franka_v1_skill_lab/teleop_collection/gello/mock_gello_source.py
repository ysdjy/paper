# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GELLO joint source abstraction with a hardware-free mock. STATUS: ready.

A ``GelloSource`` yields raw 7-DoF joint vectors (+ a gripper scalar). The real
source delegates to the legacy reader
(``projects/gello_franka_teleop/scripts/read_gello_joints.py`` /
``gello_isaac_teleop/reader.py``). The mock source generates a smooth synthetic
trajectory so teleop / collection scripts can be smoke-tested with NO hardware.

This is what makes ``--mock_gello`` / ``--no_hardware`` work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class GelloSource:
    """Interface: read() -> (q[0:7] list, gripper float in [0,1])."""

    def read(self) -> tuple[list[float], float]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass
class MockGelloSource(GelloSource):
    """Deterministic synthetic GELLO trajectory (no hardware, no randomness)."""

    num_joints: int = 7
    amplitude_rad: float = 0.2
    hz: float = 0.5
    dt: float = 1.0 / 30.0
    _t: float = 0.0

    def read(self) -> tuple[list[float], float]:
        self._t += self.dt
        phase = 2.0 * math.pi * self.hz * self._t
        q = [self.amplitude_rad * math.sin(phase + 0.3 * i) for i in range(self.num_joints)]
        # Franka joint 4 is centred near -1.5; bias it so it sits in-range.
        q[3] -= 1.5
        gripper = 0.5 * (1.0 + math.sin(phase))  # 0..1
        return q, gripper


class RealGelloSource(GelloSource):
    """Reads a real GELLO leader arm via the legacy reader. STATUS: ready.

    Delegates to ``gello_isaac_teleop.reader.ThreadedGelloReader`` (which builds on
    gello's ``DynamixelRobot``). ``dynamixel_sdk`` is importable in the
    ``env_isaaclab`` env, so this runs INSIDE the Isaac process — no IPC bridge.

    ``read()`` returns ``(arm_q[7] list rad, gripper_norm float in [0,1])`` where
    0 = open, 1 = closed (converted from the raw servo degrees using the config's
    ``gripper.open_value`` / ``close_value``). On a transient serial read failure
    it returns the last good sample so teleop never stalls.

    Default config: ``gello/configs/gello_franka.yaml`` (already calibrated for
    this GELLO: per-joint offsets/signs + gripper open/close). Override with
    ``--gello_config``.
    """

    def __init__(self, config_path: str | None = None, hz: float = 60.0, override_port: str | None = None):
        import sys
        from pathlib import Path

        import yaml

        # Resolve config: default to the V1 calibrated config next to this package.
        if config_path is None:
            config_path = str(Path(__file__).resolve().parent / "configs" / "gello_franka.yaml")
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        gcfg = cfg.get("gripper") or {}
        self._g_open = float(gcfg.get("open_value", 0.0))
        self._g_close = float(gcfg.get("close_value", 1.0))
        self._g_enabled = bool(gcfg.get("enabled", False))

        # Put the legacy reader + the vendored gello package on sys.path.
        # …/projects/franka_v1_skill_lab/teleop_collection/gello/mock_gello_source.py
        #   parents[3] = projects/  ->  projects/gello_franka_teleop
        _gello_root = Path(__file__).resolve().parents[3] / "gello_franka_teleop"
        for p in (_gello_root, _gello_root / "third_party" / "gello_software"):
            sp = str(p)
            if p.is_dir() and sp not in sys.path:
                sys.path.insert(0, sp)

        try:
            from gello_isaac_teleop.reader import GelloReader, ThreadedGelloReader
        except Exception as exc:  # pragma: no cover - import/env problem
            raise NotImplementedError(
                f"Could not import the legacy GELLO reader ({exc}). Ensure projects/gello_franka_teleop "
                "exists and dynamixel_sdk is importable in this env."
            ) from exc

        reader = GelloReader(config_path, use_gripper=self._g_enabled, override_port=override_port)
        self.num_arm = reader.num_arm
        self._reader = ThreadedGelloReader(reader, hz=hz).start()
        self._last_q = [0.0] * 7
        self._last_grip = 0.0
        print(f"[RealGelloSource] reading GELLO on {reader.port} (gripper={'on' if reader.gripper_enabled else 'off'})",
              flush=True)

    def _normalize_gripper(self, gripper_deg) -> float:
        if gripper_deg is None or self._g_close == self._g_open:
            return self._last_grip
        norm = (float(gripper_deg) - self._g_open) / (self._g_close - self._g_open)
        return 0.0 if norm < 0.0 else 1.0 if norm > 1.0 else norm

    def read(self) -> tuple[list[float], float]:
        q, gripper_deg, ok, _ms, _hz = self._reader.get_latest()
        if ok and q is not None:
            self._last_q = [float(v) for v in list(q)[:7]]
            self._last_grip = self._normalize_gripper(gripper_deg)
        return list(self._last_q), float(self._last_grip)

    def close(self) -> None:
        try:
            self._reader.stop()
        except Exception:
            pass
