# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GELLO source factory under the spec name. STATUS: ready (mock), wrapper (real).

Thin factory over :mod:`gello.mock_gello_source`. ``make_gello_source`` returns:
  * ``MockGelloSource`` for ``device in {"mock", "keyboard"}`` (no hardware), or
  * ``RealGelloSource`` for ``device == "gello"`` (delegates to the legacy reader
    in projects/gello_franka_teleop; raises NotImplementedError until wired).

Keeps the import path the project spec expects without duplicating the source
classes.
"""

from __future__ import annotations

from .mock_gello_source import GelloSource, MockGelloSource, RealGelloSource  # noqa: F401


def make_gello_source(device: str = "mock", config_path: str | None = None, hz: float = 100.0) -> GelloSource:
    """Build a GELLO source for the requested device.

    Args:
        device: ``"mock"`` / ``"keyboard"`` -> synthetic; ``"gello"`` -> real HW.
        config_path: gello yaml (real device only).
        hz: serial read rate for the real device (ignored by the mock).
    """
    device = (device or "mock").lower()
    if device in ("mock", "keyboard"):
        return MockGelloSource()
    if device == "gello":
        return RealGelloSource(config_path=config_path, hz=hz)
    raise ValueError(f"unknown teleop device {device!r}; use mock | keyboard | gello")
