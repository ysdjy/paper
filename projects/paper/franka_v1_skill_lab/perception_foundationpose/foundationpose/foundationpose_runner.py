# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Real FoundationPose runner — STATUS: stub / interface only.

FoundationPose itself is a heavy CUDA dependency (torch, pytorch3d, nvdiffrast,
custom ops). It MUST run in its own environment (see
``franka_d435_foundationpose/envs/environment_foundationpose.yml``), never inside
``env_isaaclab``. This file defines the interface the V1 perception layer expects
and a mock implementation so the rest of the pipeline can be built and tested
before FoundationPose is installed.

Interface:
    runner = FoundationPoseRunner(config)
    pose_in_camera = runner.estimate(rgb, depth, K, mesh_path, mask)   # 4x4
    pose_in_camera = runner.track(rgb, depth, K)                        # 4x4

The real implementation should wrap
``franka_d435_foundationpose/franka_d435_foundationpose/foundationpose/`` (which
already has the ZMQ server + mock + mask loader) rather than re-implementing it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FoundationPoseConfig:
    repo_path: str | None = None
    weights: str | None = None
    mock: bool = True
    meshes_dir: str | None = None
    masks_dir: str | None = None


class FoundationPoseRunner:
    """Thin façade over the real FoundationPose deployment wrapper.

    In ``mock=True`` mode (default) ``estimate``/``track`` return an identity
    pose so the pipeline runs end-to-end with no GPU. In real mode it should
    delegate to the franka_d435_foundationpose wrapper / ZMQ pose server.
    """

    def __init__(self, config: FoundationPoseConfig | None = None):
        self.config = config or FoundationPoseConfig()
        self._impl = None
        if not self.config.mock:
            self._load_real()

    def _load_real(self) -> None:  # pragma: no cover - requires FoundationPose env
        raise NotImplementedError(
            "Real FoundationPose runner is not wired up in V1 yet. Install the FoundationPose "
            "env (franka_d435_foundationpose/envs/environment_foundationpose.yml) and delegate to "
            "franka_d435_foundationpose/franka_d435_foundationpose/foundationpose/. "
            "Until then run with mock=True (or pass --mock_foundationpose)."
        )

    @staticmethod
    def _identity_4x4() -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],   # 0.5 m in front of the camera (placeholder)
            [0.0, 0.0, 0.0, 1.0],
        ]

    def estimate(self, rgb, depth, K, mesh_path: str | None = None, mask=None) -> list[list[float]]:
        """Register the object from a single RGB-D frame; return T_camera_object (4x4)."""
        if self.config.mock or self._impl is None:
            return self._identity_4x4()
        raise NotImplementedError  # pragma: no cover

    def track(self, rgb, depth, K) -> list[list[float]]:
        """Track from the previous estimate; return T_camera_object (4x4)."""
        if self.config.mock or self._impl is None:
            return self._identity_4x4()
        raise NotImplementedError  # pragma: no cover
