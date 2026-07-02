# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified object-pose source for the V1 lab — STATUS: wrapper (sim-GT ready, FP stub).

This is the ONE function downstream code calls to get object poses, regardless
of whether they come from sim ground truth (stage 1) or real FoundationPose
(stage 2). Both return the same ``PoseResult`` (list of ``PoseEntry``).

Consumers:
  * pi05_training observation adapter — injects object poses into the policy obs;
  * skill_runtime debug — overlay perceived poses vs. true poses.

Usage:
    adapter = ObjectPoseAdapter(mode="sim_gt", provider=scene_state_provider)
    result = adapter.poll()                 # -> PoseResult
    obs_objects = result.to_dict()["objects"]
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.perception_foundationpose.foundationpose.foundationpose_runner import (  # noqa: E402
    FoundationPoseConfig,
    FoundationPoseRunner,
)
from franka_v1_skill_lab.perception_foundationpose.sim.pose_entry import PoseResult  # noqa: E402
from franka_v1_skill_lab.perception_foundationpose.sim.sim_gt_pose_as_foundationpose import (  # noqa: E402
    DEFAULT_TRACKED,
    poses_from_provider,
)


class ObjectPoseAdapter:
    """Pick a pose source by mode and expose a single ``poll()`` -> PoseResult."""

    def __init__(
        self,
        mode: str = "sim_gt",
        provider=None,
        tracked: tuple[str, ...] = DEFAULT_TRACKED,
        mesh_paths: dict[str, str] | None = None,
        fp_config: FoundationPoseConfig | None = None,
    ):
        if mode not in ("sim_gt", "foundationpose"):
            raise ValueError(f"unknown pose mode: {mode!r} (expected sim_gt | foundationpose)")
        self.mode = mode
        self.provider = provider
        self.tracked = tracked
        self.mesh_paths = mesh_paths or {}
        self._fp_runner: FoundationPoseRunner | None = None
        if mode == "foundationpose":
            self._fp_runner = FoundationPoseRunner(fp_config or FoundationPoseConfig(mock=True))

    def poll(self, camera_frame=None) -> PoseResult:
        """Return the current object poses as a PoseResult.

        sim_gt: reads ground truth from the IsaacLab provider.
        foundationpose: STUB — would run FoundationPose per object then chain
        T_base_object = T_base_ee @ T_ee_camera @ T_camera_object. Until the FP
        env is wired up this raises so callers don't silently get fake poses.
        """
        if self.mode == "sim_gt":
            if self.provider is None:
                raise RuntimeError("sim_gt mode needs a SceneStateProvider (provider=...).")
            return poses_from_provider(self.provider, tracked=self.tracked, mesh_paths=self.mesh_paths)

        # foundationpose mode
        raise NotImplementedError(
            "FoundationPose pose source is a stub in V1. Use mode='sim_gt' for now, or finish the "
            "FoundationPose runner (perception_foundationpose/foundationpose/foundationpose_runner.py) "
            "and chain T_base_object here."
        )
