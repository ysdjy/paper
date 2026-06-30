# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""cuRobo collision-aware transit planner for the joint-action state machine.

Wraps IsaacLab's CuroboPlanner to plan a collision-free JOINT trajectory to a target TCP pose, for
SAFE AIR-TRANSFER between the pure-physical contact skills (grasp/place/open-door/open-drawer). The
contact skills are unchanged; this only replaces the straight-line IK used to *fly* the arm between
them.

Why the extra cuboids: the cabinet / microwave / coffee machine are INSTANCED USD prims, which
cuRobo's UsdHelper.get_obstacles_from_stage() skips (only the non-instanced handle proxies, cubes,
and the microwave stand get auto-extracted). So we add axis-aligned CUBOID obstacles for those big
bodies, read from their live world AABB (BBoxCache reads instanced bounds fine). Optionally add the
open-drawer box dynamically.

Build the planner ONCE (warmup ~12 s); reuse it for every move.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

# big static appliance bodies that cuRobo's mesh extraction misses (instanced) -> add as cuboids.
# Each: (cuboid_name, prim_path_under_env).  link paths so the box hugs the body, not the whole asset.
DEFAULT_BODY_SPECS = [
    ("microwave_body", "Microwave/link_1"),
    ("cabinet_body", "Cabinet/link_3"),     # link_3 = cabinet frame/carcass
    ("coffee_body", "CoffeeMachine/link_0"),
]


def _world_aabb(stage, prim_path):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return [float(mn[i]) for i in range(3)], [float(mx[i]) for i in range(3)]


class CuroboTransit:
    """Holds one CuroboPlanner; lets you (re)build the collision world (auto obstacles + appliance
    cuboids) and plan a collision-free joint trajectory to a TCP pose."""

    def __init__(self, env, robot, env_id: int = 0, world_config_file: str = "collision_base.yml"):
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        cfg = CuroboPlannerCfg.franka_config()
        cfg.world_config_file = world_config_file  # table-less base (our scene is ground-mounted)
        self.env = env
        self.robot = robot
        self.env_id = env_id
        self.planner = CuroboPlanner(env=env, robot=robot, config=cfg)
        self._extra_cuboids = []
        self._extra_meshes = []

    # ---- world ----------------------------------------------------------------
    def build_appliance_cuboids(self, body_specs=None, pad: float = 0.02, extra=None, extra_oriented=None):
        """Compute cuboid obstacles for the big instanced bodies from their live world AABBs.

        ``extra`` is an optional list of (name, (min_xyz), (max_xyz)) for axis-aligned dynamic boxes.
        ``extra_oriented`` is an optional list of (name, center_xyz, quat_wxyz, dims_xyz) for ORIENTED
        boxes (e.g. the OPEN microwave DOOR, posed from its live physics body so it is not stale)."""
        from curobo.geom.types import Cuboid

        stage = self.env.scene.stage
        specs = DEFAULT_BODY_SPECS if body_specs is None else body_specs
        cuboids = []
        env_prefix = f"/World/envs/env_{self.env_id}/"
        for name, rel in specs:
            ab = _world_aabb(stage, env_prefix + rel)
            if ab is None:
                continue
            mn, mx = ab
            dims = [max(0.02, (mx[i] - mn[i]) + 2 * pad) for i in range(3)]
            center = [0.5 * (mn[i] + mx[i]) for i in range(3)]
            cuboids.append(Cuboid(name=name, pose=[*center, 1.0, 0.0, 0.0, 0.0], dims=dims))
        for name, mn, mx in (extra or []):
            dims = [max(0.02, (mx[i] - mn[i]) + 2 * pad) for i in range(3)]
            center = [0.5 * (mn[i] + mx[i]) for i in range(3)]
            cuboids.append(Cuboid(name=name, pose=[*center, 1.0, 0.0, 0.0, 0.0], dims=dims))
        for name, center, quat, dims in (extra_oriented or []):
            d = [max(0.02, float(dims[i]) + 2 * pad) for i in range(3)]
            cuboids.append(Cuboid(name=name, pose=[float(center[0]), float(center[1]), float(center[2]),
                                                   float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                                  dims=d))
        self._extra_cuboids = cuboids
        return cuboids

    def set_meshes(self, meshes):
        """Set extra cuRobo Mesh obstacles (e.g. the OPEN door's real collision mesh, posed live).

        ``meshes`` is a list of curobo.geom.types.Mesh. Replaces the previous set; call before
        refresh_world()."""
        self._extra_meshes = list(meshes or [])

    def refresh_world(self):
        """Re-extract auto obstacles (cubes/proxies, live poses) and merge the appliance cuboids+meshes."""
        from curobo.geom.types import WorldConfig

        self.planner._initialize_static_world()  # re-extract auto obstacles at current poses
        wc = self.planner._static_world_config
        combined = WorldConfig(
            cuboid=list(getattr(wc, "cuboid", []) or []) + list(self._extra_cuboids),
            mesh=list(getattr(wc, "mesh", []) or []) + list(self._extra_meshes),
        )
        self.planner.motion_gen.update_world(combined.get_collision_check_world())
        return combined

    def world_object_names(self):
        try:
            return self.planner._get_world_object_names()
        except Exception:
            return []

    # ---- planning -------------------------------------------------------------
    def plan(self, pos, quat):
        """Plan to a TCP pose. pos=(x,y,z) world, quat=(w,x,y,z). Returns Nx7 joint trajectory or None."""
        device = self.robot.device
        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
        quat_t = torch.as_tensor(quat, dtype=torch.float32, device=device)
        rot = math_utils.matrix_from_quat(quat_t.unsqueeze(0))[0]
        ee_goal = math_utils.make_pose(pos_t, rot)
        ok = self.planner.plan_motion(ee_goal)
        if not ok or self.planner.current_plan is None:
            return None
        traj = self.planner.current_plan.position
        if not isinstance(traj, torch.Tensor):
            traj = torch.as_tensor(traj, dtype=torch.float32, device=device)
        return traj
