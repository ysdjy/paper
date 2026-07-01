"""Layout debug drawing for collision-free scene initialization."""

from __future__ import annotations

import math
import re
from typing import Any

import torch


class LayoutDebugVisualizer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stage = None
        self._curves: dict[str, Any] = {}
        if not enabled:
            return
        try:
            import omni.usd

            self._stage = omni.usd.get_context().get_stage()
            if self._stage is None:
                self.enabled = False
        except Exception:
            self.enabled = False

    def update(self, layout_manager, layout_result) -> None:
        if not self.enabled or self._stage is None:
            return
        from pxr import Gf, Usd, UsdGeom

        config = layout_manager.config
        region = config.movable_region
        z = config.ground_top_z + 0.001

        self._draw_rectangle(
            "/Visuals/Layout/movable_region",
            (region.x_min, region.y_min),
            (region.x_max, region.y_max),
            z,
            color=(0.2, 0.8, 0.8),
            width=0.004,
        )
        self._draw_circle(
            "/Visuals/Layout/robot_exclusion",
            (0.0, 0.0),
            config.robot_exclusion_radius,
            z,
            color=(1.0, 0.5, 0.0),
            width=0.004,
        )

        if layout_manager._cabinet_keepout is not None:
            min_x, min_y, max_x, max_y = layout_manager._cabinet_keepout
            self._draw_rectangle(
                "/Visuals/Layout/cabinet_keepout",
                (min_x, min_y),
                (max_x, max_y),
                z,
                color=(1.0, 0.2, 0.2),
                width=0.004,
            )
        if (
            layout_manager._drawer_sweep_center is not None
            and layout_manager._drawer_front is not None
            and layout_manager._drawer_side is not None
            and layout_manager._drawer_half_front is not None
            and layout_manager._drawer_half_side is not None
        ):
            self._draw_obb(
                "/Visuals/Layout/drawer_sweep",
                layout_manager._drawer_sweep_center,
                layout_manager._drawer_front,
                layout_manager._drawer_side,
                layout_manager._drawer_half_front,
                layout_manager._drawer_half_side,
                z,
                color=(0.5, 0.2, 1.0),
                width=0.004,
            )

        cabinet_aabb = layout_manager._compute_aabb_world(layout_manager.scene["cabinet"])
        self._draw_aabb(
            "/Visuals/Layout/cabinet_aabb",
            cabinet_aabb,
            color=(0.2, 1.0, 0.2),
            width=0.004,
        )

        for name, pose in layout_result.object_poses.items():
            radius = layout_manager._radii()[name]
            self._draw_circle(
                f"/Visuals/Layout/{name}_sample",
                (pose.position_base[0], pose.position_base[1]),
                radius,
                z,
                color=(0.8, 0.8, 0.2),
                width=0.003,
            )

    def _draw_rectangle(
        self,
        path: str,
        min_xy: tuple[float, float],
        max_xy: tuple[float, float],
        z: float,
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        min_x, min_y = min_xy
        max_x, max_y = max_xy
        points = [
            (min_x, min_y, z),
            (max_x, min_y, z),
            (max_x, max_y, z),
            (min_x, max_y, z),
            (min_x, min_y, z),
        ]
        self._draw_polyline(path, points, color=color, width=width)

    def _draw_circle(
        self,
        path: str,
        center: tuple[float, float],
        radius: float,
        z: float,
        color: tuple[float, float, float],
        width: float,
        segments: int = 48,
    ) -> None:
        cx, cy = center
        points = []
        for idx in range(segments + 1):
            angle = 2.0 * math.pi * idx / segments
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z))
        self._draw_polyline(path, points, color=color, width=width)

    def _draw_obb(
        self,
        path: str,
        center: tuple[float, float],
        front: torch.Tensor,
        side: torch.Tensor,
        half_front: float,
        half_side: float,
        z: float,
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        cx, cy = center
        front = front.detach().cpu()
        side = side.detach().cpu()
        corners = []
        for front_sign, side_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            offset = front * (front_sign * half_front) + side * (side_sign * half_side)
            corners.append((cx + float(offset[0]), cy + float(offset[1]), z))
        self._draw_polyline(path, corners, color=color, width=width)

    def _draw_aabb(
        self,
        path: str,
        aabb: tuple[float, float, float, float, float, float],
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        edges = [
            ((min_x, min_y, min_z), (max_x, min_y, min_z)),
            ((max_x, min_y, min_z), (max_x, max_y, min_z)),
            ((max_x, max_y, min_z), (min_x, max_y, min_z)),
            ((min_x, max_y, min_z), (min_x, min_y, min_z)),
            ((min_x, min_y, max_z), (max_x, min_y, max_z)),
            ((max_x, min_y, max_z), (max_x, max_y, max_z)),
            ((max_x, max_y, max_z), (min_x, max_y, max_z)),
            ((min_x, max_y, max_z), (min_x, min_y, max_z)),
            ((min_x, min_y, min_z), (min_x, min_y, max_z)),
            ((max_x, min_y, min_z), (max_x, min_y, max_z)),
            ((max_x, max_y, min_z), (max_x, max_y, max_z)),
            ((min_x, max_y, min_z), (min_x, max_y, max_z)),
        ]
        for idx, (start, end) in enumerate(edges):
            self._draw_polyline(f"{path}_{idx}", [start, end], color=color, width=width)

    def _draw_polyline(
        self,
        path: str,
        points: list[tuple[float, float, float]],
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        from pxr import Gf, UsdGeom

        safe_path = re.sub(r"[^A-Za-z0-9_/.]", "_", path)
        curve = self._curves.get(safe_path)
        if curve is None:
            curve = UsdGeom.BasisCurves.Define(self._stage, safe_path)
            curve.CreateTypeAttr("linear")
            curve.CreateBasisAttr("linear")
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            curve.CreateWidthsAttr([width])
            self._curves[safe_path] = curve
        curve.GetPointsAttr().Set([Gf.Vec3d(*point) for point in points])
        curve.GetCurveVertexCountsAttr().Set([len(points)])
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        curve.CreateWidthsAttr([width])
