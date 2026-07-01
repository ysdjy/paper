"""Small USD pose-frame visualizer used by the skill test entry."""

from __future__ import annotations

import re

import torch


DEFAULT_AXIS_LENGTH = 0.045
DEFAULT_AXIS_WIDTH = 0.006
OBJECT_AXIS_LENGTH = 0.045
OBJECT_AXIS_WIDTH = 0.003


class DebugVisualizer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._items = {}
        self._arrow_items = {}
        self._stage = None
        if enabled:
            try:
                import omni.usd

                self._stage = omni.usd.get_context().get_stage()
            except Exception:
                self.enabled = False

    def update_pose(
        self,
        name: str,
        pose_tensor: torch.Tensor | None,
        axis_length: float = DEFAULT_AXIS_LENGTH,
        axis_width: float = DEFAULT_AXIS_WIDTH,
        use_coordinate_arrows: bool = False,
    ):
        if not self.enabled or self._stage is None or pose_tensor is None:
            return
        if use_coordinate_arrows:
            self._update_coordinate_arrows(name, pose_tensor, axis_length)
            return
        try:
            from pxr import Gf, UsdGeom
        except Exception:
            return
        path = "/Visuals/SkillRuntime/" + re.sub(r"[^A-Za-z0-9_]", "_", name)
        curves = self._items.get(path)
        if curves is None:
            curves = self._define_axes(path, axis_width)
            self._items[path] = curves
        pos = pose_tensor[:3].detach()
        quat = pose_tensor[3:7].detach()
        axes = self._rotated_axes(quat, pos.device, pos.dtype) * axis_length
        start = pos.detach().cpu().tolist()
        for curve, axis in zip(curves, axes):
            end = (pos + axis).detach().cpu().tolist()
            curve.GetPointsAttr().Set([Gf.Vec3d(*start), Gf.Vec3d(*end)])
            curve.GetCurveVertexCountsAttr().Set([2])
            curve.GetWidthsAttr().Set([axis_width])

    def _update_coordinate_arrows(self, name: str, pose_tensor: torch.Tensor, axis_length: float):
        try:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import FRAME_MARKER_CFG
        except Exception:
            return
        path = "/Visuals/SkillRuntime/" + re.sub(r"[^A-Za-z0-9_]", "_", name)
        marker = self._arrow_items.get(path)
        if marker is None:
            cfg = FRAME_MARKER_CFG.copy()
            cfg.prim_path = path
            cfg.markers["frame"].scale = (axis_length, axis_length, axis_length)
            marker = VisualizationMarkers(cfg)
            self._arrow_items[path] = marker
        marker.visualize(
            translations=pose_tensor[:3].detach().reshape(1, 3),
            orientations=pose_tensor[3:7].detach().reshape(1, 4),
            marker_indices=[0],
        )

    def _define_axes(self, path: str, axis_width: float):
        from pxr import UsdGeom

        curves = []
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.35, 1.0)]
        for axis_name, color in zip(("x", "y", "z"), colors):
            curve = UsdGeom.BasisCurves.Define(self._stage, f"{path}_{axis_name}")
            curve.CreateTypeAttr("linear")
            curve.CreateBasisAttr("bezier")
            curve.CreateDisplayColorAttr([color])
            curve.CreateWidthsAttr([axis_width])
            curve.CreateCurveVertexCountsAttr([2])
            curves.append(curve)
        return curves

    def _rotated_axes(self, quat: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        local_axes = torch.eye(3, device=device, dtype=dtype)
        q = quat.to(device=device, dtype=dtype)
        q = q / torch.linalg.norm(q).clamp_min(1.0e-8)
        xyz = q[1:].unsqueeze(0).expand_as(local_axes)
        t = torch.linalg.cross(xyz, local_axes, dim=-1) * 2.0
        return local_axes + q[0] * t + torch.linalg.cross(xyz, t, dim=-1)
