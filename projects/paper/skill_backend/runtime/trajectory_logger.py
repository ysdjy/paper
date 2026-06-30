"""Step-level trajectory logger -> compressed NPZ with a UNIFIED schema (task spec section 6.2).

Every NPZ contains the SAME set of keys regardless of skill; fields a skill does not produce are
filled with NaN (numeric) / "" (string). This keeps place and drawer trajectories on one schema so
downstream analysis can treat them uniformly.

Usage::

    tl = TrajectoryLogger()
    tl.add(sim_time=t, elapsed_time=e, skill_state="PULL", joint_position=q.tolist(),
           tcp_position=p.tolist(), ...)
    tl.save("trajectories/drawer_000001.npz")

Numeric scalars -> (T,) float arrays; vectors -> (T, dim) float arrays; ``skill_state`` -> (T,) str.
``contact_force`` / ``collision_flag`` are kept NaN (no contact sensor yet, task spec section 6.2).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Canonical schema: name -> "scalar" | "str" | int(dim for a vector field)
UNIFIED_SCHEMA: dict[str, Any] = {
    # --- common ---
    "sim_time": "scalar",
    "elapsed_time": "scalar",
    "skill_state": "str",
    "joint_position": 9,
    "joint_velocity": 9,
    "tcp_position": 3,
    "tcp_orientation": 4,
    "target_tcp_position": 3,
    "target_tcp_orientation": 4,
    "tcp_position_error": "scalar",
    "tcp_orientation_error": "scalar",
    "measured_tcp_linear_speed": "scalar",
    "measured_tcp_angular_speed": "scalar",
    "gripper_command": "scalar",
    # --- place ---
    "object_position": 3,
    "object_orientation": 4,
    "object_linear_velocity": 3,
    "object_angular_velocity": 3,
    "object_position_error": "scalar",
    "object_orientation_error": "scalar",
    # --- drawer ---
    "drawer_joint_position": "scalar",
    "drawer_joint_velocity": "scalar",
    "handle_position": 3,
    "target_handle_grasp_position": 3,
    "handle_relative_position_error": "scalar",
    "drawer_progress": "scalar",
    # --- contact (no sensor yet: kept NaN) ---
    "contact_available": "scalar",
    "contact_force": "scalar",
    "collision_available": "scalar",
    "collision_flag": "scalar",
}


class TrajectoryLogger:
    def __init__(self, schema: dict[str, Any] | None = None):
        self.schema = schema or UNIFIED_SCHEMA
        self.rows: list[dict[str, Any]] = []

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def _column(self, name: str, kind: Any) -> np.ndarray:
        T = len(self.rows)
        if kind == "str":
            return np.array([str(r.get(name, "")) for r in self.rows], dtype=object)
        if kind == "scalar":
            out = np.full((T,), np.nan, dtype=np.float64)
            for i, r in enumerate(self.rows):
                v = r.get(name, None)
                if v is None:
                    continue
                if isinstance(v, bool):
                    out[i] = 1.0 if v else 0.0
                else:
                    try:
                        out[i] = float(v)
                    except (TypeError, ValueError):
                        out[i] = np.nan
            return out
        # vector of fixed dim
        dim = int(kind)
        out = np.full((T, dim), np.nan, dtype=np.float64)
        for i, r in enumerate(self.rows):
            v = r.get(name, None)
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            n = min(dim, arr.shape[0])
            out[i, :n] = arr[:n]
        return out

    def to_arrays(self) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        # canonical keys always present (uniform schema)
        for name, kind in self.schema.items():
            cols[name] = self._column(name, kind)
        # any extra keys the caller logged but not in schema -> keep as scalar columns
        extra = {k for r in self.rows for k in r} - set(self.schema)
        for name in sorted(extra):
            cols[name] = self._column(name, "scalar")
        return cols

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_arrays())
        return path


def measured_speed(prev_pos, cur_pos, control_dt: float) -> float:
    """||p_t - p_{t-1}|| / control_dt  (task spec section 2)."""
    if prev_pos is None:
        return float("nan")
    p0 = np.asarray(prev_pos, dtype=np.float64).reshape(-1)
    p1 = np.asarray(cur_pos, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(p1 - p0) / max(control_dt, 1e-9))


def measured_angular_speed(prev_quat, cur_quat, control_dt: float) -> float:
    """Geodesic quaternion angle / control_dt (wxyz). Sign-invariant."""
    if prev_quat is None:
        return float("nan")
    q0 = np.asarray(prev_quat, dtype=np.float64).reshape(-1)
    q1 = np.asarray(cur_quat, dtype=np.float64).reshape(-1)
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 < 1e-9 or n1 < 1e-9:
        return float("nan")
    q0 = q0 / n0
    q1 = q1 / n1
    dot = min(1.0, abs(float(np.dot(q0, q1))))
    angle = 2.0 * math.acos(dot)
    return float(angle / max(control_dt, 1e-9))
