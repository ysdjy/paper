"""Robot body-collision monitor (reads the ContactSensor net forces).

Reads ``scene['robot_contact'].data.net_forces_w`` (per robot link, set up by
``scene_props.install_collision_monitor``). A link whose net contact-force magnitude exceeds the
threshold is touching something. The two gripper FINGERS are excluded from the "body" set because
finger contact is normal grasping; every other link (arm links + the hand) counts as the robot BODY.

Intended use: while the robot is NOT holding an object and IS moving, any body-link contact is almost
certainly the robot accidentally bumping the scene (e.g. an arm/wrist link knocking the fridge door
shut on retreat). The skill controller calls this each step and logs such events.
"""

from __future__ import annotations

import torch

# Only the two gripper fingers are "allowed" contact (grasping); everything else = robot body.
_FINGER_KEYS = ("leftfinger", "rightfinger")


class CollisionMonitor:
    def __init__(self, scene, env_id: int = 0, sensor_name: str = "robot_contact",
                 force_threshold: float = 2.0):
        self.env_id = int(env_id)
        self.threshold = float(force_threshold)
        self.sensor = None
        self.body_names: list[str] = []
        self.body_idx: list[int] = []
        try:
            self.sensor = scene[sensor_name]
            self.body_names = list(self.sensor.body_names)
            self.body_idx = [
                i for i, n in enumerate(self.body_names)
                if not any(k in n.lower() for k in _FINGER_KEYS)
            ]
            print(f"[CollisionMonitor] tracking {len(self.body_names)} robot links "
                  f"({len(self.body_idx)} body links, fingers excluded)", flush=True)
        except Exception as exc:  # pragma: no cover - sensor absent
            print(f"[CollisionMonitor] disabled (no '{sensor_name}' sensor): {exc}", flush=True)

    @property
    def available(self) -> bool:
        return self.sensor is not None

    def contacts(self, body_only: bool = True) -> list[tuple[str, float]]:
        """[(link_name, force_magnitude_N)] for links in contact above the threshold this step."""
        if self.sensor is None or self.sensor.data.net_forces_w is None:
            return []
        mags = torch.linalg.norm(self.sensor.data.net_forces_w[self.env_id], dim=-1)
        idxs = self.body_idx if body_only else range(len(self.body_names))
        return [(self.body_names[i], float(mags[i])) for i in idxs if float(mags[i]) > self.threshold]
