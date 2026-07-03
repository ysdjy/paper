"""Isaac-coupled instrumentation wrappers for the band-edge experiment (v1).

Pure numeric cores are in `band_edge_instrumentation_core_v1.py` (unit-tested without Isaac). This module
adds the two Isaac-coupled pieces: `InstrumentedIKAdapter` (IK failure + clamp counting around
`IKJointAdapter.solve()`, no behaviour change) and `CollisionMonitor` (ContactSensor net force on
non-finger robot links). Field definitions are frozen by
`offline_v2/calibration_bias/band_edge_instrumentation.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FSM = Path(__file__).resolve().parents[2] / "franka_skill_state_machine"
if str(_FSM) not in sys.path:
    sys.path.insert(0, str(_FSM))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.ik_joint_adapter import IKJointAdapter   # 'runtime' == franka_skill_state_machine/runtime

from band_edge_instrumentation_core_v1 import ClampCounter, IKFailureCounter  # noqa: F401 (re-export)

FINGER_BODY_SUBSTR = ("finger",)   # panda_leftfinger / panda_rightfinger -> the INTENDED grasp contact


class InstrumentedIKAdapter(IKJointAdapter):
    """Counts IK failures + the two clamps around solve(), with NO behaviour change."""
    def attach_instruments(self, ik: IKFailureCounter, clamps: ClampCounter, phase_fn):
        self._ik = ik; self._clamps = clamps; self._phase_fn = phase_fn

    def solve(self, *a, **k):
        q_curr = self.robot.data.joint_pos[self.env_id, self._joint_ids].detach().clone()
        res = super().solve(*a, **k)
        ik = getattr(self, "_ik", None)
        if ik is None:
            return res
        ph = self._phase_fn() if getattr(self, "_phase_fn", None) else "NONE"
        self._ik.update(bool(res.success), ph)
        if res.success and res.q_des is not None:
            self._clamps.update([float(v) for v in q_curr.tolist()],
                                [float(v) for v in res.q_des.tolist()],
                                [float(v) for v in self._joint_lower.tolist()],
                                [float(v) for v in self._joint_upper.tolist()], self.max_joint_step)
        return res


class CollisionMonitor:
    """Unintended contact = net contact force on NON-finger robot links (arm + hand). Finger contact is
    the intended finger<->handle grasp and is excluded. Backend = Isaac Lab ContactSensor
    (scene['robot_contact'].data.net_forces_w) installed by enable_collision_monitor.

    Equivalence note: net_forces_w is the total contact force per link; in this scene the only bodies the
    arm/hand can touch are the cabinet/drawer, so non-finger link force == arm/hand-vs-cabinet unintended
    contact (documented in the runtime doc)."""
    def __init__(self, scene):
        self.available = False
        self.backend = "none"
        self._sensor = None
        self._nonfinger_idx = []
        self._body_names = []
        self.max_force = 0.0; self.frames = 0; self.first_phase = "NONE"
        self.by_phase = {}
        try:
            sensor = scene["robot_contact"]
            names = list(getattr(sensor, "body_names", []) or [])
            self._nonfinger_idx = [i for i, n in enumerate(names)
                                   if not any(s in n.lower() for s in FINGER_BODY_SUBSTR)]
            self._sensor = sensor
            self._body_names = names
            self.available = True
            self.backend = "isaaclab.sensors.ContactSensor(robot_contact.net_forces_w, non-finger links)"
        except Exception as ex:
            self.backend = f"unavailable: {ex}"

    def update(self, phase: str):
        if not self.available:
            return
        import torch
        try:
            f = self._sensor.data.net_forces_w  # [env, body, 3]
            fv = f[0, self._nonfinger_idx, :]
            mag = float(torch.linalg.norm(fv, dim=-1).max()) if fv.numel() else 0.0
        except Exception:
            mag = 0.0
        if mag > 1e-6:
            self.frames += 1
            if self.first_phase == "NONE":
                self.first_phase = phase
            self.by_phase[phase] = max(self.by_phase.get(phase, 0.0), mag)
        if mag > self.max_force:
            self.max_force = mag

    def result(self):
        return {"max_unintended_contact_force_N": round(float(self.max_force), 6),
                "unintended_contact_frame_count": int(self.frames),
                "first_unintended_contact_phase": self.first_phase,
                "contact_force_by_phase": {k: round(float(v), 6) for k, v in self.by_phase.items()},
                "contact_sensor_available": bool(self.available),
                "contact_sensor_backend": self.backend}
