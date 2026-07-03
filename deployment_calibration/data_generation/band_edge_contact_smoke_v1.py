"""Contact-sensor VALIDATION smoke for the band-edge experiment (v1, Isaac).

Frozen sensor-validation smoke (config `collision_sensor_plan.sensor_validation_smoke`):
  1. clean baseline    -> known collision-free hold near home; expect max unintended force ~0.
  2. positive control  -> drive the hand INTO the cabinet body; expect a large unintended force.
  3. discrimination    -> positive_force > clean_force with a gap >> noise; sensor available.
This is a SMOKE-ONLY diagnostic: it writes to its own dir, never to a band-edge data dir, and does NOT
touch the formal skill config. It does not run the 306 band.

    ./isaaclab.sh -p .../band_edge_contact_smoke_v1.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_PAPER = Path(__file__).resolve().parents[2]
for _p in (_PAPER, _PAPER / "deployment_calibration", _PAPER / "franka_skill_state_machine",
           Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run_id", default=None)
ap.add_argument("--steps", type=int, default=80)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _monitor_over(H, target_pos_w, n_steps, drive: bool, spec):
    """Step n_steps; if drive, IK-drive TCP toward target_pos_w (into cabinet); record max unintended force."""
    from band_edge_instrumentation_runtime_v1 import CollisionMonitor
    from runtime.scene_state_provider import PoseState
    provider, env = H.provider, H.env
    mon = CollisionMonitor(H.scene)
    sim_time = 0.0
    for _ in range(n_steps):
        provider.set_sim_time(sim_time)
        s = provider.get_state()
        with torch.no_grad():
            if drive:
                tgt = PoseState(target_pos_w, s.robot.tcp_pose.quat_w)
                res = H.adapter.solve(tgt)
                if res.success:
                    action = provider.make_joint_action_from_q_des(res.q_des, -1.0)  # close gripper too
                else:
                    action = provider.make_hold_joint_action(s, 1.0)
            else:
                action = provider.make_hold_joint_action(s, 1.0)
            env.step(action)
        mon.update("SMOKE")
        sim_time += H.control_dt
    return mon.result()


def main() -> int:
    from band_edge_runtime_v1 import launch_band_edge_scene
    H = launch_band_edge_scene(app_launcher, device=args.device)
    spec = H.spec("middle_drawer")

    # ---- 1. clean baseline: reset, hold at home away from cabinet ----
    H.reset(spec)
    H.set_damping(spec, 3.0)
    clean = _monitor_over(H, None, args.steps, drive=False, spec=spec)

    # ---- 2. positive control: drive the hand into the cabinet base body ----
    H.reset(spec)
    H.set_damping(spec, 3.0)
    member = H.provider.scene[spec.member]
    cabinet_base_w = member.data.body_pos_w[0, 0].clone()   # a solid cabinet body -> guaranteed collidable
    pos = _monitor_over(H, cabinet_base_w, args.steps, drive=True, spec=spec)

    gap = pos["max_unintended_contact_force_N"] - clean["max_unintended_contact_force_N"]
    discriminates = bool(pos["contact_sensor_available"] and
                         pos["max_unintended_contact_force_N"] > clean["max_unintended_contact_force_N"] and
                         pos["unintended_contact_frame_count"] > clean["unintended_contact_frame_count"])
    result = {"experiment": "band_edge_contact_sensor_smoke_v1", "smoke_only": True,
              "sensor_available": pos["contact_sensor_available"], "contact_backend": pos["contact_sensor_backend"],
              "clean_baseline": clean, "positive_control": pos, "force_gap_N": round(gap, 4),
              "discriminates": discriminates, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    outdir = (_PAPER / "deployment_calibration" / "data" /
              (args.run_id or f"band_edge_contact_smoke_v1_{time.strftime('%Y%m%d_%H%M%S')}"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "contact_smoke_result.json").write_text(json.dumps(result, indent=1))
    print(f"[contact-smoke] sensor_available={result['sensor_available']} "
          f"clean={clean['max_unintended_contact_force_N']:.3f}N pos={pos['max_unintended_contact_force_N']:.3f}N "
          f"gap={gap:.3f}N discriminates={discriminates} backend={pos['contact_sensor_backend']}", flush=True)
    print(f"[contact-smoke] DONE -> {outdir}/contact_smoke_result.json", flush=True)
    H.close()
    return 0 if result["sensor_available"] else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
