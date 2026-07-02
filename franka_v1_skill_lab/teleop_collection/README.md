# teleop_collection/ — GELLO teleop → joint targets → demos

## 1. What this module does
Brings GELLO teleoperation to the V1 joint-action scene and records demos for
pi0.5. Layers:
- `gello/` — GELLO source abstraction (`gello_reader.make_gello_source` →
  real reader wrapper + mock), the **joint-target safety pipeline**
  (`joint_teleop_safety.py`: EMA low-pass → per-step clamp → joint-limit clamp →
  e-stop), and `gello_to_franka_mapper.py` (read → safe `q_des` + gripper).
- `recording/` — the data path: `keyboard_episode_controller.py` (episode
  **state machine** + keyboard / scripted event sources), `episode_recorder.py`
  (per-frame buffer), `hdf5_writer.py` (HDF5 session writer), and
  `isaac_teleop_driver.py` (Isaac env build + frame capture + carb keyboard).
- `entries/gello_to_isaac_joint_test.py` — GELLO → safe joint target. Mapping
  smoke (no Isaac) **and** `--drive_isaac` (feed targets into the live env).
- `entries/collect_teleop_demos_joint_v1.py` — the **collection loop**: drive +
  record (obs, joint-action) demos → HDF5 for `pi05_training`.

## 2. Upstream dependencies
- Real-hardware GELLO read: `projects/gello_franka_teleop` (isolated
  `.venv-gello`, Dynamixel SDK) — only when reading actual hardware.
- `scene/` (V1 task id + registry). The live env + SceneStateProvider are reused
  from the legacy `franka_skill_state_machine` via `skill_runtime._legacy`.
- Optional wrist D435 from `sensors/d435`.

## 3. Downstream consumers
- `pi05_training/adapters/skill_demo_to_lerobot.py` consumes the HDF5 demos.

## 4. Episode state machine + keyboard
States: `IDLE → PREPARE_EPISODE → RECORDING ⇄ PAUSED →
{SUCCESS_PENDING | FAILED_PENDING | DISCARDED} → SAVED → …`, plus `RESETTING`
and `EXIT`. Keys (GUI; carb input names):

| key | action | key | action |
|-----|--------|-----|--------|
| SPACE | pause / resume | R | reset episode (discard + reset scene) |
| S | mark success + save | N | next episode |
| F | mark failed (→ failed file if `--save_failed`, else discard) | E | e-stop / freeze (toggle) |
| D | discard current episode | H | home / hold posture |
| Q | quit collection | 1–6 | select skill family |

Headless / mock (or `--auto`) uses a deterministic `ScriptedEventSource`
(start → record N steps → success → next), so collection runs with NO human.

## 5. Common commands
```bash
# offline smoke (no hardware, no Isaac): SM + recorder + HDF5 + converter
bash projects/franka_v1_skill_lab/scripts/smoke_teleop_collection_v1.sh

# mapping smoke (no Isaac)
python projects/franka_v1_skill_lab/teleop_collection/entries/gello_to_isaac_joint_test.py --mock_gello --steps 30

# collect dry-run (schema + task label, no Isaac)
python projects/franka_v1_skill_lab/teleop_collection/entries/collect_teleop_demos_joint_v1.py --dry_run

# LIVE mock collection (Isaac; auto-records one episode then exits)
./isaaclab.sh -p projects/franka_v1_skill_lab/teleop_collection/entries/collect_teleop_demos_joint_v1.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 \
  --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
  --teleop_device mock --enable_wrist_d435 --record_objects \
  --out_dir projects/franka_v1_skill_lab/data/teleop_raw_hdf5 --max_steps_per_episode 300 --seed 1
```

## 6. Current status
- `gello/joint_teleop_safety.py`, `mock_gello_source.py`, `gello_reader.py`,
  `gello_to_franka_mapper.py` — **ready**.
- `recording/keyboard_episode_controller.py`, `episode_recorder.py`,
  `hdf5_writer.py`, `isaac_teleop_driver.py` — **ready** (verified live + offline).
- `entries/gello_to_isaac_joint_test.py` — **ready** (mapping + `--drive_isaac`).
- `entries/collect_teleop_demos_joint_v1.py` — **ready** for both
  `--teleop_device mock` and `--teleop_device gello` (both verified end-to-end in
  Isaac; mock also with the wrist D435). The robot follows the device as soon as
  the episode is prepared (PREPARE/RECORDING); PAUSE / e-stop freeze it.
- Real GELLO (`--teleop_device gello`) — **ready**: `RealGelloSource` reads the
  leader arm live via `gello_isaac_teleop.reader.ThreadedGelloReader` INSIDE the
  Isaac process (`dynamixel_sdk` is importable in `env_isaaclab`; no IPC bridge).
- `--record_foundationpose_objects` — **stub** (records sim-GT, logs a NOTE).

## 7. Real GELLO — confirmed working, things to check if it misbehaves
1. Device + calibration live in `gello/configs/gello_franka.yaml` (port,
   `joint_offsets`, `joint_signs`, `start_joints`, gripper open/close). It is
   already calibrated for this GELLO and verified live.
2. Port is `/dev/serial/by-id/usb-FTDI_...-if00-port0` → `/dev/ttyUSB0`. If it
   moved, re-run the legacy `detect_gello_port.sh` and update the yaml.
3. *Robot doesn't move / all-zero reads* — usually a dialout permission issue;
   `RealGelloSource` rejects the gello fake-driver fallback loudly. Confirm the
   user is in the `dialout` group, or run via `sg dialout -c "<command>"`.
4. *Robot jumps to a wild pose on START* — GELLO offsets need re-calibration
   (`calibrate_gello_offset.sh`); the safety filter still ramps at ≤`max_step_rad`.

## 8. Control precision / responsiveness (if teleop feels laggy)
The base task runs at **20 Hz** (sim.dt=0.01, decimation=5) with **soft** arm PD
gains (stiffness 80) — both make teleop feel sluggish/imprecise. Four knobs (on
`collect_teleop_demos_joint_v1.py` and `gello_to_isaac_joint_test.py`):

| knob | default | effect |
|------|---------|--------|
| `--control_hz` | 50 | control rate; sets `decimation=round(100/hz)`. 50→2 (50 Hz), 100→1 (100 Hz). **Biggest factor.** |
| `--arm_stiffness` / `--arm_damping` | 400 / 80 | Franka arm PD gains (HIGH_PD). Base 80/4 tracks `q_des` loosely → steady-state error. |
| `--max_joint_vel` | 4.0 | max arm speed (rad/s); per-step clamp = `vel/control_hz`. Raise if fast hand motions lag. |
| `--lowpass_alpha` | 0.5 | EMA on GELLO reads (higher = snappier/less lag; 1.0 = no smoothing). |
| `--gello_hz` | 100 | real GELLO serial read rate. |

Tune live (no recording) with `gello_to_isaac_joint_test.py --drive_isaac`. For
max precision try `--control_hz 100 --arm_stiffness 600 --lowpass_alpha 0.7`
(needs the sim to keep up in real time — single env, headless or light GUI).
Trade-offs: higher stiffness = stiffer/harder contacts; higher `max_joint_vel` /
`lowpass_alpha` = snappier but less safety smoothing.

## 9. Troubleshooting
- *No GELLO hardware* — use `--teleop_device mock` (default) / `--mock_gello`.
- *No wrist camera* — omit `--enable_wrist_d435`, or pass `--no_camera`.
- *`ModuleNotFoundError: runtime`* — the legacy path bootstrap failed; ensure
  `skill_runtime/_legacy.py` resolves `projects/franka_skill_state_machine`.
- *Joints jump* — lower `--max_step_rad` or `--lowpass_alpha`.
- *No demos saved* — in mock you must reach `--max_steps_per_episode`; with the
  keyboard you must press `S` before `Q`.
