#!/usr/bin/env bash
# Franka V1 — teleop data-collection smoke test. STATUS: ready (no hardware, no Isaac).
# Exercises the FULL collection pipeline minus Isaac:
#   mock GELLO -> safety mapping  ->  episode state machine + scripted events
#   -> EpisodeRecorder -> HDF5 writer  ->  skill_demo_to_lerobot (normalized).
# The only thing not covered here is the live Isaac env loop (run that with
# ./isaaclab.sh -p .../collect_teleop_demos_joint_v1.py --teleop_device mock ...).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Pick a python with numpy + h5py (prefer an explicit $PYTHON, then env_isaaclab).
pick_python() {
  for cand in "${PYTHON:-}" \
              "/home1/banghai/miniconda3/envs/env_isaaclab/bin/python" \
              "python3"; do
    [ -n "$cand" ] || continue
    if "$cand" -c "import numpy, h5py" >/dev/null 2>&1; then echo "$cand"; return 0; fi
  done
  return 1
}
PY="$(pick_python || true)"
if [ -z "${PY:-}" ]; then
  echo "[smoke_teleop] ERROR: no python with numpy+h5py found. Set PYTHON=... (e.g. the env_isaaclab python)."
  exit 1
fi
echo "[smoke_teleop] using python: $PY"

echo "[smoke_teleop] 1/4 mock GELLO -> Franka joint target (safety mapping)"
"$PY" projects/franka_v1_skill_lab/teleop_collection/entries/gello_to_isaac_joint_test.py --mock_gello --steps 30 >/dev/null
echo "  mapping OK"

echo "[smoke_teleop] 2/4 collect entry --dry_run (schema + task label)"
"$PY" projects/franka_v1_skill_lab/teleop_collection/entries/collect_teleop_demos_joint_v1.py \
  --dry_run --task_config projects/franka_v1_skill_lab/configs/teleop_tasks.yaml \
  --task_catalog_id grasp_cube_2 >/dev/null
echo "  dry-run OK"

echo "[smoke_teleop] 3/4 state machine + recorder + HDF5 roundtrip (no Isaac)"
"$PY" - <<'PYEOF'
import sys, tempfile, os
sys.path.insert(0, "projects")
from franka_v1_skill_lab.teleop_collection.gello.gello_to_franka_mapper import GelloToFrankaMapper
from franka_v1_skill_lab.teleop_collection.gello.mock_gello_source import MockGelloSource
from franka_v1_skill_lab.teleop_collection.recording.keyboard_episode_controller import (
    CollectionStateMachine, ScriptedEventSource)
from franka_v1_skill_lab.teleop_collection.recording.episode_recorder import EpisodeRecorder, FrameBuilder
from franka_v1_skill_lab.teleop_collection.recording.hdf5_writer import TeleopHDF5Writer

# drive the state machine with the scripted source exactly like the live loop
sm = CollectionStateMachine()
src = MockGelloSource()
mapper = GelloToFrankaMapper()
mapper.reset([0.0]*7)
rec = EpisodeRecorder(record_objects=True)
ev = ScriptedEventSource(episode_len=20, num_episodes=2)

tmp = tempfile.mkdtemp()
writer = TeleopHDF5Writer(tmp, task_id="Isaac-Stack-Cube-Franka-JointPolicy-v0", scene_registry="smoke")
ep_n = [0]
def prepare():
    ep_n[0]+=1; rec.reset(episode_id=f"ep_{ep_n[0]}", task_instruction="Grasp cube_2.",
                          skill_type="grasp", target_name="cube_2", scene_registry="smoke", seed=1)
    sm.prepare_episode()
prepare()

for step in range(200):
    for e,p in ev.poll(rec.num_steps):
        tr = sm.handle(e,p)
        if tr.flush_success: writer.write_episode(rec.frames, rec.episode_meta(True))
        if tr.prepare_next: prepare()
        if tr.should_quit: break
    if sm.is_terminal(): break
    if sm.is_recording():
        q,g = src.read(); m = mapper.map(q,g)
        rec.append(FrameBuilder.build(
            step_id=rec.num_steps, timestamp=float(step), joint_pos=m["q_des"], joint_vel=[0.0]*7,
            ee_pose=[0.4,0,0.3,0,0,0,1], gripper_width=0.04, raw_q=q, filtered_q=m["q_des"],
            joint_target=m["q_des"], gripper=m["gripper_norm"],
            objects=[{"name":"cube_2","pos":[0.5,0.0,0.05],"quat_wxyz":[1,0,0,0]}]))
path = writer.close()
assert writer.num_demos == 2, f"expected 2 demos, got {writer.num_demos}"
assert sm.episodes_saved == 2, sm.episodes_saved

# verify HDF5 structure
import h5py, numpy as np
with h5py.File(path,"r") as f:
    assert f.attrs["control_mode"]=="joint"
    d = f["data/demo_0"]
    assert d["obs/joint_pos"].shape == (20,7), d["obs/joint_pos"].shape
    assert d["actions/joint_target"].shape == (20,7)
    assert d["actions/gripper_command"].shape == (20,1)
    assert d["obs/objects/cube_2"].shape == (20,7)
    assert bool(d.attrs["success"]) is True
print(f"  HDF5 OK: {writer.num_demos} demos at {os.path.basename(path)} (20 steps each, joint action)")

# run the converter on it (normalized only; lerobot is best-effort)
from franka_v1_skill_lab.pi05_training.adapters import skill_demo_to_lerobot as conv
out = os.path.join(tmp, "lerobot_out")
rep = conv.convert(path, out, task_catalog={"_default_instruction":"Grasp cube_2."},
                   build_lerobot=False, normalized_root=os.path.join(tmp,"normalized"))
assert rep["episodes"] == 2 and rep["frames"] == 40, rep
assert rep["state_dim"] == 8 and rep["action_dim"] == 8, rep
states = np.load(os.path.join(rep["normalized_dir"],"states.npy"))
actions = np.load(os.path.join(rep["normalized_dir"],"actions.npy"))
assert states.shape == (40,8) and actions.shape == (40,8), (states.shape, actions.shape)
print(f"  converter OK: normalized state{states.shape} action{actions.shape} -> {rep['normalized_dir']}")
PYEOF
echo "  pipeline OK"

echo "[smoke_teleop] 4/4 demo -> LeRobot mapping validator (dry run)"
"$PY" projects/franka_v1_skill_lab/pi05_training/adapters/skill_demo_to_lerobot.py --dry_run >/dev/null
echo "  mapping V1-valid (joint action, not EE delta)"

cat <<'NOTE'

[smoke_teleop] OFFLINE pipeline OK. To exercise the LIVE Isaac collection loop:
  conda activate env_isaaclab
  ./isaaclab.sh -p projects/franka_v1_skill_lab/teleop_collection/entries/collect_teleop_demos_joint_v1.py \
      --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 \
      --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
      --teleop_device mock --enable_wrist_d435 --record_objects \
      --out_dir projects/franka_v1_skill_lab/data/teleop_raw_hdf5 --max_steps_per_episode 300 --seed 1
NOTE
echo "[smoke_teleop] OK"
