# pi05_training/ — pi0.5 fine-tune pipeline (V1, joint action)

## 1. What this module does
The V1 view of the pi0.5 pipeline: collect → convert → train → serve → eval,
specialised for the **joint-action** task (the legacy pipeline defaulted to
IK-Rel). OpenPI stays in its own venv; IsaacLab talks to it over HTTP only.
- `WORKFLOW/` — V1 wrapper scripts (`1_collect`…`5_eval`, `stop`) that set the
  V1 task and delegate to `pi05_isaacsim_baseline/`.
- `adapters/` — V1 observation adapter, V1 joint action adapter, demo→LeRobot
  field mapping (all pure-python, unit-testable).

## 2. Upstream dependencies
- `pi05_isaacsim_baseline/` — the real pipeline (`.venv_openpi`, OpenPI, LeRobot,
  HTTP policy server). Heavy deps NEVER enter `env_isaaclab`.
- `scene/` (V1 task id), `teleop_collection/` (demo source), `configs/pi05_dataset.yaml`.

## 3. Downstream consumers
- The policy server feeds `5_eval` (closed-loop rollout in the V1 env).

## 4. Common commands
```bash
# offline adapter + mapping smoke test (no OpenPI, no GPU)
bash projects/franka_v1_skill_lab/scripts/smoke_pi05_v1.sh

# pipeline wrappers (each prints what it does / delegates)
bash projects/franka_v1_skill_lab/pi05_training/WORKFLOW/2_convert.sh --dry_run
bash projects/franka_v1_skill_lab/pi05_training/WORKFLOW/4_serve.sh          # mock server
bash projects/franka_v1_skill_lab/pi05_training/WORKFLOW/5_eval.sh           # prints real eval cmd
```

## 5. Current status
- `adapters/isaaclab_obs_adapter_v1.py`, `joint_action_adapter_v1.py` — **ready**
  (builder/clip logic; Isaac obs capture + action apply are one-line handoffs).
- `adapters/skill_demo_to_lerobot.py` — **ready** validator / **wrapper** convert.
- `WORKFLOW/*.sh` — **wrapper** (delegate to legacy with V1 task; serve has a
  mock backend that runs without OpenPI).
- Real training/serving with OpenPI — **not run here** (needs `.venv_openpi`, GPU).
  No long training, no model downloads from this project.

## 6. Troubleshooting
- *Action looks like a 7-D EE delta* — wrong; V1 is a 7-DoF joint target. The
  mapping validator rejects EE-delta action fields.
- *Server import errors about JAX/torch* — you started the **openpi** backend in
  `env_isaaclab`; run it in `.venv_openpi`, or use `POLICY_BACKEND=mock`.
- *Eval picks the wrong env kind* — confirm the legacy `run_policy_in_isaaclab.py`
  detects JOINT for `Isaac-Stack-Cube-Franka-JointPolicy-v0`.
