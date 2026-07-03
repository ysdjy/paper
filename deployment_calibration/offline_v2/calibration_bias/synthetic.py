"""Synthetic calibration-bias dataset generator (offline, TEST/DEV ONLY).

Every episode is stamped `synthetic_only: True`. This data exists solely to exercise the pipeline
before Claude A's real capability map arrives; it is NEVER a paper result.

Physics model (compensation):
  effective grasp error  = grasp_offset_local_y + bias_y      (best offset = -bias_y)
  success                = |effective error| <= grasp_tol
  task_outcome_error     = min(|effective error|, err_cap) + observable/nuisance noise
  time                   ~ base_time (+ tiny offset-magnitude term)  -> success, not time, drives U
Because the optimal offset = -bias moves across the whole range and grasp_tol is small, NO single
offset is near-optimal at every bias (no robust generalist) -> positive decision value by design.

Probes are a few FIXED offsets whose outcomes reveal the bias (informative but noisy). Nuisance
seeds are independent per session and NOT correlated with bias.
"""

from __future__ import annotations

import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def make_synthetic(*, bias_levels=None, offsets=None, probe_offsets=(-0.04, 0.0, 0.04),
                   targets=("T020",), replicates=3, grasp_tol=0.02, err_cap=0.2,
                   base_time=10.0, noise=0.004, seed=0, paired_blocks=False) -> list:
    """Return a list of episode dicts (schema-compatible). Deterministic given seed.

    paired_blocks: if True, stamp each session with nuisance_block_id = f"blk_r{rep}", i.e. the SAME
    nuisance block is reused across all bias levels for a given replicate index. This is the
    exploration-stage PAIRED design (matched nuisance context under every bias); it is independence-
    legal (block reused across bias, not 1:1 with bias) but, if split by bias level, a block would
    cross splits — which the confirmatory split audit correctly flags.
    """
    if bias_levels is None:
        # 7 continuous bias levels spanning the compensable range
        vals = np.round(np.linspace(-0.045, 0.045, 7), 4)
        bias_levels = [(f"B{i}", float(v)) for i, v in enumerate(vals)]
    if offsets is None:
        vals = np.round(np.linspace(-0.06, 0.06, 7), 4)
        offsets = [(f"o{i:+d}".replace("+", "p").replace("-", "m"), float(v)) for i, v in enumerate(vals)]

    eps = []
    seed_counter = 1000
    for bi, (bias_id, bias) in enumerate(bias_levels):
        for rep in range(replicates):
            sid = f"sess_{bias_id}_r{rep}"
            # independent nuisance seed per session, NOT a function of bias
            nseed = seed_counter
            seed_counter += 1
            # paired block id (shared across bias for a replicate) or per-session id
            block_id = f"blk_r{rep}" if paired_blocks else f"blk_{bias_id}_r{rep}"
            rng = _rng(seed * 100003 + nseed)
            order = 0

            # ---- probes: fixed offsets, reveal bias ----
            for pidx, po in enumerate(probe_offsets):
                eff = po + bias + rng.normal(0, noise)
                succ = abs(eff) <= grasp_tol
                eps.append(_episode(
                    sid=sid, role="probe", order=order, offset=po, bias=bias, bias_id=bias_id,
                    nseed=nseed, block_id=block_id, target=None, succ=succ, err=min(abs(eff), err_cap),
                    time=base_time + abs(po) * 5 + rng.normal(0, 0.05), probe_index=pidx,
                    rep=rep, grasp_tol=grasp_tol))
                order += 1

            # ---- candidates: full offset bank per target ----
            for target in targets:
                cg = f"{sid}__{target}"
                for ci, (oid, off) in enumerate(offsets):
                    eff = off + bias + rng.normal(0, noise)
                    succ = abs(eff) <= grasp_tol
                    eps.append(_episode(
                        sid=sid, role="candidate", order=order, offset=off, bias=bias, bias_id=bias_id,
                        nseed=nseed, block_id=block_id, target=target, succ=succ, err=min(abs(eff), err_cap),
                        time=base_time + abs(off) * 5 + rng.normal(0, 0.05),
                        candidate_group=cg, candidate_index=ci, offset_id=oid,
                        candidate_id=f"{target}_{oid}", matched_group_id=f"{target}__r{rep}",
                        rep=rep, grasp_tol=grasp_tol))
                    order += 1
    return eps


def _episode(*, sid, role, order, offset, bias, bias_id, nseed, target, succ, err, time,
             rep, grasp_tol, block_id=None, probe_index=None, candidate_group=None,
             candidate_index=None, offset_id=None, candidate_id=None, matched_group_id=None) -> dict:
    e = {
        "episode_id": f"{sid}_{role}_{order:03d}",
        "session_id": sid, "episode_role": role, "order_in_session": order,
        "drawer_name": "middle_drawer", "mechanism_id": "cabinet:middle_drawer",
        "x": {"mechanism_id": "cabinet:middle_drawer", "drawer_name": "middle_drawer",
              "member": "cabinet", "robot_joint_pos": [0.0] * 9, "tcp_pos": [0.4, 0.0, 0.3],
              "tcp_quat": [0, 1, 0, 0], "gripper_width": 0.08, "initial_mechanism_joint_pos": 0.0,
              # observable nuisance is allowed; the raw seed is NOT exposed
              "observable_nuisance": {"observed_sensor_noise_level": 0.01}},
        "g": {"target_open_position": 0.2, "target_tolerance": 0.02},
        "theta": {"grasp_offset_local_y": float(offset), "max_pos_step": 0.02, "pull_lead": 0.08},
        "y": {"success": bool(succ), "failure_reason": "NONE" if succ else "POSITION_TIMEOUT",
              "final_joint_position": 0.2 if succ else 0.0,
              "task_outcome_error": float(err), "skill_elapsed_time": float(time),
              "phase_durations": {"PULL": 0.6}, "handle_relative_error": float(err),
              "handle_detached": False},
        "secret_deployment_state": {"bias_y": float(bias)},   # ORACLE-ONLY
        "bias_id": bias_id, "bias_level_id": bias_id,
        "nuisance_seed": int(nseed), "nuisance_block_id": block_id,
        "history_cutoff": 3,
        "replicate_id": int(rep),
        "synthetic_only": True,
        "contract_version": "open_drawer_v2",
    }
    if role == "probe":
        e["probe_index"] = probe_index
    else:
        e.update({"candidate_group": candidate_group, "candidate_index": candidate_index,
                  "offset_id": offset_id, "candidate_id": candidate_id,
                  "target_id": target, "matched_group_id": matched_group_id})
    return e
