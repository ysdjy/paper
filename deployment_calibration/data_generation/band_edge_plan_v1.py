"""Band-edge experiment PLAN + MANIFEST (v1, runtime, pure-python — no Isaac).

Builds the frozen 306-episode plan (18 blocks x 17 offsets, no probes) from Claude B's FROZEN config
(`offline_v2/calibration_bias/band_edge_characterization_config_v1.json` + `band_edge.py`). Reuses B's
residual sampler `residual_nuisance.draw_block_residuals` (NO second residual algorithm). Draws one
residual + one nuisance context (initial joint perturbation, target jitter) per block, shared by all 17
offsets; blocks independent. Randomizes block execution order and within-block offset order from a master
RNG and writes a full, resumable manifest.

Resume rule: if a manifest exists it is LOADED and never resampled; a new run reproduces the identical
plan from the same master_seed. Duplicate (block_id, offset_id) is rejected by the validator.

This module generates ONLY the plan/manifest; it does not launch Isaac and does not generate episodes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
_DC = _PAPER / "deployment_calibration"
import sys
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from offline_v2.calibration_bias import band_edge as BE          # noqa: E402
from offline_v2.calibration_bias import residual_nuisance as RN   # noqa: E402

FROZEN_CONFIG = _DC / "offline_v2" / "calibration_bias" / "band_edge_characterization_config_v1.json"

# runtime-only nuisance magnitudes (NOT frozen by B; same verified-safe values as capability-map v2).
# The residual (which B DID freeze) is the label-moving nuisance; these are secondary realism jitter.
DEFAULT_NUISANCE = {"robot_joint_sigma": 0.015, "robot_joint_cap": 0.030, "target_jitter": 0.005,
                    "n_arm_joints": 7}


def config_sha256() -> str:
    return hashlib.sha256(FROZEN_CONFIG.read_bytes()).hexdigest()


def _git(*a) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(_PAPER), *a], text=True).strip()
    except Exception:
        return ""


def git_provenance() -> dict:
    return {"code_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
            "dirty_worktree": bool(_git("status", "--short").strip())}


def offset_id(o: float) -> str:
    return f"o{o:+.3f}"


@dataclass(frozen=True)
class PlanConfig:
    master_seed: int = 8801150
    nominal_bias_y: float = 0.0
    target_open_position: float = 0.20
    damping: float = 3.0
    drawer: str = "middle_drawer"
    robot_joint_sigma: float = DEFAULT_NUISANCE["robot_joint_sigma"]
    robot_joint_cap: float = DEFAULT_NUISANCE["robot_joint_cap"]
    target_jitter: float = DEFAULT_NUISANCE["target_jitter"]
    smoke_only: bool = False
    # smoke-only subsets (None -> full frozen plan)
    blocks_subset: tuple | None = None       # e.g. (0, 1) for 2-block smoke
    offsets_subset: tuple | None = None       # e.g. (-0.05, 0.0, 0.05) clear-zone smoke


def build_plan(pc: PlanConfig) -> dict:
    cfg = BE.DEFAULT_BAND_EDGE
    vc = BE.validate_config(cfg)
    if not vc["ok"]:
        raise ValueError(f"frozen band-edge config invalid: {vc}")

    n_blocks = cfg.n_blocks
    offsets = list(cfg.offsets)
    # --- master RNG: residual seed, block-order shuffle, per-block nuisance + offset-order seeds ---
    master = np.random.default_rng(pc.master_seed)
    residual_seed = int(master.integers(1, 2**31 - 1))
    block_residuals = RN.draw_block_residuals(n_blocks, residual_seed, cfg.residual)   # REUSE B's sampler
    block_seeds = [int(master.integers(1, 2**31 - 1)) for _ in range(n_blocks)]
    offset_order_seeds = [int(master.integers(1, 2**31 - 1)) for _ in range(n_blocks)]
    canonical_block_order = list(range(n_blocks))
    randomized_block_order = [int(i) for i in master.permutation(n_blocks)]

    blocks = []
    for b in range(n_blocks):
        brng = np.random.default_rng(block_seeds[b])
        delta = np.clip(brng.normal(0.0, pc.robot_joint_sigma, size=DEFAULT_NUISANCE["n_arm_joints"]),
                        -pc.robot_joint_cap, pc.robot_joint_cap)
        jitter = float(brng.uniform(-pc.target_jitter, pc.target_jitter))
        residual = float(block_residuals[b])
        orng = np.random.default_rng(offset_order_seeds[b])
        rand_off_idx = [int(i) for i in orng.permutation(len(offsets))]
        blocks.append({
            "block_id": b, "block_seed": block_seeds[b], "residual_seed": residual_seed,
            "nominal_bias_y": pc.nominal_bias_y, "residual_bias_y": residual,
            "actual_bias_y": RN.actual_bias(pc.nominal_bias_y, residual),
            "nuisance_robot_joint_delta": [float(v) for v in delta.tolist()],
            "nuisance_target_jitter": jitter, "offset_order_seed": offset_order_seeds[b],
            "canonical_offset_order": [offset_id(o) for o in offsets],
            "randomized_offset_order": [offset_id(offsets[i]) for i in rand_off_idx],
        })

    # --- planned episodes (respect randomized orders); apply smoke subsets if requested ---
    sel_blocks = randomized_block_order
    if pc.blocks_subset is not None:
        sel_blocks = [b for b in randomized_block_order if b in set(pc.blocks_subset)]
    off_filter = None if pc.offsets_subset is None else {offset_id(o) for o in pc.offsets_subset}
    planned = []
    exec_i = 0
    for b in sel_blocks:
        for oid in blocks[b]["randomized_offset_order"]:
            if off_filter is not None and oid not in off_filter:
                continue
            o = float(oid[1:])
            planned.append({"execution_index": exec_i, "block_id": b, "offset_id": oid,
                            "grasp_offset_local_y": o,
                            "eff_signed": round(blocks[b]["actual_bias_y"] + o, 6),
                            "abs_eff": round(abs(blocks[b]["actual_bias_y"] + o), 6)})
            exec_i += 1

    manifest = {
        "experiment": "band_edge_characterization_v1", "not_confirmatory": True,
        "smoke_only": bool(pc.smoke_only),
        "config_sha256": config_sha256(), **git_provenance(),
        "scene": {"task_id": "V1_BASE_TASK_ID", "drawer": pc.drawer, "mechanism": f"cabinet:{pc.drawer}",
                  "member": "cabinet", "robot": "franka_panda", "scene_version": "paper_scene_v2"},
        "design": {"nominal_bias_y": pc.nominal_bias_y, "n_blocks": n_blocks, "n_offsets": len(offsets),
                   "offsets": offsets, "run_probes": False, "n_episodes_full": cfg.n_episodes,
                   "target_open_position": pc.target_open_position, "damping": pc.damping},
        "residual": {**cfg.residual.to_dict(), "effective_std": RN.residual_std_effective(cfg.residual)},
        "nuisance_distribution": {"robot_joint_sigma": pc.robot_joint_sigma,
                                  "robot_joint_cap": pc.robot_joint_cap, "target_jitter": pc.target_jitter,
                                  "per_block_seed": True, "seed_independent_of_bias": True,
                                  "note": "runtime-only realism jitter; residual is the label-moving nuisance"},
        "master_seed": pc.master_seed, "residual_seed": residual_seed,
        "canonical_block_order": canonical_block_order, "randomized_block_order": randomized_block_order,
        "canonical_offset_order": [offset_id(o) for o in offsets],
        "blocks": blocks, "planned_episodes": planned, "planned_episode_count": len(planned),
        "planned_episode_count_full_design": cfg.n_episodes,
        "subset": {"blocks_subset": list(pc.blocks_subset) if pc.blocks_subset else None,
                   "offsets_subset": list(pc.offsets_subset) if pc.offsets_subset else None},
    }
    return manifest


def load_or_build(manifest_path: Path, pc: PlanConfig) -> tuple[dict, bool]:
    """Resume-safe: if manifest exists, LOAD it (no resample); else build + return (manifest, built_new)."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        return json.loads(manifest_path.read_text()), False
    m = build_plan(pc)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(m, indent=1))
    return m, True


if __name__ == "__main__":  # quick manifest dump (no Isaac)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_DC / "data" / "band_edge_manifest_preview_v1.json"))
    ap.add_argument("--master_seed", type=int, default=PlanConfig.master_seed)
    a = ap.parse_args()
    m = build_plan(PlanConfig(master_seed=a.master_seed))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=1))
    print(f"[band-edge-plan] wrote {a.out}: {m['planned_episode_count']} episodes "
          f"({len(m['blocks'])} blocks x {m['design']['n_offsets']} offsets), config_sha256={m['config_sha256'][:16]}...")
