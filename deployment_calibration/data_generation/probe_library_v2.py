"""Fixed, interpretable probe thetas (v2).

Probes are the SAME across every deployment session so that a difference in probe outcomes reflects the
hidden deployment state (damping), NOT the probe parameters. Each probe stresses the drawer's resistance
differently (pull aggressiveness / step size) so damping shows up in pull duration / final position / vel.
"""

from __future__ import annotations

# standard probe target (fixed)
PROBE_TARGET = {"target_open_position": 0.20, "target_tolerance": 0.02}

# K=3 standard probes. offset 0 (centered grasp) so probes don't miss the handle for reasons unrelated
# to the hidden state; they differ in pull_lead and max_pos_step which interact with joint damping.
PROBES = [
    {"name": "P0_nominal", "grasp_offset_local_y": 0.0, "max_pos_step": 0.020, "pull_lead": 0.08},
    {"name": "P1_aggressive_pull", "grasp_offset_local_y": 0.0, "max_pos_step": 0.028, "pull_lead": 0.12},
    {"name": "P2_fine_step", "grasp_offset_local_y": 0.0, "max_pos_step": 0.012, "pull_lead": 0.06},
]


def probe_theta(i: int) -> dict:
    p = PROBES[i % len(PROBES)]
    return {k: p[k] for k in ("grasp_offset_local_y", "max_pos_step", "pull_lead")}


def probe_name(i: int) -> str:
    return PROBES[i % len(PROBES)]["name"]


def n_probes() -> int:
    return len(PROBES)
