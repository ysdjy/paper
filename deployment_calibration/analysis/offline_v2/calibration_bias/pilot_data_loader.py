"""Scientific pilot data loader (EXPLORATORY / PILOT — NOT CONFIRMATORY).

Loads the existing 306-episode band-edge exploratory run and exposes it as an 18-block x 17-offset capability
map. IMPORTANT inventory fact (see scientific_pilot_data_inventory.md): this run has NO native probe+3-candidate
session structure — every episode is role='candidate', one trial per (block, offset). The pilot therefore
EMULATES a probe by designating the offset=0.0 episode of each block as the probe observation; its execution
outcome (`y`) is the probe history. Hidden fields (residual/actual/nominal bias, true handle error) are used
ONLY as analysis labels, never as model inputs (enforced in pilot_features.leakage_check).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Pinned exploratory run (gitignored dataset). Override with PILOT_DATA_PATH.
PINNED_SHA256 = "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
_DEFAULT_CANDIDATES = [
    os.environ.get("PILOT_DATA_PATH", ""),
    "/home1/banghai/Documents/IsaacLab/projects/paper_runtime_calibration_band_edge_reliability_v1/"
    "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl",
    "/home1/banghai/Documents/IsaacLab/projects/paper_runtime_calibration_band_edge_v1/"
    "deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl",
]
PROBE_OFFSET = 0.0                       # the emulated-probe offset (its y history = probe observation)


def resolve_data_path() -> Path:
    for c in _DEFAULT_CANDIDATES:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError("306 episodes.jsonl not found; set PILOT_DATA_PATH")


def data_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episodes(path: Path | None = None) -> list[dict]:
    path = path or resolve_data_path()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _offset(ep: dict) -> float:
    return round(float(ep["theta"]["grasp_offset_local_y"]), 3)


def build_capability_map(episodes: list[dict]) -> dict:
    """Return a structured view: cells[(block, offset)] -> episode; per-block hidden labels + probe episode."""
    cells = {(ep["block_id"], _offset(ep)): ep for ep in episodes}
    blocks = sorted({ep["block_id"] for ep in episodes})
    offsets = sorted({_offset(ep) for ep in episodes})
    hidden = {}
    probe = {}
    for b in blocks:
        any_ep = cells[(b, offsets[0])]
        sds = any_ep["secret_deployment_state"]
        hidden[b] = {"residual_bias_y": sds["residual_bias_y"], "actual_bias_y": sds["actual_bias_y"],
                     "nominal_bias_y": sds["nominal_bias_y"], "block_seed": any_ep["block_seed"]}
        probe[b] = cells.get((b, PROBE_OFFSET))
    return {"cells": cells, "blocks": blocks, "offsets": offsets, "hidden": hidden, "probe": probe}


def success(ep: dict) -> int:
    return int(bool(ep["y"]["success"]))
