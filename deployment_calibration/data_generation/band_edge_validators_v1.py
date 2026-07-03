"""Band-edge validators (v1, runtime): plan/count, matched-block, leakage, smoke exclusion, duplicates.

Reuses Claude B's frozen validators where they exist:
  * `band_edge_instrumentation.validate_record` — six instrumentation groups + secret-in-audit + no
    privileged key in x.
  * `residual_nuisance.assert_residual_not_in_x` — residual/actual/nominal bias not in x.
  * `schema.OBSERVABLE_NUISANCE_KEYS` / `FORBIDDEN_X_SUBSTRINGS` — leakage substrings + observable-nuisance
    gate (DISABLED for this experiment: none may enter x/g/theta/H).
This module adds the runtime-plan checks (306/no-probes/matched-block/duplicate/smoke) on top.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DC = Path(__file__).resolve().parents[1]
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from offline_v2.calibration_bias import band_edge as BE                     # noqa: E402
from offline_v2.calibration_bias import band_edge_instrumentation as BEI    # noqa: E402
from offline_v2.calibration_bias import residual_nuisance as RN             # noqa: E402
from offline_v2.calibration_bias import schema as CBS                       # noqa: E402

MODEL_INPUT_KEYS = ("x", "g", "theta", "H")   # channels a deployable model may read


# --------------------------------------------------------------- plan / count / matched-block
def validate_plan(manifest: dict, *, expect_full: bool = True) -> dict:
    errs = []
    d = manifest["design"]
    if expect_full:
        if manifest["planned_episode_count"] != 306:
            errs.append(f"planned_episode_count {manifest['planned_episode_count']} != 306")
        if d["n_blocks"] != 18 or d["n_offsets"] != 17:
            errs.append(f"blocks/offsets {d['n_blocks']}x{d['n_offsets']} != 18x17")
    if d["run_probes"] is not False:
        errs.append("run_probes must be False (no probes)")
    if d["nominal_bias_y"] != 0.0:
        errs.append(f"nominal_bias_y {d['nominal_bias_y']} != 0.0")
    if list(d["offsets"]) != list(BE.OFFSET_GRID):
        errs.append("offsets != frozen OFFSET_GRID")
    # residual matches frozen v3
    r = manifest["residual"]
    if not (r["sigma"] == 0.005 and r["lo"] == -0.01 and r["hi"] == 0.01 and r["mean"] == 0.0):
        errs.append(f"residual != frozen v3 {r}")
    # one residual per block; blocks independent (distinct residual draws expected but not required equal)
    for b in manifest["blocks"]:
        if abs(b["actual_bias_y"] - (b["nominal_bias_y"] + b["residual_bias_y"])) > 1e-12:
            errs.append(f"block {b['block_id']} actual != nominal+residual")
    # planned episode uniqueness (no duplicate (block, offset))
    seen = set()
    for e in manifest["planned_episodes"]:
        key = (e["block_id"], e["offset_id"])
        if key in seen:
            errs.append(f"duplicate planned episode {key}")
        seen.add(key)
    # each selected block has each offset exactly once (matched block plan)
    if expect_full:
        for b in manifest["randomized_block_order"]:
            got = sorted(e["offset_id"] for e in manifest["planned_episodes"] if e["block_id"] == b)
            want = sorted(BEI_offset_ids(manifest))
            if got != want:
                errs.append(f"block {b} offsets {len(got)} != full 17"); break
    return {"ok": not errs, "errors": errs, "n_planned": manifest["planned_episode_count"]}


def BEI_offset_ids(manifest: dict):
    return list(manifest["canonical_offset_order"])


def validate_matched_block(records: list[dict]) -> dict:
    """Within one block: nominal/residual/actual, initial x, g, mechanism, target, joint nuisance and
    target jitter identical; ONLY grasp_offset_local_y / candidate id differ."""
    errs = []
    import numpy as np
    by_block = {}
    for e in records:
        by_block.setdefault(e.get("nuisance_block_id", e.get("block_id")), []).append(e)
    for bid, es in by_block.items():
        def sec(e, k):
            return (e.get("secret_deployment_state") or {}).get(k)
        for k in ("nominal_bias_y", "residual_bias_y", "actual_bias_y"):
            vals = {round(float(sec(e, k)), 9) for e in es if sec(e, k) is not None}
            if len(vals) > 1:
                errs.append(f"block {bid}: {k} varies within block ({vals})")
        # g identical
        gts = {round(float(e["g"]["target_open_position"]), 9) for e in es}
        if len(gts) > 1:
            errs.append(f"block {bid}: target varies within block")
        # nuisance identical
        for k in ("nuisance_target_jitter",):
            vals = {round(float(e.get(k)), 9) for e in es if e.get(k) is not None}
            if len(vals) > 1:
                errs.append(f"block {bid}: {k} varies within block")
        deltas = {tuple(round(v, 9) for v in e.get("nuisance_robot_joint_delta", [])) for e in es}
        if len(deltas) > 1:
            errs.append(f"block {bid}: nuisance_robot_joint_delta varies within block")
        # initial x identical within tolerance (only offset should differ; offset is in theta, not x)
        rj = np.array([e["x"]["robot_joint_pos"] for e in es], dtype=float)
        tc = np.array([e["x"]["tcp_pos"] for e in es], dtype=float)
        spread = float(max(np.ptp(rj, axis=0).max(), np.ptp(tc, axis=0).max())) if len(es) > 1 else 0.0
        if spread > 5e-3:
            errs.append(f"block {bid}: initial x spread {spread:.2e} > 5e-3 (more than offset differs)")
        # only offset / candidate id vary
        offs = {round(float(e["theta"]["grasp_offset_local_y"]), 9) for e in es}
        if len(offs) != len(es):
            errs.append(f"block {bid}: candidate offsets not unique within block")
    return {"ok": not errs, "errors": errs, "n_blocks": len(by_block)}


# --------------------------------------------------------------- leakage / secret / observable-nuisance
def validate_leakage(record: dict) -> dict:
    errs = []
    # 1) residual/actual/nominal not in x  (reuse B)
    try:
        RN.assert_residual_not_in_x(record)
    except ValueError as ex:
        errs.append(str(ex))
    # 2) forbidden substrings not in x  (reuse B schema)
    for k in record.get("x", {}):
        low = str(k).lower()
        if any(s in low for s in CBS.FORBIDDEN_X_SUBSTRINGS):
            errs.append(f"x carries forbidden key '{k}'")
    # 3) OBSERVABLE_NUISANCE_KEYS gate: DISABLED for this experiment -> none may enter model channels
    for ch in MODEL_INPUT_KEYS:
        blk = record.get(ch)
        if isinstance(blk, dict):
            for k in blk:
                if k in CBS.OBSERVABLE_NUISANCE_KEYS:
                    errs.append(f"observable nuisance key '{k}' present in model channel '{ch}' (disabled here)")
    # 4) secret bias fields only in audit/secret block
    for sf in BEI.SECRET_BIAS_FIELDS:
        for ch in MODEL_INPUT_KEYS:
            blk = record.get(ch)
            if isinstance(blk, dict) and sf in blk:
                errs.append(f"secret field '{sf}' present in '{ch}'")
    return {"ok": not errs, "errors": errs}


def validate_record_full(record: dict) -> dict:
    """Schema (B) + leakage (this module) for one episode record."""
    r1 = BEI.validate_record(record)
    r2 = validate_leakage(record)
    return {"ok": r1["ok"] and r2["ok"], "schema_errors": r1["errors"], "leakage_errors": r2["errors"]}


# --------------------------------------------------------------- smoke exclusion / duplicate episodes
def is_smoke(record_or_meta: dict) -> bool:
    return bool(record_or_meta.get("smoke_only", False))


def exclude_smoke(records: list[dict]) -> list[dict]:
    """Formal analysis MUST call this: drop any smoke_only episode."""
    return [e for e in records if not is_smoke(e)]


def reject_duplicate_episodes(records: list[dict]) -> dict:
    """Duplicate episode_id OR duplicate (block, offset) in a formal set is an error."""
    errs = []
    ids, keys = set(), set()
    for e in records:
        eid = e.get("episode_id")
        if eid in ids:
            errs.append(f"duplicate episode_id {eid}")
        ids.add(eid)
        key = (e.get("nuisance_block_id", e.get("block_id")), e.get("offset_id"))
        if key in keys:
            errs.append(f"duplicate (block,offset) {key}")
        keys.add(key)
    return {"ok": not errs, "errors": errs, "n": len(records)}


def assert_no_probes(records: list[dict]) -> dict:
    n_probe = sum(1 for e in records if e.get("episode_role") == "probe")
    return {"ok": n_probe == 0, "n_probe": n_probe}
