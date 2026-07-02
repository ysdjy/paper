"""Stage-1 analysis: from a capability-map run, characterize the success band and freeze formal theta.

Reports overall + per-theta-dim binned success, terminal-error spread, failure modes; proposes a frozen
formal candidate theta range that yields a 20-80% mix and orderable candidates. Writes
configs/drawer_theta_frozen_v2.yaml + a summary. Pure-python. Run:
    python projects/paper/deployment_calibration/evaluation/analyze_capability_map_v2.py [run_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
DATA = _PAPER / "deployment_calibration" / "data"
KEYS = ["grasp_offset_local_y", "max_pos_step", "pull_lead"]


def _latest():
    runs = sorted(DATA.glob("capability_map_v2_*"))
    if not runs:
        raise SystemExit("no capability_map_v2 run")
    return runs[-1]


def _bin_success(vals, succ, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (vals >= a) & (vals < b)
        out.append((round(a, 4), round(b, 4), int(m.sum()), round(float(succ[m].mean()) if m.sum() else float("nan"), 3)))
    return out


def _band_range(vals, succ, lo, hi):
    """Sub-range of [lo,hi] whose per-quartile success stays roughly inside (0.2,0.85); else full range."""
    qs = np.quantile(vals, [0, .25, .5, .75, 1.0])
    keep_lo, keep_hi = lo, hi
    for a, b in zip(qs[:-1], qs[1:]):
        m = (vals >= a) & (vals < b)
        if m.sum() >= 3:
            sr = float(succ[m].mean())
            if sr < 0.05:                 # near-certain-fail quartile -> trim from the candidate space
                if abs(a - lo) < abs(b - hi):
                    keep_lo = max(keep_lo, b)
                else:
                    keep_hi = min(keep_hi, a)
    if keep_lo >= keep_hi:
        keep_lo, keep_hi = lo, hi
    return round(float(keep_lo), 4), round(float(keep_hi), 4)


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest()
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    cfg = json.loads((run / "metadata.json").read_text()).get("config", {})
    tr = cfg.get("theta_ranges", {k: None for k in KEYS})

    lines = [f"# Stage-1 capability map — v2\n", f"Run: `{run.name}`  ({len(eps)} episodes)\n"]
    frozen = {"drawers": [], "theta_ranges": {}, "targets": cfg.get("targets", [0.12, 0.20, 0.28]),
              "source_run": run.name, "note": "frozen formal candidate theta space (20-80% band, orderable)"}
    drawers = sorted({e["drawer_name"] for e in eps})
    for dn in drawers:
        rows = [e for e in eps if e["drawer_name"] == dn]
        succ = np.array([1.0 if e["y"]["success"] else 0.0 for e in rows])
        finals = np.array([e["y"]["final_joint_position"] for e in rows], dtype=float)
        sr = float(succ.mean())
        reasons = {}
        for e in rows:
            reasons[e["y"]["failure_reason"]] = reasons.get(e["y"]["failure_reason"], 0) + 1
        lines.append(f"\n## {dn}  (n={len(rows)}, success={sr:.2f})")
        lines.append(f"- terminal position: [{np.nanmin(finals):.3f}, {np.nanmax(finals):.3f}], "
                     f"std={np.nanstd(finals):.3f}")
        lines.append(f"- failure modes: {reasons}")
        band = {}
        for k in KEYS:
            v = np.array([e["theta"][k] for e in rows], dtype=float)
            lo, hi = (tr[k] if tr.get(k) else (float(v.min()), float(v.max())))
            edges = np.linspace(lo, hi, 4)
            binned = _bin_success(v, succ, edges)
            lines.append(f"- {k}: " + " ".join(f"[{a},{b}]={sr_}({n})" for a, b, n, sr_ in binned))
            band[k] = _band_range(v, succ, lo, hi)
        # only trust a drawer whose overall success is a genuine mix
        if 0.15 <= sr <= 0.9:
            frozen["drawers"].append(dn)
            for k in KEYS:
                cur = frozen["theta_ranges"].get(k)
                frozen["theta_ranges"][k] = band[k] if cur is None else [min(cur[0], band[k][0]), max(cur[1], band[k][1])]
        lines.append(f"- proposed band per dim: {band}")
        lines.append(f"- verdict: {'MIX (usable)' if 0.15 <= sr <= 0.9 else ('ALL-SUCCESS' if sr > 0.9 else 'ALL-FAIL')}")

    if not frozen["theta_ranges"]:
        # fallback: keep full ranges from config
        frozen["theta_ranges"] = {k: list(tr[k]) for k in KEYS if tr.get(k)}
        frozen["note"] += " (no per-drawer mix found; kept full config ranges -- may need damping to create band)"

    (run / "capability_analysis.md").write_text("\n".join(lines))
    import yaml
    outcfg = _PAPER / "deployment_calibration" / "configs" / "drawer_theta_frozen_v2.yaml"
    yaml.safe_dump(frozen, open(outcfg, "w"), sort_keys=False)
    print("\n".join(lines))
    print(f"\n[analyze] frozen theta -> {outcfg}\n{json.dumps(frozen, indent=1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
