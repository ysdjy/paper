#!/usr/bin/env python3
"""Stage-1 pilot / sensitivity analysis (task spec section 16).

Reads one or more pilot run directories (each ``<run>/episodes.jsonl``) and produces, per parameter:

* per-level summary (n, success rate, failure-mode counts);
* effect of the parameter on execution time, terminal accuracy, tracking error, settling time;
* a sensitivity verdict (does the parameter move a metric beyond within-level noise?);
* time-vs-accuracy scatter for SUCCESS episodes;
* lists of effective / ineffective parameters and any "similar success rate, different quality" levels.

No Isaac dependency -- runs under any python with numpy (+ matplotlib if available).

Usage::
    python analysis/analyze_stage1_pilot.py --root experiments/stage1 --out experiments/stage1/_analysis
    python analysis/analyze_stage1_pilot.py --run_dir experiments/stage1/place_one_param_... --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# primary terminal-accuracy metric per skill
PRIMARY_ERROR = {"place": "object_position_error", "open_drawer": "drawer_position_error"}


def load_run(run_dir: Path) -> list[dict]:
    p = run_dir / "episodes.jsonl"
    if not p.exists():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            r["_run"] = run_dir.name
            recs.append(r)
        except Exception:
            continue
    return recs


def param_of(run_dir: Path) -> str | None:
    mp = run_dir / "metadata.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text()).get("param")
        except Exception:
            return None
    return None


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def summarize_param(recs: list[dict], skill: str, param: str) -> dict:
    """Group episodes (one param's run) by level; compute per-level stats + a sensitivity verdict."""
    err_key = PRIMARY_ERROR.get(skill, "object_position_error")
    by_level: dict[str, list[dict]] = {}
    for r in recs:
        if r.get("setup_failure"):
            continue
        by_level.setdefault(str(r.get("level")), []).append(r)

    levels = []
    for level, rs in sorted(by_level.items()):
        succ = [r for r in rs if r.get("success")]
        out = [r.get("outcomes", {}) for r in rs]
        out_s = [r.get("outcomes", {}) for r in succ]
        fmodes: dict[str, int] = {}
        for r in rs:
            if not r.get("success"):
                fmodes[str(r.get("failure_reason"))] = fmodes.get(str(r.get("failure_reason")), 0) + 1
        # numeric level value if scalar (for correlation)
        lvl_val = None
        for r in rs:
            ep = r.get("requested_parameters", {})
            if param in ep and _num(ep[param]) is not None:
                lvl_val = _num(ep[param]); break
        levels.append({
            "level": level,
            "level_value": lvl_val,
            "n": len(rs),
            "n_success": len(succ),
            "success_rate": len(succ) / len(rs) if rs else 0.0,
            "elapsed_time": _agg([_num(o.get("elapsed_time")) for o in out]),
            "primary_error_success": _agg([_num(o.get(err_key)) for o in out_s]),
            "max_tcp_tracking_error_success": _agg([_num(o.get("maximum_tcp_tracking_error")) for o in out_s]),
            "settling_time_success": _agg([_num(o.get("settling_time")) for o in out_s]) if skill == "place" else None,
            "failure_modes": fmodes,
        })

    verdict = sensitivity_verdict(levels, skill)
    return {"param": param, "skill": skill, "levels": levels, "verdict": verdict}


def _spread_vs_noise(level_means, level_stds):
    """Effect size: spread of level means vs typical within-level std. >1 => effect exceeds noise."""
    means = [m for m in level_means if m is not None]
    stds = [s for s in level_stds if s is not None]
    if len(means) < 2:
        return None
    spread = max(means) - min(means)
    noise = st.mean(stds) if stds else 0.0
    if noise <= 1e-12:
        return float("inf") if spread > 1e-9 else 0.0
    return spread / noise


def _corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xa = np.array([p[0] for p in pairs]); ya = np.array([p[1] for p in pairs])
    if xa.std() < 1e-12 or ya.std() < 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def sensitivity_verdict(levels, skill) -> dict:
    metrics = {
        "elapsed_time": [l["elapsed_time"] for l in levels],
        "primary_error": [l["primary_error_success"] for l in levels],
        "max_tcp_tracking_error": [l["max_tcp_tracking_error_success"] for l in levels],
    }
    if skill == "place":
        metrics["settling_time"] = [l["settling_time_success"] for l in levels]
    lvl_vals = [l["level_value"] for l in levels]
    out = {}
    for name, aggs in metrics.items():
        means = [a["mean"] if a else None for a in aggs]
        stds = [a["std"] if a else None for a in aggs]
        ratio = _spread_vs_noise(means, stds)
        corr = _corr(lvl_vals, means)
        # "effective" on this metric: between-level mean spread clearly exceeds within-level noise
        effective = ratio is not None and ratio >= 1.5
        out[name] = {"effect_spread_over_noise": ratio, "corr_level_vs_metric": corr,
                     "level_means": means, "effective": bool(effective)}
    sr = [l["success_rate"] for l in levels]
    out["success_rate_range"] = (min(sr) - max(sr)) if sr else None
    out["success_rates"] = sr
    out["affects_failure"] = bool(sr and (max(sr) - min(sr)) >= 0.2)
    out["any_effective"] = bool(any(out[m].get("effective") for m in metrics) or out["affects_failure"])
    return out


def plot_param(summary, out_dir: Path, recs):
    if not HAVE_MPL:
        return []
    param = summary["param"]; skill = summary["skill"]
    err_key = PRIMARY_ERROR.get(skill)
    figs = []
    # 1) time-vs-accuracy scatter (success episodes), colored by level
    succ = [r for r in recs if r.get("success") and not r.get("setup_failure")]
    if succ:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        levels = sorted({str(r.get("level")) for r in succ})
        cmap = plt.get_cmap("viridis", max(2, len(levels)))
        for i, lv in enumerate(levels):
            xs = [_num(r["outcomes"].get("elapsed_time")) for r in succ if str(r.get("level")) == lv]
            ys = [_num(r["outcomes"].get(err_key)) for r in succ if str(r.get("level")) == lv]
            ax.scatter(xs, ys, s=28, color=cmap(i), label=lv, alpha=0.8)
        ax.set_xlabel("elapsed_time (s)"); ax.set_ylabel(f"{err_key} (m)")
        ax.set_title(f"{skill} / {param}: time vs accuracy (success)")
        ax.legend(fontsize=7); fig.tight_layout()
        f = out_dir / f"{skill}_{param}_time_vs_accuracy.png"
        fig.savefig(f, dpi=120); plt.close(fig); figs.append(f)
    # 2) per-level metric bars
    levels = summary["levels"]
    labels = [l["level"].split("=")[-1] for l in levels]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, title) in zip(axes, [("elapsed_time", "elapsed_time (s)"),
                                       ("primary_error_success", f"{err_key} (m)"),
                                       ("max_tcp_tracking_error_success", "max tcp tracking err (m)")]):
        means = [(l[key]["mean"] if l[key] and l[key]["mean"] is not None else 0.0) for l in levels]
        errs = [(l[key]["std"] if l[key] and l[key]["std"] is not None else 0.0) for l in levels]
        ax.bar(range(len(labels)), means, yerr=errs, capsize=3, color="#4C78A8")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, fontsize=7)
        ax.set_title(title)
    fig.suptitle(f"{skill} / {param}: per-level means (+-std)")
    fig.tight_layout()
    f = out_dir / f"{skill}_{param}_per_level.png"
    fig.savefig(f, dpi=120); plt.close(fig); figs.append(f)
    return figs


def write_summary_md(summaries, out_dir: Path, regressions):
    lines = ["# Stage-1 Pilot / Sensitivity Analysis", ""]
    if regressions:
        lines += ["## Default regression", "", "| run | skill | n | success | rate |", "|---|---|---|---|---|"]
        for r in regressions:
            lines.append(f"| {r['run']} | {r['skill']} | {r['n']} | {r['n_success']} | {r['rate']:.0%} |")
        lines.append("")
    eff, ineff = [], []
    for s in summaries:
        p = s["param"]; v = s["verdict"]
        (eff if v["any_effective"] else ineff).append(p)
        lines += [f"## {s['skill']} :: parameter `{p}`", ""]
        lines += ["| level | n | succ | rate | elapsed mean(s) | err mean(m) | maxTrack(m) |",
                  "|---|---|---|---|---|---|---|"]
        for l in s["levels"]:
            et = l["elapsed_time"]["mean"]; er = l["primary_error_success"]["mean"]; tr = l["max_tcp_tracking_error_success"]["mean"]
            lines.append(f"| {l['level'].split('=')[-1]} | {l['n']} | {l['n_success']} | {l['success_rate']:.0%} | "
                         f"{'-' if et is None else f'{et:.2f}'} | {'-' if er is None else f'{er:.4f}'} | "
                         f"{'-' if tr is None else f'{tr:.4f}'} |")
        lines.append("")
        for m, d in v.items():
            if not isinstance(d, dict):
                continue
            lines.append(f"- **{m}**: effective={d.get('effective')} "
                         f"spread/noise={_fmt(d.get('effect_spread_over_noise'))} corr={_fmt(d.get('corr_level_vs_metric'))}")
        lines.append(f"- success-rate spread across levels: {_fmt(max(v['success_rates'])-min(v['success_rates']) if v['success_rates'] else None)} "
                     f"(affects_failure={v['affects_failure']})")
        lines += [f"- **VERDICT: {'EFFECTIVE' if v['any_effective'] else 'NO MEASURABLE EFFECT'}**", ""]
    lines += ["## Effective parameters", "", ", ".join(sorted(set(eff))) or "(none)", "",
              "## Ineffective parameters (drop from first model inputs)", "",
              ", ".join(sorted(set(ineff))) or "(none)", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / "summary.md"


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf"
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Parent dir; analyze all run subdirs.")
    ap.add_argument("--run_dir", action="append", default=[], help="Specific run dir(s).")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    run_dirs = []
    if args.root:
        run_dirs += [d for d in sorted(Path(args.root).iterdir()) if d.is_dir() and not d.name.startswith("_")]
    run_dirs += [Path(d) for d in args.run_dir]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    regressions = []
    for rd in run_dirs:
        recs = load_run(rd)
        if not recs:
            continue
        skill = recs[0].get("skill", "place")
        param = param_of(rd)
        if not param:  # a fixed/regression run
            succ = [r for r in recs if r.get("success")]
            regressions.append({"run": rd.name, "skill": skill, "n": len(recs), "n_success": len(succ),
                                "rate": len(succ) / len(recs) if recs else 0.0})
            continue
        s = summarize_param(recs, skill, param)
        summaries.append(s)
        plot_param(s, out_dir, recs)

    md = write_summary_md(summaries, out_dir, regressions)
    print(f"[analyze] wrote {md}")
    print(f"[analyze] {len(summaries)} parameter run(s), {len(regressions)} regression run(s), "
          f"plots in {out_dir} (matplotlib={'on' if HAVE_MPL else 'OFF'})")
    # also dump machine-readable
    (out_dir / "analysis.json").write_text(json.dumps({"summaries": summaries, "regressions": regressions},
                                                      indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
