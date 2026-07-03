"""Command-line entry for offline_v2.

  python -m deployment_calibration.offline_v2.cli validate       <run_dir> [--out DIR]
  python -m deployment_calibration.offline_v2.cli evaluate-pilot <run_dir> [--out DIR] [--test-mode]
  python -m deployment_calibration.offline_v2.cli evaluate       <run_dir> [--out DIR] ...
  python -m deployment_calibration.offline_v2.cli train          <run_dir> --model NAME [--out DIR]
  python -m deployment_calibration.offline_v2.cli run-all        <run_dir> [--out DIR]

Only writes under the given --out (default: <run_dir>/../../evaluation/offline_v2/<run_name>).
Never launches Isaac; never mutates the input run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_out(run_dir: Path) -> Path:
    # deployment_calibration/evaluation/offline_v2/<run_name>/
    dc = Path(__file__).resolve().parents[1]
    return dc / "evaluation" / "offline_v2" / run_dir.name


def _cfg_from_args(a):
    from deployment_calibration.offline_v2.utility import UtilityConfig
    if getattr(a, "utility_config", None):
        return UtilityConfig.load(a.utility_config)
    return UtilityConfig(lambda_error=a.lambda_error, lambda_time=a.lambda_time)


def cmd_validate(a):
    from deployment_calibration.offline_v2.data import load_run
    from deployment_calibration.offline_v2.pairing import validate_bank_report
    from deployment_calibration.offline_v2.splits import audit_split, split_sessions, split_manifest
    run = load_run(a.run_dir)
    split = split_sessions(run.episodes, fracs=(0.34, 0.33, 0.33), seed=a.split_seed)
    man = split_manifest(run.episodes, split, a.split_seed, (0.34, 0.33, 0.33))
    rep = {"provenance": run.provenance(), "split_audit": man["audit"],
           "matched_bank": validate_bank_report(run.episodes),
           "n_probes": len(run.probes()), "n_candidates": len(run.candidates()),
           "n_sessions": len(run.sessions())}
    out = Path(a.out) if a.out else _default_out(Path(a.run_dir))
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"split_ok": man["audit"]["ok"],
                      "matched_bank": rep["matched_bank"]["is_matched_bank"],
                      "reason": rep["matched_bank"]["reason"]}, indent=2))
    return 0


def cmd_evaluate(a):
    from deployment_calibration.evaluation.offline_v2.pipeline import evaluate_run
    cfg = _cfg_from_args(a)
    out = Path(a.out) if a.out else _default_out(Path(a.run_dir))
    seeds = tuple(range(a.seeds))
    n_boot = 50 if getattr(a, "test_mode", False) else a.n_boot
    res = evaluate_run(a.run_dir, out, utility_cfg=cfg, seeds=seeds, n_boot=n_boot,
                       split_seed=a.split_seed)
    print(res["summary_md"])
    print(f"\n[wrote artifacts to {out}]")
    return 0


def cmd_train(a):
    from deployment_calibration.evaluation.offline_v2.pipeline import build_pairs
    from deployment_calibration.models_v2 import REGISTRY
    from deployment_calibration.models_v2.train import save_models, train_seeds
    from deployment_calibration.offline_v2.data import load_run
    from deployment_calibration.offline_v2.splits import split_sessions
    run = load_run(a.run_dir)
    split = split_sessions(run.episodes, fracs=(0.34, 0.33, 0.33), seed=a.split_seed)
    kmax = max([e.get("history_cutoff", 3) for e in run.episodes] + [3])
    tr = build_pairs(run.episodes, set(split["train"] + split["val"]), k=kmax)
    va = build_pairs(run.episodes, set(split["val"]), k=kmax) or None
    if a.model not in REGISTRY:
        print(f"unknown model {a.model}; choices: {list(REGISTRY)}"); return 2
    models = train_seeds(a.model, tr, va, seeds=tuple(range(a.seeds)))
    out = Path(a.out) if a.out else _default_out(Path(a.run_dir))
    man = save_models(models, out / "models", a.model)
    print(json.dumps(man, indent=2))
    return 0


def cmd_run_all(a):
    cmd_validate(a)
    return cmd_evaluate(a)


def build_parser():
    p = argparse.ArgumentParser(prog="offline_v2")
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(sp):
        sp.add_argument("run_dir")
        sp.add_argument("--out", default=None)
        sp.add_argument("--split-seed", type=int, default=0, dest="split_seed")
        sp.add_argument("--seeds", type=int, default=5)
        sp.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
        sp.add_argument("--lambda-error", type=float, default=1.0, dest="lambda_error")
        sp.add_argument("--lambda-time", type=float, default=0.02, dest="lambda_time")
        sp.add_argument("--utility-config", default=None, dest="utility_config")
        sp.add_argument("--test-mode", action="store_true", dest="test_mode")
    for nm in ("validate", "evaluate", "evaluate-pilot", "train", "run-all"):
        sp = sub.add_parser(nm); common(sp)
        if nm == "train":
            sp.add_argument("--model", required=True)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    fn = {"validate": cmd_validate, "evaluate": cmd_evaluate, "evaluate-pilot": cmd_evaluate,
          "train": cmd_train, "run-all": cmd_run_all}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
