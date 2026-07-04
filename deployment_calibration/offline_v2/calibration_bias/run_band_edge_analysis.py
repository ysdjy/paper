"""Driver: read-only analysis of the authorized 306-episode band-edge run -> all frozen artifacts.

    python -m deployment_calibration.offline_v2.calibration_bias.run_band_edge_analysis \
        <data_dir> --out <docs_dir>

Emits (per the protocol): integrity audit (md+json), binned success (csv+json), logistic/isotonic/
bootstrap json, directional/residual/collision/continuous/analysis md, exit verdict json. Never mutates
the source data. Bin width / models / bootstrap unit+count / success definition / exit criteria are all
frozen in band_edge.py and band_edge_analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import band_edge as BE          # noqa: E402
from . import band_edge_analysis as A   # noqa: E402

FROZEN_CONFIG_SHA = "2f20429b5273cb7d269ae0e78f6361e98d2d7cd138b55324549b36a98a2dac86"
FROZEN_SCIENCE_SHA = "e2c8216ed2b2bae19d03815ed922eaf49a6870afce72d7d3fb544e98cd7c86ba"
FROZEN_EPISODES_SHA = "131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e"
AUTH_SOURCE_COMMIT = "e447f00728b9b3dd26cbc85ecc3f0c2de17822f5"


def integrity_audit(data_dir):
    import band_edge_validators_v1 as VAL   # from data_generation (on sys.path when run in-tree)
    import band_edge_reliability_v1 as REL
    eps, man, meta = A.load_run(data_dir)
    st = json.loads((Path(data_dir) / "run_status.json").read_text())
    sc = json.loads((Path(data_dir) / "self_check.json").read_text())
    grid = set(round(o, 6) for o in BE.OFFSET_GRID)
    from collections import Counter
    bc = Counter(e["block_id"] for e in eps)
    planned_ids = {pe["planned_episode_id"] for pe in man["planned_episodes"]}
    checks = {
        "run_status_COMPLETE": st["run_status"] == "COMPLETE",
        "self_check_ok": bool(sc.get("ok")),
        "episode_count_306": len(eps) == 306,
        "unique_ids_306": len(set(e["episode_id"] for e in eps)) == 306,
        "unique_block_offset_306": len(set((e["block_id"], e["offset_id"]) for e in eps)) == 306,
        "blocks_18": len(set(e["block_id"] for e in eps)) == 18,
        "each_block_17": set(bc.values()) == {17},
        "offset_grid_matches_frozen": all(round(e["theta"]["grasp_offset_local_y"], 6) in grid for e in eps),
        "nominal_bias_0": all(e["secret_deployment_state"]["nominal_bias_y"] == 0.0 for e in eps),
        "no_probes": sum(1 for e in eps if e.get("episode_role") == "probe") == 0,
        "no_smoke": sum(1 for e in eps if e.get("smoke_only")) == 0,
        "all_planned_no_unplanned": {e.get("planned_episode_id") for e in eps} == planned_ids,
        "config_sha256_match_frozen": meta["config_sha256"] == FROZEN_CONFIG_SHA,
        "science_sha256_match_frozen": meta["science_manifest_sha256"] == FROZEN_SCIENCE_SHA,
        "episodes_sha256_match_frozen": A.episodes_sha256(data_dir) == FROZEN_EPISODES_SHA,
        "code_commit_e447f00": all(str(e.get("code_commit", "")).startswith("e447f00") for e in eps),
        "schema_leakage_all_valid": all(VAL.validate_record_full(e)["ok"] for e in eps),
        "matched_block_valid": VAL.validate_matched_block(eps)["ok"],
        "instrumentation_complete": A.instrumentation_confound(A.to_records(eps), eps)["INSTRUMENTATION_COMPLETE"],
        "contact_sensor_available_all": all(e.get("contact_sensor_available") for e in eps),
        "records_match_episodes": set(e["episode_id"] for e in eps) == set(REL.committed_records(data_dir).keys()),
    }
    # identities
    id_ok = True
    for e in eps:
        s = e["secret_deployment_state"]; off = e["theta"]["grasp_offset_local_y"]
        if abs(s["actual_bias_y"] - (s["nominal_bias_y"] + s["residual_bias_y"])) > 1e-9 \
           or abs(e["eff_signed"] - (s["actual_bias_y"] + off)) > 1e-6 \
           or abs(e["abs_eff"] - abs(e["eff_signed"])) > 1e-9:
            id_ok = False; break
    checks["identities_actual_eff_abseff"] = id_ok
    return {"ok": all(checks.values()), "checks": checks,
            "n_episodes": len(eps), "n_success": sum(e["y"]["success"] for e in eps),
            "n_failure": sum(1 for e in eps if not e["y"]["success"])}


def _env_provenance(data_dir, out_dir):
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[3].parent), *a], text=True).strip()
        except Exception:
            return ""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "analysis_git_commit": git("rev-parse", "HEAD"),
            "analysis_dirty": bool(git("status", "--porcelain").strip()),
            "input_episodes_sha256": A.episodes_sha256(data_dir),
            "bootstrap_seed": A.derive_bootstrap_seed(data_dir),
            "n_boot": A.N_BOOT, "bin_width": A.BIN_WIDTH, "ci": A.CI}


def _write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], dict) else r[k]) for k in keys})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # make data_generation importable for validators/reliability
    dc = Path(a.data_dir).resolve().parents[1]
    sys.path.insert(0, str(dc)); sys.path.insert(0, str(dc / "data_generation"))

    prov = _env_provenance(a.data_dir, out)
    seed = prov["bootstrap_seed"]

    # 3. integrity (gate)
    integ = integrity_audit(a.data_dir)
    (out / "band_edge_data_integrity_v1.json").write_text(json.dumps({"provenance": prov, **integ}, indent=2))
    if not integ["ok"]:
        (out / "band_edge_exit_verdict_v1.json").write_text(json.dumps(
            {"verdict": "DATA_INTEGRITY_FAILURE", "failed_checks": [k for k, v in integ["checks"].items() if not v]}, indent=2))
        print("DATA_INTEGRITY_FAILURE:", [k for k, v in integ["checks"].items() if not v]); return 1

    eps, man, meta = A.load_run(a.data_dir)
    recs = A.to_records(eps)

    # 5.1 binned
    bins_all = A.binned_success(recs, seed=seed)
    bins_neg = A.binned_success(recs, direction="neg", seed=seed)
    bins_pos = A.binned_success(recs, direction="pos", seed=seed)
    keys = ["bin_lo", "bin_hi", "bin_center", "n_episodes", "n_blocks", "n_success", "n_failure",
            "success_rate", "ci_low", "ci_high", "failure_reasons", "failure_phases"]
    _write_csv(out / "band_edge_binned_success_rates_v1.csv", bins_all, keys)
    (out / "band_edge_binned_success_rates_v1.json").write_text(json.dumps(
        {"all": bins_all, "neg_direction": bins_neg, "pos_direction": bins_pos, "provenance": prov}, indent=2))

    # 5.2 / 5.3
    logi = A.logistic_analysis(recs, seed=seed)
    iso = A.isotonic_analysis(recs, seed=seed)
    (out / "band_edge_logistic_fit_v1.json").write_text(json.dumps({**logi, "provenance": prov}, indent=2))
    (out / "band_edge_isotonic_fit_v1.json").write_text(json.dumps({**iso, "provenance": prov}, indent=2))

    # block bootstrap bundle (edge center/scale + isotonic crossing)
    bb = {"logistic_center": {"point": logi["edge_center"], "ci": logi["center_ci"]},
          "logistic_scale": {"point": logi["edge_scale"], "ci": logi["scale_ci"]},
          "isotonic_crossing": {"point": iso["crossing_0.5"], "ci": [iso["crossing_ci_low"], iso["crossing_ci_high"]]},
          "n_boot": A.N_BOOT, "resample_unit": "block", "seed": seed, "provenance": prov}
    (out / "band_edge_block_bootstrap_v1.json").write_text(json.dumps(bb, indent=2))

    # 6 directional
    asym = A.directional_asymmetry(recs, seed=seed)
    # 7 boundary
    boundary = A.boundary_mixed_labels(recs)
    # 8.1 / 8.2
    per_off = A.per_offset_variation(recs)
    future = A.future_operating_region(recs, logi, iso)
    residual_verdict = bool(per_off["any_offset_mixed_across_blocks"] and future["best_single_enters_nondegenerate_edge"])
    # 9 collision
    coll = A.collision_analysis(recs, eps)
    # 10 instrumentation
    instr = A.instrumentation_confound(recs, eps)
    # 11 continuous
    cont = A.continuous_outcomes(eps, seed=seed)

    # exit verdict (frozen)
    exit_pass = (boundary["BOUNDARY_MIXED_LABELS"] and residual_verdict and logi["scale_identifiable"]
                 and asym["verdict"] == "NO_SEVERE_ASYMMETRY" and coll["contact_sensor_available_all"]
                 and instr["INSTRUMENTATION_COMPLETE"] and coll["EDGE_NOT_COLLISION_DRIVEN"] and integ["ok"])
    verdict = "PASS_TO_V4" if exit_pass else "MODIFY_RESIDUAL_OR_DESIGN"
    exit_obj = {"verdict": verdict,
                "criteria": {
                    "boundary_mixed_labels_0.025_0.035": boundary["BOUNDARY_MIXED_LABELS"],
                    "residual_induces_block_label_variation": residual_verdict,
                    "edge_center_scale_estimable": logi["scale_identifiable"],
                    "no_severe_asymmetry": asym["verdict"] == "NO_SEVERE_ASYMMETRY",
                    "contact_sensor_available": coll["contact_sensor_available_all"],
                    "instrumentation_complete": instr["INSTRUMENTATION_COMPLETE"],
                    "edge_not_collision_driven": coll["EDGE_NOT_COLLISION_DRIVEN"],
                    "data_integrity_ok": integ["ok"]},
                "key_numbers": {"edge_center": logi["edge_center"], "edge_scale": logi["edge_scale"],
                                "edge_center_ci": logi["center_ci"], "isotonic_crossing": iso["crossing_0.5"],
                                "boundary_success_rate": boundary["success_rate"]},
                "provenance": prov}
    (out / "band_edge_exit_verdict_v1.json").write_text(json.dumps(exit_obj, indent=2))

    # stash intermediate objects for the md writers / tests
    bundle = {"integ": integ, "logi": logi, "iso": iso, "asym": asym, "boundary": boundary,
              "per_off": per_off, "future": future, "residual_verdict": residual_verdict,
              "coll": coll, "instr": instr, "cont": cont, "bins_all": bins_all, "exit": exit_obj, "prov": prov}
    (out / "_band_edge_analysis_bundle_v1.json").write_text(json.dumps(bundle, indent=2, default=str))
    print(json.dumps({"integrity_ok": integ["ok"], "verdict": verdict,
                      "edge_center": logi["edge_center"], "edge_scale": logi["edge_scale"],
                      "edge_center_ci": logi["center_ci"], "isotonic_crossing": iso["crossing_0.5"],
                      "boundary_mixed": boundary["BOUNDARY_MIXED_LABELS"],
                      "residual_block_variation": residual_verdict,
                      "asymmetry": asym["verdict"], "edge_not_collision": coll["EDGE_NOT_COLLISION_DRIVEN"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
