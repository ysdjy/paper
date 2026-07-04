"""Reliability harness helpers for the band-edge full run (v1).

Adds the run-reliability machinery the pre-run audit (MODIFY_RUNTIME) required, WITHOUT touching any
science: manifest-first run identity, crash-safe atomic per-episode persistence, resume/skip-completed
with per-record re-validation, fail-fast, dirty-source refusal for the full run, and a hard end-of-run
self-check that controls the process exit status. The pure functions here are unit-testable without Isaac.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import sys
_DC = Path(__file__).resolve().parents[1]
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_edge_validators_v1 as VAL
from offline_v2.calibration_bias import band_edge as BE

# run_status states (explicit transitions only)
PLANNED, IN_PROGRESS, COMPLETE, FAILED, INVALID = "PLANNED", "IN_PROGRESS", "COMPLETE", "FAILED", "INVALID"

# manifest keys that are volatile provenance / self-referential (NOT frozen science) -> excluded from hash
_VOLATILE_MANIFEST_KEYS = {"run_id", "git_commit", "dirty_worktree", "branch", "code_commit",
                          "science_manifest_sha256"}


# --------------------------------------------------------------------- hashing / identity
def science_manifest_sha256(manifest: dict) -> str:
    """Stable SHA over the SCIENCE content of a manifest (excludes volatile git/run provenance), so the
    same frozen plan hashes identically across runs and machines."""
    core = {k: v for k, v in manifest.items() if k not in _VOLATILE_MANIFEST_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------- atomic persistence
def _fsync_dir(d: Path):
    try:
        fd = os.open(str(d), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def records_dir(outdir: Path) -> Path:
    d = Path(outdir) / "records"
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_write_record(outdir: Path, episode_id: str, record: dict) -> Path:
    """Persist ONE validated record atomically: records/<episode_id>.json via tmp+fsync+rename."""
    rp = records_dir(outdir) / f"{episode_id}.json"
    write_json_atomic(rp, record)
    return rp


def committed_records(outdir: Path) -> dict:
    """Load all committed per-episode records (skips malformed *.tmp; ignores half-written)."""
    out = {}
    rd = Path(outdir) / "records"
    if not rd.exists():
        return out
    for p in sorted(rd.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue   # malformed committed file -> treated as absent; resume re-executes that episode
        out[r.get("episode_id", p.stem)] = r
    return out


def rebuild_episodes_jsonl(outdir: Path, manifest: dict) -> int:
    """Rebuild episodes.jsonl in manifest planned order from committed records (only present ones)."""
    recs = committed_records(outdir)
    by_pid = {}
    for r in recs.values():
        by_pid[r.get("planned_episode_id")] = r
    n = 0
    tmp = Path(outdir) / "episodes.jsonl.tmp"
    with open(tmp, "w") as f:
        for pe in manifest["planned_episodes"]:
            r = by_pid.get(pe["planned_episode_id"])
            if r is not None:
                f.write(json.dumps(r) + "\n"); n += 1
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, Path(outdir) / "episodes.jsonl")
    return n


# --------------------------------------------------------------------- run status
def write_run_status(outdir: Path, status: str, **extra) -> None:
    payload = {"run_status": status, "updated_at": extra.pop("updated_at", None) or _ts(), **extra}
    write_json_atomic(Path(outdir) / "run_status.json", payload)


def read_run_status(outdir: Path) -> dict:
    p = Path(outdir) / "run_status.json"
    return json.loads(p.read_text()) if p.exists() else {"run_status": None}


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_resume_log(outdir: Path, entry: dict) -> None:
    p = Path(outdir) / "resume_log.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"time": _ts(), **entry}) + "\n")
        f.flush(); os.fsync(f.fileno())


# --------------------------------------------------------------------- resume: verify existing records
def validate_existing_record(rec: dict, manifest: dict, planned_ids: set, planned_keys: set,
                             blk_by_id: dict, sci_sha: str) -> list:
    """Per audit §4.2: a committed record must be self-consistent with the frozen manifest."""
    errs = []
    expect_smoke = bool(manifest.get("smoke_only", False))
    if bool(rec.get("smoke_only", False)) != expect_smoke:
        errs.append(f"smoke_only={rec.get('smoke_only')} != run smoke_only={expect_smoke}")
    pid = rec.get("planned_episode_id")
    if pid not in planned_ids:
        errs.append(f"planned_episode_id {pid} not in manifest")
    key = (rec.get("block_id"), rec.get("offset_id"))
    if key not in planned_keys:
        errs.append(f"(block,offset) {key} not in manifest")
    if rec.get("config_sha256") != manifest.get("config_sha256"):
        errs.append("config_sha256 mismatch")
    if rec.get("science_manifest_sha256") != sci_sha:
        errs.append("science_manifest_sha256 mismatch")
    b = blk_by_id.get(rec.get("block_id"))
    if b is not None:
        sec = rec.get("secret_deployment_state") or {}
        for k in ("nominal_bias_y", "residual_bias_y", "actual_bias_y"):
            if round(float(sec.get(k, 1e9)), 9) != round(float(b[k]), 9):
                errs.append(f"secret {k} != manifest for block {b['block_id']}")
        if [round(v, 9) for v in rec.get("nuisance_robot_joint_delta", [])] != \
           [round(v, 9) for v in b["nuisance_robot_joint_delta"]]:
            errs.append("nuisance_robot_joint_delta != manifest")
        if round(float(rec.get("nuisance_target_jitter", 1e9)), 9) != round(float(b["nuisance_target_jitter"]), 9):
            errs.append("nuisance_target_jitter != manifest")
    vr = VAL.validate_record_full(rec)
    if not vr["ok"]:
        errs.append(f"schema/leakage: {vr['schema_errors'] + vr['leakage_errors']}")
    return errs


def load_and_verify_completed(outdir: Path, manifest: dict, sci_sha: str):
    """Return (completed_planned_ids:set, records:dict) or raise ValueError (-> run_status INVALID)."""
    planned_ids = {pe["planned_episode_id"] for pe in manifest["planned_episodes"]}
    planned_keys = {(pe["block_id"], pe["offset_id"]) for pe in manifest["planned_episodes"]}
    blk_by_id = {b["block_id"]: b for b in manifest["blocks"]}
    recs = committed_records(outdir)
    seen_ids, seen_keys, completed = set(), set(), set()
    problems = []
    for eid, rec in recs.items():
        errs = validate_existing_record(rec, manifest, planned_ids, planned_keys, blk_by_id, sci_sha)
        pid = rec.get("planned_episode_id"); key = (rec.get("block_id"), rec.get("offset_id"))
        if pid in seen_ids:
            errs.append(f"duplicate planned_episode_id {pid}")
        if key in seen_keys:
            errs.append(f"duplicate (block,offset) {key}")
        seen_ids.add(pid); seen_keys.add(key)
        if errs:
            problems.append({"episode": eid, "errors": errs})
        else:
            completed.add(pid)
    if problems:
        raise ValueError(json.dumps(problems)[:4000])
    return completed, recs


# --------------------------------------------------------------------- dirty-source refusal (full run)
def _classify_porcelain(porcelain_lines, allowed_prefixes) -> dict:
    """Given `git status --porcelain` lines, split changes into source-tree vs allowed-output.
    A change is ALLOWED only if its path starts with one of allowed_prefixes."""
    offending = []
    allowed = []
    for ln in porcelain_lines:
        ln = ln.rstrip("\n")
        if not ln.strip():
            continue
        path = ln[3:].strip().strip('"')
        if " -> " in path:          # rename
            path = path.split(" -> ", 1)[1].strip().strip('"')
        if any(path.startswith(pref) for pref in allowed_prefixes):
            allowed.append(path)
        else:
            offending.append(path)
    return {"clean_for_full": len(offending) == 0, "offending": offending, "allowed": allowed}


def source_tree_status(paper_root: Path, allowed_prefixes) -> dict:
    try:
        out = subprocess.check_output(["git", "-C", str(paper_root), "status", "--porcelain"], text=True)
    except Exception as ex:
        return {"clean_for_full": False, "offending": [f"git error: {ex}"], "allowed": []}
    return _classify_porcelain(out.splitlines(), list(allowed_prefixes))


def assert_clean_source_tree_for_full_run(paper_root: Path, outdir: Path, extra_allowed=()) -> dict:
    """Full 306 run must start from a clean SOURCE tree. Only the run's own output dir (+ explicitly
    allowed cache dirs) may differ. Raises RuntimeError otherwise. Resume passes when ONLY output changed."""
    rel_out = str(Path(outdir).resolve().relative_to(Path(paper_root).resolve())) + "/"
    allowed = [rel_out, *extra_allowed]
    st = source_tree_status(paper_root, allowed)
    if not st["clean_for_full"]:
        raise RuntimeError(f"dirty source tree — full run refused. Offending paths: {st['offending'][:20]}")
    return st


# --------------------------------------------------------------------- hard end-of-run self-check
def hard_self_check(records: list, manifest: dict) -> dict:
    """Per audit §7.2. Returns {ok, errors, counts}. Caller: ok False -> run_status INVALID, non-zero exit."""
    errs = []
    design = manifest["design"]
    is_full = manifest.get("subset", {}).get("blocks_subset") is None and \
        manifest.get("subset", {}).get("offsets_subset") is None
    planned = manifest["planned_episodes"]
    planned_ids = {pe["planned_episode_id"] for pe in planned}
    frozen_grid = set(round(o, 6) for o in BE.OFFSET_GRID)

    # count / structure
    if len(records) != len(planned):
        errs.append(f"episode count {len(records)} != planned {len(planned)}")
    if is_full:
        if len({r["block_id"] for r in records}) != 18:
            errs.append("blocks != 18")
        for bid in {r["block_id"] for r in records}:
            offs = [r for r in records if r["block_id"] == bid]
            if len(offs) != 17:
                errs.append(f"block {bid} has {len(offs)} offsets != 17")
    # offsets match frozen grid
    for r in records:
        if round(float(r["theta"]["grasp_offset_local_y"]), 6) not in frozen_grid:
            errs.append(f"offset {r['theta']['grasp_offset_local_y']} not in frozen grid"); break
    # duplicates
    ids = [r.get("planned_episode_id") for r in records]
    if len(set(ids)) != len(ids):
        errs.append("duplicate planned_episode_id")
    keys = [(r.get("block_id"), r.get("offset_id")) for r in records]
    if len(set(keys)) != len(keys):
        errs.append("duplicate (block,offset)")
    # probes / smoke (expected smoke flag comes from the run mode)
    if any(r.get("episode_role") == "probe" for r in records):
        errs.append("probe episode present")
    expect_smoke = bool(manifest.get("smoke_only", False))
    if any(bool(r.get("smoke_only", False)) != expect_smoke for r in records):
        errs.append(f"record smoke_only flag inconsistent with run smoke_only={expect_smoke}")
    # all planned completed, none unplanned
    got = set(ids)
    if is_full and got != planned_ids:
        missing = planned_ids - got; extra = got - planned_ids
        errs.append(f"planned mismatch missing={len(missing)} extra={len(extra)}")
    # schema + leakage + matched-block
    for r in records:
        vr = VAL.validate_record_full(r)
        if not vr["ok"]:
            errs.append(f"record {r.get('episode_id')} invalid: {vr['schema_errors']+vr['leakage_errors']}"); break
    mb = VAL.validate_matched_block(records)
    if not mb["ok"]:
        errs.append(f"matched-block: {mb['errors'][:5]}")
    # config / manifest hash consistency
    sci = science_manifest_sha256(manifest)
    if any(r.get("config_sha256") != manifest.get("config_sha256") for r in records):
        errs.append("config_sha256 inconsistent across records")
    if any(r.get("science_manifest_sha256") != sci for r in records):
        errs.append("science_manifest_sha256 inconsistent across records")
    # instrumentation required fields + collision availability field present
    from offline_v2.calibration_bias.band_edge_instrumentation import REQUIRED_FIELD_GROUPS
    req = [f for g in REQUIRED_FIELD_GROUPS.values() for f in g]
    for r in records:
        miss = [f for f in req if f not in r]
        if miss:
            errs.append(f"record {r.get('episode_id')} missing instrumentation {miss}"); break
        if "contact_sensor_available" not in r:
            errs.append("contact_sensor_available field missing"); break

    return {"ok": not errs, "errors": errs,
            "counts": {"records": len(records), "planned": len(planned), "is_full": is_full}}
