"""Episode-level logger (task spec sections 6.1 / 10).

Owns one run directory::

    <output_dir>/<run_id>/
        episodes.jsonl            # one JSON object per episode
        trajectories/<id>.npz     # per-episode step trajectory
        metadata.json
        parameter_ranges.json
        environment_config.json
        git_commit.txt
        run_command.txt

The Episode JSON keeps the four variable groups STRICTLY separate (task spec section 1):
``initial_state`` (x), ``task_target`` (g), ``execution_parameters`` (theta), ``outcomes`` (y).
Task target is never mixed into the parameter dict.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


def _json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-safe values; NaN/Inf -> None (so JSON stays valid)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            v = obj.item()
            return _json_safe(v)
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    except Exception:
        pass
    return obj


def git_commit(repo_root: str | Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        commit = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class EpisodeLogger:
    def __init__(self, output_dir: str | Path, run_id: str):
        self.run_dir = Path(output_dir) / run_id
        self.traj_dir = self.run_dir / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.run_id = run_id

    # -- run metadata -------------------------------------------------------
    def write_metadata(
        self,
        metadata: dict[str, Any],
        parameter_ranges: dict[str, Any],
        environment_config: dict[str, Any],
        run_command: str,
        repo_root: str | Path,
    ) -> None:
        (self.run_dir / "metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
        (self.run_dir / "parameter_ranges.json").write_text(
            json.dumps(_json_safe(parameter_ranges), indent=2, sort_keys=True), encoding="utf-8")
        (self.run_dir / "environment_config.json").write_text(
            json.dumps(_json_safe(environment_config), indent=2, sort_keys=True), encoding="utf-8")
        (self.run_dir / "git_commit.txt").write_text(git_commit(repo_root) + "\n", encoding="utf-8")
        (self.run_dir / "run_command.txt").write_text(run_command + "\n", encoding="utf-8")

    # -- per-episode --------------------------------------------------------
    def trajectory_relpath(self, episode_id: str) -> str:
        return f"trajectories/{episode_id}.npz"

    def trajectory_abspath(self, episode_id: str) -> Path:
        return self.traj_dir / f"{episode_id}.npz"

    def completed_episode_ids(self) -> set[str]:
        """For resumable runs: ids already present in episodes.jsonl."""
        ids: set[str] = set()
        if not self.episodes_path.exists():
            return ids
        for line in self.episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["episode_id"])
            except Exception:
                continue
        return ids

    def log_episode(self, record: dict[str, Any]) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

    @staticmethod
    def build_episode_record(
        *,
        episode_id: str,
        seed: int,
        skill: str,
        env_id: int,
        controller: dict[str, Any],
        initial_state: dict[str, Any],
        task_target: dict[str, Any],
        requested_parameters: dict[str, Any],
        effective_parameters: list[dict[str, Any]] | dict[str, Any],
        outcomes: dict[str, Any],
        success: bool,
        failure_reason: str | None,
        trajectory_file: str | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec = {
            "episode_id": episode_id,
            "seed": seed,
            "skill": skill,
            "env_id": env_id,
            "controller": controller,
            "initial_state": initial_state,            # x
            "task_target": task_target,                # g
            "requested_parameters": requested_parameters,
            "execution_parameters": effective_parameters,  # theta (effective, with provenance)
            "outcomes": outcomes,                      # y
            "success": bool(success),
            "failure_reason": failure_reason,
            "trajectory_file": trajectory_file,
        }
        if extra:
            rec.update(extra)
        return rec
