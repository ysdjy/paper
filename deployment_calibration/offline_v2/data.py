"""Read-only loaders and grouping helpers for episodes.jsonl (offline_v2).

Never mutates A's data. Provides typed accessors, session/candidate-group indexes, and a
provenance fingerprint (source path + git commit + sha256) recorded in every report.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Keys that must never appear in a normal model's feature input.
PRIVILEGED_KEYS = ("damping", "secret", "hidden")
THETA_SAMPLED_KEYS = ("grasp_offset_local_y", "max_pos_step", "pull_lead")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunData:
    run_dir: Path
    episodes: list[dict]
    metadata: dict
    episodes_sha256: str

    # ---- provenance for reports ----
    def provenance(self) -> dict:
        return {
            "source_run_path": str(self.run_dir.resolve()),
            "source_git_commit": self.metadata.get("git_commit"),
            "source_branch": self.metadata.get("branch"),
            "contract_version": self.metadata.get("contract_version"),
            "episodes_sha256": self.episodes_sha256,
            "n_episodes": len(self.episodes),
            "damping_verified": self.metadata.get("damping_verified"),
        }

    # ---- selectors ----
    def probes(self) -> list[dict]:
        return [e for e in self.episodes if e.get("episode_role") == "probe"]

    def candidates(self) -> list[dict]:
        return [e for e in self.episodes if e.get("episode_role") == "candidate"]

    def sessions(self) -> dict[str, dict]:
        """session_id -> {drawer, hidden_state_id, damping, n, episodes}."""
        out: dict[str, dict] = {}
        for e in self.episodes:
            sid = e["session_id"]
            s = out.setdefault(sid, {
                "session_id": sid,
                "drawer": e.get("drawer_name"),
                "hidden_state_id": e.get("hidden_state_id"),
                "damping": (e.get("secret_deployment_state") or {}).get("damping"),
                "n": 0,
            })
            s["n"] += 1
        return out

    def candidate_groups(self) -> dict[str, list[dict]]:
        """candidate_group id -> [candidate episodes] (sorted by candidate_index)."""
        g: dict[str, list[dict]] = defaultdict(list)
        for e in self.candidates():
            g[e.get("candidate_group") or e["session_id"]].append(e)
        return {k: sorted(v, key=lambda z: z.get("candidate_index", 0)) for k, v in g.items()}

    def group_to_session(self) -> dict[str, str]:
        return {gid: cands[0]["session_id"] for gid, cands in self.candidate_groups().items()}


def load_run(run_dir: str | Path) -> RunData:
    run = Path(run_dir)
    ep_path = run / "episodes.jsonl"
    if not ep_path.exists():
        raise FileNotFoundError(f"no episodes.jsonl in {run}")
    episodes = [json.loads(l) for l in ep_path.read_text().splitlines() if l.strip()]
    meta_path = run / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return RunData(run_dir=run, episodes=episodes, metadata=metadata,
                   episodes_sha256=sha256_file(ep_path))


def theta_signature(e: dict, ndigits: int = 6) -> tuple:
    th = e.get("theta", {})
    return tuple(round(float(th.get(k, 0.0)), ndigits) for k in THETA_SAMPLED_KEYS)


def assert_no_privileged_in_x(e: dict) -> None:
    """Guard mirroring the contract: x must not carry damping/secret/hidden keys."""
    for k in e.get("x", {}):
        low = k.lower()
        if any(p in low for p in PRIVILEGED_KEYS):
            raise ValueError(f"episode {e.get('episode_id')} x carries privileged key {k!r}")
