"""Session-level split (v2). Splits deployment sessions (NOT episodes) into train/val/test.

Splitting by session is mandatory: a session's hidden state must not appear in two splits, and history
must be built AFTER the split. Stratifies by (drawer, hidden_state_id) so each split sees each condition.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_episodes(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "episodes.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def sessions_of(episodes: list[dict]) -> dict[str, dict]:
    s = {}
    for e in episodes:
        sid = e["session_id"]
        s.setdefault(sid, {"session_id": sid, "drawer": e["drawer_name"],
                           "hidden_state_id": e.get("hidden_state_id"), "n": 0})
        s[sid]["n"] += 1
    return s


def split_sessions(episodes: list[dict], fracs=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    """Deterministic stratified split by (drawer, hidden_state_id). Returns {train:[sids],val:[],test:[]}."""
    sess = sessions_of(episodes)
    strata = defaultdict(list)
    for sid, s in sorted(sess.items()):
        strata[(s["drawer"], s["hidden_state_id"])].append(sid)
    out = {"train": [], "val": [], "test": []}
    for key, sids in sorted(strata.items()):
        sids = sorted(sids)
        # deterministic rotation offset by seed for reproducibility
        off = seed % max(1, len(sids))
        sids = sids[off:] + sids[:off]
        n = len(sids); n_tr = max(1, round(fracs[0] * n)); n_va = max(0, round(fracs[1] * n))
        out["train"] += sids[:n_tr]
        out["val"] += sids[n_tr:n_tr + n_va]
        out["test"] += sids[n_tr + n_va:]
    # ensure test non-empty when possible
    if not out["test"] and out["train"]:
        out["test"].append(out["train"].pop())
    return out


def split_manifest(episodes, split) -> dict:
    sess = sessions_of(episodes)
    return {"fracs_by_session": True,
            "counts": {k: len(v) for k, v in split.items()},
            "sessions": {k: [{"session_id": s, **{kk: sess[s][kk] for kk in ("drawer", "hidden_state_id")}}
                             for s in v] for k, v in split.items()},
            "disjoint": len(set(split["train"]) & set(split["val"]) & set(split["test"])) == 0}
