"""Session-level train/val/test split + strict integrity audit (offline_v2).

Splitting is BY SESSION (a session's fixed hidden state must never straddle two splits).
The audit enforces:
  * pairwise disjointness  train ∩ val = val ∩ test = train ∩ test = ∅
  * every session assigned to exactly one split (partition, no drops/dupes)
  * every candidate group lives entirely within one split (never split a group)
  * (reported) stratification coverage of hidden states per split
"""

from __future__ import annotations

from collections import defaultdict


def _sessions(episodes):
    s = {}
    for e in episodes:
        sid = e["session_id"]
        s.setdefault(sid, {"drawer": e.get("drawer_name"), "hidden_state_id": e.get("hidden_state_id")})
    return s


def split_sessions(episodes, fracs=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    """Deterministic stratified split by (drawer, hidden_state_id).

    Returns {"train": [sids], "val": [...], "test": [...]}. Deterministic given seed.
    Guarantees a partition of all sessions (no session dropped or duplicated).
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, "fracs must sum to 1"
    sess = _sessions(episodes)
    strata = defaultdict(list)
    for sid, s in sorted(sess.items()):
        strata[(s["drawer"], s["hidden_state_id"])].append(sid)
    out = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        sids = sorted(strata[key])
        off = seed % max(1, len(sids))
        sids = sids[off:] + sids[:off]
        n = len(sids)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        # keep at least one in train and one in test when the stratum allows it
        n_tr = min(max(n_tr, 1), n) if n >= 1 else 0
        n_va = min(max(n_va, 0), n - n_tr)
        if n - n_tr - n_va == 0 and n - n_tr >= 1:
            # ensure test gets a session if any remain after train
            n_va = max(0, n - n_tr - 1)
        out["train"] += sids[:n_tr]
        out["val"] += sids[n_tr:n_tr + n_va]
        out["test"] += sids[n_tr + n_va:]
    return out


def audit_split(episodes, split: dict) -> dict:
    """Strict integrity checks. Returns a report; sets report['ok'] False on any violation."""
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    all_sids = set(_sessions(episodes))

    pair_tr_va = tr & va
    pair_tr_te = tr & te
    pair_va_te = va & te
    assigned = tr | va | te
    missing = all_sids - assigned
    # a duplicate = a session listed in >1 split OR twice in one list
    listed = split["train"] + split["val"] + split["test"]
    duplicates = sorted({x for x in listed if listed.count(x) > 1})

    # candidate group must not cross split
    group_splits = defaultdict(set)
    for e in episodes:
        if e.get("episode_role") != "candidate":
            continue
        gid = e.get("candidate_group") or e["session_id"]
        sid = e["session_id"]
        for name, group in (("train", tr), ("val", va), ("test", te)):
            if sid in group:
                group_splits[gid].add(name)
    crossing_groups = sorted(g for g, names in group_splits.items() if len(names) > 1)

    # stratification coverage
    sess = _sessions(episodes)
    coverage = {}
    for name, group in (("train", tr), ("val", va), ("test", te)):
        states = sorted({sess[s]["hidden_state_id"] for s in group})
        coverage[name] = {"n_sessions": len(group), "hidden_states": states}

    ok = (not pair_tr_va and not pair_tr_te and not pair_va_te
          and not missing and not duplicates and not crossing_groups)
    return {
        "ok": bool(ok),
        "pairwise_disjoint": {
            "train_val": sorted(pair_tr_va),
            "train_test": sorted(pair_tr_te),
            "val_test": sorted(pair_va_te),
            "all_empty": not (pair_tr_va or pair_tr_te or pair_va_te),
        },
        "partition": {
            "missing_sessions": sorted(missing),
            "duplicate_sessions": duplicates,
            "n_total": len(all_sids),
            "n_assigned": len(assigned),
        },
        "candidate_groups": {
            "crossing_split": crossing_groups,
            "all_within_one_split": not crossing_groups,
        },
        "coverage": coverage,
    }


def split_manifest(episodes, split, seed, fracs) -> dict:
    audit = audit_split(episodes, split)
    sess = _sessions(episodes)
    return {
        "seed": seed,
        "fracs": list(fracs),
        "counts": {k: len(v) for k, v in split.items()},
        "sessions": {k: [{"session_id": s, **sess[s]} for s in v] for k, v in split.items()},
        "audit": audit,
    }
