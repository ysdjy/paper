"""Bias-level split for the calibration-bias stage (offline).

The confirmatory protocol must avoid rewarding pure discrete-state memorisation, so we split by
BIAS LEVEL (not merely by session) and hold out UNSEEN INTERIOR bias levels for test — these
measure interpolation across the bias range, not extrapolation past its edges.

Guarantees / audits:
  * every bias level assigned to exactly one of {train, val, test} (partition).
  * pairwise-disjoint bias levels AND their sessions AND their nuisance seeds.
  * each TEST bias level is bracketed by TRAIN levels on both sides (true interpolation).
  * extremes of the bias range are kept in train (anchor the range).
"""

from __future__ import annotations

from collections import defaultdict

from .schema import DEFAULT_FIELDS, FieldMap, bias_level_of


def level_values(episodes, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    """bias_level_id -> representative numeric bias value."""
    out = {}
    for e in episodes:
        try:
            lvl = bias_level_of(e, fm)
            val = float((e.get(fm.secret) or {}).get(fm.bias_value_in_secret))
        except (KeyError, TypeError):
            continue
        out.setdefault(lvl, val)
    return out


def split_bias_levels(levels: dict, n_test_interior: int = 2, n_val_interior: int = 1) -> dict:
    """Deterministic level split. `levels` = {level_id: numeric_value}.

    Returns {"train":[ids], "val":[ids], "test":[ids]}. Test levels are interior and each is
    bracketed by train levels; extremes go to train.
    """
    items = sorted(levels.items(), key=lambda kv: kv[1])
    ids = [k for k, _ in items]
    n = len(ids)
    if n < 5:
        raise ValueError(f"need >=5 bias levels for a held-out interpolation split, got {n}")
    train, val, test = set(), set(), set()
    train.add(ids[0]); train.add(ids[-1])
    interior_idx = list(range(1, n - 1))
    # pick test from the center outward, keeping >=1 gap so no two test are adjacent
    by_center = sorted(interior_idx, key=lambda i: abs(i - (n - 1) / 2))
    for i in by_center:
        if len(test) >= n_test_interior:
            break
        if ids[i - 1] not in test and ids[i + 1] not in test:
            test.add(ids[i])
    # force both neighbours of each test level into train (bracketing / interpolation)
    for i in interior_idx:
        if ids[i] in test:
            train.add(ids[i - 1]); train.add(ids[i + 1])
    # val from remaining interior, near the middle
    remaining = [ids[i] for i in by_center if ids[i] not in test and ids[i] not in train]
    for lid in remaining[:n_val_interior]:
        val.add(lid)
    # everything else -> train
    for lid in ids:
        if lid not in test and lid not in val:
            train.add(lid)
    return {"train": sorted(train, key=lambda k: levels[k]),
            "val": sorted(val, key=lambda k: levels[k]),
            "test": sorted(test, key=lambda k: levels[k])}


def build_manifest(episodes, split: dict, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    levels = level_values(episodes, fm)
    # map sessions + seeds per level
    lvl_sessions = defaultdict(set)
    lvl_seeds = defaultdict(set)
    for e in episodes:
        try:
            lvl = bias_level_of(e, fm)
        except KeyError:
            continue
        lvl_sessions[lvl].add(e.get(fm.session_id))
        if e.get(fm.nuisance_seed) is not None:
            lvl_seeds[lvl].add(e.get(fm.nuisance_seed))

    def sessions_of(split_name):
        s = set()
        for lid in split[split_name]:
            s |= lvl_sessions[lid]
        return s

    def seeds_of(split_name):
        s = set()
        for lid in split[split_name]:
            s |= lvl_seeds[lid]
        return s

    audit = audit_split(split, levels, lvl_sessions, lvl_seeds)
    return {
        "levels": {k: levels[k] for k in levels},
        "split_level_ids": split,
        "split_values": {k: [levels[l] for l in v] for k, v in split.items()},
        "sessions": {k: sorted(sessions_of(k)) for k in ("train", "val", "test")},
        "nuisance_seeds": {k: sorted(seeds_of(k), key=str) for k in ("train", "val", "test")},
        "audit": audit,
    }


def audit_split(split, levels, lvl_sessions, lvl_seeds) -> dict:
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    all_lvls = set(levels)
    # pairwise-disjoint levels
    pair = {"train_val": sorted(tr & va), "train_test": sorted(tr & te), "val_test": sorted(va & te)}
    levels_ok = not (tr & va or tr & te or va & te)
    partition_ok = (tr | va | te) == all_lvls and len(tr) + len(va) + len(te) == len(all_lvls)

    def collect(sset, name):
        s = set()
        for lid in split[name]:
            s |= sset[lid]
        return s

    tr_s, va_s, te_s = (collect(lvl_sessions, "train"), collect(lvl_sessions, "val"), collect(lvl_sessions, "test"))
    sessions_disjoint = not (tr_s & va_s or tr_s & te_s or va_s & te_s)
    tr_z, va_z, te_z = (collect(lvl_seeds, "train"), collect(lvl_seeds, "val"), collect(lvl_seeds, "test"))
    seeds_disjoint = not (tr_z & va_z or tr_z & te_z or va_z & te_z)

    # interpolation: every test level bracketed by a train level below and above
    ordered = [k for k, _ in sorted(levels.items(), key=lambda kv: kv[1])]
    bracket = {}
    for lid in split["test"]:
        i = ordered.index(lid)
        below = any(ordered[j] in tr for j in range(i - 1, -1, -1))
        above = any(ordered[j] in tr for j in range(i + 1, len(ordered)))
        bracket[lid] = bool(below and above)
    interpolation_ok = all(bracket.values()) and len(split["test"]) > 0

    ok = levels_ok and partition_ok and sessions_disjoint and seeds_disjoint and interpolation_ok
    return {"ok": bool(ok),
            "levels_pairwise_disjoint": levels_ok, "level_overlaps": pair,
            "partition_ok": partition_ok,
            "sessions_disjoint": sessions_disjoint,
            "nuisance_seeds_disjoint": seeds_disjoint,
            "test_levels_bracketed_by_train": bracket, "interpolation_ok": interpolation_ok,
            "counts": {k: len(v) for k, v in split.items()}}
