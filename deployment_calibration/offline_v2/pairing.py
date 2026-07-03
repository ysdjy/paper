"""Candidate-bank pairing validator + matched-bank construction (offline_v2).

A "matched candidate bank" lets us compare the SAME candidates across damping levels. Before
any switch / reversal / VSI is computed, the bank must pass validation:

For each matched group (candidates sharing target + initial-condition semantics across damping):
  * the candidate_id set is identical across damping levels
  * theta is identical (per candidate_id) across damping levels
  * target g is identical
  * initial-condition semantics are identical (same reset / initial mechanism joint pos)

If any check fails, the group is EXCLUDED and switch/reversal/VSI are not computed for it.
If no group qualifies, the bank is reported not-matched with a reason (the 72-ep pilot case).

Matched-group keying
--------------------
We group candidate episodes by (drawer, target, initial-condition signature, candidate theta
SET). The candidate_key within a matched group is the per-candidate theta signature. Two
sessions belong to the same matched group iff they share the same theta SET and the same
(target, initial-condition) signature but (ideally) different damping. This is robust to A's
exact bookkeeping: it relies on the data, not on trusting an index alignment.
"""

from __future__ import annotations

from collections import defaultdict

from .data import theta_signature


def _init_signature(e: dict, ndigits: int = 4) -> tuple:
    x = e.get("x", {})
    g = e.get("g", {})
    return (
        e.get("drawer_name"),
        round(float(g.get("target_open_position", 0.0)), ndigits),
        round(float(x.get("initial_mechanism_joint_pos", 0.0)), ndigits),
    )


def _matched_key(e: dict):
    """Key that identifies a matched group across damping levels.

    Prefer explicit (drawer, target_id, replicate_id) when the run carries them (v1 selection-
    interaction bank). This holds target + initial-condition semantics fixed while damping varies.
    Fall back to (init_signature, theta-SET) clustering for runs that predate those fields.
    """
    if e.get("target_id") is not None and e.get("replicate_id") is not None:
        return ("tr", e.get("drawer_name"), e.get("target_id"), e.get("replicate_id"))
    return ("legacy", _init_signature(e))


def _candidate_key(e: dict):
    """Comparable candidate identity within a matched group across damping."""
    return e.get("candidate_id") or theta_signature(e)


def build_matched_bank(episodes) -> dict:
    """Construct + validate the matched candidate bank. Returns a dict consumed by oracle.py.

    Structure on success:
      {"matched": True,
       "matched_groups": [
          {"matched_group_id": str, "target": float,
           "candidate_keys": [candidate_id,...], "dampings": [float,...],
           "by_damping": {damping_value: {candidate_key: episode}},
           "n_replicate_sessions_per_damping": {damping: n}}, ...],
       "excluded": [...], "reason": None}
    """
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    legacy = any(_matched_key(e)[0] == "legacy" for e in cands)

    if legacy:
        return _build_legacy(cands)

    # explicit (target, replicate) keying
    buckets = defaultdict(list)
    for e in cands:
        buckets[_matched_key(e)].append(e)

    matched_groups, excluded = [], []
    for key, eps in sorted(buckets.items()):
        # by_damping[d] = {candidate_key: episode}; a legal matched cell has ONE episode per
        # (damping, candidate_key). A duplicate would mean the same candidate ran twice in the
        # same session-damping -> flag it rather than silently overwrite.
        by_damping = defaultdict(dict)
        dup = False
        for e in eps:
            d = float((e.get("secret_deployment_state") or {}).get("damping"))
            ck = _candidate_key(e)
            if ck in by_damping[d]:
                dup = True
            by_damping[d][ck] = e
        dampings = sorted(by_damping)
        if len(dampings) < 2:
            excluded.append({"matched_key": list(key), "reason": "spans <2 damping levels"})
            continue
        key_sets = [set(by_damping[d]) for d in dampings]
        ok, detail = True, None
        if dup:
            ok, detail = False, "duplicate candidate within a (damping) cell"
        if ok and not all(ks == key_sets[0] for ks in key_sets):
            ok, detail = False, "candidate_id set differs across damping"
        if ok:  # theta identical per candidate across damping
            for ck in key_sets[0]:
                sigs = {theta_signature(by_damping[d][ck]) for d in dampings}
                if len(sigs) != 1:
                    ok, detail = False, f"theta differs across damping for {ck}"
                    break
        if ok:  # single target across the group
            tgs = {round(float(by_damping[d][next(iter(key_sets[0]))]["g"]["target_open_position"]), 6)
                   for d in dampings}
            if len(tgs) != 1:
                ok, detail = False, "target differs across damping"
        if not ok:
            excluded.append({"matched_key": list(key), "reason": detail})
            continue
        target = round(float(by_damping[dampings[0]][next(iter(key_sets[0]))]["g"]["target_open_position"]), 6)
        matched_groups.append({
            "matched_group_id": eps[0].get("matched_group_id") or "__".join(str(x) for x in key[1:]),
            "target": target,
            "target_id": eps[0].get("target_id"),
            "replicate_id": eps[0].get("replicate_id"),
            "candidate_keys": sorted(key_sets[0]),
            "dampings": dampings,
            "by_damping": {d: dict(by_damping[d]) for d in dampings},
        })

    matched = len(matched_groups) > 0
    reason = None if matched else "no matched candidate group spans >=2 damping levels"
    return {"matched": matched, "matched_groups": matched_groups, "excluded": excluded,
            "reason": reason, "n_matched_groups": len(matched_groups)}


def _build_legacy(cands) -> dict:
    """Theta-set clustering for runs without target_id/replicate_id (e.g. damping_pilot_v2)."""
    buckets = defaultdict(list)
    for e in cands:
        buckets[(_init_signature(e),)] = buckets[(_init_signature(e),)] + [e]
    matched_groups, excluded = [], []
    for init_sig_key, eps in buckets.items():
        by_session = defaultdict(list)
        for e in eps:
            by_session[e["session_id"]].append(e)
        session_theta_set = {sid: frozenset(theta_signature(x) for x in ceps)
                             for sid, ceps in by_session.items()}
        clusters = defaultdict(list)
        for sid, tset in session_theta_set.items():
            clusters[tset].append(sid)
        for tset, sids in clusters.items():
            damping_of = {sid: (by_session[sid][0].get("secret_deployment_state") or {}).get("damping")
                          for sid in sids}
            distinct = sorted(set(damping_of.values()))
            if len(distinct) < 2:
                excluded.append({"init_signature": init_sig_key[0],
                                 "reason": "candidate theta-set spans <2 damping levels (not matched)"})
                continue
            by_damping = {}
            for d in distinct:
                sid = sorted(s for s in sids if damping_of[s] == d)[0]
                by_damping[d] = {theta_signature(x): x for x in by_session[sid]}
            key_sets = [set(cm) for cm in by_damping.values()]
            if not all(ks == key_sets[0] for ks in key_sets):
                excluded.append({"init_signature": init_sig_key[0], "reason": "key set differs"})
                continue
            matched_groups.append({
                "matched_group_id": f"mg_{len(matched_groups):03d}",
                "target": round(float(list(by_damping[distinct[0]].values())[0]["g"]["target_open_position"]), 6),
                "candidate_keys": sorted(key_sets[0]), "dampings": distinct, "by_damping": by_damping})
    matched = len(matched_groups) > 0
    reason = None if matched else (
        "no candidate theta-set is shared across >=2 damping levels; this run is NOT a matched "
        "candidate bank, so switch-rate / rank-reversal / VSI are not computable")
    return {"matched": matched, "matched_groups": matched_groups, "excluded": excluded,
            "reason": reason, "n_matched_groups": len(matched_groups)}


def validate_bank_report(episodes) -> dict:
    """Human/JSON-friendly summary for validation_report.json."""
    bank = build_matched_bank(episodes)
    return {
        "is_matched_bank": bank["matched"],
        "n_matched_groups": bank["n_matched_groups"],
        "n_excluded_clusters": len(bank["excluded"]),
        "reason": bank["reason"],
        "excluded_sample": bank["excluded"][:5],
        "matched_group_summ": [
            {"matched_group_id": mg["matched_group_id"], "dampings": mg["dampings"],
             "n_candidate_keys": len(mg["candidate_keys"])}
            for mg in bank["matched_groups"][:10]
        ],
    }


def full_pairing_validation(episodes, *, expect_n_episodes=None, expect_n_groups=None,
                            expect_dampings=None) -> dict:
    """Strict gate for Phase-2 (returns ok=False if any check fails).

    Checks: episode completeness, matched-group count, per-group candidate-id set identical
    across all dampings, theta identical per candidate-id, target/replicate consistency, no
    candidate outcome leaking into any candidate's decision-legal history, and no secret/hidden
    key inside any x. Purely read-only.
    """
    from .history import assert_history_legal, build_history

    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    probes = [e for e in episodes if e.get("episode_role") == "probe"]
    add("episode_count", expect_n_episodes is None or len(episodes) == expect_n_episodes,
        f"n_episodes={len(episodes)} (expected {expect_n_episodes})")

    bank = build_matched_bank(episodes)
    add("is_matched_bank", bank["matched"], bank["reason"] or "matched")
    add("n_matched_groups", expect_n_groups is None or bank["n_matched_groups"] == expect_n_groups,
        f"n_matched_groups={bank['n_matched_groups']} (expected {expect_n_groups})")

    # each group spans expected dampings with identical candidate-id set + identical theta
    per_group_ok = True
    per_group_detail = []
    for mg in bank["matched_groups"]:
        dset = set(mg["dampings"])
        exp = set(expect_dampings) if expect_dampings else dset
        key_sets_ok = all(set(mg["by_damping"][d]) == set(mg["candidate_keys"]) for d in mg["dampings"])
        theta_ok = True
        for ck in mg["candidate_keys"]:
            sigs = {theta_signature(mg["by_damping"][d][ck]) for d in mg["dampings"]}
            if len(sigs) != 1:
                theta_ok = False
        g_ok = (dset == exp) and key_sets_ok and theta_ok
        if not g_ok:
            per_group_ok = False
            per_group_detail.append({"mg": mg["matched_group_id"], "dampings_ok": dset == exp,
                                     "candset_ok": key_sets_ok, "theta_ok": theta_ok})
    add("matched_groups_damping_candset_theta_consistent", per_group_ok,
        per_group_detail[:5] if per_group_detail else "all groups consistent")

    # no candidate outcome in any candidate's history (build + assert legal)
    hist_ok, hist_detail = True, ""
    for e in cands:
        try:
            H = build_history(episodes, e, k=e.get("history_cutoff"))
            assert_history_legal(e, H, all_episodes=episodes, k=e.get("history_cutoff"))
        except ValueError as ex:
            hist_ok = False
            hist_detail = f"{e.get('episode_id')}: {ex}"
            break
    add("no_candidate_outcome_in_history", hist_ok, hist_detail or "all candidate histories legal")

    # no secret/hidden key inside any x
    x_ok, x_detail = True, ""
    for e in episodes:
        for k in e.get("x", {}):
            if any(p in k.lower() for p in ("damping", "secret", "hidden")):
                x_ok, x_detail = False, f"{e.get('episode_id')} x has {k}"
                break
        if not x_ok:
            break
    add("no_secret_in_x", x_ok, x_detail or "x clean in all episodes")

    # selection-group structure
    sel_groups = {e.get("candidate_group") for e in cands}
    add("selection_groups_present", len(sel_groups) > 0, f"n_selection_groups={len(sel_groups)}")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks, "n_episodes": len(episodes),
            "n_candidates": len(cands), "n_probes": len(probes),
            "n_matched_groups": bank["n_matched_groups"],
            "n_selection_groups": len(sel_groups)}
