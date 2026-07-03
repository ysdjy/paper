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


def build_matched_bank(episodes) -> dict:
    """Construct + validate the matched candidate bank. Returns a dict consumed by oracle.py.

    Structure on success:
      {"matched": True,
       "matched_groups": [
          {"matched_group_id": str,
           "target": float, "init_signature": tuple,
           "candidate_keys": [theta_sig,...],
           "dampings": [float,...],
           "by_damping": {damping_value: {candidate_key: episode}}}, ...],
       "excluded": [...], "reason": None}
    """
    cands = [e for e in episodes if e.get("episode_role") == "candidate"]
    # bucket by (init_signature, theta SET)
    buckets = defaultdict(list)
    for e in cands:
        buckets_key = (_init_signature(e),)
        buckets[buckets_key].append(e)

    matched_groups, excluded = [], []
    # Within each init-signature bucket, group sessions by their theta-set; a matched group
    # needs >=2 damping levels sharing the SAME theta set.
    for init_sig_key, eps in buckets.items():
        # organize by session -> theta set
        by_session = defaultdict(list)
        for e in eps:
            by_session[e["session_id"]].append(e)
        # theta set per session
        session_theta_set = {sid: frozenset(theta_signature(x) for x in ceps)
                             for sid, ceps in by_session.items()}
        # cluster sessions by identical theta set
        clusters = defaultdict(list)
        for sid, tset in session_theta_set.items():
            clusters[tset].append(sid)

        for tset, sids in clusters.items():
            damping_of = {}
            for sid in sids:
                d = (by_session[sid][0].get("secret_deployment_state") or {}).get("damping")
                damping_of[sid] = d
            distinct_dampings = sorted(set(damping_of.values()))
            if len(distinct_dampings) < 2:
                excluded.append({"init_signature": init_sig_key[0], "n_sessions": len(sids),
                                 "reason": "candidate theta-set spans <2 damping levels (not matched)"})
                continue
            # Build by_damping: for each damping choose ONE session (first) — validate identity.
            by_damping = {}
            ok = True
            detail = None
            for d in distinct_dampings:
                d_sids = [s for s in sids if damping_of[s] == d]
                sid = sorted(d_sids)[0]
                cand_map = {theta_signature(x): x for x in by_session[sid]}
                by_damping[d] = cand_map
            # identity checks across damping
            key_sets = [set(cm) for cm in by_damping.values()]
            if not all(ks == key_sets[0] for ks in key_sets):
                ok, detail = False, "candidate_key set differs across damping"
            targets = {round(float(list(cm.values())[0]["g"]["target_open_position"]), 6)
                       for cm in by_damping.values()}
            if ok and len(targets) != 1:
                ok, detail = False, "target differs across damping"
            if ok:
                matched_groups.append({
                    "matched_group_id": f"mg_{len(matched_groups):03d}",
                    "target": next(iter(targets)),
                    "init_signature": list(init_sig_key[0]),
                    "candidate_keys": sorted(key_sets[0]),
                    "dampings": distinct_dampings,
                    "by_damping": by_damping,
                })
            else:
                excluded.append({"init_signature": init_sig_key[0], "reason": detail})

    matched = len(matched_groups) > 0
    reason = None if matched else (
        "no candidate theta-set is shared across >=2 damping levels; this run is NOT a matched "
        "candidate bank, so switch-rate / rank-reversal / VSI are not computable")
    return {"matched": matched, "matched_groups": matched_groups,
            "excluded": excluded, "reason": reason,
            "n_matched_groups": len(matched_groups)}


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
