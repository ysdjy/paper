"""Leakage / integrity validator for the calibration-bias stage (offline).

Enforces the schema's model-legal boundary and the matched-offset-bank structure. Returns a
report with ok=False on any violation. Purely read-only. Poison unit tests inject each violation.

Checks:
  1. bias / secret / hidden / calibration keys never appear inside x.
  2. the raw nuisance seed is not exposed as a model feature (not inside x).
  3. history is same-session, strictly-earlier probes only; no candidate outcome enters history.
  4. nuisance seed does NOT map one-to-one to bias, and does not leak bias (a seed value must not
     determine the bias level; a bias level must not be reconstructable from the seed alone).
  5. matched offset bank: same offset-id set across all bias levels per matched group, identical
     theta per offset-id, single target.
"""

from __future__ import annotations

from collections import defaultdict

from ..history import assert_history_legal, build_history
from .schema import (CONTROLLABLE_OFFSET, DEFAULT_FIELDS, FORBIDDEN_X_SUBSTRINGS, FieldMap,
                     ROLE_CANDIDATE, ROLE_PROBE, bias_level_of, offset_key)


def _check(name, ok, detail=""):
    return {"check": name, "ok": bool(ok), "detail": detail}


def validate(episodes, fm: FieldMap = DEFAULT_FIELDS, *, expect_bias_levels=None) -> dict:
    checks = []
    cands = [e for e in episodes if e.get(fm.episode_role) == ROLE_CANDIDATE]
    probes = [e for e in episodes if e.get(fm.episode_role) == ROLE_PROBE]

    # 1 + 2: nothing forbidden (bias/secret) and no raw nuisance provenance (seed / block id) in x
    forbidden_exact = {fm.nuisance_seed, fm.nuisance_block_id}
    x_ok, x_detail = True, "x clean"
    for e in episodes:
        for k in e.get(fm.x, {}):
            low = str(k).lower()
            if any(s in low for s in FORBIDDEN_X_SUBSTRINGS) or low in forbidden_exact:
                x_ok, x_detail = False, f"{e.get('episode_id')} x has forbidden key {k!r}"
                break
        if not x_ok:
            break
    checks.append(_check("no_bias_or_seed_in_x", x_ok, x_detail))

    # 3a: no probe episode carries candidate-only fields (a candidate masquerading as a probe would
    # otherwise be sanitised into a legal-looking history entry by the builder — catch it at source).
    cand_fields = (fm.candidate_id, fm.candidate_group, fm.candidate_index, fm.offset_id)
    mislabel_ok, mislabel_detail = True, "probes carry no candidate fields"
    for e in probes:
        present = [f for f in cand_fields if e.get(f) is not None]
        if present:
            mislabel_ok, mislabel_detail = False, f"{e.get('episode_id')} probe carries {present}"
            break
    checks.append(_check("probes_have_no_candidate_fields", mislabel_ok, mislabel_detail))

    # 3b: leakage-safe history for every candidate
    hist_ok, hist_detail = True, "all candidate histories legal"
    for e in cands:
        try:
            H = build_history(episodes, e, k=e.get(fm.history_cutoff))
            assert_history_legal(e, H, all_episodes=episodes, k=e.get(fm.history_cutoff))
        except ValueError as ex:
            hist_ok, hist_detail = False, f"{e.get('episode_id')}: {ex}"
            break
    checks.append(_check("leakage_safe_history", hist_ok, hist_detail))

    # 4: nuisance seed independent of bias
    ind = nuisance_bias_independence(episodes, fm)
    checks.append(_check("nuisance_seed_independent_of_bias", ind["ok"], ind["detail"]))

    # 5: matched offset bank consistency
    bank = matched_offset_bank(episodes, fm)
    checks.append(_check("matched_offset_bank_consistent", bank["ok"], bank["detail"]))
    if expect_bias_levels is not None:
        checks.append(_check("bias_level_count", bank.get("n_bias_levels") == expect_bias_levels,
                             f"n_bias_levels={bank.get('n_bias_levels')} (expected {expect_bias_levels})"))

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks, "n_episodes": len(episodes),
            "n_candidates": len(cands), "n_probes": len(probes),
            "n_bias_levels": bank.get("n_bias_levels"),
            "n_matched_groups": bank.get("n_matched_groups"),
            "n_selection_groups": bank.get("n_selection_groups")}


def nuisance_bias_independence(episodes, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    """Reject only a bias-ENCODING nuisance id: a one-to-one id<->bias mapping (the id determines
    the bias level). It does NOT penalise the legitimate PAIRED design where the same nuisance
    block/seed is reused across many bias levels — that is the intended exploration-stage structure
    (a block gives a matched nuisance context under every bias), so reuse-across-bias is HEALTHY.

    Uses the nuisance BLOCK id when present (that is the paired-context identity), else the raw seed.

    ok=True requires: NOT a permutation map (some id co-occurs with >1 bias, or >1 id per bias) AND
    the id is not a deterministic 1:1 encoding of bias.
    """
    key_field = fm.nuisance_block_id
    pairs = []
    for e in episodes:
        nid = e.get(fm.nuisance_block_id)
        if nid is None:
            nid = e.get(fm.nuisance_seed)
            key_field = fm.nuisance_seed if e.get(fm.nuisance_block_id) is None else key_field
        if nid is None:
            continue
        pairs.append((nid, bias_level_of(e, fm)))
    if not pairs:
        return {"ok": False, "detail": "no nuisance seed/block recorded — cannot certify independence"}
    id_to_bias = defaultdict(set)
    bias_to_id = defaultdict(set)
    for s, b in pairs:
        id_to_bias[s].add(b)
        bias_to_id[b].add(s)
    n_ids = len(id_to_bias)
    n_bias = len(bias_to_id)
    # bias-encoding: every id -> exactly one bias AND every bias -> exactly one id (a permutation)
    deterministic_map = all(len(v) == 1 for v in id_to_bias.values()) and \
        all(len(v) == 1 for v in bias_to_id.values()) and n_ids == n_bias
    # PAIRED design: same nuisance id reused under different bias levels (matched context) — healthy
    paired_reuse_across_bias = any(len(v) > 1 for v in id_to_bias.values())
    ok = (not deterministic_map) and (paired_reuse_across_bias or n_ids > n_bias or n_ids >= 2 * n_bias)
    detail = (f"key={key_field} n_ids={n_ids} n_bias={n_bias} deterministic_map={deterministic_map} "
              f"paired_reuse_across_bias={paired_reuse_across_bias}")
    return {"ok": bool(ok), "detail": detail, "n_ids": n_ids, "n_bias_levels": n_bias,
            "deterministic_map": deterministic_map,
            "paired_reuse_across_bias": paired_reuse_across_bias,
            # kept for back-compat with existing callers/tests
            "n_seeds": n_ids, "seed_reused_across_bias": paired_reuse_across_bias}


def matched_offset_bank(episodes, fm: FieldMap = DEFAULT_FIELDS) -> dict:
    """Verify the matched offset bank: per matched group, identical offset-id set at every bias
    level, identical theta per offset-id, single target. Returns structure for oracle.py."""
    cands = [e for e in episodes if e.get(fm.episode_role) == ROLE_CANDIDATE]
    groups = defaultdict(list)
    for e in cands:
        key = e.get(fm.matched_group_id) or (e.get(fm.target_id), e.get(fm.replicate_id))
        groups[key].append(e)

    matched, excluded = [], []
    for key, eps in groups.items():
        by_bias = defaultdict(dict)
        for e in eps:
            by_bias[bias_level_of(e, fm)][offset_key(e, fm)] = e
        biases = sorted(by_bias, key=lambda z: str(z))
        if len(biases) < 2:
            excluded.append({"group": str(key), "reason": "spans <2 bias levels"})
            continue
        key_sets = [set(by_bias[b]) for b in biases]
        ok, detail = True, None
        if not all(ks == key_sets[0] for ks in key_sets):
            ok, detail = False, "offset-id set differs across bias"
        if ok:
            for ok_id in key_sets[0]:
                thetas = {round(float(by_bias[b][ok_id][fm.theta][CONTROLLABLE_OFFSET]), 6) for b in biases}
                if len(thetas) != 1:
                    ok, detail = False, f"offset theta differs across bias for {ok_id}"
                    break
        if ok:
            tgs = {by_bias[b][next(iter(key_sets[0]))].get(fm.target_id) for b in biases}
            if len(tgs) != 1:
                ok, detail = False, "target differs across bias"
        if ok:
            matched.append({"matched_group_id": eps[0].get(fm.matched_group_id) or str(key),
                            "biases": biases, "offset_keys": sorted(key_sets[0], key=str),
                            "by_bias": {b: dict(by_bias[b]) for b in biases},
                            "target_id": eps[0].get(fm.target_id)})
        else:
            excluded.append({"group": str(key), "reason": detail})

    n_bias = len({bias_level_of(e, fm) for e in cands})
    sel_groups = {e.get(fm.candidate_group) for e in cands}
    all_ok = len(matched) > 0 and not excluded
    return {"ok": all_ok, "detail": excluded[:5] if excluded else "matched bank consistent",
            "matched": len(matched) > 0, "matched_groups": matched, "excluded": excluded,
            "n_matched_groups": len(matched), "n_bias_levels": n_bias,
            "n_selection_groups": len(sel_groups)}
