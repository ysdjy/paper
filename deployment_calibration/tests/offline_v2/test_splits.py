"""Unit tests for offline_v2.splits — pairwise disjointness + candidate-group integrity."""

from deployment_calibration.offline_v2 import splits as S


def _mk_episodes():
    eps = []
    # 6 sessions, 2 hidden states, each session 2 probes + 1 group of 2 candidates
    for i in range(6):
        sid = f"s{i}"
        hs = "L1" if i < 3 else "L2"
        for o in range(2):
            eps.append({"session_id": sid, "drawer_name": "d", "hidden_state_id": hs,
                        "episode_role": "probe", "order_in_session": o})
        for c in range(2):
            eps.append({"session_id": sid, "drawer_name": "d", "hidden_state_id": hs,
                        "episode_role": "candidate", "order_in_session": 2 + c,
                        "candidate_group": f"{sid}_g", "candidate_index": c})
    return eps


def test_split_is_partition_and_pairwise_disjoint():
    eps = _mk_episodes()
    split = S.split_sessions(eps, fracs=(0.34, 0.33, 0.33), seed=0)
    audit = S.audit_split(eps, split)
    assert audit["ok"]
    assert audit["pairwise_disjoint"]["all_empty"]
    assert not audit["partition"]["missing_sessions"]
    assert not audit["partition"]["duplicate_sessions"]
    assert audit["candidate_groups"]["all_within_one_split"]


def test_audit_detects_pairwise_overlap():
    eps = _mk_episodes()
    bad = {"train": ["s0", "s1"], "val": ["s1", "s2"], "test": ["s3", "s4", "s5"]}
    audit = S.audit_split(eps, bad)
    assert not audit["ok"]
    assert audit["pairwise_disjoint"]["train_val"] == ["s1"]


def test_audit_detects_group_crossing_split():
    # force a candidate group's sessions... a group lives in ONE session here, so instead
    # fabricate an episode set where a group spans two sessions to prove the check fires.
    eps = _mk_episodes()
    # make s0's group also contain a candidate whose session is s3
    eps.append({"session_id": "s3", "drawer_name": "d", "hidden_state_id": "L2",
                "episode_role": "candidate", "order_in_session": 9,
                "candidate_group": "s0_g", "candidate_index": 5})
    split = {"train": ["s0", "s1", "s2"], "val": ["s3"], "test": ["s4", "s5"]}
    audit = S.audit_split(eps, split)
    assert not audit["ok"]
    assert "s0_g" in audit["candidate_groups"]["crossing_split"]


def test_audit_detects_missing_session():
    eps = _mk_episodes()
    split = {"train": ["s0", "s1"], "val": ["s2"], "test": ["s3", "s4"]}  # s5 dropped
    audit = S.audit_split(eps, split)
    assert not audit["ok"]
    assert "s5" in audit["partition"]["missing_sessions"]
