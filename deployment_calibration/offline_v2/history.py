"""Leakage-safe probe-history builder (offline_v2).

For a candidate episode e, its decision-legal history H is EXACTLY the probe episodes with:
  * same session_id as e
  * episode_role == "probe"
  * order_in_session < e.order_in_session
  * (optionally) truncated to the first `history_cutoff` (K) probes
sorted by order_in_session.

FORBIDDEN and actively rejected by `assert_history_legal`:
  * any candidate outcome (self / same-group / future)
  * any probe from a different session
  * any probe at or after the candidate's order (a "future" probe)
  * any probe beyond history_cutoff

This module is deliberately paranoid: the builder returns only whitelisted fields, and the
audit function raises on any contamination so unit tests can inject poison and confirm it dies.
"""

from __future__ import annotations

# Whitelisted probe-outcome fields a model may read from history.
HISTORY_FIELDS = (
    "g", "theta", "success", "failure_reason", "task_outcome_error",
    "skill_elapsed_time", "pull_phase_duration", "handle_relative_error",
    "final_joint_position", "mechanism_id", "probe_index", "order_in_session",
)


def history_entry(probe: dict) -> dict:
    y = probe["y"]
    return {
        "g": probe["g"],
        "theta": probe["theta"],
        "success": bool(y["success"]),
        "failure_reason": y.get("failure_reason", "NONE"),
        "task_outcome_error": float(y["task_outcome_error"]),
        "skill_elapsed_time": float(y["skill_elapsed_time"]),
        "pull_phase_duration": float(y.get("phase_durations", {}).get("PULL", 0.0)),
        "handle_relative_error": float(y.get("handle_relative_error", 0.0)),
        "final_joint_position": float(y["final_joint_position"]),
        "mechanism_id": probe.get("mechanism_id"),
        "probe_index": int(probe.get("probe_index", -1)),
        "order_in_session": int(probe["order_in_session"]),
    }


def build_history(episodes, candidate: dict, k: int | None = None) -> list[dict]:
    """Return the leakage-safe history for `candidate`, truncated to first k probes."""
    sid = candidate["session_id"]
    before = candidate["order_in_session"]
    probes = [e for e in episodes
              if e["session_id"] == sid
              and e.get("episode_role") == "probe"
              and e["order_in_session"] < before]
    probes.sort(key=lambda e: e["order_in_session"])
    if k is not None:
        probes = probes[:k]
    return [history_entry(p) for p in probes]


def assert_history_legal(candidate: dict, history: list[dict], *,
                         all_episodes=None, k: int | None = None) -> None:
    """Raise ValueError on any leakage. Used by tests and defensively in the pipeline.

    Checks structural legality of a built history against its candidate. If `all_episodes`
    is provided, also verifies each history entry corresponds to a real probe in the same
    session (catches cross-session / candidate-sourced injection).
    """
    sid = candidate["session_id"]
    before = candidate["order_in_session"]
    cutoff = candidate.get("history_cutoff") if k is None else k

    # index of legal probes for this session, keyed by order -> canonical entry content
    legal = None
    if all_episodes is not None:
        legal = {}
        for e in all_episodes:
            if (e["session_id"] == sid and e.get("episode_role") == "probe"
                    and e["order_in_session"] < before):
                legal[int(e["order_in_session"])] = history_entry(e)

    orders_seen = []
    for h in history:
        order = h.get("order_in_session")
        if order is None:
            raise ValueError("history entry missing order_in_session (cannot verify legality)")
        if order >= before:
            raise ValueError(f"leakage: history probe order {order} >= candidate order {before} (future)")
        orders_seen.append(order)
        for bad in ("candidate_index", "candidate_group"):
            if bad in h:
                raise ValueError(f"leakage: history entry carries candidate field {bad!r}")
        if any(p in str(kk).lower() for kk in h for p in ("damping", "secret", "hidden")):
            raise ValueError("leakage: history entry carries privileged (damping/secret/hidden) key")
        if legal is not None:
            if order not in legal:
                raise ValueError(f"leakage: history probe order {order} not a legal probe of session {sid}")
            # content must match the canonical legal probe (catches cross-session injection
            # that happens to reuse a legal order number).
            canon = legal[order]
            for fld in ("theta", "success", "task_outcome_error", "skill_elapsed_time",
                        "final_joint_position", "mechanism_id"):
                if h.get(fld) != canon.get(fld):
                    raise ValueError(
                        f"leakage: history entry at order {order} is not a legal probe of "
                        f"session {sid} (field {fld!r} mismatch)")

    if cutoff is not None and len(history) > cutoff:
        raise ValueError(f"leakage: history length {len(history)} exceeds cutoff {cutoff}")

    if orders_seen != sorted(orders_seen):
        raise ValueError("history not sorted by order_in_session")
    if len(set(orders_seen)) != len(orders_seen):
        raise ValueError("duplicate probe in history")
