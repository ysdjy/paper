"""Calibration-bias stage schema (offline, spec-only).

This module DOES NOT modify the frozen episode contract. It declares, in one place, what the
calibration-bias experiment expects each episode to carry and which fields are model-legal vs
hidden. Claude A's generator should emit episodes matching this spec; the validator enforces it.

Hidden deployment state
------------------------
`bias_y` = the handle local-Y calibration bias actually used by the controller (metres). A
candidate's controllable `grasp_offset_local_y` can COMPENSATE it. The optimal offset is expected
to move with bias, so knowing bias has (hypothesised) decision value.

Model-legal inputs (a deployable model MAY read)
  * x observable keys (existing contract): mechanism_id, drawer_name, member, robot_joint_pos,
    tcp_pos, tcp_quat, gripper_width, initial_mechanism_joint_pos
  * g (target)
  * theta, especially `grasp_offset_local_y`
  * H = same-session earlier probe outcomes (leakage-safe)
  * recorded OBSERVABLE nuisance features (see OBSERVABLE_NUISANCE_KEYS)

Hidden / oracle-only (a deployable model MUST NOT read)
  * bias_y, bias_id / hidden_state_id, bias_level_id
  * secret_deployment_state, any effective/ground-truth calibration error
  * future probes, candidate outcomes, other-session history
  * the raw nuisance SEED (it is provenance, not a feature, and must not encode bias)
"""

from __future__ import annotations

from dataclasses import dataclass, field

CB_STAGE_VERSION = "calibration_bias_v1"

# the hidden state and the controllable that compensates it
HIDDEN_STATE_NAME = "bias_y"
CONTROLLABLE_OFFSET = "grasp_offset_local_y"

# substrings that must NEVER appear as a key inside x (leakage guard)
FORBIDDEN_X_SUBSTRINGS = (
    "bias", "secret", "hidden", "calib", "ground_truth", "groundtruth",
    "effective_error", "true_offset", "z_secret",
)

# observable, model-legal nuisance features (values a deployment sensor could actually see).
# The raw seed is NOT here — only realised, observable quantities may be exposed to a model.
OBSERVABLE_NUISANCE_KEYS = (
    "observed_init_offset", "observed_friction_proxy", "observed_sensor_noise_level",
)

# fields the offline stage reads from each episode (names; validator tolerates a field map)
ROLE_PROBE = "probe"
ROLE_CANDIDATE = "candidate"


@dataclass
class FieldMap:
    """Where the calibration-bias fields live in an episode dict. Defaults match the expected
    generator output; override if Claude A names them differently (documented in prereg)."""
    session_id: str = "session_id"
    episode_role: str = "episode_role"
    order_in_session: str = "order_in_session"
    x: str = "x"
    g: str = "g"
    theta: str = "theta"
    y: str = "y"
    secret: str = "secret_deployment_state"            # oracle-only
    bias_value_in_secret: str = "bias_y"               # secret[bias_value_in_secret]
    bias_id: str = "bias_id"                            # oracle-only label, e.g. "B-03"
    bias_level_id: str = "bias_level_id"               # oracle-only, e.g. "L2"
    nuisance_seed: str = "nuisance_seed"               # provenance only, NOT a feature
    nuisance_block_id: str = "nuisance_block_id"        # paired-block id; provenance only, NOT a feature
    candidate_group: str = "candidate_group"           # selection group (session x target)
    candidate_index: str = "candidate_index"
    candidate_id: str = "candidate_id"                 # offset identity, matched across bias
    offset_id: str = "offset_id"                        # e.g. "o+0.02"
    target_id: str = "target_id"
    matched_group_id: str = "matched_group_id"         # (target, [replicate]) across bias
    replicate_id: str = "replicate_id"
    probe_index: str = "probe_index"
    history_cutoff: str = "history_cutoff"
    # observable nuisance block (optional): a dict of realised observable values
    observable_nuisance: str = "observable_nuisance"


DEFAULT_FIELDS = FieldMap()


def bias_of(e: dict, fm: FieldMap = DEFAULT_FIELDS):
    """Oracle-only accessor for the true bias value (raises if absent)."""
    sec = e.get(fm.secret) or {}
    if fm.bias_value_in_secret not in sec:
        raise KeyError(f"episode {e.get('episode_id')} has no secret {fm.bias_value_in_secret}")
    return float(sec[fm.bias_value_in_secret])


def bias_level_of(e: dict, fm: FieldMap = DEFAULT_FIELDS):
    """Oracle-only bias level id (falls back to the numeric bias if no discrete id)."""
    return e.get(fm.bias_level_id) or e.get(fm.bias_id) or bias_of(e, fm)


def offset_of(e: dict, fm: FieldMap = DEFAULT_FIELDS) -> float:
    """The controllable grasp offset actually used by this episode (model-legal)."""
    return float(e.get(fm.theta, {}).get(CONTROLLABLE_OFFSET, 0.0))


def offset_key(e: dict, fm: FieldMap = DEFAULT_FIELDS):
    """Comparable offset-candidate identity within a matched group across bias levels."""
    return e.get(fm.offset_id) or e.get(fm.candidate_id) or round(offset_of(e, fm), 6)
