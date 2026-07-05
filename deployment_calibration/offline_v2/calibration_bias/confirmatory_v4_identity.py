"""Canonical planned-identity + domain-subseed formats for confirmatory v4 (offline; frozen strings).

FIX2 / BLOCKER_PLANNED_IDENTITY_FORMAT_NOT_FROZEN.

Every planned identity string, its padding/sign/precision, the planned_episode_id template, the domain
subseed labels, and the canonical manifest sort/serialization are pinned here as PURE functions so the test
manifest is uniquely determined (no A-chosen strings, no dependence on Python iteration order). No Isaac,
no manifest instance, no data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from deployment_calibration.offline_v2.calibration_bias import preregistration_v4 as PRE

# ---------------- frozen literal alphabet ----------------
PREFIX = "v4"
ENCODING = "utf-8"
FIELD_SEP = "|"
KV_SEP = "="
SPLITS = ("train", "validation", "test")
ROLES = ("probe", "candidate")
SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}
ROLE_RANK = {"probe": 0, "candidate": 1}
EPISODE_PREFIX = "v4ep-"
EPISODE_HASH_LEN = 24

# frozen block ranges (zero-based, 2-digit) and per-split nominals
BLOCKS = {"train": range(0, 9), "validation": range(0, 6), "test": range(0, 9)}   # 9 / 6 / 9
NOMINALS = {"train": (-0.04, -0.02, 0.0, 0.02, 0.04),
            "validation": (-0.01, 0.01),
            "test": (-0.035, 0.035)}
PROBE_OFFSET = -0.04
CANDIDATE_BANK = (-0.04, 0.0, 0.04)

# frozen domain labels (A may NOT rename these)
DOMAIN_RESIDUAL = "residual"
DOMAIN_NUISANCE = "nuisance"
DOMAIN_SESSION_ORDER = "session_order"
DOMAIN_NOMINAL_ORDER = "nominal_order"
DOMAIN_CANDIDATE_ORDER = "candidate_order"
DOMAIN_TRIAL_INIT = "trial_init"


MAX_ATTEMPT_INDEX = 1                    # initial attempt (0) + at most 1 retry (1)
PID_RE = re.compile(r"^v4ep-[0-9a-f]{24}$")
_ABS_TOL = 1e-12


def fmt_m(x) -> str:
    """Signed fixed-point metres, exactly 3 decimals; +0.000 for zero; negative zero normalized to +0.000."""
    s = f"{float(x):+.3f}"
    if s == "-0.000":
        s = "+0.000"
    return s


# ---------------- FIX3 BLOCKER 2: frozen-design DOMAIN validation ----------------
def _v_split(split) -> str:
    if split not in SPLITS:
        raise ValueError(f"illegal split {split!r} (allowed {SPLITS})")
    return split


def _v_block(split, block_index) -> int:
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise ValueError(f"block_index must be a non-bool int, got {block_index!r}")
    if block_index not in BLOCKS[split]:
        raise ValueError(f"block_index {block_index} out of range for split {split!r} "
                         f"({min(BLOCKS[split])}..{max(BLOCKS[split])})")
    return block_index


def _canon_numeric(value, allowed, what):
    """finite; match one frozen allowed value by isclose(abs_tol=1e-12); return the EXACT frozen value.
    Formatting rounding may NOT turn an illegal value legal (e.g. 0.0346 -> +0.035)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a real number, got {value!r}")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{what} must be finite, got {value!r}")
    for a in allowed:
        if math.isclose(fv, a, rel_tol=0.0, abs_tol=_ABS_TOL):
            return a
    raise ValueError(f"illegal {what} {value!r} (allowed {tuple(allowed)})")


def _v_role(role) -> str:
    if role not in ROLES:
        raise ValueError(f"illegal role {role!r} (allowed {ROLES})")
    return role


def _canon_offset(role, offset):
    if role == "probe":
        return _canon_numeric(offset, (PROBE_OFFSET,), "probe offset")
    return _canon_numeric(offset, CANDIDATE_BANK, "candidate offset")


@dataclass(frozen=True)
class PlannedTrialKey:
    """Validated + canonicalized planned identity. __post_init__ enforces the frozen design domain; every
    public identity/subseed API is built from this key so a generator can never hand-craft an illegal string."""
    split: Literal["train", "validation", "test"]
    block_index: int
    nominal_m: float
    role: Literal["probe", "candidate"]
    offset_m: float

    def __post_init__(self):
        s = _v_split(self.split)
        b = _v_block(s, self.block_index)
        nom = _canon_numeric(self.nominal_m, NOMINALS[s], f"{s} nominal")
        role = _v_role(self.role)
        off = _canon_offset(role, self.offset_m)
        object.__setattr__(self, "split", s)
        object.__setattr__(self, "block_index", b)
        object.__setattr__(self, "nominal_m", nom)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "offset_m", off)

    def block_str(self) -> str:
        return f"{PREFIX}{FIELD_SEP}split{KV_SEP}{self.split}{FIELD_SEP}block{KV_SEP}{self.block_index:02d}"

    def session_str(self) -> str:
        return f"{self.block_str()}{FIELD_SEP}nominal{KV_SEP}{fmt_m(self.nominal_m)}"

    def trial_str(self) -> str:
        return (f"{self.session_str()}{FIELD_SEP}role{KV_SEP}{self.role}"
                f"{FIELD_SEP}offset{KV_SEP}{fmt_m(self.offset_m)}")


def block_identity(split, block_index) -> str:
    s = _v_split(split); b = _v_block(s, block_index)
    return f"{PREFIX}{FIELD_SEP}split{KV_SEP}{s}{FIELD_SEP}block{KV_SEP}{b:02d}"


def session_identity(split, block_index, nominal) -> str:
    s = _v_split(split); b = _v_block(s, block_index)
    nom = _canon_numeric(nominal, NOMINALS[s], f"{s} nominal")
    return f"{block_identity(s, b)}{FIELD_SEP}nominal{KV_SEP}{fmt_m(nom)}"


def trial_identity(split, block_index, nominal, role, offset) -> str:
    # full domain validation + canonicalization via the validated key
    return PlannedTrialKey(split, block_index, nominal, role, offset).trial_str()


def planned_episode_id(trial_identity_str) -> str:
    h = hashlib.sha256(trial_identity_str.encode(ENCODING)).hexdigest()[:EPISODE_HASH_LEN]
    return f"{EPISODE_PREFIX}{h}"


def attempt_id(planned_episode_id_str, attempt_index) -> str:
    if not PID_RE.match(planned_episode_id_str or ""):
        raise ValueError(f"malformed planned_episode_id {planned_episode_id_str!r}")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise ValueError(f"attempt_index must be a non-bool int, got {attempt_index!r}")
    if attempt_index < 0 or attempt_index > MAX_ATTEMPT_INDEX:
        raise ValueError(f"attempt_index {attempt_index} out of range 0..{MAX_ATTEMPT_INDEX}")
    return f"{planned_episode_id_str}{FIELD_SEP}attempt{KV_SEP}{attempt_index:02d}"


# ---------------- domain subseeds (identity-addressed, order-independent) ----------------
def _sub(seed_label, identity, domain):
    return PRE.subseed(PRE.seed(seed_label), f"{identity}{FIELD_SEP}domain{KV_SEP}{domain}")


def block_residual_subseed(split, block_index):
    return _sub(f"{split}_residual_seed", block_identity(split, block_index), DOMAIN_RESIDUAL)


def block_nuisance_subseed(split, block_index):
    return _sub("master_seed", block_identity(split, block_index), DOMAIN_NUISANCE)


def session_order_subseed(split, block_index, nominal):
    return _sub(f"{split}_session_order_seed", session_identity(split, block_index, nominal),
                DOMAIN_SESSION_ORDER)


def nominal_order_subseed(split, block_index, nominal):
    return _sub(f"{split}_nominal_order_seed", session_identity(split, block_index, nominal),
                DOMAIN_NOMINAL_ORDER)


def candidate_order_subseed(split, block_index, nominal):
    return _sub("candidate_order_seed", session_identity(split, block_index, nominal),
                DOMAIN_CANDIDATE_ORDER)


def trial_init_subseed(split, block_index, nominal, role, offset):
    return _sub("master_seed", trial_identity(split, block_index, nominal, role, offset), DOMAIN_TRIAL_INIT)


# ==================== FINAL-001: frozen seed -> canonical order resolution ====================
# The ONLY legal ordering: SHA256-derived integer key + stable lexicographic sort + canonical-index
# tie-break. NO random.shuffle, NO RNG, NO Python hash(), NO dict/set iteration order. A manifest's order
# is uniquely determined by the frozen seeds + canonical identity, so a generator cannot pick any legal
# permutation.
DOMAIN_BLOCK_ORDER = "block_order"
_KEY_MODULUS = 2 ** 63 - 1


def resolve_order(items, *, key_fn, canonical_index_fn):
    """Return tuple(items) sorted by (key_fn(item), canonical_index_fn(item)).

    items must be unique; key_fn(item) a non-bool int in [0, 2**63-1); canonical_index_fn(item) unique
    (deterministic tie-break). Any input permutation yields the same output; key collisions are resolved
    stably by the canonical index."""
    items = list(items)
    seen_item, seen_ci = set(), set()
    decorated = []
    for it in items:
        if it in seen_item:
            raise ValueError(f"resolve_order: duplicate item {it!r}")
        seen_item.add(it)
        k = key_fn(it)
        if isinstance(k, bool) or not isinstance(k, int) or not (0 <= k < _KEY_MODULUS):
            raise ValueError(f"resolve_order: key for {it!r} must be a non-bool int in [0,2**63-1), got {k!r}")
        ci = canonical_index_fn(it)
        if ci in seen_ci:
            raise ValueError(f"resolve_order: duplicate canonical index {ci!r}")
        seen_ci.add(ci)
        decorated.append((k, ci, it))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return tuple(t[2] for t in decorated)


def block_order_key(split, block_index):
    return _sub(f"{split}_block_order_seed", block_identity(split, block_index), DOMAIN_BLOCK_ORDER)


def resolve_block_order(split):
    """Canonical execution order of block indices within a split (by block_order_key, then block index)."""
    s = _v_split(split)
    return resolve_order(list(BLOCKS[s]), key_fn=lambda bi: block_order_key(s, bi),
                         canonical_index_fn=lambda bi: bi)


def resolve_session_order(split, block_index):
    """Canonical execution order of nominals (sessions) within a block: sort by
    (session_order_key, nominal_order_key, canonical nominal index in NOMINALS[split])."""
    s = _v_split(split); b = _v_block(s, block_index)
    noms = list(NOMINALS[s])
    ci = {round(float(n), 6): i for i, n in enumerate(noms)}
    # composite key packed into two-level sort via resolve_order on a surrogate (session_order_key primary)
    def kfn(n):
        return session_order_subseed(s, b, n)
    # secondary tie-break: (nominal_order_key, canonical nominal index) -> encode in canonical_index_fn
    def cifn(n):
        return (nominal_order_subseed(s, b, n), ci[round(float(n), 6)])
    return resolve_order(noms, key_fn=kfn, canonical_index_fn=cifn)


def candidate_item_order_key(split, block_index, nominal, offset):
    tid = trial_identity(split, block_index, nominal, "candidate", offset)
    return PRE.subseed(PRE.seed("candidate_order_seed"), f"{tid}{FIELD_SEP}domain{KV_SEP}{DOMAIN_CANDIDATE_ORDER}")


def candidate_order_key_map(split, block_index, nominal):
    return {fmt_m(o): candidate_item_order_key(split, block_index, nominal, o) for o in CANDIDATE_BANK}


def resolve_candidate_order(split, block_index, nominal):
    """Canonical execution order of the 3 candidate offsets (by candidate_item_order_key, then bank index)."""
    s = _v_split(split); b = _v_block(s, block_index)
    nom = _canon_numeric(nominal, NOMINALS[s], f"{s} nominal")
    bank_idx = {round(float(o), 6): i for i, o in enumerate(CANDIDATE_BANK)}
    return resolve_order(list(CANDIDATE_BANK),
                         key_fn=lambda o: candidate_item_order_key(s, b, nom, o),
                         canonical_index_fn=lambda o: bank_idx[round(float(o), 6)])


def resolve_phase_execution_plan(phase):
    """Canonical_trial_identity sequence in EXACT execution order for a phase.
    split order (train,validation for train_validation; test for test) -> resolve_block_order ->
    resolve_session_order -> probe first, then resolve_candidate_order."""
    splits = ("train", "validation") if phase == "train_validation" else ("test",) if phase == "test" \
        else None
    if splits is None:
        raise ValueError(f"unknown phase {phase!r}")
    plan = []
    for split in splits:
        for b in resolve_block_order(split):
            for nom in resolve_session_order(split, b):
                plan.append(trial_identity(split, b, nom, "probe", PROBE_OFFSET))
                for off in resolve_candidate_order(split, b, nom):
                    plan.append(trial_identity(split, b, nom, "candidate", off))
    return tuple(plan)


# ---------------- enumerate the frozen planned structure (in-memory only) ----------------
def enumerate_trials():
    """All 300 planned trials as dicts, in canonical STORAGE order. No manifest instance is written."""
    rows = []
    for split in SPLITS:
        for b in BLOCKS[split]:
            for nom in sorted(NOMINALS[split]):
                # probe first, then candidates in bank order
                seq = [("probe", PROBE_OFFSET)] + [("candidate", o) for o in sorted(CANDIDATE_BANK)]
                for role, off in seq:
                    tid = trial_identity(split, b, nom, role, off)
                    rows.append({
                        "canonical_block_identity": block_identity(split, b),
                        "canonical_session_identity": session_identity(split, b, nom),
                        "canonical_trial_identity": tid,
                        "planned_episode_id": planned_episode_id(tid),
                        "split": split, "block_index": b, "nominal": nom, "role": role, "offset": off,
                    })
    return rows


def _sort_key(row):
    return (SPLIT_RANK[row["split"]], int(row["block_index"]), float(row["nominal"]),
            ROLE_RANK[row["role"]], float(row["offset"]))


def canonical_manifest_rows():
    return sorted(enumerate_trials(), key=_sort_key)


def canonical_json(obj) -> str:
    """Frozen canonical JSON: keys sorted, tight separators, non-ascii kept, UTF-8, no trailing whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_planned_structure_hash():
    """STRUCTURE-ONLY hash: covers ONLY planned identities + planned_episode_ids. It does NOT cover
    residual/nuisance/execution-order/seeds/environment/generator-commit. It is NOT a fully-resolved
    manifest integrity anchor (see confirmatory_v4_manifest_integrity for the phase full-manifest hashes)."""
    rows = [[r["canonical_trial_identity"], r["planned_episode_id"]] for r in canonical_manifest_rows()]
    return hashlib.sha256(canonical_json(rows).encode(ENCODING)).hexdigest()


# deprecated alias (structure-only; do NOT treat as a full-manifest hash)
canonical_manifest_hash = canonical_planned_structure_hash


IDENTITY_TEMPLATE = {
    "encoding": ENCODING, "field_sep": FIELD_SEP, "kv_sep": KV_SEP,
    "block_identity": "v4|split={split}|block={block_index:02d}",
    "session_identity": "v4|split={split}|block={block_index:02d}|nominal={nominal:+.3f}",
    "trial_identity": "v4|split={split}|block={block_index:02d}|nominal={nominal:+.3f}"
                      "|role={role}|offset={offset:+.3f}",
    "planned_episode_id": "v4ep- + sha256(trial_identity.utf8).hexdigest()[:24]",
    "attempt_id": "{planned_episode_id}|attempt={attempt_index:02d}",
    "float_format": "signed fixed-point, exactly 3 decimals, unit metre; zero=+0.000; negative zero forbidden",
    "block_index": "zero-based, exactly 2 digits",
    "splits": list(SPLITS), "roles": list(ROLES),
    "block_ranges": {s: [min(BLOCKS[s]), max(BLOCKS[s])] for s in SPLITS},
    "domain_labels": {"residual": DOMAIN_RESIDUAL, "nuisance": DOMAIN_NUISANCE,
                      "session_order": DOMAIN_SESSION_ORDER, "nominal_order": DOMAIN_NOMINAL_ORDER,
                      "candidate_order": DOMAIN_CANDIDATE_ORDER, "trial_init": DOMAIN_TRIAL_INIT},
    "domain_subseed_rule": "subseed(<seed>, canonical_identity + '|domain=<label>')",
    "storage_sort": "split rank(train=0,validation=1,test=2) -> block asc -> nominal asc -> "
                    "role(probe=0,candidate=1) -> offset asc",
    "canonical_json": "sort_keys=True, separators=(',',':'), ensure_ascii=False, UTF-8, no trailing newline",
    "resume_key": "planned_episode_id (attempt_index does NOT participate in scientific randomization)",
}


if __name__ == "__main__":
    rows = enumerate_trials()
    pids = [r["planned_episode_id"] for r in rows]
    print("planned trials:", len(rows), "unique ids:", len(set(pids)))
    print("example trial identity:", rows[0]["canonical_trial_identity"])
    print("example planned_episode_id:", rows[0]["planned_episode_id"])
    print("canonical_manifest_hash:", canonical_manifest_hash())
