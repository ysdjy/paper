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


def fmt_m(x) -> str:
    """Signed fixed-point metres, exactly 3 decimals; +0.000 for zero; negative zero normalized to +0.000."""
    s = f"{float(x):+.3f}"
    if s == "-0.000":
        s = "+0.000"
    return s


def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"bad split {split!r}")


def _check_role(role):
    if role not in ROLES:
        raise ValueError(f"bad role {role!r}")


def block_identity(split, block_index) -> str:
    _check_split(split)
    return f"{PREFIX}{FIELD_SEP}split{KV_SEP}{split}{FIELD_SEP}block{KV_SEP}{int(block_index):02d}"


def session_identity(split, block_index, nominal) -> str:
    return f"{block_identity(split, block_index)}{FIELD_SEP}nominal{KV_SEP}{fmt_m(nominal)}"


def trial_identity(split, block_index, nominal, role, offset) -> str:
    _check_role(role)
    return (f"{session_identity(split, block_index, nominal)}"
            f"{FIELD_SEP}role{KV_SEP}{role}{FIELD_SEP}offset{KV_SEP}{fmt_m(offset)}")


def planned_episode_id(trial_identity_str) -> str:
    h = hashlib.sha256(trial_identity_str.encode(ENCODING)).hexdigest()[:EPISODE_HASH_LEN]
    return f"{EPISODE_PREFIX}{h}"


def attempt_id(planned_episode_id_str, attempt_index) -> str:
    return f"{planned_episode_id_str}{FIELD_SEP}attempt{KV_SEP}{int(attempt_index):02d}"


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


def canonical_manifest_hash():
    rows = [[r["canonical_trial_identity"], r["planned_episode_id"]] for r in canonical_manifest_rows()]
    return hashlib.sha256(canonical_json(rows).encode(ENCODING)).hexdigest()


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
