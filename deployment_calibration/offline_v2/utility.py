"""Decision utility definition and its frozen config (offline_v2).

Utility of choosing a candidate with (predicted or true) outcomes:
    U = p_success - lambda_error * error - lambda_time * time

Rules:
  * lambda_error / lambda_time are determined ONLY on validation and then FROZEN.
  * true_utility uses the realized success/error/time from y.
  * predicted error/time may be clipped to a physically sensible range; the clip bounds
    are FIXED here and recorded in utility_config.json (never tuned on test).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Physically sensible clip ranges for predicted regression outputs.
# task_outcome_error is a joint-position error in metres for a <=0.28 m drawer travel;
# skill_elapsed_time is bounded by the per-phase timeouts in the contract (~ <=45 s).
ERROR_CLIP = (0.0, 0.30)
TIME_CLIP = (0.0, 60.0)


@dataclass
class UtilityConfig:
    lambda_error: float = 1.0
    lambda_time: float = 0.02
    error_clip: tuple = ERROR_CLIP
    time_clip: tuple = TIME_CLIP
    selected_on: str = "validation"   # provenance: never "test"
    note: str = "weights frozen from validation before touching test"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_clip"] = list(self.error_clip)
        d["time_clip"] = list(self.time_clip)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "UtilityConfig":
        d = json.loads(Path(path).read_text())
        return cls(
            lambda_error=d["lambda_error"], lambda_time=d["lambda_time"],
            error_clip=tuple(d.get("error_clip", ERROR_CLIP)),
            time_clip=tuple(d.get("time_clip", TIME_CLIP)),
            selected_on=d.get("selected_on", "validation"),
            note=d.get("note", ""),
        )


def _clip(v, lo_hi):
    lo, hi = lo_hi
    return min(max(v, lo), hi)


def predicted_utility(p_success: float, pred_error: float, pred_time: float,
                      cfg: UtilityConfig) -> float:
    e = _clip(pred_error, cfg.error_clip)
    t = _clip(pred_time, cfg.time_clip)
    return p_success - cfg.lambda_error * e - cfg.lambda_time * t


def true_utility(episode: dict, cfg: UtilityConfig) -> float:
    """Utility from realized outcomes. True error/time are NOT clipped (they are physical)."""
    y = episode["y"]
    s = 1.0 if y["success"] else 0.0
    return s - cfg.lambda_error * float(y["task_outcome_error"]) - cfg.lambda_time * float(y["skill_elapsed_time"])


# Sensitivity variants (for the frozen sensitivity analysis; main metric uses full utility).
def utility_variants():
    return {
        "success_only": UtilityConfig(lambda_error=0.0, lambda_time=0.0),
        "success_error": UtilityConfig(lambda_error=1.0, lambda_time=0.0),
        "full": UtilityConfig(lambda_error=1.0, lambda_time=0.02),
    }
