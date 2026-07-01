"""Unified experiment-parameter parsing + provenance tracking for stage-1 skills.

Priority (task spec section 3):

    request.parameters  >  skill Config default  >  legacy code constant

Every resolved parameter yields a :class:`ParamTrace` recording the requested value, the default,
the effective value actually used, the allowed band, whether it was clamped, its source, and the
code location where it is applied. The full set of traces is what the episode logger serialises as
``effective_parameters`` (and lets us prove a parameter really entered the execution path).

Illegal values are NOT silently ignored:

    * wrong type                 -> raise InvalidExperimentParameter (episode rejected)
    * NaN / Inf                  -> raise InvalidExperimentParameter
    * beyond the hard safety band-> raise InvalidExperimentParameter
    * slightly outside [min,max] -> clamp to the band, trace.clamped = True (episode kept)
    * not provided               -> use default, trace.source = "default"

The pilot runner catches :class:`InvalidExperimentParameter`, marks the episode
``failure_reason = "INVALID_EXPERIMENT_PARAMETER"`` and does not run the skill.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


class InvalidExperimentParameter(ValueError):
    """Raised for a request parameter that must reject the whole episode."""

    code = "INVALID_EXPERIMENT_PARAMETER"

    def __init__(self, name: str, message: str, requested: Any = None):
        self.param_name = name
        self.requested_value = requested
        self.detail = message
        super().__init__(f"{self.code}[{name}]: {message}")


@dataclass
class ParamTrace:
    name: str
    requested_value: Any
    default_value: Any
    effective_value: Any
    minimum: Any = None
    maximum: Any = None
    clamped: bool = False
    rejected: bool = False
    source: str = "default"  # "request" | "default"
    applied_at: str = ""
    dtype: str = "float"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_number(v: Any) -> bool:
    # bool is a subclass of int -- treat it as NOT a number for float/int params.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class ParamResolver:
    """Stateful resolver: pull typed params from a request dict and accumulate traces.

    Usage inside a skill::

        r = ParamResolver(request.parameters)
        step, _ = r.float("descend_max_position_step", cfg.descend_position_step,
                          minimum=0.001, maximum=0.05, hard_maximum=0.2,
                          applied_at="place_skill.DESCEND_TO_RELEASE")
        ...
        effective = r.effective()      # {name: value}
        traces = r.trace_dicts()       # list[dict] for the episode log
    """

    def __init__(self, params: dict[str, Any] | None):
        self.params: dict[str, Any] = dict(params or {})
        self.traces: list[ParamTrace] = []
        self._effective: dict[str, Any] = {}

    # -- public typed getters ------------------------------------------------
    def float(
        self,
        name: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        hard_minimum: float | None = None,
        hard_maximum: float | None = None,
        applied_at: str = "",
    ) -> tuple[float, ParamTrace]:
        if not self._present(name):
            return self._store(ParamTrace(name, None, float(default), float(default),
                                          minimum, maximum, False, False, "default", applied_at, "float"))
        raw = self.params[name]
        if not _is_number(raw):
            self._reject(name, raw, default, minimum, maximum, applied_at, "float",
                         f"expected a number, got {type(raw).__name__}")
        val = float(raw)
        if not math.isfinite(val):
            self._reject(name, raw, default, minimum, maximum, applied_at, "float", "value is NaN/Inf")
        self._check_hard(name, raw, val, hard_minimum, hard_maximum, minimum, maximum, applied_at, "float")
        val, clamped = _clamp(val, minimum, maximum)
        return self._store(ParamTrace(name, float(raw), float(default), val,
                                      minimum, maximum, clamped, False, "request", applied_at, "float"))

    def int(
        self,
        name: str,
        default: int,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        hard_minimum: int | None = None,
        hard_maximum: int | None = None,
        applied_at: str = "",
    ) -> tuple[int, ParamTrace]:
        if not self._present(name):
            return self._store(ParamTrace(name, None, int(default), int(default),
                                          minimum, maximum, False, False, "default", applied_at, "int"))
        raw = self.params[name]
        if not _is_number(raw):
            self._reject(name, raw, default, minimum, maximum, applied_at, "int",
                         f"expected an integer, got {type(raw).__name__}")
        fval = float(raw)
        if not math.isfinite(fval):
            self._reject(name, raw, default, minimum, maximum, applied_at, "int", "value is NaN/Inf")
        if abs(fval - round(fval)) > 1e-9:
            self._reject(name, raw, default, minimum, maximum, applied_at, "int", "value is not integral")
        ival = int(round(fval))
        self._check_hard(name, raw, ival, hard_minimum, hard_maximum, minimum, maximum, applied_at, "int")
        cval, clamped = _clamp(ival, minimum, maximum)
        ival = int(cval)
        return self._store(ParamTrace(name, raw, int(default), ival,
                                      minimum, maximum, clamped, False, "request", applied_at, "int"))

    def bool(self, name: str, default: bool, *, applied_at: str = "") -> tuple[bool, ParamTrace]:
        if not self._present(name):
            return self._store(ParamTrace(name, None, bool(default), bool(default),
                                          None, None, False, False, "default", applied_at, "bool"))
        raw = self.params[name]
        if isinstance(raw, bool):
            val = raw
        elif _is_number(raw) and float(raw) in (0.0, 1.0):
            val = bool(raw)
        elif isinstance(raw, str) and raw.strip().lower() in ("true", "false", "0", "1"):
            val = raw.strip().lower() in ("true", "1")
        else:
            self._reject(name, raw, default, None, None, applied_at, "bool",
                         f"expected a boolean, got {raw!r}")
        return self._store(ParamTrace(name, raw, bool(default), val,
                                      None, None, False, False, "request", applied_at, "bool"))

    def vec3(
        self,
        name: str,
        default: Sequence[float],
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        hard_minimum: float | None = None,
        hard_maximum: float | None = None,
        applied_at: str = "",
    ) -> tuple[tuple[float, float, float], ParamTrace]:
        """Per-component float vector of length 3. Band/hard limits applied component-wise."""
        dflt = tuple(float(v) for v in default)
        if len(dflt) != 3:
            raise ValueError(f"vec3 default for '{name}' must have 3 entries")
        if not self._present(name):
            return self._store(ParamTrace(name, None, list(dflt), list(dflt),
                                          minimum, maximum, False, False, "default", applied_at, "vec3"))
        raw = self.params[name]
        if not isinstance(raw, (list, tuple)) or len(raw) != 3 or not all(_is_number(v) for v in raw):
            self._reject(name, raw, list(dflt), minimum, maximum, applied_at, "vec3",
                         "expected a list of exactly 3 numbers")
        vals = [float(v) for v in raw]
        if not all(math.isfinite(v) for v in vals):
            self._reject(name, raw, list(dflt), minimum, maximum, applied_at, "vec3", "contains NaN/Inf")
        for v in vals:
            self._check_hard(name, raw, v, hard_minimum, hard_maximum, minimum, maximum, applied_at, "vec3")
        clamped = False
        out = []
        for v in vals:
            cv, c = _clamp(v, minimum, maximum)
            out.append(cv)
            clamped = clamped or c
        return self._store(ParamTrace(name, list(vals), list(dflt), list(out),
                                      minimum, maximum, clamped, False, "request", applied_at, "vec3"))

    # -- outputs -------------------------------------------------------------
    def effective(self) -> dict[str, Any]:
        return dict(self._effective)

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.traces]

    def requested_subset(self) -> dict[str, Any]:
        """Only the params the resolver actually consumed AND that came from the request."""
        return {t.name: t.requested_value for t in self.traces if t.source == "request"}

    def unknown_params(self) -> list[str]:
        """Request keys never consumed by a typed getter -- flagged so silent typos surface."""
        consumed = {t.name for t in self.traces}
        return [k for k in self.params if k not in consumed]

    # -- internals -----------------------------------------------------------
    def _present(self, name: str) -> bool:
        return name in self.params and self.params[name] is not None

    def _store(self, trace: ParamTrace) -> tuple[Any, ParamTrace]:
        self.traces.append(trace)
        self._effective[trace.name] = trace.effective_value
        return trace.effective_value, trace

    def _reject(self, name, raw, default, minimum, maximum, applied_at, dtype, message):
        self.traces.append(ParamTrace(name, raw, default, None, minimum, maximum,
                                      False, True, "request", applied_at, dtype))
        raise InvalidExperimentParameter(name, message, requested=raw)

    def _check_hard(self, name, raw, val, hard_min, hard_max, minimum, maximum, applied_at, dtype):
        if hard_min is not None and val < hard_min:
            self._reject(name, raw, None, minimum, maximum, applied_at, dtype,
                         f"value {val} below hard safety limit {hard_min}")
        if hard_max is not None and val > hard_max:
            self._reject(name, raw, None, minimum, maximum, applied_at, dtype,
                         f"value {val} above hard safety limit {hard_max}")


def _clamp(val, minimum, maximum):
    clamped = False
    if minimum is not None and val < minimum:
        val = minimum
        clamped = True
    if maximum is not None and val > maximum:
        val = maximum
        clamped = True
    return val, clamped


# -- stateless module-level helpers (thin wrappers, same semantics) ----------
def get_float_parameter(params, name, default, **kw):
    return ParamResolver(params).float(name, default, **kw)


def get_int_parameter(params, name, default, **kw):
    return ParamResolver(params).int(name, default, **kw)


def get_bool_parameter(params, name, default, **kw):
    return ParamResolver(params).bool(name, default, **kw)


def get_vec3_parameter(params, name, default, **kw):
    return ParamResolver(params).vec3(name, default, **kw)
