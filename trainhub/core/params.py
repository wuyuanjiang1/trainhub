"""Declarative parameter schema.

Trainers describe their tunable parameters with :class:`ParamSpec` and the GUI
builds the whole form from that description.  Adding a new trainer therefore
never requires touching GUI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

ParamType = Literal["int", "float", "bool", "str", "choice", "path"]


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    type: ParamType
    default: Any
    group: str = "基本"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 4
    choices: tuple[str, ...] = ()
    allow_custom: bool = False
    suffix: str = ""
    help: str = ""

    def coerce(self, value: Any) -> Any:
        if self.type == "int":
            return int(round(float(value)))
        if self.type == "float":
            return float(value)
        if self.type == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        return value

    def clamp(self, value: Any) -> Any:
        value = self.coerce(value)
        if self.type in ("int", "float"):
            if self.minimum is not None:
                value = max(value, self.minimum)
            if self.maximum is not None:
                value = min(value, self.maximum)
        # allow_custom means the dropdown is a convenience list, not a whitelist.
        if (
            self.type == "choice"
            and self.choices
            and not self.allow_custom
            and value not in self.choices
        ):
            raise ValueError(f"{self.label} 只能是 {self.choices} 之一，收到 {value!r}")
        return value


def default_params(specs: list[ParamSpec]) -> dict[str, Any]:
    return {spec.key: spec.default for spec in specs}


def normalize_params(specs: list[ParamSpec], values: dict[str, Any]) -> dict[str, Any]:
    """Coerce/clamp ``values`` against ``specs``; unknown keys are dropped."""
    out: dict[str, Any] = {}
    for spec in specs:
        raw = values.get(spec.key, spec.default)
        try:
            out[spec.key] = spec.clamp(raw)
        except (TypeError, ValueError):
            out[spec.key] = spec.default
    return out


def grouped(specs: list[ParamSpec]) -> dict[str, list[ParamSpec]]:
    """Preserve declaration order while grouping specs by their ``group``."""
    groups: dict[str, list[ParamSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.group, []).append(spec)
    return groups
