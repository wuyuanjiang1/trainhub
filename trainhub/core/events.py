"""Transport-agnostic event types emitted by trainers.

Trainers never import Qt.  They push plain dataclasses into an
:class:`EventSink`; the GUI supplies an implementation that forwards them to Qt
signals (see :mod:`trainhub.ui.runner`).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable


@dataclass
class LogEvent:
    message: str
    level: str = "info"


@dataclass
class ProgressEvent:
    phase: str
    current: int = 0
    total: int = 0
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.total))


@dataclass
class MetricEvent:
    """One row of the live charts.

    ``metrics`` maps a series name such as ``"train/loss"`` to its value.  The
    chart widget discovers series dynamically, so trainers are free to report
    whatever they have (loss, mAP50, Dice, lr, ...).
    """

    epoch: int
    total_epochs: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    phase: str = "train"


@dataclass
class ArtifactEvent:
    """A file the trainer produced (checkpoint, plot, config, ...)."""

    path: str
    kind: str = "file"
    label: str = ""


class EventSink:
    """Default sink; override the hooks you care about.

    Subclasses/instances may pass plain callables instead, which keeps trainers
    testable without a GUI.
    """

    def __init__(
        self,
        on_log: Callable[[LogEvent], None] | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        on_metric: Callable[[MetricEvent], None] | None = None,
        on_artifact: Callable[[ArtifactEvent], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_metric = on_metric
        self._on_artifact = on_artifact
        self._is_cancelled = is_cancelled

    def log(self, message: str, level: str = "info") -> None:
        if self._on_log is not None:
            self._on_log(LogEvent(message=message, level=level))

    def progress(
        self,
        phase: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
    ) -> None:
        if self._on_progress is not None:
            self._on_progress(
                ProgressEvent(
                    phase=phase, current=current, total=total, message=message
                )
            )

    def metric(
        self,
        epoch: int,
        metrics: dict[str, Any],
        total_epochs: int = 0,
        phase: str = "train",
    ) -> None:
        clean = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        if clean and self._on_metric is not None:
            self._on_metric(
                MetricEvent(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    metrics=clean,
                    phase=phase,
                )
            )

    def artifact(self, path: str, kind: str = "file", label: str = "") -> None:
        if self._on_artifact is not None:
            self._on_artifact(ArtifactEvent(path=path, kind=kind, label=label))

    def cancelled(self) -> bool:
        return bool(self._is_cancelled and self._is_cancelled())


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return value == value  # drop NaN
    return False
