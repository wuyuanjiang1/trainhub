"""Live training curves (loss / metrics / learning rate).

Series names are discovered from whatever the trainer reports, so the same
widget plots YOLO's ``val/metrics/mAP50(B)`` and nnU-Net's ``val/mean_dice``
without any per-trainer code.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import matplotlib

matplotlib.use("QtAgg")

from cycler import cycler
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6 import QtWidgets

from ..core.events import MetricEvent
from .theme import ACCENT
from .theme import BG
from .theme import BORDER
from .theme import SURFACE
from .theme import TEXT
from .theme import TEXT_DIM

matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti TC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# Dark figure / axes so the chart matches the app; indigo leads the cycle.
matplotlib.rcParams["figure.facecolor"] = BG
matplotlib.rcParams["axes.facecolor"] = SURFACE
matplotlib.rcParams["axes.edgecolor"] = BORDER
matplotlib.rcParams["axes.labelcolor"] = TEXT_DIM
matplotlib.rcParams["text.color"] = TEXT_DIM
matplotlib.rcParams["xtick.color"] = TEXT_DIM
matplotlib.rcParams["ytick.color"] = TEXT_DIM
matplotlib.rcParams["grid.color"] = BORDER
matplotlib.rcParams["legend.facecolor"] = SURFACE
matplotlib.rcParams["legend.edgecolor"] = BORDER
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=[
        ACCENT,
        "#6A9BCC",
        "#788C5D",
        "#D98E2B",
        "#8A6F4D",
        "#7C6BC4",
        "#C0392B",
        "#4A8FB5",
        "#B45309",
        "#5E7D8C",
    ]
)

_AXIS_LOSS = 0
_AXIS_METRIC = 1
_AXIS_LR = 2
_AXIS_TITLES = ("损失 Loss", "指标 Metrics", "学习率 LR")

_PREFIX_LABELS = (
    ("train/", "训练 "),
    ("val/", "验证 "),
    ("metrics/", ""),
)


def _axis_for(key: str) -> int:
    lowered = key.lower()
    if "loss" in lowered:
        return _AXIS_LOSS
    if lowered == "lr" or "learning rate" in lowered or lowered.startswith("lr/"):
        return _AXIS_LR
    return _AXIS_METRIC


def _pretty(key: str) -> str:
    for prefix, replacement in _PREFIX_LABELS:
        if key.startswith(prefix):
            return replacement + key[len(prefix) :]
    return key


class MetricChart(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: dict[str, dict[int, float]] = {}

        self._figure = Figure(figsize=(6, 7), constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = [
            self._figure.add_subplot(3, 1, index + 1) for index in range(3)
        ]
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self.reset()

    # ------------------------------------------------------------------ data
    def reset(self) -> None:
        self._series.clear()
        self._redraw()

    def add_point(self, epoch: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._series.setdefault(key, {})[int(epoch)] = float(value)
        self._redraw()

    def load_events(self, events: list[MetricEvent]) -> None:
        self._series.clear()
        for event in events:
            for key, value in event.metrics.items():
                self._series.setdefault(key, {})[int(event.epoch)] = float(value)
        self._redraw()

    @property
    def series_names(self) -> list[str]:
        return sorted(self._series)

    # ---------------------------------------------------------------- render
    def _redraw(self) -> None:
        buckets: list[list[str]] = [[], [], []]
        for key in self._series:
            buckets[_axis_for(key)].append(key)

        for index, axis in enumerate(self._axes):
            axis.clear()
            keys = sorted(buckets[index])
            if not keys:
                axis.set_visible(False)
                continue
            axis.set_visible(True)
            for key in keys:
                points = sorted(self._series[key].items())
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                axis.plot(xs, ys, linewidth=1.6, label=_pretty(key))
            axis.set_title(_AXIS_TITLES[index], fontsize=10, color=TEXT)
            axis.grid(True, alpha=0.4)
            legend = axis.legend(fontsize=8, loc="best")
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(TEXT_DIM)
            axis.tick_params(labelsize=8, colors=TEXT_DIM)

        self._axes[-1].set_xlabel("Epoch", fontsize=9)
        self._canvas.draw_idle()
