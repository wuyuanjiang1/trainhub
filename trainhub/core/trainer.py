"""Trainer plugin interface.

Adding support for a new model means writing one subclass of
:class:`BaseTrainer` and decorating it with :func:`trainhub.core.registry.register`.
The GUI discovers it automatically: the parameter form is generated from
:meth:`BaseTrainer.param_specs`, the task dropdown from :attr:`BaseTrainer.tasks`,
and live charts from whatever series the trainer reports via :class:`EventSink`.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from .events import EventSink
from .params import ParamSpec
from .project import Project


@dataclass(frozen=True)
class TaskSpec:
    key: str
    display_name: str
    annotation: str
    image: str = "2d"
    description: str = ""


@dataclass
class TrainJob:
    project: Project
    task_key: str
    params: dict[str, Any]
    run_dir: Path
    resume: bool = False
    device: str = "auto"


@dataclass
class PreparedData:
    dataset_dir: Path
    train_count: int = 0
    val_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    run_dir: Path
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


class BaseTrainer(ABC):
    key: str = ""
    display_name: str = ""
    description: str = ""
    tasks: tuple[TaskSpec, ...] = ()

    # ------------------------------------------------------------- metadata
    def is_available(self) -> bool:
        """Whether this trainer's third-party dependency is importable."""
        return True

    def task(self, task_key: str) -> TaskSpec:
        for spec in self.tasks:
            if spec.key == task_key:
                return spec
        raise KeyError(f"{self.key} 不支持任务类型 {task_key!r}")

    def all_params(self, task_key: str) -> list[ParamSpec]:
        """Dataset-split params are shared by every trainer."""
        return [*_COMMON_SPECS, *self.param_specs(task_key)]

    # ------------------------------------------------------------ overrides
    @abstractmethod
    def param_specs(self, task_key: str) -> list[ParamSpec]:
        """Trainer-specific tunables (epochs, lr, batch size, ...)."""

    @abstractmethod
    def prepare(self, job: TrainJob, sink: EventSink) -> PreparedData:
        """Convert annotations into the format this trainer consumes."""

    @abstractmethod
    def train(
        self,
        job: TrainJob,
        prepared: PreparedData,
        sink: EventSink,
    ) -> TrainResult:
        """Run training, reporting per-epoch metrics through ``sink``."""


_COMMON_SPECS: list[ParamSpec] = [
    ParamSpec(
        key="val_ratio",
        label="验证集比例",
        type="float",
        default=0.2,
        group="数据集划分",
        minimum=0.05,
        maximum=0.5,
        step=0.05,
        decimals=2,
        help="留作验证集的样本比例，其余用于训练",
    ),
    ParamSpec(
        key="split_seed",
        label="划分随机种子",
        type="int",
        default=42,
        group="数据集划分",
        minimum=0,
        maximum=10_000,
        help="固定种子可让多次训练使用完全相同的划分，便于对比",
    ),
]
