"""Trainer plugin registry.

Trainers register themselves at import time::

    @register_trainer
    class YoloTrainer(BaseTrainer):
        key = "yolo"
        ...

:func:`load_builtin_trainers` imports every bundled trainer package so the GUI
can simply list what is available.
"""

from __future__ import annotations

import importlib
from typing import TypeVar

from .trainer import BaseTrainer
from .trainer import TaskSpec

_TRAINERS: dict[str, BaseTrainer] = {}

T = TypeVar("T", bound=type[BaseTrainer])


def register_trainer(cls: T) -> T:
    instance = cls()
    if not instance.key:
        raise ValueError(f"{cls.__name__} 必须定义非空的 key")
    if instance.key in _TRAINERS:
        raise ValueError(f"训练器 key 重复: {instance.key!r}")
    _TRAINERS[instance.key] = instance
    return cls


def get_trainer(key: str) -> BaseTrainer:
    if key not in _TRAINERS:
        raise KeyError(f"未注册的训练器: {key!r}（可用: {sorted(_TRAINERS)}）")
    return _TRAINERS[key]


def all_trainers() -> list[BaseTrainer]:
    return [_TRAINERS[k] for k in sorted(_TRAINERS)]


def trainers_for_task(task_key: str) -> list[BaseTrainer]:
    return [t for t in all_trainers() if any(s.key == task_key for s in t.tasks)]


def all_tasks() -> list[tuple[BaseTrainer, TaskSpec]]:
    return [(t, spec) for t in all_trainers() for spec in t.tasks]


_BUILTIN_MODULES = (
    "trainhub.trainers.yolo",
    "trainhub.trainers.nnunet",
)


def load_builtin_trainers() -> None:
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            # Optional trainer whose heavy dependency is missing: skip it rather
            # than taking the whole GUI down.
            continue
