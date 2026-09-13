"""Project layout and persistence.

A *project* is just a directory with a ``trainhub.yaml`` inside it.  Layout::

    <root>/
        trainhub.yaml          project metadata (labels, chosen trainer/task, params)
        images/                原始图像
        annotations/           labelme 格式的 JSON
        datasets/              转换后可直接喂给训练器的数据集
        runs/                  每次训练的产物（权重、日志、图表）

Everything is derived from ``root``, so moving a project directory keeps it
valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "trainhub.yaml"


@dataclass
class Project:
    root: Path
    name: str = "未命名项目"
    trainer_key: str = "yolo"
    task_key: str = "detect"
    labels: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    # VLM 预标注的译名登记表：类别原名称（如中文/模型自由命名）-> 项目内
    # 唯一的英文标准名。模型对同一目标的英文写法不稳定（wooden stick /
    # wooden_stick / stick），靠这张表把变体收敛到一个标签上。
    label_translations: dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- paths
    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: str | Path, name: str | None = None) -> Project:
        root = Path(root).expanduser().resolve()
        project = cls(root=root, name=name or root.name)
        for directory in (
            project.images_dir,
            project.annotations_dir,
            project.datasets_dir,
            project.runs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        project.save()
        return project

    @classmethod
    def load(cls, path: str | Path) -> Project:
        path = Path(path).expanduser().resolve()
        config_path = path / CONFIG_NAME if path.is_dir() else path
        if not config_path.exists():
            raise FileNotFoundError(f"找不到项目配置文件: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        root = config_path.parent
        return cls(
            root=root,
            name=data.get("name", root.name),
            trainer_key=data.get("trainer", "yolo"),
            task_key=data.get("task", "detect"),
            labels=list(data.get("labels") or []),
            params=dict(data.get("params") or {}),
            label_translations=dict(data.get("label_translations") or {}),
        )

    @classmethod
    def open_or_create(cls, root: str | Path) -> Project:
        root = Path(root).expanduser().resolve()
        if (root / CONFIG_NAME).exists():
            return cls.load(root)
        return cls.create(root)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "trainer": self.trainer_key,
            "task": self.task_key,
            "labels": list(self.labels),
            "params": self.params,
            "label_translations": self.label_translations,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    # ---------------------------------------------------------------- labels
    def add_labels(self, labels: list[str]) -> bool:
        """Merge ``labels`` into the project label set; returns True if changed."""
        changed = False
        for label in labels:
            label = label.strip()
            if label and label not in self.labels:
                self.labels.append(label)
                changed = True
        return changed

    def remove_label(self, label: str) -> bool:
        """Remove ``label`` from the project label set; returns True if removed."""
        if label in self.labels:
            self.labels.remove(label)
            return True
        return False

    def label_index(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def canonicalize_label(self, reference: str | None, label: str) -> str:
        """把模型输出的英文标注词收敛成项目内唯一的标准名。

        有类别原名称（reference，如 label_cn）时，优先复用登记表里既有的
        英文译名；首次遇到则把归一化结果登记进去，之后所有变体都映射回
        同一个标签。无对照时只做归一化。
        """
        from .vlm import normalize_english_label

        normalized = normalize_english_label(label)
        if not reference or reference == normalized:
            return normalized
        known = self.label_translations.get(reference)
        if known:
            return known
        self.label_translations[reference] = normalized
        return normalized
