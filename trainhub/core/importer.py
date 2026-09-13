"""项目数据文件操作：导入（含命名冲突处理）、清空与删除的 .trash 归档。

从 UI 层抽出的纯文件操作，不依赖 Qt。设计原则：标注数据永不静默丢失——
被替换/删除的内容一律移入项目 ``.trash/<时间戳>/``，可随时手动找回。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from .dataset import IMAGE_SUFFIXES
from .project import Project


def new_trash_dir(project: Project) -> Path:
    """本项目本次操作的垃圾桶目录 ``.trash/<时间戳>/``。"""
    trash = project.root / ".trash" / datetime.now().strftime("%Y%m%d-%H%M%S")
    trash.mkdir(parents=True, exist_ok=True)
    return trash


def move_into(trash: Path, path: Path, bucket: str) -> None:
    """把文件或目录移入 trash/bucket/ 保留原名；跨设备时回退为复制后删除。"""
    target = trash / bucket / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.rename(target)
    except OSError:
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
            shutil.rmtree(path)
        else:
            shutil.copy2(path, target)
            path.unlink(missing_ok=True)


def clear_dataset_files(project: Project) -> None:
    """清空项目的 images/ 与 annotations/；旧数据整体移入 .trash。"""
    trash = new_trash_dir(project)
    for directory, bucket in (
        (project.images_dir, "images"),
        (project.annotations_dir, "annotations"),
    ):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            continue
        for child in sorted(directory.iterdir()):
            move_into(trash, child, bucket)


def labels_from_json(json_path: Path) -> set[str]:
    """从 labelme JSON 里收集标签名；解析失败返回空集。"""
    labels: set[str] = set()
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes") or []:
            label = shape.get("label")
            if label:
                labels.add(str(label))
    except (OSError, ValueError, TypeError):
        pass
    return labels


def import_image_file(
    project: Project, src: Path, used: dict[str, str], warnings: list[str]
) -> set[str]:
    """复制一张图像（连同同名 JSON）进项目，返回带进来的标签集。

    used 是项目内已占用的 stem→后缀 表，随导入更新。命名冲突策略：
    - 同名同后缀重复：改名 name_2、name_3…，JSON 一并改名（归属明确）；
    - 同名不同后缀（cat.png 与 cat.jpg）：改名导入但不携带标注——
      同一个 cat.json 无法判断属于哪张图，宁可丢标注也不标错图。
    """
    images_dir = project.images_dir
    stem, suffix = src.stem, src.suffix.lower()
    candidate = stem
    renamed = 0
    while candidate in used:
        renamed += 1
        candidate = f"{stem}_{renamed}"
    target = images_dir / f"{candidate}{src.suffix}"
    shutil.copy2(src, target)
    used[candidate] = suffix

    labels: set[str] = set()
    json_src = src.with_suffix(".json")
    ambiguous = renamed > 0 and used.get(stem) != suffix
    if json_src.exists() and ambiguous:
        warnings.append(
            f"{src.name} 与已有图像 {stem}{used.get(stem, '')} 同名（后缀不同），"
            f"已改名为 {target.name} 且未携带标注（{json_src.name} 归属不明确）"
        )
        return labels
    if json_src.exists():
        json_target = project.annotations_dir / f"{candidate}.json"
        json_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_src, json_target)
        labels = labels_from_json(json_target)
    if renamed:
        note = "标注一并改名" if json_src.exists() and not ambiguous else "无标注"
        warnings.append(f"{src.name} 与已有图像重名，已改名为 {target.name}（{note}）")
    return labels


def scan_image_files(directory: Path) -> list[Path]:
    """目录（含子目录）下的全部图像文件，按名称排序。"""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
