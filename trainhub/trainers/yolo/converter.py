"""labelme annotations -> Ultralytics YOLO dataset.

Layout produced (detect / segment)::

    <out>/
        images/train/*.png      labels/train/*.txt
        images/val/*.png        labels/val/*.txt
        data.yaml

For ``classify`` the Ultralytics directory format is used instead::

    <out>/train/<class_name>/*.png
    <out>/val/<class_name>/*.png
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import yaml

from ...core.dataset import AnnotationItem
from ...core.dataset import find_images
from ...core.dataset import split_items
from ...core.geometry import bbox_of
from ...core.geometry import polygon_of


@dataclass
class YoloExport:
    dataset_dir: Path
    classes: list[str]
    train_count: int
    val_count: int
    task: str
    data_yaml: Path | None = None
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _unique_stem(item: AnnotationItem, used: set[str]) -> str:
    stem = item.image_path.stem
    if stem not in used:
        used.add(stem)
        return stem
    index = 2
    while f"{stem}_{index}" in used:
        index += 1
    unique = f"{stem}_{index}"
    used.add(unique)
    return unique


def _normalize(value: float, size: int) -> float:
    if size <= 0:
        return 0.0
    return min(1.0, max(0.0, value / size))


def _collect_classes(items: list[AnnotationItem], classes: list[str]) -> list[str]:
    result = list(classes)
    for item in items:
        for label in item.labels:
            if label not in result:
                result.append(label)
    return result


def export_yolo_dataset(
    items: list[AnnotationItem],
    *,
    task: str,
    classes: list[str],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> YoloExport:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = _collect_classes(items, classes)
    class_index = {name: i for i, name in enumerate(classes)}

    train_items, val_items = split_items(items, val_ratio, seed=seed)
    warnings: list[str] = []
    skipped = 0

    if task == "classify":
        for split, split_items_ in (("train", train_items), ("val", val_items)):
            for item in split_items_:
                label = item.labels[0] if item.labels else None
                if label is None:
                    skipped += 1
                    continue
                if len(set(item.labels)) > 1:
                    warnings.append(
                        f"{item.image_path.name} 有多个标签，分类任务只取第一个 {label!r}"
                    )
                _link_or_copy(item.image_path, out_dir / split / label / item.image_path.name)
        return YoloExport(
            dataset_dir=out_dir,
            classes=classes,
            train_count=len(train_items),
            val_count=len(val_items),
            task=task,
            skipped=skipped,
            warnings=warnings,
        )

    if task not in ("detect", "segment"):
        raise ValueError(f"YOLO 不支持的任务类型: {task!r}")

    for split, split_items_ in (("train", train_items), ("val", val_items)):
        used: set[str] = set()
        for item in split_items_:
            stem = _unique_stem(item, used)
            _link_or_copy(
                item.image_path,
                out_dir / "images" / split / f"{stem}{item.image_path.suffix}",
            )
            lines: list[str] = []
            for shape in item.shapes:
                if shape.label not in class_index:
                    continue
                cls = class_index[shape.label]
                if task == "detect":
                    box = bbox_of(shape)
                    if box is None:
                        skipped += 1
                        continue
                    x1, y1, x2, y2 = box
                    cx = _normalize((x1 + x2) / 2, item.width)
                    cy = _normalize((y1 + y2) / 2, item.height)
                    w = _normalize(x2 - x1, item.width)
                    h = _normalize(y2 - y1, item.height)
                    if w <= 0 or h <= 0:
                        skipped += 1
                        continue
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                else:
                    polygon = polygon_of(shape)
                    if len(polygon) < 3:
                        skipped += 1
                        continue
                    coords = " ".join(
                        f"{_normalize(x, item.width):.6f} {_normalize(y, item.height):.6f}"
                        for x, y in polygon
                    )
                    lines.append(f"{cls} {coords}")
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines), encoding="utf-8")

    data_yaml = out_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {i: name for i, name in enumerate(classes)},
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    return YoloExport(
        dataset_dir=out_dir,
        classes=classes,
        train_count=len(train_items),
        val_count=len(val_items),
        task=task,
        data_yaml=data_yaml,
        skipped=skipped,
        warnings=warnings,
    )


# --------------------------------------------------------------------- import
@dataclass
class YoloImport:
    imported: int
    annotated: int
    labels: list[str]
    warnings: list[str] = field(default_factory=list)


def _read_yolo_names(src_dir: Path) -> list[str]:
    for filename in ("dataset.yaml", "data.yaml"):
        path = src_dir / filename
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        names = data.get("names")
        if isinstance(names, dict):
            names = [str(names[i]) for i in sorted(names)]
        if isinstance(names, list):
            return [str(n) for n in names]
    return []


def _pixel_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _yolo_txt_to_labelme_shapes(
    label_path: Path,
    names: list[str],
    width: int,
    height: int,
) -> list[dict]:
    shapes: list[dict] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return shapes
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        label = names[cls] if 0 <= cls < len(names) else str(cls)
        if len(values) == 4:  # detect: cx cy w h
            cx, cy, w, h = values
            x1 = (cx - w / 2.0) * width
            y1 = (cy - h / 2.0) * height
            x2 = (cx + w / 2.0) * width
            y2 = (cy + h / 2.0) * height
            shapes.append(
                {
                    "label": label,
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "flags": {},
                    "group_id": None,
                    "description": "",
                }
            )
        elif len(values) >= 6 and len(values) % 2 == 0:  # segment polygon
            points = [
                [values[i] * width, values[i + 1] * height]
                for i in range(0, len(values), 2)
            ]
            shapes.append(
                {
                    "label": label,
                    "points": points,
                    "shape_type": "polygon",
                    "flags": {},
                    "group_id": None,
                    "description": "",
                }
            )
    return shapes


def _unique_name(stem: str, used: set[str]) -> str:
    if stem not in used:
        used.add(stem)
        return stem
    index = 2
    while f"{stem}_{index}" in used:
        index += 1
    name = f"{stem}_{index}"
    used.add(name)
    return name


def import_yolo_dataset(
    src_dir: Path,
    images_dir: Path,
    annotations_dir: Path,
) -> YoloImport:
    """Convert a YOLO dataset (``images/`` + ``labels/`` + ``dataset.yaml``)
    into the app's flat ``images/`` + labelme JSON layout."""
    src_dir = Path(src_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    names = _read_yolo_names(src_dir)
    used: set[str] = set()
    seen_labels: list[str] = []
    imported = 0
    annotated = 0
    warnings: list[str] = []

    for image_path in sorted(find_images(src_dir / "images")):
        rel = image_path.relative_to(src_dir / "images")
        label_path = src_dir / "labels" / rel.with_suffix(".txt")
        try:
            width, height = _pixel_size(image_path)
        except OSError:
            warnings.append(f"无法读取图像尺寸，跳过: {image_path.name}")
            continue

        stem = _unique_name(image_path.stem, used)
        dst_image = images_dir / f"{stem}{image_path.suffix}"
        shutil.copy2(image_path, dst_image)
        imported += 1

        shapes = (
            _yolo_txt_to_labelme_shapes(label_path, names, width, height)
            if label_path.exists()
            else []
        )
        if not shapes:
            continue
        annotated += 1
        for shape in shapes:
            if shape["label"] not in seen_labels:
                seen_labels.append(shape["label"])

        payload = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.relpath(dst_image, annotations_dir),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        with open(annotations_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    labels = list(names) + [label for label in seen_labels if label not in names]
    return YoloImport(
        imported=imported,
        annotated=annotated,
        labels=labels,
        warnings=warnings,
    )
