"""labelme annotations -> nnUNet raw dataset (2D, PNG).

nnUNet's ``NaturalImage2DIO`` reader is selected automatically for ``.png`` and
handles RGB (channels moved to front) or grayscale, so no NIfTI conversion is
needed::

    nnUNet_raw/Dataset001_MyProject/
        imagesTr/case_0000.png
        labelsTr/case.png          # uint8 label map, 0 = background
        dataset.json
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

from ...core.dataset import AnnotationItem
from ...core.dataset import Shape
from ...core.dataset import split_items
from ...core.geometry import area_of
from ...core.geometry import polygon_of

_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")
MAX_UINT8_LABEL = 255


def sanitize_dataset_name(name: str) -> str:
    cleaned = _NAME_RE.sub("_", name).strip("_")
    return cleaned or "Project"


def dataset_folder_name(dataset_id: int, name: str) -> str:
    """nnUNet requires exactly ``Dataset<3 digits>_<name>``."""
    return f"Dataset{int(dataset_id):03d}_{sanitize_dataset_name(name)}"


@dataclass
class NnunetExport:
    raw_dir: Path
    dataset_dir: Path
    dataset_id: int
    dataset_json: Path
    labels: list[str]
    channels: dict[str, str]
    train_count: int
    holdout_count: int
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def _detect_mode(path: Path) -> str:
    with Image.open(path) as image:
        return "RGB" if image.mode in ("RGB", "RGBA", "P", "CMYK") else "L"


def _unique_case_name(item: AnnotationItem, used: set[str]) -> str:
    name = _NAME_RE.sub("_", item.image_path.stem) or "case"
    if name not in used:
        used.add(name)
        return name
    index = 2
    while f"{name}_{index}" in used:
        index += 1
    unique = f"{name}_{index}"
    used.add(unique)
    return unique


def _rasterize(
    item: AnnotationItem,
    class_index: dict[str, int],
    dtype: type,
) -> tuple[np.ndarray, int]:
    """Paint every shape into one label map.

    Shapes are processed largest-first so that small objects drawn on top win
    the pixels they overlap, matching what the annotator showed the user.
    """
    width, height = item.width, item.height
    canvas = np.zeros((height, width), dtype=np.uint8)

    shapes: list[tuple[float, Shape]] = [
        (area_of(shape), shape) for shape in item.shapes if shape.label in class_index
    ]
    shapes.sort(key=lambda pair: pair[0], reverse=True)

    skipped = 0
    for _, shape in shapes:
        value = class_index[shape.label]

        brush_mask = shape.decode_mask()
        if brush_mask is not None:
            if brush_mask.shape != (height, width):
                resized = Image.fromarray(brush_mask.astype(np.uint8) * 255).resize(
                    (width, height), Image.NEAREST
                )
                brush_mask = np.asarray(resized) > 0
            canvas[brush_mask] = value
            continue

        polygon = polygon_of(shape)
        if len(polygon) < 3:
            skipped += 1
            continue
        layer = Image.new("L", (width, height), 0)
        ImageDraw.Draw(layer).polygon(
            [(float(x), float(y)) for x, y in polygon], fill=1
        )
        canvas[np.asarray(layer) > 0] = value

    return canvas.astype(dtype), skipped


def export_nnunet_dataset(
    items: list[AnnotationItem],
    *,
    dataset_id: int,
    dataset_name: str,
    classes: list[str],
    raw_dir: Path,
    holdout_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> NnunetExport:
    raw_dir = Path(raw_dir)
    dataset_dir = raw_dir / dataset_folder_name(dataset_id, dataset_name)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    for directory in (images_tr, labels_tr):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    usable = [item for item in items if item.shapes and item.width > 0]
    train_items, holdout_items = split_items(usable, val_ratio, seed=seed)

    # nnUNet requires one consistent channel layout across the whole dataset.
    mode = _detect_mode(train_items[0].image_path) if train_items else "RGB"
    channels = (
        {"0": "R", "1": "G", "2": "B"}
        if mode == "RGB"
        else {"0": "grayscale"}
    )

    labels = list(classes)
    for item in usable:
        for shape in item.shapes:
            if shape.label not in labels:
                labels.append(shape.label)

    if len(labels) + 1 > MAX_UINT8_LABEL:
        dtype: type = np.uint16
    else:
        dtype = np.uint8
    label_dict: dict[str, int] = {"background": 0}
    for index, label in enumerate(labels, start=1):
        label_dict[label] = index
    class_index = {label: label_dict[label] for label in labels}

    warnings: list[str] = []
    skipped = 0
    used: set[str] = set()

    for item in train_items:
        case = _unique_case_name(item, used)
        with Image.open(item.image_path) as image:
            image.convert(mode).save(images_tr / f"{case}_0000.png")
        mask, shape_skipped = _rasterize(item, class_index, dtype)
        skipped += shape_skipped
        if mask.max() == 0:
            warnings.append(f"{item.image_path.name} 没有可用标注，标签图为全背景")
        Image.fromarray(mask).save(labels_tr / f"{case}.png")

    # Held-out cases stay out of nnUNet's 5-fold CV so the user can evaluate on
    # data the model never saw.
    holdout_count = 0
    if holdout_items:
        target = Path(holdout_dir) / dataset_dir.name
        h_images = target / "images"
        h_labels = target / "labels"
        for directory in (h_images, h_labels):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        holdout_used: set[str] = set()
        for item in holdout_items:
            case = _unique_case_name(item, holdout_used)
            with Image.open(item.image_path) as image:
                image.convert(mode).save(h_images / f"{case}.png")
            mask, shape_skipped = _rasterize(item, class_index, dtype)
            skipped += shape_skipped
            Image.fromarray(mask).save(h_labels / f"{case}.png")
            holdout_count += 1

    dataset_json = dataset_dir / "dataset.json"
    with open(dataset_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channel_names": channels,
                "labels": label_dict,
                "numTraining": len(train_items),
                "file_ending": ".png",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return NnunetExport(
        raw_dir=raw_dir,
        dataset_dir=dataset_dir,
        dataset_id=int(dataset_id),
        dataset_json=dataset_json,
        labels=labels,
        channels=channels,
        train_count=len(train_items),
        holdout_count=holdout_count,
        skipped=skipped,
        warnings=warnings,
    )
