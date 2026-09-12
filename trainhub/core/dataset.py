"""Scanning and splitting of labelme-format annotations.

Converters for YOLO / nnUNet both start from the same list of
:class:`AnnotationItem`, so dataset splitting and label statistics stay in one
place and behave identically for every trainer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class Shape:
    label: str
    points: list[tuple[float, float]]
    shape_type: str
    flags: dict = field(default_factory=dict)
    group_id: int | None = None
    mask_b64: str | None = None

    @property
    def is_bbox(self) -> bool:
        return self.shape_type == "rectangle" and len(self.points) >= 2

    def decode_mask(self) -> "np.ndarray | None":
        """Decode the brush mask (base64 PNG) that labelme stores inline."""
        if not self.mask_b64:
            return None
        import base64
        import io

        import numpy as np
        from PIL import Image

        raw = base64.b64decode(self.mask_b64)
        with Image.open(io.BytesIO(raw)) as mask:
            return np.asarray(mask.convert("L")) > 0


@dataclass
class AnnotationItem:
    image_path: Path
    json_path: Path | None
    width: int
    height: int
    shapes: list[Shape]

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.shapes]


def find_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    files = [
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _parse_shapes(raw_shapes: list[dict]) -> list[Shape]:
    shapes: list[Shape] = []
    for raw in raw_shapes:
        label = raw.get("label")
        points = raw.get("points") or []
        if not label or not points:
            continue
        raw_mask = raw.get("mask")
        shapes.append(
            Shape(
                label=str(label),
                points=[(float(x), float(y)) for x, y in points],
                shape_type=str(raw.get("shape_type", "polygon")),
                flags=dict(raw.get("flags") or {}),
                group_id=raw.get("group_id"),
                mask_b64=raw_mask if isinstance(raw_mask, str) else None,
            )
        )
    return shapes


def load_annotation(
    image_path: Path,
    annotation_dir: Path | None = None,
) -> AnnotationItem:
    """Load one image plus its labelme JSON (if it has been annotated)."""
    json_path = (
        (annotation_dir / f"{image_path.stem}.json")
        if annotation_dir is not None
        else image_path.with_suffix(".json")
    )
    shapes: list[Shape] = []
    width, height = 0, 0
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            width = int(data.get("imageWidth") or 0)
            height = int(data.get("imageHeight") or 0)
            shapes = _parse_shapes(data.get("shapes") or [])
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            shapes = []
    if width <= 0 or height <= 0:
        try:
            width, height = _image_size(image_path)
        except OSError:
            width, height = 0, 0
    return AnnotationItem(
        image_path=image_path,
        json_path=json_path if json_path.exists() else None,
        width=width,
        height=height,
        shapes=shapes,
    )


def scan_dataset(
    images_dir: Path,
    annotations_dir: Path | None = None,
    *,
    require_annotation: bool = False,
) -> list[AnnotationItem]:
    items = [
        load_annotation(path, annotations_dir) for path in find_images(images_dir)
    ]
    if require_annotation:
        items = [item for item in items if item.json_path is not None]
    return items


def label_histogram(items: list[AnnotationItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for label in item.labels:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def split_items(
    items: list[AnnotationItem],
    val_ratio: float,
    *,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[list[AnnotationItem], list[AnnotationItem]]:
    """Split into (train, val).

    ``val_ratio`` is clamped so that a non-empty dataset always leaves at least
    one sample on each side, which avoids the classic "empty val set" crash.
    """
    ordered = list(items)
    if shuffle:
        random.Random(seed).shuffle(ordered)

    total = len(ordered)
    if total == 0:
        return [], []
    if total == 1:
        return ordered, ordered

    n_val = int(round(total * max(0.0, min(1.0, val_ratio))))
    n_val = max(1, min(total - 1, n_val))
    return ordered[n_val:], ordered[:n_val]
