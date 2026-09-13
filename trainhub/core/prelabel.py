"""AI 预标注：用 Ultralytics YOLO 模型为图像生成候选标注。

只复用训练后端已有的 ultralytics，全部本地离线推理。返回的 shape 字典与
标注 JSON 同构（labelme 兼容），由界面层转成画布形状或直接写盘。
检测模型输出矩形框，分割模型把实例掩膜转成多边形；分类模型没有空间输出，
返回空列表。
"""

from __future__ import annotations

from pathlib import Path

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def find_project_weights(project_root: Path, limit: int = 5) -> list[Path]:
    """收集 runs/*/weights/*.pt，最近训练的排在前面。"""
    runs_dir = Path(project_root) / "runs"
    if not runs_dir.is_dir():
        return []
    weights = [
        p
        for p in sorted(
            runs_dir.glob("*/weights/*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if p.is_file()
    ]
    return weights[:limit]


def _load_model(weights: str, device: str):
    from ultralytics import YOLO

    key = (weights, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(weights)
        _MODEL_CACHE[key] = model
    return model


def _base_shape(label: str, points: list[list[float]], score: float) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": f"AI {score:.2f}",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
        "other_data": {},
    }


def predict_shapes(
    image_path: str,
    weights: str,
    conf: float = 0.25,
    device: str | None = None,
) -> list[dict]:
    """对单张图像推理，返回候选标注字典列表（检测=rectangle，分割=polygon）。

    ``device`` 为 None 时由 Ultralytics 自动选择（优先 CUDA）。
    """
    model = _load_model(weights, device or "auto")
    result = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
        **({"device": device} if device else {}),
    )[0]
    names = result.names or {}

    shapes: list[dict] = []
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)

    if masks is not None and boxes is not None and masks.xy is not None:
        for points, cls, score in zip(
            masks.xy,
            boxes.cls.tolist(),
            boxes.conf.tolist(),
        ):
            if points is None or len(points) < 3:
                continue
            shape = _base_shape(
                names.get(int(cls), str(int(cls))),
                [[float(x), float(y)] for x, y in points],
                float(score),
            )
            shape["shape_type"] = "polygon"
            shapes.append(shape)
        return shapes

    if boxes is not None:
        for box, cls, score in zip(
            boxes.xyxy.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            shapes.append(
                _base_shape(
                    names.get(int(cls), str(int(cls))),
                    [[x1, y1], [x2, y2]],
                    float(score),
                )
            )
    return shapes
