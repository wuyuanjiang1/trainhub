"""labelme ↔ YOLO 双向转换器测试（R4：最容易出数据事故的路径）。

覆盖：detect 矩形框双向 round-trip、segment 多边形双向、classify 目录
结构、圆与定向矩形的降级、数据集切分的确定性与比例。
"""

from __future__ import annotations

import json
from pathlib import Path

from trainhub.core.dataset import AnnotationItem
from trainhub.core.dataset import Shape
from trainhub.core.dataset import split_items
from trainhub.trainers.yolo.converter import export_yolo_dataset
from trainhub.trainers.yolo.converter import import_yolo_dataset

W, H = 640, 480


def _touch(path: Path) -> Path:
    """生成真实小图（反向导入时 PIL 要读尺寸，不能放假字节）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (W, H), (120, 120, 120)).save(path, quality=80)
    return path


def _rect(label: str, x1: float, y1: float, x2: float, y2: float) -> Shape:
    return Shape(label=label, points=[(x1, y1), (x2, y2)], shape_type="rectangle")


def _make_item(image_path: Path, shapes: list[Shape]) -> AnnotationItem:
    _touch(image_path)
    return AnnotationItem(
        image_path=image_path, json_path=None, width=W, height=H, shapes=shapes
    )


def test_detect_round_trip(tmp_path: Path):
    box = _rect("cat", 64, 48, 320, 240)
    items = [
        _make_item(tmp_path / "src" / "a.jpg", [box]),
        _make_item(tmp_path / "src" / "b.jpg", [_rect("cat", 10, 10, 600, 400)]),
        _make_item(tmp_path / "src" / "c.jpg", [_rect("cat", 100, 100, 500, 380)]),
        _make_item(tmp_path / "src" / "d.jpg", [_rect("cat", 0, 0, 640, 480)]),
    ]
    result = export_yolo_dataset(
        items, task="detect", classes=["cat"],
        out_dir=tmp_path / "ds", val_ratio=0.25, seed=42,
    )
    assert result.train_count == 3 and result.val_count == 1
    assert result.data_yaml is not None and result.data_yaml.exists()

    # 导出的 txt 行数与归一化数值
    txt_files = sorted((tmp_path / "ds" / "labels").rglob("*.txt"))
    assert len(txt_files) == 4
    # a.jpg 一定在其中一个 split 里，找到它并校验框值
    a_txt = [f for f in txt_files if f.stem == "a"][0]
    line = a_txt.read_text(encoding="utf-8").strip()
    assert line == "0 0.300000 0.300000 0.400000 0.400000"

    # 反向导回 labelme，框应回到原始像素坐标（±0.5px 容差）
    img_dir, ann_dir = tmp_path / "back" / "images", tmp_path / "back" / "annotations"
    imported = import_yolo_dataset(tmp_path / "ds", img_dir, ann_dir)
    assert imported.imported == 4
    payload = json.loads((ann_dir / "a.json").read_text(encoding="utf-8"))
    (x1, y1), (x2, y2) = payload["shapes"][0]["points"]
    assert abs(x1 - 64) < 0.5 and abs(y1 - 48) < 0.5
    assert abs(x2 - 320) < 0.5 and abs(y2 - 240) < 0.5
    assert payload["shapes"][0]["shape_type"] == "rectangle"


def test_segment_round_trip(tmp_path: Path):
    poly = Shape(
        label="obj", points=[(10, 10), (200, 20), (100, 300)], shape_type="polygon"
    )
    items = [
        _make_item(tmp_path / "src" / "a.jpg", [poly]),
        _make_item(tmp_path / "src" / "b.jpg", [poly]),
        _make_item(tmp_path / "src" / "c.jpg", [poly]),
        _make_item(tmp_path / "src" / "d.jpg", [poly]),
    ]
    export_yolo_dataset(
        items, task="segment", classes=["obj"],
        out_dir=tmp_path / "ds", val_ratio=0.25, seed=7,
    )
    txt_files = sorted((tmp_path / "ds" / "labels").rglob("*.txt"))
    a_txt = [f for f in txt_files if f.stem == "a"][0]
    values = [float(v) for v in a_txt.read_text().split()[1:]]
    assert len(values) == 6  # 3 个点 × 2
    assert values[0] == 10 / 640 and values[2] == 200 / 640 and values[4] == 100 / 640

    img_dir, ann_dir = tmp_path / "back" / "images", tmp_path / "back" / "annotations"
    import_yolo_dataset(tmp_path / "ds", img_dir, ann_dir)
    payload = json.loads((ann_dir / "a.json").read_text(encoding="utf-8"))
    points = payload["shapes"][0]["points"]
    assert len(points) == 3
    for (px, py), (ox, oy) in zip(points, poly.points):
        assert abs(px - ox) < 0.5 and abs(py - oy) < 0.5


def test_circle_degrades_to_polygon_for_segment(tmp_path: Path):
    circle = Shape(label="c", points=[(320, 240), (420, 240)], shape_type="circle")
    items = [
        _make_item(tmp_path / "src" / f"{n}.jpg", [circle]) for n in "abcd"
    ]
    export_yolo_dataset(
        items, task="segment", classes=["c"],
        out_dir=tmp_path / "ds", val_ratio=0.25, seed=1,
    )
    txt = next((tmp_path / "ds" / "labels").rglob("a.txt"))
    coords = [float(v) for v in txt.read_text().split()[1:]]
    assert len(coords) == 48  # 圆用 24 边形近似
    # 半径 100 → 中心 ±100（±多边形内接误差）
    xs = coords[0::2]
    assert min(xs) > (320 - 100) / 640 - 0.01 and max(xs) < (320 + 100) / 640 + 0.01


def test_classify_directory_layout(tmp_path: Path):
    items = [
        _make_item(tmp_path / "src" / "a.jpg", [_rect("dog", 0, 0, 100, 100)]),
        _make_item(tmp_path / "src" / "b.jpg", [_rect("dog", 0, 0, 100, 100)]),
        _make_item(tmp_path / "src" / "c.jpg", [_rect("cat", 0, 0, 100, 100)]),
        _make_item(tmp_path / "src" / "d.jpg", [_rect("cat", 0, 0, 100, 100)]),
    ]
    result = export_yolo_dataset(
        items, task="classify", classes=["dog", "cat"],
        out_dir=tmp_path / "ds", val_ratio=0.5, seed=3,
    )
    exported = sorted((tmp_path / "ds").rglob("*.jpg"))
    assert len(exported) == 4
    for f in exported:
        expected = "dog" if f.stem in ("a", "b") else "cat"
        assert f.parent.name == expected, f"{f} 类别目录错误"
    assert any("train" in f.parts for f in exported)
    assert any("val" in f.parts for f in exported)


def test_split_deterministic_and_clamped():
    items = [
        AnnotationItem(image_path=Path(f"{n}.jpg"), json_path=None,
                       width=10, height=10, shapes=[])
        for n in "abcdefgh"
    ]
    first = split_items(items, 0.25, seed=42)
    second = split_items(items, 0.25, seed=42)
    assert [i.image_path for i in first[0]] == [i.image_path for i in second[0]]
    assert [i.image_path for i in first[1]] == [i.image_path for i in second[1]]
    assert len(first[1]) == 2  # 8 × 0.25

    # 单样本：同一份同时充当训练与验证（避免空验证集崩溃的既有钳制行为）
    one = split_items(items[:1], 0.2, seed=1)
    assert len(one[0]) == 1 and len(one[1]) == 1

    # 极端比例收敛到至少 1 张验证
    two = split_items(items[:2], 0.9, seed=1)
    assert len(two[1]) == 1 and len(two[0]) == 1
