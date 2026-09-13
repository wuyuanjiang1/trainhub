"""标签归一化与译名登记表测试（针对模型英文写法不稳定的问题）。"""

from __future__ import annotations

import json

from trainhub.core.prelabel_vlm import normalize_english_label
from trainhub.core.prelabel_vlm import parse_detections
from trainhub.core.project import Project


def test_normalize_english_label_merges_variants():
    assert normalize_english_label("Wooden Stick") == "wooden_stick"
    assert normalize_english_label("wooden stick") == "wooden_stick"
    assert normalize_english_label("wooden_stick") == "wooden_stick"
    assert normalize_english_label("  Stick  ") == "stick"
    assert normalize_english_label("red-cap/cylinder") == "red_cap_cylinder"
    assert normalize_english_label("") == "object"
    # 中文目标名保持原样（仅去空白）
    assert normalize_english_label("木棍") == "木棍"


def test_canonicalize_seeds_mapping_on_first_use():
    project = Project.create(tmp_root := __import__("tempfile").mkdtemp())
    try:
        first = project.canonicalize_label("木棍", "wooden stick")
        assert first == "wooden_stick"
        assert project.label_translations["木棍"] == "wooden_stick"
        # 之后模型换成完全不同的英文写法，也强制映射回登记的标准名
        assert project.canonicalize_label("木棍", "stick") == "wooden_stick"
        assert project.canonicalize_label("木棍", "WoodenStick") == "wooden_stick"
    finally:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def test_canonicalize_without_reference_only_normalizes():
    project = Project.create(tmp_root := __import__("tempfile").mkdtemp())
    try:
        assert project.canonicalize_label(None, "Wooden Stick") == "wooden_stick"
        assert project.label_translations == {}
    finally:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def test_label_translations_roundtrip():
    project = Project.create(tmp_root := __import__("tempfile").mkdtemp())
    try:
        project.canonicalize_label("木棍", "wooden stick")
        project.save()
        reloaded = Project.load(project.root)
        assert reloaded.label_translations == {"木棍": "wooden_stick"}
        assert reloaded.canonicalize_label("木棍", "stick") == "wooden_stick"
    finally:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def test_parse_translator_converges_variants():
    """同一目标的两种英文写法 + 同一中文对照，必须收敛成一个标签。"""
    text = json.dumps(
        [
            {"label": "wooden stick", "label_cn": "木棍", "bbox_2d": [0, 0, 300, 300]},
            {"label": "WoodenStick", "label_cn": "木棍", "bbox_2d": [400, 0, 700, 300]},
            {"label": "stick", "label_cn": "木棍", "bbox_2d": [0, 400, 300, 700]},
        ]
    )
    mapping: dict[str, str] = {}

    def translator(reference, label):
        from trainhub.core.prelabel_vlm import normalize_english_label

        normalized = normalize_english_label(label)
        if reference:
            return mapping.setdefault(reference, normalized)
        return normalized

    shapes, warnings = parse_detections(
        text, width=1000, height=1000, label_translator=translator
    )
    assert warnings == []
    assert {s["label"] for s in shapes} == {"wooden_stick"}
    assert all(s["other_data"]["label_cn"] == "木棍" for s in shapes)


def test_parse_translator_failure_falls_back_to_raw_label():
    text = json.dumps([{"label": "box", "bbox_2d": [0, 0, 200, 200]}])
    shapes, _ = parse_detections(
        text,
        width=1000,
        height=1000,
        label_translator=lambda ref, label: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert [s["label"] for s in shapes] == ["box"]


def test_whitelist_check_happens_before_translation():
    """白名单仍按原名称（label_cn）校验，翻译只影响入库的名字。"""
    text = json.dumps([{"label": "cat", "label_cn": "猫", "bbox_2d": [0, 0, 200, 200]}])
    shapes, warnings = parse_detections(
        text,
        width=1000,
        height=1000,
        allowed_labels=["木棍"],
        label_translator=lambda ref, label: "whatever",
    )
    assert shapes == []
    assert any("猫" in w for w in warnings)
