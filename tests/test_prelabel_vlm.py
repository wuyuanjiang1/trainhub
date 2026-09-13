"""prelabel_vlm 的解析与坐标转换测试（不联网）。"""

from __future__ import annotations

import json

import pytest

from trainhub.core.prelabel_vlm import VLMError
from trainhub.core.prelabel_vlm import build_prompt
from trainhub.core.prelabel_vlm import parse_detections
from trainhub.core.prelabel_vlm import provider_from_key
from trainhub.core.prelabel_vlm import VLM_PROVIDERS
from trainhub.core.prelabel_vlm import _extract_json_array


def _labels(result):
    return [shape["label"] for shape in result]


def _boxes(result):
    return [shape["points"] for shape in result]


def test_parse_clean_json_array():
    payload = json.dumps(
        [
            {"label": "cat", "bbox_2d": [100, 200, 500, 600], "confidence": 0.9},
            {"label": "dog", "bbox_2d": [600, 100, 900, 400]},
        ]
    )
    shapes, warnings = parse_detections(payload, width=1000, height=800)
    assert warnings == []
    assert _labels(shapes) == ["cat", "dog"]
    # 0-1000 归一化 -> 像素
    assert _boxes(shapes)[0] == [[100.0, 160.0], [500.0, 480.0]]
    assert shapes[0]["description"] == "AI 0.90"
    assert shapes[1]["description"] == "AI"
    assert shapes[0]["shape_type"] == "rectangle"


def test_parse_with_fences_and_prose():
    text = (
        "好的，检测结果如下：\n```json\n"
        '[{"label": "car", "bbox_2d": [0, 0, 1000, 1000], "confidence": 0.5}]\n'
        "```\n以上供参考。"
    )
    shapes, warnings = parse_detections(text, width=640, height=480)
    assert warnings == []
    assert _boxes(shapes) == [[[0.0, 0.0], [640.0, 480.0]]]


def test_parse_wrapped_object():
    text = json.dumps(
        {"detections": [{"label": "a", "bbox": [10, 10, 20, 20]}]}
    )
    shapes, _ = parse_detections(text, width=1000, height=1000)
    assert _labels(shapes) == ["a"]


def test_parse_two_point_bbox():
    text = json.dumps(
        [{"label": "x", "bbox_2d": [[100, 100], [300, 500]]}]
    )
    shapes, _ = parse_detections(text, width=500, height=1000)
    assert _boxes(shapes) == [[[50.0, 100.0], [150.0, 500.0]]]


def test_parse_qwen_ref_box_tags():
    text = '<ref>person</ref><box>(100,200),(300,400)</box> <ref>car</ref><box>(0,0),(500,500)</box>'
    shapes, warnings = parse_detections(text, width=1000, height=1000)
    assert warnings == []
    assert _labels(shapes) == ["person", "car"]
    assert _boxes(shapes)[0] == [[100.0, 200.0], [300.0, 400.0]]


def test_parse_think_blocks_ignored():
    text = (
        "<think>让我看看这张图……</think>"
        '[{"label": "a", "bbox_2d": [0, 0, 500, 500]}]'
    )
    shapes, _ = parse_detections(text, width=100, height=100)
    assert _labels(shapes) == ["a"]


def test_pixel_coords_detected_and_reinterpreted():
    # 模型没守归一化约定，返回了 1920x1080 图内的像素坐标（数值超出 0-1000）
    text = json.dumps(
        [{"label": "a", "bbox_2d": [10, 20, 1800, 1000]}]
    )
    shapes, warnings = parse_detections(text, width=1920, height=1080)
    assert _boxes(shapes) == [[[10.0, 20.0], [1800.0, 1000.0]]]
    assert any("像素坐标" in w for w in warnings)


def test_out_of_range_coords_dropped():
    text = json.dumps([{"label": "a", "bbox_2d": [5000, 0, 9000, 900]}])
    shapes, warnings = parse_detections(text, width=100, height=100)
    assert shapes == []
    assert warnings


def test_reversed_corners_and_clamping():
    text = json.dumps(
        [{"label": "a", "bbox_2d": [900, 990, 100, 1050]}]  # 反角 + 轻微出界
    )
    shapes, _ = parse_detections(text, width=1000, height=1000)
    assert _boxes(shapes) == [[[100.0, 990.0], [900.0, 1000.0]]]


def test_unknown_labels_dropped_when_allowed_given():
    text = json.dumps(
        [
            {"label": "cat", "bbox_2d": [0, 0, 100, 100]},
            {"label": "dog", "bbox_2d": [200, 0, 300, 100]},
        ]
    )
    shapes, warnings = parse_detections(
        text, width=500, height=500, allowed_labels=["cat"]
    )
    assert _labels(shapes) == ["cat"]
    assert any("dog" in w for w in warnings)


# --------------------------- 英文标注词 + label_cn 中文对照 ---------------------------
def test_english_label_with_cn_reference_kept():
    text = json.dumps(
        [{"label": "cat", "label_cn": "猫", "bbox_2d": [0, 0, 200, 200], "confidence": 0.8}]
    )
    shapes, warnings = parse_detections(text, width=1000, height=1000, allowed_labels=["猫"])
    assert warnings == []
    assert _labels(shapes) == ["cat"]
    assert shapes[0]["other_data"] == {"label_cn": "猫"}
    assert shapes[0]["description"] == "AI 0.80"


def test_reference_equal_label_no_other_data():
    text = json.dumps([{"label": "cat", "label_cn": "cat", "bbox_2d": [0, 0, 200, 200]}])
    shapes, _ = parse_detections(text, width=1000, height=1000, allowed_labels=["cat"])
    assert shapes[0]["other_data"] == {}


def test_reference_mismatch_dropped():
    text = json.dumps(
        [
            {"label": "cat", "label_cn": "猫", "bbox_2d": [0, 0, 200, 200]},
            {"label": "bird", "label_cn": "狗", "bbox_2d": [400, 0, 600, 200]},
        ]
    )
    shapes, warnings = parse_detections(text, width=1000, height=1000, allowed_labels=["猫"])
    assert _labels(shapes) == ["cat"]
    assert any("狗" in w for w in warnings)


def test_missing_reference_dropped_for_chinese_project_labels():
    # 项目类别是中文，模型没回传 label_cn，英文 label 无法对应 -> 丢弃
    text = json.dumps([{"label": "cat", "bbox_2d": [0, 0, 200, 200]}])
    shapes, warnings = parse_detections(text, width=1000, height=1000, allowed_labels=["猫"])
    assert shapes == []
    assert any("cat" in w for w in warnings)


def test_swapped_cn_en_fields_fixed():
    # 模型把中英填反：label 中文、label_cn 英文
    text = json.dumps([{"label": "猫", "label_cn": "cat", "bbox_2d": [0, 0, 200, 200]}])
    shapes, warnings = parse_detections(text, width=1000, height=1000, allowed_labels=["猫"])
    assert warnings == []
    assert _labels(shapes) == ["cat"]
    assert shapes[0]["other_data"] == {"label_cn": "猫"}


def test_chinese_label_without_reference_warns():
    text = json.dumps([{"label": "猫", "bbox_2d": [0, 0, 200, 200]}])
    shapes, warnings = parse_detections(text, width=1000, height=1000)
    assert _labels(shapes) == ["猫"]  # 不丢框，留给人工处理
    assert any("中文" in w for w in warnings)


def test_free_labeling_when_no_allowed_labels():
    text = json.dumps([{"label": "bolt", "bbox_2d": [0, 0, 100, 100]}])
    shapes, warnings = parse_detections(text, width=100, height=100)
    assert warnings == []
    assert _labels(shapes) == ["bolt"]


def test_tiny_boxes_dropped():
    text = json.dumps([{"label": "a", "bbox_2d": [500, 500, 500.5, 500.5]}])
    shapes, warnings = parse_detections(text, width=2000, height=2000)
    assert shapes == []
    assert any("过小" in w for w in warnings)


def test_unparseable_reply_reported():
    shapes, warnings = parse_detections("我找不到任何目标。", width=100, height=100)
    assert shapes == []
    assert any("没有找到可解析" in w for w in warnings)


def test_empty_array_is_clean_empty_result():
    shapes, warnings = parse_detections("[]", width=100, height=100)
    assert shapes == []
    assert warnings == []


def test_build_prompt_teaches_empty_array():
    prompt = build_prompt(["box"])
    assert "空数组" in prompt


def test_extract_json_array_skips_brackets_in_strings():
    text = '前文 [备注] 里的干扰 {"a": "]"}，结果: [{"label":"a","bbox_2d":[1,2,3,4]}] 尾巴'
    parsed = _extract_json_array(text)
    assert parsed == [{"label": "a", "bbox_2d": [1, 2, 3, 4]}]


def test_pixel_coord_system_uses_scale():
    # 发送前图被缩到一半（scale=0.5），模型返回的是缩放图的像素坐标
    shapes, _ = parse_detections(
        json.dumps([{"label": "a", "bbox_2d": [100, 100, 200, 200]}]),
        width=800,
        height=600,
        coord_system="pixel",
        scale=0.5,
    )
    assert _boxes(shapes) == [[[200.0, 200.0], [400.0, 400.0]]]


def test_build_prompt_injects_labels_and_hint():
    prompt = build_prompt(["猫", "狗"], hint="夜间图像")
    assert "猫" in prompt and "狗" in prompt
    assert "0-1000" in prompt
    assert "补充要求：夜间图像" in prompt
    assert "JSON" in prompt


def test_build_prompt_custom_override_keeps_format_block():
    prompt = build_prompt(
        ["猫"], prompt="只标注画面中央的猫，忽略边缘的。", hint="配合夜间图像"
    )
    # 自定义任务描述替换默认那句
    assert prompt.startswith("只标注画面中央的猫，忽略边缘的。")
    assert "默认" not in prompt.split("\n")[0]
    # 标签约束与格式约定仍然自动附加
    assert "猫" in prompt
    assert "0-1000" in prompt
    assert "JSON" in prompt
    assert "补充要求：配合夜间图像" in prompt


def test_build_prompt_empty_falls_back_to_default():
    from trainhub.core.prelabel_vlm import DEFAULT_DETECT_PROMPT

    assert build_prompt([]).startswith(DEFAULT_DETECT_PROMPT)
    assert build_prompt([], prompt="   ").startswith(DEFAULT_DETECT_PROMPT)


def test_build_prompt_requires_english_labels():
    with_labels = build_prompt(["猫", "狗"])
    assert "英文" in with_labels
    assert "label_cn" in with_labels
    assert '"cat"' in with_labels  # JSON 示例里 label 是英文
    free = build_prompt([])
    assert "英文" in free
    assert "禁止使用中文" in free


def test_provider_presets_complete():
    for key, provider in VLM_PROVIDERS.items():
        assert provider.key == key
        assert provider.display_name
        if key != "custom":
            assert provider.base_url.startswith("https://")
            assert provider.default_model
    assert provider_from_key("deepseek").default_model == "deepseek-flash"
    with pytest.raises(VLMError):
        provider_from_key("nope")
