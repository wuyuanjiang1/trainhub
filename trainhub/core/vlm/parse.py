"""模型回复解析：抽取检测条目、坐标换算、标注词归一化。"""

from __future__ import annotations

import json
import re
from typing import Any
from typing import Sequence

_LABEL_KEYS = ("label", "name", "class", "category", "标签", "类别", "type")
# label_cn 是模型回传的"项目类别原名称"（可能为中文），用于白名单校验与对照
_REFERENCE_KEYS = (
    "label_cn",
    "label_zh",
    "category_cn",
    "class_cn",
    "original_label",
    "原始类别",
    "中文类别",
    "类别",
    "标签",
)
_BBOX_KEYS = (
    "bbox_2d",
    "bbox",
    "box",
    "bndbox",
    "bounding_box",
    "boundingbox",
    "rect",
    "坐标",
)
_CONF_KEYS = ("confidence", "conf", "score", "prob", "probability")
_RESULT_LIST_KEYS = ("detections", "objects", "results", "items", "data", "annotations")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_NON_LABEL_WORD_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")

# Qwen grounding 的另一种输出：<ref>标签</ref><box>(x1,y1),(x2,y2)</box>
_QWEN_BOX_RE = re.compile(
    r"<ref>\s*(?P<label>.*?)\s*</ref>\s*<box>\s*"
    r"\(\s*(?P<x1>[\d.]+)\s*,\s*(?P<y1>[\d.]+)\s*\)\s*,\s*"
    r"\(\s*(?P<x2>[\d.]+)\s*,\s*(?P<y2>[\d.]+)\s*\)\s*</box>"
)
_BARE_BOX_RE = re.compile(
    r"<box>\s*\[\s*\[?\s*(?P<x1>[\d.]+)\s*,\s*(?P<y1>[\d.]+)\s*,\s*"
    r"(?P<x2>[\d.]+)\s*,\s*(?P<y2>[\d.]+)\s*\]?\s*\]\s*</box>"
)


def normalize_english_label(text: str) -> str:
    """把模型输出的标注词归一成稳定形式：小写、非字母数字并成下划线。

    "Wooden Stick" / "wooden stick" / "wooden_stick" 都归一为
    ``wooden_stick``，避免同一目标因空格/大小写/连字符差异分裂成多个标签。
    """
    cleaned = _NON_LABEL_WORD_RE.sub("_", str(text).strip().lower()).strip("_")
    return cleaned or "object"


def parse_detections(
    text: str,
    *,
    width: int,
    height: int,
    coord_system: str = "norm1000",
    scale: float = 1.0,
    allowed_labels: Sequence[str] | None = None,
    label_translator: "Callable[[str | None, str], str] | None" = None,
) -> tuple[list[dict], list[str]]:
    """把模型回复解析成 labelme shape 字典列表。

    返回 (shapes, warnings)。width/height 是原图尺寸；输出坐标始终落在
    原图像素范围内。shape 的 ``label`` 是模型输出的英文标注词（若提供
    ``label_translator``，会先归一/映射成项目内唯一的标准名），项目类别
    原名称（如有）存在 ``other_data.label_cn`` 便于对照。无法解析的条目
    跳过并记入 warnings，不抛异常。
    """
    warnings: list[str] = []
    entries = _extract_entries(text, warnings)
    allowed = {str(label) for label in allowed_labels} if allowed_labels else None

    shapes: list[dict] = []
    dropped_labels: set[str] = set()
    chinese_labels = False
    for entry in entries:
        label, reference, values, confidence = _parse_entry(entry)
        if values is None:
            warnings.append(f"跳过无法识别坐标的条目: {_short(entry)}")
            continue
        if label is None:
            label = "object"
        label = str(label).strip() or "object"
        reference = (str(reference).strip() if reference else "") or None

        # 模型把中英填反了（label 是中文、label_cn 是英文）：换回来
        if reference and _CJK_RE.search(label) and not _CJK_RE.search(reference):
            label, reference = reference, label
        if _CJK_RE.search(label):
            chinese_labels = True

        # 白名单按项目类别原名称（label_cn）校验；没有回传时退回 label 本身
        if allowed is not None:
            check = reference or label
            if check not in allowed:
                dropped_labels.add(check)
                continue

        # 归一/映射成项目内唯一的标准名（模型对同一目标的英文写法不稳定）
        final_label = label
        if label_translator is not None:
            try:
                final_label = str(label_translator(reference, label)) or label
            except Exception:
                final_label = label

        other_data: dict = {}
        if reference and reference != final_label:
            other_data["label_cn"] = reference

        pixels = _to_pixels(values, width=width, height=height,
                            coord_system=coord_system, scale=scale,
                            warnings=warnings)
        if pixels is None:
            continue
        x1, y1, x2, y2 = pixels
        if x2 - x1 < 2 or y2 - y1 < 2:
            warnings.append(f"跳过过小的目标框: {final_label}")
            continue
        shapes.append(_shape_dict(final_label, x1, y1, x2, y2, confidence, other_data))

    if dropped_labels:
        warnings.append(
            "忽略了不属于项目类别的目标: " + "、".join(sorted(dropped_labels))
        )
    if chinese_labels:
        warnings.append("部分标注词仍是中文，未能转换为英文，请人工检查")
    return shapes, warnings


def _extract_entries(text: str, warnings: list[str]) -> list[Any]:
    """从回复文本里抠出检测结果条目（JSON 数组 / 包一层 / Qwen 标签）。"""
    cleaned = re.sub(r"```[a-zA-Z]*", "", text).strip()
    candidate = _extract_json_array(cleaned)
    if isinstance(candidate, list) and not candidate:
        # 模型明确回复空数组：图中无目标，属正常空结果
        return []
    if candidate is not None:
        entries = _unwrap_entries(candidate)
        if entries:
            return entries

    # JSON 路径失败：回退解析 Qwen / GLM 的 <box> 标签写法
    entries = _entries_from_tags(cleaned)
    if entries:
        return entries
    if cleaned:
        warnings.append("模型回复中没有找到可解析的检测结果")
    return []


def _extract_json_array(text: str) -> Any | None:
    """定位第一个可解析的完整 JSON 数组（括号配平扫描）。

    正文里可能夹杂带括号的说明文字，扫描到一组配平但解析失败的括号时
    不放弃，继续找下一个候选数组。
    """
    start = text.find("[")
    while 0 <= start < len(text):
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        start = text.find("[", index + 1)
                        break
        else:
            return None
    return None


def _unwrap_entries(candidate: Any) -> list[Any]:
    """允许 {"detections": [...]} 之类的包裹结构，返回条目列表。"""
    if isinstance(candidate, list):
        return [item for item in candidate if isinstance(item, (dict, list))]
    if isinstance(candidate, dict):
        for key in _RESULT_LIST_KEYS:
            value = candidate.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, (dict, list))]
        for value in candidate.values():
            if isinstance(value, list) and value:
                return [item for item in value if isinstance(item, (dict, list))]
    return []


def _entries_from_tags(text: str) -> list[dict]:
    entries: list[dict] = []
    for match in _QWEN_BOX_RE.finditer(text):
        entries.append(
            {
                "label": match.group("label"),
                "bbox_2d": [
                    float(match.group("x1")),
                    float(match.group("y1")),
                    float(match.group("x2")),
                    float(match.group("y2")),
                ],
            }
        )
    if entries:
        return entries
    for match in _BARE_BOX_RE.finditer(text):
        entries.append(
            {
                "label": "object",
                "bbox_2d": [
                    float(match.group("x1")),
                    float(match.group("y1")),
                    float(match.group("x2")),
                    float(match.group("y2")),
                ],
            }
        )
    return entries


def _parse_entry(
    entry: Any,
) -> tuple[str | None, str | None, list[float] | None, float | None]:
    """取出 (标注词 label, 项目类别原名称 reference, 坐标, 置信度)。"""
    if isinstance(entry, list):
        numbers = [
            float(v) for v in entry if isinstance(v, int | float) and not isinstance(v, bool)
        ]
        if len(numbers) == 4:
            return None, None, numbers, None
        return None, None, None, None

    if not isinstance(entry, dict):
        return None, None, None, None

    def _pick(keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    label = _pick(_LABEL_KEYS)
    reference = _pick(_REFERENCE_KEYS)

    confidence = None
    for key in _CONF_KEYS:
        value = entry.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= 1:
            confidence = float(value)
            break

    values = None
    for key in _BBOX_KEYS:
        raw = entry.get(key)
        values = _flatten_bbox(raw)
        if values is not None:
            break
    if values is None:
        # 兜底：条目里除标签外唯一的 4 数字列表
        for value in entry.values():
            if isinstance(value, list | tuple):
                flat = _flatten_bbox(value)
                if flat is not None:
                    values = flat
                    break
    return label, reference, values, confidence


def _flatten_bbox(raw: Any) -> list[float] | None:
    """[x1,y1,x2,y2] 或 [[x1,y1],[x2,y2]] 两种写法都接受。"""
    if not isinstance(raw, list | tuple):
        return None
    if len(raw) == 4 and all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in raw
    ):
        return [float(v) for v in raw]
    if (
        len(raw) == 2
        and all(isinstance(point, list | tuple) and len(point) == 2 for point in raw)
        and all(
            isinstance(v, int | float) and not isinstance(v, bool)
            for point in raw
            for v in point
        )
    ):
        (x1, y1), (x2, y2) = raw
        return [float(x1), float(y1), float(x2), float(y2)]
    return None


def _to_pixels(
    values: list[float],
    *,
    width: int,
    height: int,
    coord_system: str,
    scale: float,
    warnings: list[str],
) -> tuple[float, float, float, float] | None:
    """把模型坐标换算成原图像素坐标，交换反角、clamp 出界。"""
    if len(values) != 4 or not all(v == v and abs(v) != float("inf") for v in values):
        return None
    x1, y1, x2, y2 = values

    if coord_system != "norm1000":
        # pixel：模型看到的是缩放后的图，先除回缩放比
        if scale <= 0:
            scale = 1.0
        x1, y1, x2, y2 = (v / scale for v in (x1, y1, x2, y2))
    elif max(x1, y1, x2, y2) > 1050 or min(x1, y1, x2, y2) < -50:
        # 模型没守归一化约定、直接吐了像素坐标：只要数值在图尺寸内就按像素解释
        longest = max(width, height)
        if max(x1, y1, x2, y2) <= longest * 1.05:
            warnings.append("模型返回了像素坐标而非 0-1000 归一化坐标，已按像素解析")
            coord_system = "pixel"
        else:
            warnings.append(
                f"坐标 [{', '.join(f'{v:g}' for v in values)}] 超出图像范围，已丢弃"
            )
            return None

    if coord_system == "norm1000":
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def _shape_dict(
    label: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float | None,
    other_data: dict | None = None,
) -> dict:
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": f"AI {confidence:.2f}" if confidence is not None else "AI",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
        "other_data": other_data or {},
    }


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, dict | list) else str(value)
    return text if len(text) <= limit else text[:limit] + "…"
