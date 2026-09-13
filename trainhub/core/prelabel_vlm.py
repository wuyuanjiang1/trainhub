"""大模型（VLM）预标注后端：调用 OpenAI 兼容接口生成候选标注。

服务商只需要一个 OpenAI 兼容的 ``chat/completions`` 端点和一把 API Key，
因此所有预设共用同一套请求与解析代码（DeepSeek / 智谱 / 百炼 / 硅基流动
/ 自定义）。模型被要求以 0-1000 归一化坐标输出矩形框——这是 GLM-4.5V 与
Qwen-VL grounding 的通用约定；换算回像素后产出与
:mod:`trainhub.core.prelabel` 同构的 labelme shape 字典，画布回填与批量
写盘因此零改动。

注意：VLM 输出的是矩形框；segment 任务的多边形候选需在画布上人工调整，
后续可接"VLM 出框 + SAM 出掩膜"的两阶段管线。全部请求走标准库 urllib，
不引入新依赖。
"""

from __future__ import annotations

import base64
import io
import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Sequence

MAX_PROMPT_LABELS = 60


class VLMError(RuntimeError):
    """调用大模型接口失败（网络 / 鉴权 / 响应无法解析）。"""


@dataclass(frozen=True)
class VLMProvider:
    """一个 OpenAI 兼容服务商的预标注预设。"""

    key: str
    display_name: str
    base_url: str
    default_model: str
    models: tuple[str, ...] = ()
    # norm1000: 0-1000 归一化坐标（GLM / Qwen grounding 约定）；pixel: 像素坐标
    coord_system: str = "norm1000"
    # 服务商是否官方声明了检测框（grounding）能力；未声明的建议先试标一张
    grounding_verified: bool = False
    docs_url: str = ""


VLM_PROVIDERS: dict[str, VLMProvider] = {
    "deepseek": VLMProvider(
        key="deepseek",
        display_name="DeepSeek（deepseek-flash）",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-flash",
        models=("deepseek-flash",),
        grounding_verified=False,
        docs_url="https://api-docs.deepseek.com/zh-cn/guides/vision",
    ),
    "zhipu": VLMProvider(
        key="zhipu",
        display_name="智谱 GLM（glm-4.5v）",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4.5v",
        models=("glm-4.5v",),
        grounding_verified=True,
        docs_url="https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.5v",
    ),
    "dashscope": VLMProvider(
        key="dashscope",
        display_name="阿里百炼（Qwen-VL）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3-vl-plus",
        models=("qwen3-vl-plus", "qwen3-vl-max", "qwen2.5-vl-72b-instruct"),
        grounding_verified=True,
        docs_url="https://help.aliyun.com/zh/model-studio/getting-started/models",
    ),
    "siliconflow": VLMProvider(
        key="siliconflow",
        display_name="硅基流动（聚合平台）",
        base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-VL-72B-Instruct",
        models=("Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct"),
        grounding_verified=False,
        docs_url="https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions",
    ),
    "custom": VLMProvider(
        key="custom",
        display_name="自定义（OpenAI 兼容）",
        base_url="",
        default_model="",
        grounding_verified=False,
    ),
}

ENV_KEYS: dict[str, tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "zhipu": ("ZHIPU_API_KEY", "ZHIPUAI_API_KEY"),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "siliconflow": ("SILICONFLOW_API_KEY",),
    "custom": (),
}


def provider_env_key(provider_key: str) -> str | None:
    """该服务商在本机环境变量里配置的 API Key（如有）。"""
    import os

    for name in ENV_KEYS.get(provider_key, ()):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


# --------------------------------------------------------------------- prompt
# 默认检测任务描述；界面上可被用户自定义提示词替换，坐标与 JSON 格式约定
# 由 build_prompt 自动附加，用户无需也不应手写。
DEFAULT_DETECT_PROMPT = (
    "你是图像数据标注助手。请找出图像中的所有目标物体，逐一用矩形框标出。"
)


def build_prompt(labels: Sequence[str], hint: str = "", prompt: str = "") -> str:
    """构造检测框 prompt：任务描述（可自定义）+ 标签约束 + 格式要求。

    标注词（label）统一要求英文输出：项目标签可能是中文，由模型给出英文
    译名并在 ``label_cn`` 里回传原类别名，白名单校验与中英对照都靠它。
    """
    task = prompt.strip() or DEFAULT_DETECT_PROMPT
    lines = [task, ""]
    if labels:
        shown = list(labels)[:MAX_PROMPT_LABELS]
        lines.append(
            "类别约束：每个目标从以下类别中选择语义对应的一项，"
            "label 用简洁、通用的英文译名输出（同一类别始终用同一个英文词），"
            "label_cn 原样填写所选类别的名称。类别列表："
            + json.dumps(shown, ensure_ascii=False)
        )
    else:
        lines.append(
            "类别由你根据图像内容命名：label 必须使用简洁的小写英文单词或短语，"
            "禁止使用中文。"
        )
    lines += [
        "",
        "坐标要求：使用 0-1000 的归一化坐标（相对图像宽度和高度），",
        "每个框输出 [x1, y1, x2, y2]，(x1, y1) 为左上角，(x2, y2) 为右下角。",
        "框要紧贴目标边缘；同一个目标只输出一个框，不要把多个同类目标合并。",
        "",
        "只输出一个 JSON 数组，不要输出任何解释文字、Markdown 代码块或其他内容。格式：",
        '[{"label": "cat", "label_cn": "猫", "bbox_2d": [x1, y1, x2, y2], "confidence": 0.85}]',
        "label 必须是英文；label_cn 仅在提供了类别列表时填写列表中的原名称，否则省略。",
    ]
    if hint.strip():
        lines += ["", f"补充要求：{hint.strip()}"]
    return "\n".join(lines)


# ------------------------------------------------------------------- request
def _encode_image(image_path: str, *, max_side: int) -> tuple[str, int, int, float]:
    """读图、压到 max_side 内并转 base64 JPEG。

    返回 (base64, 原图宽, 原图高, 缩放比)；像素坐标系的服务商需要用缩放比
    把"模型看到的坐标"换算回原图。
    """
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
        rgb = image.convert("RGB")
    scale = 1.0
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), width, height, scale


def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: float,
    retries: int,
) -> str:
    """调一次 OpenAI 兼容的 chat/completions，返回回复文本。"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(min(2**attempt * 2, 15))
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return _extract_content(body)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(500).decode("utf-8", "replace")
            except OSError:
                pass
            last_error = VLMError(f"HTTP {exc.code}: {detail or exc.reason}")
            # 4xx 是请求本身的问题（Key 无效 / 模型名错误 / 无视觉能力），重试无意义
            if exc.code not in (429, 500, 502, 503, 504):
                raise last_error from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            last_error = VLMError(f"网络请求失败: {exc}")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            last_error = VLMError(f"响应格式无法解析: {exc}")
    raise last_error or VLMError("请求失败")


def _extract_content(body: dict) -> str:
    """从 chat/completions 响应里取出文本；兼容 content 为分块列表的情况。"""
    message = body["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        text = ""
    if not text and message.get("reasoning_content"):
        # 个别推理模型把答案也塞进 reasoning_content，兜底取用
        text = str(message["reasoning_content"])
    # 推理型模型可能把 <think> 混进正文
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


# -------------------------------------------------------------------- parsing
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


def parse_detections(
    text: str,
    *,
    width: int,
    height: int,
    coord_system: str = "norm1000",
    scale: float = 1.0,
    allowed_labels: Sequence[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """把模型回复解析成 labelme shape 字典列表。

    返回 (shapes, warnings)。width/height 是原图尺寸；输出坐标始终落在
    原图像素范围内。shape 的 ``label`` 是模型输出的英文标注词，项目类别
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

        other_data: dict = {}
        if reference and reference != label:
            other_data["label_cn"] = reference

        pixels = _to_pixels(values, width=width, height=height,
                            coord_system=coord_system, scale=scale,
                            warnings=warnings)
        if pixels is None:
            continue
        x1, y1, x2, y2 = pixels
        if x2 - x1 < 2 or y2 - y1 < 2:
            warnings.append(f"跳过过小的目标框: {label}")
            continue
        shapes.append(_shape_dict(label, x1, y1, x2, y2, confidence, other_data))

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
    """允许 {""detections": [...]} 之类的包裹结构，返回条目列表。"""
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


# ------------------------------------------------------------------- entries
def predict_shapes_vlm(
    image_path: str,
    *,
    provider: VLMProvider,
    api_key: str,
    model: str = "",
    base_url: str = "",
    labels: Sequence[str] = (),
    hint: str = "",
    prompt: str = "",
    max_side: int = 1600,
    timeout: float = 90.0,
    retries: int = 2,
) -> list[dict]:
    """对单张图像调用视觉大模型，返回候选标注（labelme shape 字典）。

    ``prompt`` 为自定义检测任务描述，留空用默认的标框提示词；
    ``labels`` 为项目标签列表：非空时只保留这些类别，空列表表示自由命名。
    解析失败抛 :class:`VLMError`；个别目标框解析失败只记 warning 并跳过。
    """
    if not api_key or not api_key.strip():
        raise VLMError("API Key 为空，请先在预标注对话框中填写")
    base = (base_url or provider.base_url).strip().rstrip("/")
    if not base:
        raise VLMError("base_url 为空，请填写服务商的 OpenAI 兼容接口地址")
    model = (model or provider.default_model).strip()
    if not model:
        raise VLMError("模型名为空，请填写要调用的视觉模型名称")

    image_b64, width, height, scale = _encode_image(image_path, max_side=max_side)
    content = _chat_completion(
        base_url=base,
        api_key=api_key.strip(),
        model=model,
        prompt=build_prompt(labels=labels, hint=hint, prompt=prompt),
        image_b64=image_b64,
        timeout=timeout,
        retries=retries,
    )
    if not content:
        raise VLMError("模型返回了空内容，请检查模型是否支持图像输入")
    shapes, _warnings = parse_detections(
        content,
        width=width,
        height=height,
        coord_system=provider.coord_system,
        scale=scale,
        allowed_labels=labels if labels else None,
    )
    return shapes


def test_connection(
    *,
    provider: VLMProvider,
    api_key: str,
    model: str = "",
    base_url: str = "",
    timeout: float = 30.0,
) -> None:
    """发一张内置小图验证 Key、端点与视觉能力；失败抛 VLMError。"""
    if not api_key or not api_key.strip():
        raise VLMError("API Key 为空")
    buffer = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (64, 64), (200, 40, 40)).save(buffer, format="JPEG", quality=80)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    content = _chat_completion(
        base_url=(base_url or provider.base_url).strip().rstrip("/"),
        api_key=api_key.strip(),
        model=(model or provider.default_model).strip(),
        prompt='这张图是什么颜色？只输出 JSON：{"ok": true}',
        image_b64=image_b64,
        timeout=timeout,
        retries=0,
    )
    if not content:
        raise VLMError("模型返回了空内容")


def provider_from_key(key: str) -> VLMProvider:
    try:
        return VLM_PROVIDERS[key]
    except KeyError:
        raise VLMError(f"未知的服务商预设: {key!r}（可用: {sorted(VLM_PROVIDERS)}）") from None
