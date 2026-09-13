"""检测提示词构造与"裸目标名清单"白名单解析。"""

from __future__ import annotations

import json
import re
from typing import Sequence

# 类别列表在 prompt 里最多列这么多，防止超长
MAX_PROMPT_LABELS = 60

# 默认检测任务描述；界面上可被用户自定义提示词替换，坐标与 JSON 格式约定
# 由 build_prompt 自动附加，用户无需也不应手写。
DEFAULT_DETECT_PROMPT = (
    "你是图像数据标注助手。请找出图像中的所有目标物体，逐一用矩形框标出。"
)

# 判断提示词是否"指令性语句"的标记词；命中则不把提示词当类别清单解析。
# 允许误放（宁可将指令句不当清单），不允许误杀常见目标名词。"
_INSTRUCTION_MARKERS = (
    "标",
    "找",
    "输出",
    "检测",
    "识别",
    "忽略",
    "请",
    "所有",
    "完整",
    "可见",
    "帮助",
    "助手",
    "图像",
    "图中",
    "label",
    "detect",
)

_MAX_WHITELIST_PROMPT_LEN = 40
_MAX_WHITELIST_TOKEN_LEN = 12


def extract_label_whitelist(prompt: str, *, max_labels: int = 8) -> list[str] | None:
    """把"裸目标名清单"式的检测提示词解析成类别白名单。

    用户在提示词里只填目标名称（如 ``木棍`` 或 ``木棍、纸箱``）时，这些词
    就是本次预标注唯一允许的输出标签：提到几种就只允许这几种，模型不得
    新增类别。提示词为空、是默认模板或含指令性语句（"只标…"、"忽略…"等）
    时返回 None，维持原有类别逻辑。
    """
    text = (prompt or "").strip()
    if not text or text == DEFAULT_DETECT_PROMPT or len(text) > _MAX_WHITELIST_PROMPT_LEN:
        return None
    tokens = [t for t in re.split(r"[,，、;；/\s]+", text) if t]
    if not 1 <= len(tokens) <= max_labels:
        return None
    for token in tokens:
        if len(token) > _MAX_WHITELIST_TOKEN_LEN:
            return None
        if any(marker in token.lower() for marker in _INSTRUCTION_MARKERS):
            return None
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


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
            "label_cn 原样填写所选类别的名称。"
            "只允许输出这些类别：提示词没提到的目标一律跳过、不得新增标签，"
            "类别列表里有几种目标就最多输出这几种标签；"
            "同一类别的多个实例要分别画框。类别列表："
            + json.dumps(shown, ensure_ascii=False)
        )
    else:
        lines.append(
            "类别由你根据图像内容命名：label 必须使用简洁的小写英文单词或短语，"
            "禁止使用中文。提示词提到几种目标，就只允许输出这几种标签，"
            "不得增加提示词未提到的类别；同一类别的多个实例要分别画框。"
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
        "如果图中没有目标，输出空数组 []。",
    ]
    if hint.strip():
        lines += ["", f"补充要求：{hint.strip()}"]
    return "\n".join(lines)
