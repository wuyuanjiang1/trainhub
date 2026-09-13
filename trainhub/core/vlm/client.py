"""OpenAI 兼容请求、响应抽取，以及预标注总入口。"""

from __future__ import annotations

import base64
import io
import json
import re
import socket
import time
import urllib.error
import urllib.request

from .parse import parse_detections
from .prompt import build_prompt
from .prompt import extract_label_whitelist
from .providers import VLMError
from .providers import VLMProvider


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
) -> tuple[str, dict]:
    """调一次 OpenAI 兼容的 chat/completions，返回（回复文本, usage）。"""
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
            return _extract_content(body), _extract_usage(body)
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


def _extract_usage(body: dict) -> dict:
    """取响应里的 token 用量；缺失时返回空字典。"""
    usage = body.get("usage")
    return usage if isinstance(usage, dict) else {}


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


def predict_shapes_vlm(
    image_path: str,
    *,
    provider: VLMProvider,
    api_key: str,
    model: str = "",
    base_url: str = "",
    labels=(),
    hint: str = "",
    prompt: str = "",
    max_side: int = 1600,
    timeout: float = 90.0,
    retries: int = 2,
    label_translator: "Callable[[str | None, str], str] | None" = None,
    usage_tracker=None,
) -> list[dict]:
    """对单张图像调用视觉大模型，返回候选标注（labelme shape 字典）。

    ``prompt`` 为自定义检测任务描述，留空用默认的标框提示词；提示词是裸
    目标名清单（如"木棍"或"木棍、纸箱"）时，这些词成为唯一允许的标签：
    提到几种就最多输出几种，多出的标签在解析时被强制丢弃。
    ``labels`` 为项目标签列表（提示词清单优先于它）：非空时只保留这些
    类别，两者都为空表示自由命名。
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

    effective_labels = extract_label_whitelist(prompt) or list(labels)
    image_b64, width, height, scale = _encode_image(image_path, max_side=max_side)
    content, usage = _chat_completion(
        base_url=base,
        api_key=api_key.strip(),
        model=model,
        prompt=build_prompt(labels=effective_labels, hint=hint, prompt=prompt),
        image_b64=image_b64,
        timeout=timeout,
        retries=retries,
    )
    if usage_tracker is not None:
        usage_tracker.add_usage(usage)
    if not content:
        raise VLMError("模型返回了空内容，请检查模型是否支持图像输入")
    shapes, _warnings = parse_detections(
        content,
        width=width,
        height=height,
        coord_system=provider.coord_system,
        scale=scale,
        allowed_labels=effective_labels or None,
        label_translator=label_translator,
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
    content, _usage = _chat_completion(
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
