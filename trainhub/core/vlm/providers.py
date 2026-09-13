"""VLM 服务商预设与 API Key 查找。"""

from __future__ import annotations

from dataclasses import dataclass


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
    # 元 / 百万 token（空闲时段价，用于费用估算）；None 表示未内置价格，
    # 只显示 token 计数。实际费用以服务商账单为准（DeepSeek 高峰时段翻倍）。
    input_price: float | None = None
    output_price: float | None = None
    cache_hit_price: float | None = None


VLM_PROVIDERS: dict[str, VLMProvider] = {
    "deepseek": VLMProvider(
        key="deepseek",
        display_name="DeepSeek（deepseek-flash）",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-flash",
        models=("deepseek-flash",),
        grounding_verified=False,
        docs_url="https://api-docs.deepseek.com/zh-cn/guides/vision",
        input_price=1.0,
        output_price=4.0,
        cache_hit_price=0.02,
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


def provider_from_key(key: str) -> VLMProvider:
    try:
        return VLM_PROVIDERS[key]
    except KeyError:
        raise VLMError(f"未知的服务商预设: {key!r}（可用: {sorted(VLM_PROVIDERS)}）") from None
