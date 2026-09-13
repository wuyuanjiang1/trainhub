"""一次预标注运行的 token / 检出 / 费用统计。"""

from __future__ import annotations


def _fmt_tokens(count: int) -> str:
    return f"{count / 10000:.2f}万" if count >= 10000 else str(count)


class UsageTracker:
    """一次预标注运行的 token / 检出统计，用于界面上的花费标识。

    ``add_usage`` 吃 chat/completions 响应里的 ``usage`` 字段（DeepSeek 的
    ``prompt_cache_hit_tokens`` 和 OpenAI 风格的
    ``prompt_tokens_details.cached_tokens`` 都认）；``note_image`` 逐张记录
    检出与否。QThread 里累加、主线程读快照，纯 int/dict 操作无需加锁。
    """

    def __init__(self) -> None:
        self.images = 0
        self.found_images = 0
        self.empty_images = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0

    # ---------------------------------------------------------------- update
    def add_usage(self, usage: dict) -> None:
        self.requests += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        hit = usage.get("prompt_cache_hit_tokens")  # DeepSeek
        if hit is None:
            details = usage.get("prompt_tokens_details")  # OpenAI 风格
            if isinstance(details, dict):
                hit = details.get("cached_tokens")
        self.cache_hit_tokens += int(hit or 0)

    def note_image(self, found: bool) -> None:
        self.images += 1
        if found:
            self.found_images += 1
        else:
            self.empty_images += 1

    # ----------------------------------------------------------------- view
    def estimate_cost(self, provider) -> float | None:
        """按服务商空闲档单价估算费用（元）；价格未内置时返回 None。"""
        if provider is None or provider.input_price is None or provider.output_price is None:
            return None
        if provider.cache_hit_price is not None:
            hit = self.cache_hit_tokens
        else:
            hit = 0
        cost = (self.prompt_tokens - hit) / 1e6 * provider.input_price
        cost += hit / 1e6 * (provider.cache_hit_price or provider.input_price)
        cost += self.completion_tokens / 1e6 * provider.output_price
        return cost

    def format(self, provider=None) -> str:
        if self.images:
            rate = self.found_images / self.images
            parts = [
                f"已处理 {self.images} 张",
                f"检出 {self.found_images}（{rate:.0%}）",
                f"空 {self.empty_images}",
            ]
        else:
            parts = ["尚未处理图像"]
        if self.requests:
            parts.append(
                f"tokens ↑{_fmt_tokens(self.prompt_tokens)}"
                f" ↓{_fmt_tokens(self.completion_tokens)}"
            )
        if self.cache_hit_tokens and self.prompt_tokens:
            parts.append(f"缓存命中 {self.cache_hit_tokens / self.prompt_tokens:.0%}")
        cost = self.estimate_cost(provider)
        if cost is not None:
            shown = f"{cost:.4f}".rstrip("0").rstrip(".") or "0"
            parts.append(f"≈¥{shown}")
        return " · ".join(parts)
