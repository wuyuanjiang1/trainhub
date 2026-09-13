"""UsageTracker：token 用量、缓存命中率、检出率与费用估算测试。"""

from __future__ import annotations

from trainhub.core.prelabel_vlm import UsageTracker
from trainhub.core.prelabel_vlm import VLM_PROVIDERS


def test_add_usage_deepseek_style_cache_fields():
    tracker = UsageTracker()
    tracker.add_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_cache_hit_tokens": 400,
        }
    )
    assert tracker.prompt_tokens == 1000
    assert tracker.completion_tokens == 200
    assert tracker.cache_hit_tokens == 400


def test_add_usage_openai_style_cached_tokens():
    tracker = UsageTracker()
    tracker.add_usage(
        {
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 128},
        }
    )
    assert tracker.cache_hit_tokens == 128


def test_note_image_and_detection_rate():
    tracker = UsageTracker()
    for found in (True, True, False):
        tracker.note_image(found)
    assert tracker.images == 3
    assert tracker.found_images == 2
    assert tracker.empty_images == 1
    text = tracker.format()
    assert "已处理 3 张" in text
    assert "检出 2（67%）" in text
    assert "空 1" in text


def test_cost_estimate_deepseek_pricing():
    tracker = UsageTracker()
    tracker.add_usage(
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 500_000,
            "prompt_cache_hit_tokens": 400_000,
        }
    )
    # 未命中 60 万 × 1 元 + 命中 40 万 × 0.02 元 + 输出 50 万 × 4 元
    cost = tracker.estimate_cost(VLM_PROVIDERS["deepseek"])
    assert cost is not None
    assert abs(cost - (0.6 + 0.008 + 2.0)) < 1e-9
    text = tracker.format(VLM_PROVIDERS["deepseek"])
    assert "缓存命中 40%" in text
    assert "≈¥2.608" in text


def test_cost_none_when_price_not_configured():
    tracker = UsageTracker()
    tracker.add_usage({"prompt_tokens": 100, "completion_tokens": 10})
    assert tracker.estimate_cost(VLM_PROVIDERS["zhipu"]) is None
    assert "¥" not in tracker.format(VLM_PROVIDERS["zhipu"])


def test_token_formatting_wan_unit():
    tracker = UsageTracker()
    tracker.add_usage({"prompt_tokens": 123_456, "completion_tokens": 789})
    text = tracker.format()
    assert "↑12.35万" in text
    assert "↓789" in text


def test_empty_tracker_format():
    assert UsageTracker().format() == "尚未处理图像"
