"""core/evaluate 的框匹配与再训练报告测试。"""

from __future__ import annotations

from trainhub.core.evaluate import build_retrain_report
from trainhub.core.evaluate import iou
from trainhub.core.evaluate import match_detections


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_match_counts_tp_fp_fn():
    preds = [
        ("cat", (0, 0, 10, 10), 0.9),        # 命中 gt1
        ("cat", (100, 100, 110, 110), 0.5),  # 无真值 → FP
        ("dog", (0, 0, 10, 10), 0.8),        # 类别不符 gt1 → FP(dog)
    ]
    gts = [("cat", (0, 0, 10, 10)), ("cat", (50, 50, 60, 60))]
    result = match_detections(preds, gts)
    assert result["cat"] == {"tp": 1, "fp": 1, "fn": 1}
    assert result["dog"] == {"tp": 0, "fp": 1, "fn": 0}


def test_low_conf_pred_loses_to_high_conf():
    preds = [
        ("cat", (0, 0, 10, 10), 0.3),
        ("cat", (1, 1, 11, 11), 0.9),  # 同一真值，高置信度者胜出
    ]
    gts = [("cat", (0, 0, 10, 10))]
    result = match_detections(preds, gts)
    assert result["cat"]["tp"] == 1
    assert result["cat"]["fp"] == 1  # 低分框成 FP
    assert result["cat"]["fn"] == 0


def test_report_undertrained_with_zero_detections():
    stats = {
        "images": 4, "boxes": 0, "zero_images": 4,
        "avg_conf": None, "low_conf_boxes": 0,
        "per_class_pred": {}, "gt_images": 0, "per_class_gt": {},
        "training": {"run": "yolo_x", "epochs": 10, "last_map50": 0.197},
    }
    report = build_retrain_report(stats)
    assert "处理 4 张" in report
    assert "欠训练" in report and "150–300" in report
    assert "全部零检出" in report


def test_report_with_gt_comparison():
    stats = {
        "images": 3, "boxes": 2, "zero_images": 1,
        "avg_conf": 0.44, "low_conf_boxes": 1,
        "per_class_pred": {"stick": {"count": 2, "avg_conf": 0.44}},
        "gt_images": 2,
        "per_class_gt": {"stick": {"tp": 1, "fp": 1, "fn": 1}},
        "training": {"run": "yolo_x", "epochs": 120, "last_map50": 0.81},
    }
    report = build_retrain_report(stats)
    assert "检出率 50%" in report
    assert "漏检偏多" in report          # 50% < 60% 阈值
    assert "平均置信度 0.44" in report
    assert "欠训练" not in report        # mAP50 0.81 不触发


def test_report_fp_heavy_advice():
    stats = {
        "images": 2, "boxes": 5, "zero_images": 0,
        "avg_conf": 0.5, "low_conf_boxes": 0,
        "per_class_pred": {"cat": {"count": 5, "avg_conf": 0.5}},
        "gt_images": 2,
        "per_class_gt": {"cat": {"tp": 1, "fp": 4, "fn": 0}},
        "training": {"run": "yolo_x", "epochs": 200, "last_map50": 0.75},
    }
    report = build_retrain_report(stats)
    assert "误检偏多" in report
    assert "漏检偏多" not in report


def test_report_no_gt_hint():
    stats = {
        "images": 2, "boxes": 1, "zero_images": 1,
        "avg_conf": 0.4, "low_conf_boxes": 0,
        "per_class_pred": {"a": {"count": 1, "avg_conf": 0.4}},
        "gt_images": 0, "per_class_gt": {},
        "training": None,
    }
    report = build_retrain_report(stats)
    assert "没有真值标注" in report
