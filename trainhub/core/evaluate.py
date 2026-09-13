"""推理结果评估：与真值对比的框匹配，以及"再训练参考"报告生成。

无 Qt 依赖，供推理页与测试共用。匹配规则：同类目标按 IoU≥0.5 贪心
配对（置信度从高到低），配上的计 TP，预测没配上计 FP，真值没配上计 FN。
"""

from __future__ import annotations


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_detections(
    preds: list[tuple[str, tuple[float, float, float, float], float]],
    gts: list[tuple[str, tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """同类贪心匹配：返回 {类别: {"tp": n, "fp": n, "fn": n}}。

    preds: (标签, 框, 置信度)；gts: (标签, 框)。置信度高的先配。
    """
    per_class: dict[str, dict[str, int]] = {}
    for label in {label for label, _, _ in preds} | {label for label, _ in gts}:
        per_class[label] = {"tp": 0, "fp": 0, "fn": 0}

    remaining = list(range(len(gts)))
    order = sorted(range(len(preds)), key=lambda i: -preds[i][2])
    for pi in order:
        pred_label, pred_box, _conf = preds[pi]
        best_gi, best_iou = -1, 0.0
        for gi in remaining:
            gt_label, gt_box = gts[gi]
            if gt_label != pred_label:
                continue
            score = iou(pred_box, gt_box)
            if score > best_iou:
                best_gi, best_iou = gi, score
        if best_gi >= 0 and best_iou >= iou_threshold:
            per_class[pred_label]["tp"] += 1
            remaining.remove(best_gi)
        else:
            per_class[pred_label]["fp"] += 1

    for gi, (gt_label, _gt_box) in enumerate(gts):
        if gi in remaining:
            per_class[gt_label]["fn"] += 1
    return per_class


def build_retrain_report(stats: dict) -> str:
    """把推理统计转成"再训练参考"文字报告。

    ``stats`` 结构（由推理页组装）::

        {
            "images": 4, "boxes": 1, "zero_images": 2,
            "avg_conf": 0.31 | None,
            "low_conf_boxes": 1,                # conf < 0.15 的框数
            "per_class_pred": {"stick": {"count": 1, "avg_conf": 0.31}},
            "gt_images": 2,                     # 有真值标注的图像数
            "per_class_gt": {"stick": {"tp": 1, "fp": 0, "fn": 1}},
            "training": {"run": "...", "epochs": 10, "last_map50": 0.197} | None,
        }
    """
    lines: list[str] = []
    images = stats.get("images", 0)
    boxes = stats.get("boxes", 0)
    zero = stats.get("zero_images", 0)
    avg_conf = stats.get("avg_conf")

    lines.append(f"【汇总】处理 {images} 张 · 检出 {boxes} 框 · 零检出 {zero} 张")
    if avg_conf is not None:
        lines.append(f"检出框平均置信度 {avg_conf:.2f}")
    if stats.get("low_conf_boxes"):
        lines.append(f"低置信度框（<0.15）{stats['low_conf_boxes']} 个")

    per_class_pred = stats.get("per_class_pred") or {}
    if per_class_pred:
        lines.append("【每类检出】")
        for label, info in sorted(per_class_pred.items()):
            conf_text = (
                f" · 平均置信度 {info['avg_conf']:.2f}"
                if info.get("avg_conf") is not None else ""
            )
            lines.append(f"  {label}: {info['count']} 框{conf_text}")

    gt_images = stats.get("gt_images", 0)
    per_class_gt = stats.get("per_class_gt") or {}
    total_tp = total_fp = total_fn = 0
    if gt_images and per_class_gt:
        lines.append(f"【与真值对比】（{gt_images} 张有标注，IoU≥0.5）")
        total_tp = total_fn = 0
        for label, info in sorted(per_class_gt.items()):
            gt_total = info["tp"] + info["fn"]
            recall = info["tp"] / gt_total if gt_total else None
            recall_text = f"检出率 {recall:.0%}" if recall is not None else "—"
            lines.append(
                f"  {label}: 正确 {info['tp']} · 漏检 {info['fn']} · 误检 {info['fp']}"
                f" → {recall_text}"
            )
            total_tp += info["tp"]
            total_fn += info["fn"]

    training = stats.get("training")
    if training:
        epochs = training.get("epochs")
        map50 = training.get("last_map50")
        text = f"【训练信息】{training.get('run', '')}："
        text += f"{epochs if epochs is not None else '?'} 轮"
        if map50 is not None:
            text += f" · 最终 mAP50 {map50:.3f}"
        lines.append(text)

    # -------------------------------------------------- 规则化再训练建议
    advice: list[str] = []
    if training:
        epochs = training.get("epochs") or 0
        map50 = training.get("last_map50")
        if map50 is not None and map50 < 0.5:
            advice.append(
                f"模型明显欠训练（mAP50 仅 {map50:.2f}）：建议把轮数提到 150–300 "
                "重新训练，观察训练曲线 mAP50 爬平后再推理"
            )
        elif epochs and epochs < 50:
            advice.append("训练轮数偏少（<50），曲线仍在爬升时早停会低估模型，建议加轮数重训")
    if images and boxes == 0:
        advice.append(
            "全部零检出：先把置信度阈值降到 0.05 观察是否有低分框；"
            "仍无框则确认权重与数据是同一批目标"
        )
    if total_fn:
        gt_all = total_tp + total_fn
        recall = total_tp / gt_all if gt_all else 0
        if recall < 0.6:
            advice.append(
                f"漏检偏多（检出率 {recall:.0%}）：优先检查漏检图像的标注质量与目标大小，"
                "必要时补充该类样本后重训"
            )
    total_fp = sum(info["fp"] for info in per_class_gt.values())
    total_pred_gt = sum(info["tp"] + info["fp"] for info in per_class_gt.values())
    if total_pred_gt and total_fp / total_pred_gt > 0.3:
        advice.append("误检偏多（>30%）：可提高置信度阈值，或补充易混淆的负样本/背景图后重训")
    if zero and images and zero / images > 0.3:
        advice.append(f"零检出占 {zero}/{images}：多为模型能力不足所致，重训后再评估")
    if gt_images == 0 and images:
        advice.append("本批图像没有真值标注，无法对比漏检；可先在标注页标注一部分再推理对比")

    if advice:
        lines.append("【再训练参考】")
        lines.extend(f"• {text}" for text in advice)
    return "\n".join(lines)
