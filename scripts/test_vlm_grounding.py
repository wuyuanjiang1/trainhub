"""Grounding 能力验证：用项目里已标注的图像评测 VLM 预标注的框质量。

接入大模型预标注前，先跑这个脚本确认所选服务商的检测框"能不能用"：

    python scripts/test_vlm_grounding.py --provider deepseek --project /path/to/proj
    python scripts/test_vlm_grounding.py --provider zhipu --api-key KEY \
        --model glm-4.5v --limit 10 --json-out report.json

对带标注的图像逐张调用 VLM 预标注，与 labelme 真值做同类贪心匹配，输出
IoU 与漏检/误检统计。粗略判断标准：IoU>=0.5 的框占多数即可用于预标注
（候选框反正要人工微调）。

API Key 优先级：--api-key > 环境变量（DEEPSEEK_API_KEY / ZHIPU_API_KEY /
DASHSCOPE_API_KEY / SILICONFLOW_API_KEY）。注意：脚本会把所选图像上传到
对应服务商。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainhub.core.dataset import scan_dataset
from trainhub.core.geometry import bbox_of
from trainhub.core.prelabel_vlm import ENV_KEYS
from trainhub.core.prelabel_vlm import VLM_PROVIDERS
from trainhub.core.prelabel_vlm import VLMError
from trainhub.core.prelabel_vlm import predict_shapes_vlm
from trainhub.core.project import Project

IOU_GOOD = 0.5


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _gt_boxes(item) -> list[tuple[str, tuple[float, float, float, float]]]:
    boxes = []
    for shape in item.shapes:
        box = bbox_of(shape)
        if box is not None:
            boxes.append((shape.label, box))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM 预标注 grounding 质量验证")
    parser.add_argument(
        "--provider", required=True, choices=sorted(VLM_PROVIDERS), help="服务商预设"
    )
    parser.add_argument("--api-key", default="", help="不填则读环境变量")
    parser.add_argument("--model", default="", help="默认用该服务商的预设模型")
    parser.add_argument("--base-url", default="", help="覆盖服务商预设接口地址")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="trainhub 项目目录")
    group.add_argument("--images", help="图像目录（配合 --annotations）")
    parser.add_argument("--annotations", default="", help="labelme JSON 目录")
    parser.add_argument("--labels", default="", help="逗号分隔的类别白名单；默认读项目标签")
    parser.add_argument("--limit", type=int, default=5, help="最多评测的图像数")
    parser.add_argument("--max-side", type=int, default=1600, help="发送前压到的最长边")
    parser.add_argument("--json-out", default="", help="把明细写入 JSON 文件")
    args = parser.parse_args()

    provider = VLM_PROVIDERS[args.provider]
    api_key = args.api_key.strip()
    if not api_key:
        import os

        for name in ENV_KEYS.get(provider.key, ()):
            api_key = os.environ.get(name, "").strip()
            if api_key:
                print(f"使用环境变量 {name} 中的 API Key")
                break
    if not api_key:
        print("错误：未提供 API Key（--api-key 或对应环境变量）", file=sys.stderr)
        return 2

    labels = [s for s in (x.strip() for x in args.labels.split(",")) if s]
    if args.project:
        project = Project.open_or_create(args.project)
        images_dir, annotations_dir = project.images_dir, project.annotations_dir
        if not labels:
            labels = list(project.labels)
    else:
        if not args.annotations:
            print("错误：--images 需要同时提供 --annotations", file=sys.stderr)
            return 2
        images_dir = Path(args.images)
        annotations_dir = Path(args.annotations)

    items = [
        item for item in scan_dataset(images_dir, annotations_dir, require_annotation=True)
        if item.shapes
    ]
    if not items:
        print("错误：没有找到带标注的图像", file=sys.stderr)
        return 2
    items = items[: max(1, args.limit)]
    print(f"评测 {len(items)} 张图像；服务商 {provider.display_name}；类别 {labels or '（自由命名）'}\n")

    per_image = []
    ious_by_label: dict[str, list[float]] = defaultdict(list)
    total_gt = total_pred = total_good = 0

    for item in items:
        try:
            shapes = predict_shapes_vlm(
                str(item.image_path),
                provider=provider,
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                labels=labels,
                max_side=args.max_side,
            )
        except VLMError as exc:
            print(f"[失败] {item.image_path.name}: {exc}")
            per_image.append({"image": item.image_path.name, "error": str(exc)})
            continue

        preds = [(s["label"], (s["points"][0][0], s["points"][0][1],
                               s["points"][1][0], s["points"][1][1])) for s in shapes]
        gts = _gt_boxes(item)
        used: set[int] = set()
        matched: list[float] = []
        for gt_label, gt_box in gts:
            best, best_iou = -1, 0.0
            for index, (pred_label, pred_box) in enumerate(preds):
                if index in used or pred_label != gt_label:
                    continue
                iou = _iou(gt_box, pred_box)
                if iou > best_iou:
                    best, best_iou = index, iou
            if best >= 0:
                used.add(best)
                matched.append(best_iou)
                ious_by_label[gt_label].append(best_iou)

        good = sum(1 for iou in matched if iou >= IOU_GOOD)
        total_gt += len(gts)
        total_pred += len(preds)
        total_good += good
        mean_iou = sum(matched) / len(matched) if matched else 0.0
        per_image.append(
            {
                "image": item.image_path.name,
                "gt": len(gts),
                "predicted": len(preds),
                "matched": len(matched),
                "good_boxes": good,
                "mean_iou": round(mean_iou, 3),
            }
        )
        print(
            f"[完成] {item.image_path.name}: 真值 {len(gts)} / 预测 {len(preds)} / "
            f"匹配 {len(matched)}，IoU>=0.5 {good} 个，平均 IoU {mean_iou:.2f}"
        )

    if total_pred == 0:
        print("\n结论：模型没有输出任何有效框，该服务商当前不可用。")
        report = {"provider": provider.key, "images": per_image, "verdict": "unusable"}
        _dump(args.json_out, report)
        return 1

    precision = total_good / total_pred
    recall = total_good / total_gt if total_gt else 0.0
    all_ious = [iou for values in ious_by_label.values() for iou in values]
    overall_mean = sum(all_ious) / len(all_ious) if all_ious else 0.0

    print("\n========== 汇总 ==========")
    print(f"预测框 {total_pred} 个，真值框 {total_gt} 个")
    print(f"框质量（IoU>=0.5 占预测框比例）: {precision:.1%}")
    print(f"召回（IoU>=0.5 占真值框比例）:   {recall:.1%}")
    print(f"匹配框平均 IoU: {overall_mean:.3f}")
    for label in sorted(ious_by_label):
        values = ious_by_label[label]
        mean = sum(values) / len(values)
        print(f"  - {label}: n={len(values)}, 平均 IoU {mean:.3f}")

    if precision >= 0.6 and overall_mean >= 0.5:
        verdict = "usable"
        print("\n结论：框质量可用于预标注（候选仍需人工微调）。")
    elif precision >= 0.3:
        verdict = "borderline"
        print("\n结论：框质量一般，可作参考但仍需较多修改；建议对比其他服务商。")
    else:
        verdict = "poor"
        print("\n结论：框质量较差，不建议用该服务商做预标注。")

    if args.json_out:
        _dump(
            args.json_out,
            {
                "provider": provider.key,
                "model": args.model or provider.default_model,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "mean_iou": round(overall_mean, 3),
                "verdict": verdict,
                "images": per_image,
            },
        )
        print(f"\n明细已写入 {args.json_out}")
    return 0


def _dump(path: str, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
