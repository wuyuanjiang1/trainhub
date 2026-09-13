"""Grounding 能力验证：评测服务商预标注的检出率与框质量。

两种模式：

1. 有真值（默认）：对项目里已标注的图像调用 VLM 预标注，与 labelme 真值
   做同类贪心匹配，输出 IoU 与漏检/误检统计：

       python scripts/test_vlm_grounding.py --provider deepseek --project /path/to/proj

2. 无真值（--no-gt）：不需要任何标注，只统计检出率与框面积分布，适合在
   接入前横向对比服务商的稳定性：

       python scripts/test_vlm_grounding.py --provider deepseek \
           --images /path/to/images --limit 20 --no-gt

判断标准：框质量模式看 IoU>=0.5 占比；无真值模式看检出率是否稳定、
框面积是否与目标尺度相符（全图大小的大框通常是坏框）。

API Key 优先级：--api-key > 环境变量（DEEPSEEK_API_KEY / ZHIPU_API_KEY /
DASHSCOPE_API_KEY / SILICONFLOW_API_KEY）。注意：脚本会把所选图像上传到
对应服务商。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainhub.core.dataset import scan_dataset
from trainhub.core.geometry import bbox_of
from trainhub.core.vlm import ENV_KEYS
from trainhub.core.vlm import VLM_PROVIDERS
from trainhub.core.vlm import VLMError
from trainhub.core.vlm import predict_shapes_vlm
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


def _resolve_api_key(args, provider) -> str:
    api_key = args.api_key.strip()
    if api_key:
        return api_key
    import os

    for name in ENV_KEYS.get(provider.key, ()):
        value = os.environ.get(name, "").strip()
        if value:
            print(f"使用环境变量 {name} 中的 API Key")
            return value
    print("错误：未提供 API Key（--api-key 或对应环境变量）", file=sys.stderr)
    return ""


def _predict(provider, api_key, args, image_path: Path, labels: list[str]) -> list[dict]:
    return predict_shapes_vlm(
        str(image_path),
        provider=provider,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        labels=labels,
        prompt=args.prompt,
        max_side=args.max_side,
    )


def run_detection_rate(provider, api_key, args, image_paths: list[Path], labels: list[str]) -> int:
    """无真值模式：检出率 + 框面积分布。"""
    print(f"无真值模式：评测 {len(image_paths)} 张图像的检出率；类别 {labels or '（自由命名）'}\n")
    hits = 0
    box_counts: list[int] = []
    area_fractions: list[float] = []
    missed: list[str] = []
    errors = 0

    for image_path in image_paths:
        try:
            shapes = _predict(provider, api_key, args, image_path, labels)
        except VLMError as exc:
            print(f"[失败] {image_path.name}: {exc}")
            errors += 1
            continue
        count = len(shapes)
        box_counts.append(count)
        if count:
            hits += 1
            from PIL import Image

            with Image.open(image_path) as im:
                image_area = im.width * im.height
            for shape in shapes:
                (x1, y1), (x2, y2) = shape["points"]
                area_fractions.append(max(0.0, (x2 - x1) * (y2 - y1)) / image_area)
        else:
            missed.append(image_path.name)
        print(f"[{'检出' if count else '空  '}] {image_path.name}: {count} 个框")

    total = len(image_paths) - errors
    if total == 0:
        print("\n没有成功调用过模型（网络/鉴权失败）。")
        return 2

    rate = hits / total
    print("\n========== 汇总 ==========")
    print(f"成功 {total} 张（失败 {errors}），检出 {hits} 张，检出率 {rate:.0%}")
    if box_counts:
        print(f"命中图的框数：平均 {statistics.mean(box_counts):.2f}，最多 {max(box_counts)}")
    if area_fractions:
        print(
            "框面积占全图比例："
            f"中位数 {statistics.median(area_fractions):.1%}，"
            f"最大 {max(area_fractions):.1%}"
        )
        big = sum(1 for f in area_fractions if f > 0.5)
        if big:
            print(f"⚠ 有 {big} 个框超过全图一半面积——通常是坏框（模型在糊答案）")
    if missed:
        shown = "、".join(missed[:10])
        more = f" 等 {len(missed)} 张" if len(missed) > 10 else ""
        print(f"未检出：{shown}{more}")

    if rate >= 0.8:
        print("\n结论：检出率良好。")
    elif rate >= 0.4:
        print("\n结论：检出率偏低，模型 grounding 不稳定，建议换服务商对比。")
    else:
        print("\n结论：检出率很差，不建议用该服务商做预标注。")
    return 0


def run_with_ground_truth(provider, api_key, args, items, labels: list[str]) -> int:
    """有真值模式：IoU 贪心匹配。"""
    print(f"评测 {len(items)} 张图像；服务商 {provider.display_name}；类别 {labels or '（自由命名）'}\n")
    per_image = []
    ious_by_label: dict[str, list[float]] = defaultdict(list)
    total_gt = total_pred = total_good = 0

    for item in items:
        try:
            shapes = _predict(provider, api_key, args, item.image_path, labels)
        except VLMError as exc:
            print(f"[失败] {item.image_path.name}: {exc}")
            per_image.append({"image": item.image_path.name, "error": str(exc)})
            continue

        preds = [(s["label"], (s["points"][0][0], s["points"][0][1],
                               s["points"][1][0], s["points"][1][1])) for s in shapes]
        gts = [
            (shape.label, bbox_of(shape))
            for shape in item.shapes
            if bbox_of(shape) is not None
        ]
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


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM 预标注 grounding 质量验证")
    parser.add_argument(
        "--provider", required=True, choices=sorted(VLM_PROVIDERS), help="服务商预设"
    )
    parser.add_argument("--api-key", default="", help="不填则读环境变量")
    parser.add_argument("--model", default="", help="默认用该服务商的预设模型")
    parser.add_argument("--base-url", default="", help="覆盖服务商预设接口地址")
    parser.add_argument("--prompt", default="", help="自定义检测任务提示词")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="trainhub 项目目录")
    group.add_argument("--images", help="图像目录")
    parser.add_argument("--annotations", default="", help="labelme JSON 目录（无真值模式可省略）")
    parser.add_argument("--no-gt", action="store_true", help="无真值模式：只统计检出率")
    parser.add_argument("--labels", default="", help="逗号分隔的类别白名单；默认读项目标签")
    parser.add_argument("--limit", type=int, default=5, help="最多评测的图像数")
    parser.add_argument("--max-side", type=int, default=1600, help="发送前压到的最长边")
    parser.add_argument("--json-out", default="", help="把明细写入 JSON 文件（仅真值模式）")
    args = parser.parse_args()

    provider = VLM_PROVIDERS[args.provider]
    api_key = _resolve_api_key(args, provider)
    if not api_key:
        return 2

    labels = [s for s in (x.strip() for x in args.labels.split(",")) if s]
    items = []
    image_paths: list[Path] = []
    if args.project:
        project = Project.open_or_create(args.project)
        if not labels:
            labels = list(project.labels)
        if args.no_gt or not (project.annotations_dir).exists():
            image_paths = [
                item.image_path
                for item in scan_dataset(project.images_dir, project.annotations_dir)
            ]
        else:
            items = [
                item
                for item in scan_dataset(
                    project.images_dir, project.annotations_dir, require_annotation=True
                )
                if item.shapes
            ]
            if not items:
                print("提示：项目里没有已标注图像，改用无真值模式（只统计检出率）\n")
                image_paths = [
                    item.image_path
                    for item in scan_dataset(project.images_dir, project.annotations_dir)
                ]
    else:
        from trainhub.core.dataset import find_images

        image_paths = find_images(Path(args.images))

    if args.no_gt or (not items and image_paths):
        if not image_paths:
            print("错误：没有找到图像", file=sys.stderr)
            return 2
        return run_detection_rate(
            provider, api_key, args, image_paths[: max(1, args.limit)], labels
        )

    if not items:
        print("错误：没有找到带标注的图像（无真值模式请加 --no-gt）", file=sys.stderr)
        return 2
    return run_with_ground_truth(
        provider, api_key, args, items[: max(1, args.limit)], labels
    )


def _dump(path: str, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
