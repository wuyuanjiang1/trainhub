"""Ultralytics YOLO trainer plugin (detect / segment / classify)."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from ...core.dataset import scan_dataset
from ...core.devices import DEVICE_CHOICES
from ...core.devices import DEVICE_HELP
from ...core.devices import validate_device
from ...core.events import EventSink
from ...core.events import MetricEvent
from ...core.events import clean_log_text
from ...core.params import ParamSpec
from ...core.recorder import MetricRecorder
from ...core.registry import register_trainer
from ...core.trainer import BaseTrainer
from ...core.trainer import PreparedData
from ...core.trainer import TaskSpec
from ...core.trainer import TrainJob
from ...core.trainer import TrainResult
from .converter import export_yolo_dataset

WEIGHTS_CACHE = Path("~/.cache/trainhub/weights").expanduser()

_KNOWN_WEIGHTS = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
)


# 每轮摘要固定包含的训练损失项（Ultralytics 键名）。
_TRAIN_SUMMARY_KEYS = (
    ("train/box_loss", "box"),
    ("train/cls_loss", "cls"),
    ("train/dfl_loss", "dfl"),
    ("train/loss", "loss"),
)
# 验证指标按片段匹配 results_dict 的键（如 "metrics/mAP50(B)"）；顺序重要：
# "mAP50-95" 必须先于 "mAP50" 匹配，命中过的键不再复用。
_VAL_SUMMARY_KEYS = (
    ("mAP50-95", "mAP50-95"),
    ("mAP50", "mAP50"),
    ("precision", "P"),
    ("recall", "R"),
    ("accuracy_top1", "top1"),
    ("accuracy_top5", "top5"),
)


def _fmt_metric(value: float) -> str:
    return f"{value:.4g}"


def _epoch_summary(epoch: int, total: int, metrics: dict[str, float]) -> str:
    """把本轮指标压成一行易读摘要；没有可用指标时返回空串。"""
    parts: list[str] = []
    for key, label in _TRAIN_SUMMARY_KEYS:
        if key in metrics:
            parts.append(f"{label} {_fmt_metric(metrics[key])}")
    used: set[str] = set()
    for fragment, label in _VAL_SUMMARY_KEYS:
        for key, value in metrics.items():
            if key in used or not isinstance(value, int | float):
                continue
            if fragment in key:
                parts.append(f"{label} {_fmt_metric(value)}")
                used.add(key)
                break
    if "lr" in metrics:
        parts.append(f"lr {_fmt_metric(metrics['lr'])}")
    if not parts:
        return ""
    return f"第 {epoch}/{total} 轮 · " + " · ".join(parts)


class _SinkLogHandler(logging.Handler):
    """Forwards Ultralytics' own log records into the GUI log pane.

    Ultralytics re-prints the epoch table header every epoch, dumps the full
    ~80-key args dict at startup, and prints one raw validation row per epoch
    — all noise on top of the charts and the concise per-epoch summary line
    emitted by the training callback, so those records are dropped here.
    """

    _EPOCH_HEADER_RE = re.compile(r"^\s*Epoch\s+GPU_mem\s+box_loss")
    _VAL_ROW_RE = re.compile(r"^\s*all\s+\d+\s+\d+(\s|$)")
    _ARGS_DUMP_RE = re.compile(r"^engine[/\\]trainer:")

    def __init__(self, sink: EventSink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = clean_log_text(self.format(record)).strip()
            if (
                self._EPOCH_HEADER_RE.match(text)
                or self._VAL_ROW_RE.match(text)
                or self._ARGS_DUMP_RE.match(text)
            ):
                return
            # Ultralytics logs warnings at INFO level with a "WARNING" prefix.
            if record.levelno >= logging.ERROR or text.startswith("ERROR"):
                level = "error"
            elif record.levelno >= logging.WARNING or text.startswith("WARNING"):
                level = "warning"
            else:
                level = "info"
            self._sink.log(text, level)
        except Exception:
            pass


def _resolve_weights(name: str, sink: EventSink) -> str:
    """Return a local path for a weight name, downloading on first use."""
    path = Path(name).expanduser()
    if path.exists():
        return str(path)
    if name not in _KNOWN_WEIGHTS:
        return name
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    cached = WEIGHTS_CACHE / name
    if cached.exists():
        return str(cached)
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        sink.log(f"正在下载预训练权重 {name} ...")
        return str(attempt_download_asset(name, dir=WEIGHTS_CACHE))
    except Exception as exc:  # offline / API change: let ultralytics try itself
        sink.log(f"预训练权重下载失败（{exc}），交由 Ultralytics 自行处理", "warning")
        return name


@register_trainer
class YoloTrainer(BaseTrainer):
    key = "yolo"
    display_name = "YOLO (Ultralytics)"
    description = "基于 Ultralytics 的检测 / 分割 / 分类训练，自动适配 CUDA / MPS / CPU 加速"
    tasks = (
        TaskSpec("detect", "目标检测", "bbox", "2d", "矩形框标注，输出检测框"),
        TaskSpec("segment", "实例分割", "polygon", "2d", "多边形标注，输出掩膜轮廓"),
        TaskSpec("classify", "图像分类", "classify", "2d", "整图一个类别，取首个标签"),
    )

    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("ultralytics") is not None

    def param_specs(self, task_key: str) -> list[ParamSpec]:
        return [
            ParamSpec(
                key="model",
                label="预训练权重",
                type="choice",
                default="yolo11n.pt",
                group="模型",
                choices=_KNOWN_WEIGHTS,
                allow_custom=True,
                help="首次使用会下载到 ~/.cache/trainhub/weights，也可直接填本地 .pt 路径",
            ),
            ParamSpec(
                key="pretrained",
                label="加载预训练参数",
                type="bool",
                default=True,
                group="模型",
            ),
            ParamSpec(
                key="epochs",
                label="训练轮数",
                type="int",
                default=100,
                group="训练",
                minimum=1,
                maximum=100_000,
            ),
            ParamSpec(
                key="resume",
                label="断点续训",
                type="bool",
                default=False,
                group="训练",
                help="从上次中断的 last.pt 继续（运行名称需填中断那次训练的名字，"
                "沿用中断时的全部参数，其余表单项被忽略）",
            ),
            ParamSpec(
                key="imgsz",
                label="输入尺寸",
                type="int",
                default=640,
                group="训练",
                minimum=32,
                maximum=4096,
                step=32,
                help="训练时图像缩放的短边长度",
            ),
            ParamSpec(
                key="batch",
                label="批大小",
                type="int",
                default=16,
                group="训练",
                minimum=-1,
                maximum=512,
                help="-1 表示自动估算（显存/内存不足时优先调小）",
            ),
            ParamSpec(
                key="lr0",
                label="初始学习率",
                type="float",
                default=0.01,
                group="训练",
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                decimals=5,
            ),
            ParamSpec(
                key="optimizer",
                label="优化器",
                type="choice",
                default="auto",
                group="训练",
                choices=("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam"),
            ),
            ParamSpec(
                key="weight_decay",
                label="权重衰减",
                type="float",
                default=0.0005,
                group="训练",
                minimum=0.0,
                maximum=0.1,
                step=0.0001,
                decimals=5,
            ),
            ParamSpec(
                key="patience",
                label="早停耐心值",
                type="int",
                default=50,
                group="训练",
                minimum=0,
                maximum=10_000,
                help="验证指标连续 N 轮不提升就停止，0 表示关闭早停",
            ),
            ParamSpec(
                key="cos_lr",
                label="余弦学习率调度",
                type="bool",
                default=False,
                group="训练",
            ),
            ParamSpec(
                key="warmup_epochs",
                label="预热轮数",
                type="float",
                default=3.0,
                group="训练",
                minimum=0.0,
                maximum=50.0,
                step=0.5,
                decimals=1,
            ),
            ParamSpec(
                key="close_mosaic",
                label="关闭 Mosaic 的轮数",
                type="int",
                default=10,
                group="训练",
                minimum=0,
                maximum=1000,
                help="最后 N 轮关闭 Mosaic 增强，通常能提升精度",
            ),
            ParamSpec(
                key="device",
                label="计算设备",
                type="choice",
                default="auto",
                group="性能",
                choices=DEVICE_CHOICES,
                help=DEVICE_HELP,
            ),
            ParamSpec(
                key="workers",
                label="数据加载进程数",
                type="int",
                default=8,
                group="性能",
                minimum=0,
                maximum=64,
            ),
            ParamSpec(
                key="cache",
                label="缓存图像到内存",
                type="bool",
                default=False,
                group="性能",
                help="小数据集可开启，显著加快每个 epoch",
            ),
            ParamSpec(
                key="amp",
                label="混合精度 (AMP)",
                type="bool",
                default=False,
                group="性能",
                help="NVIDIA GPU（CUDA）上建议开启；MPS 支持不完整，Apple 机器建议关闭",
            ),
            ParamSpec(
                key="seed",
                label="训练随机种子",
                type="int",
                default=0,
                group="性能",
                minimum=0,
                maximum=100_000,
            ),
        ]

    # ------------------------------------------------------------------ prep
    def prepare(self, job: TrainJob, sink: EventSink) -> PreparedData:
        project = job.project
        sink.progress("扫描数据集", 0, 1, "正在读取标注")
        items = [
            item
            for item in scan_dataset(project.images_dir, project.annotations_dir)
            if item.width > 0 and item.height > 0
        ]
        if not items:
            raise RuntimeError(
                f"在 {project.images_dir} 中没有找到图像，请先导入并标注数据"
            )
        sink.log(f"共扫描到 {len(items)} 张图像")

        annotated = sum(1 for item in items if item.shapes)
        if annotated == 0:
            raise RuntimeError("所有图像都还没有标注，无法开始训练")
        if annotated < len(items):
            sink.log(f"其中 {annotated} 张有标注，其余作为背景图参与训练", "warning")

        out_dir = job.run_dir / "dataset"
        sink.progress("转换数据集", 0, len(items), "正在生成 YOLO 格式标注")
        exported = export_yolo_dataset(
            items,
            task=job.task_key,
            classes=project.labels,
            out_dir=out_dir,
            val_ratio=job.params.get("val_ratio", 0.2),
            seed=job.params.get("split_seed", 42),
        )
        for message in exported.warnings:
            sink.log(message, "warning")
        if exported.skipped:
            sink.log(f"{exported.skipped} 个形状无法转换，已跳过", "warning")
        sink.log(
            f"数据集已生成：训练 {exported.train_count} 张 / 验证 {exported.val_count} 张，"
            f"类别 {exported.classes}"
        )
        sink.artifact(
            str(exported.dataset_dir),
            "dataset",
            f"数据集切分（训练 {exported.train_count} / 验证 {exported.val_count}，双击打开）",
        )
        sink.progress("转换数据集", len(items), len(items), "完成")

        return PreparedData(
            dataset_dir=exported.dataset_dir,
            train_count=exported.train_count,
            val_count=exported.val_count,
            meta={"classes": exported.classes, "data_yaml": str(exported.data_yaml or "")},
        )

    # ----------------------------------------------------------------- train
    def train(
        self,
        job: TrainJob,
        prepared: PreparedData,
        sink: EventSink,
    ) -> TrainResult:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 ultralytics，请在 trainhub 环境中执行 "
                "`pip install ultralytics`"
            ) from exc

        params = job.params
        classes: list[str] = list(prepared.meta.get("classes") or [])
        task = job.task_key

        weights = _resolve_weights(str(params.get("model", "yolo11n.pt")), sink)
        sink.log(f"使用权重: {weights}")

        handler = _SinkLogHandler(sink)
        handler.setFormatter(logging.Formatter("%(message)s"))
        ul_logger = logging.getLogger("ultralytics")
        ul_logger.addHandler(handler)

        epochs = int(params.get("epochs", 100))
        device, device_warning = validate_device(str(params.get("device", "auto")))
        if device_warning:
            sink.log(device_warning, "warning")
        sink.log(f"计算设备: {device}")

        kwargs = {
            "data": prepared.meta.get("data_yaml") or str(prepared.dataset_dir),
            "epochs": epochs,
            "imgsz": int(params.get("imgsz", 640)),
            "batch": int(params.get("batch", 16)),
            "lr0": float(params.get("lr0", 0.01)),
            "optimizer": str(params.get("optimizer", "auto")),
            "weight_decay": float(params.get("weight_decay", 0.0005)),
            "patience": int(params.get("patience", 50)),
            "cos_lr": bool(params.get("cos_lr", False)),
            "warmup_epochs": float(params.get("warmup_epochs", 3.0)),
            "close_mosaic": int(params.get("close_mosaic", 10)),
            "workers": int(params.get("workers", 8)),
            "cache": bool(params.get("cache", False)),
            "amp": bool(params.get("amp", False)),
            "seed": int(params.get("seed", 0)),
            "pretrained": bool(params.get("pretrained", True)),
            "project": str(job.run_dir.parent),
            "name": job.run_dir.name,
            "exist_ok": True,
            "verbose": True,
            "plots": True,
            "device": device,
        }

        resume = bool(params.get("resume", False))
        if resume:
            last = job.run_dir / "weights" / "last.pt"
            if not last.exists():
                raise RuntimeError(
                    f"运行目录 {job.run_dir} 下没有 weights/last.pt，无法续训"
                    "（运行名称需与中断那次训练一致）"
                )
            sink.log(f"断点续训：从 {last} 继续（沿用中断时的全部参数）")
            model = YOLO(str(last))
        else:
            model = YOLO(weights)

        state = {"best": {}}

        def on_fit_epoch_end(trainer) -> None:
            try:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                total = int(getattr(trainer, "epochs", epochs) or epochs)
                if epoch > total:
                    # final_eval() re-fires this callback with epoch temporarily
                    # bumped past the budget; that row would be a phantom epoch.
                    return
                metrics: dict[str, float] = {}

                total_loss = getattr(trainer, "tloss", None)
                if total_loss is not None:
                    try:
                        metrics.update(trainer.label_loss_items(total_loss, prefix="train"))
                    except Exception:
                        pass

                # `trainer.metrics` is a plain dict in some versions and a
                # DetMetrics/SegmentMetrics/ClassifyMetrics object in others;
                # results_dict is the common flat view.
                raw = getattr(trainer, "metrics", None)
                results = getattr(raw, "results_dict", None)
                if not isinstance(results, dict):
                    results = raw if isinstance(raw, dict) else {}
                for key, value in results.items():
                    if not isinstance(value, int | float):
                        continue
                    key = str(key)
                    # "fitness" is re-added under a stable name below, and the
                    # per-group lr keys are collapsed into a single series.
                    if key == "fitness" or key.startswith("lr/"):
                        continue
                    # Keys arrive already namespaced ("metrics/mAP50(B)",
                    # "val/box_loss"); anything bare needs a val/ prefix.
                    metrics[key if "/" in key else f"val/{key}"] = float(value)

                fitness = getattr(trainer, "fitness", None)
                if isinstance(fitness, int | float):
                    metrics["val/fitness"] = float(fitness)

                lr = getattr(trainer, "lr", None)
                if isinstance(lr, dict) and lr:
                    first = next(iter(lr.values()))
                    if isinstance(first, int | float):
                        metrics["lr"] = float(first)

                if metrics:
                    record.write(
                        MetricEvent(
                            epoch=epoch,
                            total_epochs=total,
                            metrics={k: float(v) for k, v in metrics.items()},
                            phase="epoch",
                        )
                    )
                    sink.metric(epoch, metrics, total_epochs=total, phase="epoch")

                    if epoch != state.get("last_summary_epoch"):
                        state["last_summary_epoch"] = epoch
                        summary = _epoch_summary(epoch, total, metrics)
                        if summary:
                            sink.log(summary)

                if len(state["best"]) == 0 and metrics:
                    state["best"] = dict(metrics)

                if sink.cancelled():
                    sink.log("收到停止请求，将在本轮结束后中断训练", "warning")
                    trainer.stop_training = True
            except Exception:
                # Never let metric reporting abort training.
                pass

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        run_dir = job.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        with MetricRecorder(run_dir) as record:
            sink.progress("训练", 0, epochs, "断点续训" if resume else "开始训练")
            try:
                if resume:
                    # Ultralytics 的 resume 完全沿用断点里保存的参数与输出目录
                    model.train(resume=True)
                else:
                    model.train(**kwargs)
            except Exception as exc:
                if sink.cancelled():
                    sink.log(f"训练已中止: {exc}", "warning")
                else:
                    raise
            finally:
                ul_logger.removeHandler(handler)

        weights_dir = run_dir / "weights"
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"

        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            # Older/newer ultralytics may nest output under a task subfolder.
            for candidate in run_dir.rglob("results.csv"):
                results_csv = candidate
                break

        best_metrics: dict[str, float] = {}
        if results_csv.exists():
            best_metrics = _best_row(results_csv)
            sink.artifact(str(results_csv), "csv", "训练指标 (results.csv)")
        for png in sorted(run_dir.rglob("*.png")):
            sink.artifact(str(png), "plot", png.stem)
        for weight in (best, last):
            if weight.exists():
                sink.artifact(str(weight), "weight", weight.name)

        sink.log("训练完成")
        return TrainResult(
            run_dir=run_dir,
            best_checkpoint=best if best.exists() else None,
            last_checkpoint=last if last.exists() else None,
            best_metrics=best_metrics,
            artifacts=sorted(run_dir.rglob("*.*")),
        )


def _best_row(results_csv: Path) -> dict[str, float]:
    """Pick the row with the best fitness/mAP as a summary of the run."""
    import csv

    try:
        with open(results_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    if not rows:
        return {}

    def clean(row: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in row.items():
            if key is None:
                continue
            try:
                out[key.strip()] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    key = None
    for candidate in ("fitness", "metrics/mAP50-95(B)", "metrics/accuracy_top1"):
        if candidate in rows[0]:
            key = candidate
            break
    if key is None:
        return clean(rows[-1])
    best = max(rows, key=lambda row: _as_float(row.get(key)))
    return clean(best)


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")


def clear_weights_cache() -> None:
    if WEIGHTS_CACHE.exists():
        shutil.rmtree(WEIGHTS_CACHE, ignore_errors=True)
