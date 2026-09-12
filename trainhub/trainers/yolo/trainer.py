"""Ultralytics YOLO trainer plugin (detect / segment / classify)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ...core.dataset import scan_dataset
from ...core.events import EventSink
from ...core.events import MetricEvent
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


class _SinkLogHandler(logging.Handler):
    """Forwards Ultralytics' own log records into the GUI log pane."""

    def __init__(self, sink: EventSink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = "error" if record.levelno >= logging.ERROR else "info"
            if record.levelno >= logging.WARNING:
                level = "warning"
            self._sink.log(self.format(record), level)
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
    description = "基于 Ultralytics 的检测 / 分割 / 分类训练，支持 MPS 加速"
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
                choices=("auto", "mps", "cpu"),
                help="Apple Silicon 选 mps；显存不足或算子不支持时回退 cpu",
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
                help="MPS 后端对 AMP 支持不完整，建议保持关闭",
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
        device = params.get("device", "auto")
        device = None if device == "auto" else str(device)

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
        }
        if device is not None:
            kwargs["device"] = device

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
            sink.progress("训练", 0, epochs, "开始训练")
            try:
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
