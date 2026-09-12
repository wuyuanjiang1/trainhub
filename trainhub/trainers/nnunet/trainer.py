"""nnU-Net trainer plugin (2D semantic segmentation).

nnU-Net is driven through its CLI.  Every project gets its own
``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results`` tree so datasets
never collide between projects, and the trainer streams the CLI output to parse
per-epoch loss / pseudo-Dice into the live charts.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
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
from .converter import export_nnunet_dataset

# nnU-Net selects the epoch budget by trainer class; only these names exist.
EPOCH_TRAINERS: dict[int, str] = {
    1: "nnUNetTrainer_1epoch",
    5: "nnUNetTrainer_5epochs",
    10: "nnUNetTrainer_10epochs",
    20: "nnUNetTrainer_20epochs",
    50: "nnUNetTrainer_50epochs",
    100: "nnUNetTrainer_100epochs",
    250: "nnUNetTrainer_250epochs",
    500: "nnUNetTrainer_500epochs",
    750: "nnUNetTrainer_750epochs",
    1000: "nnUNetTrainer",
    2000: "nnUNetTrainer_2000epochs",
    4000: "nnUNetTrainer_4000epochs",
    8000: "nnUNetTrainer_8000epochs",
}

_EPOCH_RE = re.compile(r"^Epoch (\d+)$")
_TRAIN_LOSS_RE = re.compile(r"^train_loss\s+(.+)$")
_VAL_LOSS_RE = re.compile(r"^val_loss\s+(.+)$")
_DICE_RE = re.compile(r"^Pseudo dice\s+\[(.+)\]$")
_LR_RE = re.compile(r"Current learning rate:\s*([-+0-9.eE]+)")
_BEST_EMA_RE = re.compile(r"New best EMA pseudo Dice:\s*([-+0-9.eE]+)")
_MEAN_DICE_RE = re.compile(r"Mean Validation Dice:\s*([-+0-9.eE]+)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# nnU-Net's print_to_log_file prepends "2026-09-12 20:00:00.123456:" to stdout.
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?:\s*")
# numpy>=2 renders scalars as np.float64(0.12); unwrap before reading numbers.
_NP_SCALAR_RE = re.compile(r"\bnp\.\w+\(([^()]*)\)")

_ENTRY_POINTS = {
    "nnUNetv2_plan_and_preprocess": (
        "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "plan_and_preprocess_entry",
    ),
    "nnUNetv2_train": ("nnunetv2.run.run_training", "run_training_entry"),
}


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _command(name: str) -> list[str]:
    """Prefer the console script next to the running interpreter."""
    script = Path(sys.executable).parent / name
    if script.exists():
        return [str(script)]
    module, func = _ENTRY_POINTS[name]
    return [sys.executable, "-c", f"from {module} import {func}; {func}()"]


def _nnunet_env(raw: Path, preprocessed: Path, results: Path, *, compile_model: bool) -> dict:
    env = dict(os.environ)
    env["nnUNet_raw"] = str(raw)
    env["nnUNet_preprocessed"] = str(preprocessed)
    env["nnUNet_results"] = str(results)
    env["PYTHONUNBUFFERED"] = "1"
    # Some ops are still unimplemented on MPS; let them fall back to CPU.
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["nnUNet_compile"] = "true" if compile_model else "false"
    return env


class _LogParser:
    """Turns nnU-Net's CLI chatter into per-epoch metric rows."""

    def __init__(self, sink: EventSink, record: MetricRecorder, total_epochs: int) -> None:
        self._sink = sink
        self._record = record
        self._total_epochs = total_epochs
        self._epoch = -1
        self._pending: dict[str, float] = {}

    def feed(self, line: str) -> None:
        line = _TIMESTAMP_RE.sub("", line.strip())

        match = _EPOCH_RE.match(line)
        if match:
            self._epoch = int(match.group(1)) + 1
            return

        for regex, key in (
            (_TRAIN_LOSS_RE, "train/loss"),
            (_VAL_LOSS_RE, "val/loss"),
        ):
            match = regex.match(line)
            if match:
                number = _NUMBER_RE.search(match.group(1))
                if number:
                    self._pending[key] = float(number.group(0))
                return

        match = _LR_RE.search(line)
        if match:
            self._pending["lr"] = float(match.group(1))
            return

        match = _BEST_EMA_RE.search(line)
        if match:
            # Printed after the epoch row, so report it on its own instead of
            # carrying it over into the next epoch.
            self._emit(max(self._epoch, 1), {"val/best_ema_dice": float(match.group(1))})
            return

        match = _MEAN_DICE_RE.search(line)
        if match:
            self._emit(
                max(self._epoch, 1),
                {"val/final_dice": float(match.group(1))},
                phase="validation",
            )
            return

        match = _DICE_RE.match(line)
        if match:
            # Unwrap np.float64(...) first, otherwise "float64" leaks a bogus 64.
            payload = _NP_SCALAR_RE.sub(r"\1", match.group(1))
            values = [float(v) for v in _NUMBER_RE.findall(payload)]
            if values:
                self._pending["val/mean_dice"] = sum(values) / len(values)
                if len(values) > 1:
                    for index, value in enumerate(values, start=1):
                        self._pending[f"val/dice_{index}"] = value
            self._flush()
            return

    def _emit(self, epoch: int, metrics: dict[str, float], phase: str = "epoch") -> None:
        self._record.write(
            MetricEvent(
                epoch=epoch,
                total_epochs=self._total_epochs,
                metrics=metrics,
                phase=phase,
            )
        )
        self._sink.metric(epoch, metrics, total_epochs=self._total_epochs, phase=phase)

    def _flush(self) -> None:
        if not self._pending:
            return
        metrics = dict(self._pending)
        self._pending.clear()
        self._emit(max(self._epoch, 1), metrics)


def _run(
    command: list[str],
    env: dict,
    sink: EventSink,
    parser: _LogParser | None = None,
    phase: str = "",
    total: int = 0,
) -> int:
    """Stream a subprocess' combined output, optionally parsing metrics."""
    sink.log(f"$ {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(Path(env["nnUNet_results"]).parent),
    )
    assert process.stdout is not None

    step = 0
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if line:
            sink.log(line)
            if parser is not None:
                parser.feed(line)
        step += 1
        if sink.cancelled():
            sink.log("收到停止请求，正在终止 nnU-Net 进程 ...", "warning")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        if total and step % 20 == 0:
            sink.progress(phase, min(step, total), total, phase)
    return process.wait()


@register_trainer
class NnunetTrainer(BaseTrainer):
    key = "nnunet"
    display_name = "nnU-Net"
    description = "医学影像分割，自动决定网络结构与预处理；使用 2D PNG 数据集"
    tasks = (
        TaskSpec(
            "semantic_seg",
            "语义分割 (2D)",
            "polygon",
            "2d",
            "多边形/笔刷标注，导出多类别分割掩膜",
        ),
    )

    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("nnunetv2") is not None

    def param_specs(self, task_key: str) -> list[ParamSpec]:
        return [
            ParamSpec(
                key="dataset_id",
                label="数据集编号",
                type="int",
                default=1,
                group="数据集",
                minimum=1,
                maximum=999,
                help="nnUNet 要求 1-999，同一项目内不要与其他数据集重复",
            ),
            ParamSpec(
                key="dataset_name",
                label="数据集名称",
                type="str",
                default="",
                group="数据集",
                help="留空则使用项目名称",
            ),
            ParamSpec(
                key="epochs",
                label="训练轮数",
                type="choice",
                default="250",
                group="训练",
                choices=tuple(str(n) for n in EPOCH_TRAINERS),
                help="nnUNet 的轮数由训练器类决定，只能从这些预设值中选择",
            ),
            ParamSpec(
                key="fold",
                label="交叉验证折 (fold)",
                type="choice",
                default="0",
                group="训练",
                choices=("0", "1", "2", "3", "4"),
                help="nnUNet 内部做 5 折交叉验证，验证曲线来自这里的折",
            ),
            ParamSpec(
                key="device",
                label="计算设备",
                type="choice",
                default="auto",
                group="训练",
                choices=("auto", "mps", "cpu"),
                help="auto 会优先选 mps，其次 cuda，最后 cpu",
            ),
            ParamSpec(
                key="num_processes",
                label="预处理进程数",
                type="int",
                default=8,
                group="训练",
                minimum=1,
                maximum=32,
            ),
            ParamSpec(
                key="verify_integrity",
                label="校验数据集完整性",
                type="bool",
                default=True,
                group="训练",
                help="首次处理时建议开启，能提前暴露标注问题",
            ),
            ParamSpec(
                key="export_npz",
                label="保存验证集预测概率",
                type="bool",
                default=True,
                group="训练",
                help="训练后保存 softmax 概率，便于做集成与后处理",
            ),
            ParamSpec(
                key="run_validation",
                label="训练后自动验证",
                type="bool",
                default=True,
                group="训练",
                help="额外跑一次 --val，得到最终 Mean Validation Dice",
            ),
            ParamSpec(
                key="continue_training",
                label="续训 (--c)",
                type="bool",
                default=False,
                group="训练",
                help="从该折已有的 checkpoint_final.pth 继续训练",
            ),
            ParamSpec(
                key="torch_compile",
                label="启用 torch.compile",
                type="bool",
                default=False,
                group="性能",
                help="MPS 上不支持，会自动跳过",
            ),
        ]

    # ------------------------------------------------------------------ prep
    def prepare(self, job: TrainJob, sink: EventSink) -> PreparedData:
        project = job.project
        dataset_name = str(job.params.get("dataset_name") or "").strip() or project.name
        dataset_id = int(job.params.get("dataset_id", 1))

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
        if not any(item.shapes for item in items):
            raise RuntimeError("所有图像都还没有标注，无法开始训练")

        nnunet_root = project.root / "nnunet"
        raw_dir = nnunet_root / "raw"
        preprocessed = nnunet_root / "preprocessed"
        results = nnunet_root / "results"
        holdout = nnunet_root / "holdout"
        for directory in (raw_dir, preprocessed, results):
            directory.mkdir(parents=True, exist_ok=True)

        sink.progress("转换数据集", 0, len(items), "正在生成 nnUNet 数据集")
        exported = export_nnunet_dataset(
            items,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            classes=project.labels,
            raw_dir=raw_dir,
            holdout_dir=holdout,
            val_ratio=job.params.get("val_ratio", 0.2),
            seed=job.params.get("split_seed", 42),
        )
        for message in exported.warnings[:10]:
            sink.log(message, "warning")
        if len(exported.warnings) > 10:
            sink.log(f"... 另有 {len(exported.warnings) - 10} 条类似提示", "warning")
        if exported.skipped:
            sink.log(f"{exported.skipped} 个形状无法转换，已跳过", "warning")

        sink.log(
            f"数据集 {exported.dataset_dir.name}：训练 {exported.train_count} 例，"
            f"留出测试 {exported.holdout_count} 例，通道 {list(exported.channels.values())}"
        )
        sink.log(
            "说明：nnUNet 在训练集内部自行做 5 折交叉验证，验证曲线来自所选折；"
            "留出测试集不参与训练，仅供后续评估"
        )
        sink.progress("转换数据集", len(items), len(items), "完成")

        return PreparedData(
            dataset_dir=exported.dataset_dir,
            train_count=exported.train_count,
            val_count=exported.holdout_count,
            meta={
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_json": str(exported.dataset_json),
                "labels": exported.labels,
                "channels": exported.channels,
                "nnunet_root": str(nnunet_root),
                "preprocessed": str(preprocessed),
                "results": str(results),
            },
        )

    # ----------------------------------------------------------------- train
    def train(
        self,
        job: TrainJob,
        prepared: PreparedData,
        sink: EventSink,
    ) -> TrainResult:
        meta = prepared.meta
        dataset_id = int(meta["dataset_id"])
        configuration = "2d"
        fold = str(job.params.get("fold", "0"))
        epochs = int(job.params.get("epochs", 250))
        trainer_name = EPOCH_TRAINERS.get(epochs, "nnUNetTrainer")
        device = _resolve_device(str(job.params.get("device", "auto")))
        raw = Path(meta["nnunet_root"]) / "raw"
        preprocessed = Path(meta["preprocessed"])
        results = Path(meta["results"])

        env = _nnunet_env(
            raw,
            preprocessed,
            results,
            compile_model=bool(job.params.get("torch_compile", False)),
        )

        sink.log(
            f"nnUNet 训练：数据集 {dataset_id}，配置 {configuration}，fold {fold}，"
            f"轮数 {epochs}（{trainer_name}），设备 {device}"
        )
        if device == "mps":
            sink.log(
                "MPS 提示：nnUNet 并非为 Apple GPU 调优，显存/算子受限时请改用 cpu",
                "warning",
            )

        job.run_dir.mkdir(parents=True, exist_ok=True)

        with MetricRecorder(job.run_dir) as record:
            parser = _LogParser(sink, record, epochs)

            plan_cmd = _command("nnUNetv2_plan_and_preprocess") + [
                "-d",
                str(dataset_id),
                "-c",
                configuration,
                "-np",
                str(int(job.params.get("num_processes", 8))),
            ]
            if job.params.get("verify_integrity", True):
                plan_cmd.append("--verify_dataset_integrity")
            sink.progress("预处理", 0, 0, "正在生成计划并预处理")
            code = _run(plan_cmd, env, sink, parser=None, phase="预处理")
            if code != 0:
                raise RuntimeError(
                    "nnUNet 预处理失败，请查看上方日志（常见原因：标注类别数不一致、图像尺寸异常）"
                )
            sink.log("预处理完成")

            train_cmd = _command("nnUNetv2_train") + [
                str(dataset_id),
                configuration,
                fold,
                "-tr",
                trainer_name,
                "-device",
                device,
            ]
            if job.params.get("export_npz", True):
                train_cmd.append("--npz")
            if job.params.get("continue_training", False):
                train_cmd.append("--c")

            sink.progress("训练", 0, epochs, "开始训练")
            code = _run(train_cmd, env, sink, parser, phase="训练", total=epochs)
            if code != 0 and not sink.cancelled():
                raise RuntimeError("nnUNet 训练失败，请查看上方日志")

            if job.params.get("run_validation", True) and not sink.cancelled():
                val_cmd = _command("nnUNetv2_train") + [
                    str(dataset_id),
                    configuration,
                    fold,
                    "-tr",
                    trainer_name,
                    "-device",
                    device,
                    "--val",
                ]
                sink.progress("验证", 0, 1, "正在运行最终验证")
                _run(val_cmd, env, sink, parser, phase="验证")
                sink.progress("验证", 1, 1, "验证完成")

        output_dir = results / prepared.dataset_dir.name / f"{trainer_name}__nnUNetPlans__{configuration}"
        fold_dir = output_dir / f"fold_{fold}"
        best = fold_dir / "checkpoint_best.pth"
        last = fold_dir / "checkpoint_final.pth"

        for pattern, kind in (
            ("**/checkpoint_best.pth", "weight"),
            ("**/checkpoint_final.pth", "weight"),
            ("**/progress.png", "plot"),
            ("**/training_log_*.txt", "log"),
            ("**/validation/summary.json", "json"),
        ):
            for path in sorted(results.rglob(pattern)):
                sink.artifact(str(path), kind, path.name)

        sink.log("nnUNet 训练完成")
        return TrainResult(
            run_dir=job.run_dir,
            best_checkpoint=best if best.exists() else None,
            last_checkpoint=last if last.exists() else None,
            best_metrics=_summary_metrics(fold_dir),
            artifacts=sorted(p for p in results.rglob("*") if p.is_file())[:200],
        )


def _summary_metrics(fold_dir: Path) -> dict[str, float]:
    """Mean foreground Dice from nnU-Net's validation summary, if available."""
    import json

    summary = fold_dir / "validation" / "summary.json"
    if not summary.exists():
        return {}
    try:
        with open(summary, encoding="utf-8") as f:
            data = json.load(f)
        dice = data.get("foreground_mean", {}).get("Dice")
        return {"val/final_dice": float(dice)} if dice is not None else {}
    except (OSError, ValueError, TypeError):
        return {}
