"""AI 预标注对话框：选权重 / 置信度 / 范围，后台线程推理，进度可视化。

推理跑在 QThread 里，界面不卡顿；单图模式把候选 shape 发回画布，批量模式
直接写标注 JSON（与手动保存同格式），逐张回报进度。
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtWidgets

from ..annotator.label_file import write_label_file
from ..core.devices import DEVICE_CHOICES
from ..core.devices import validate_device
from ..core.prelabel import predict_shapes

_DEVICE_LABELS = {
    "auto": "自动（按平台选择，优先 CUDA）",
    "cuda": "CUDA（NVIDIA 显卡）",
    "mps": "MPS（Apple 芯片）",
    "cpu": "CPU",
}


class PrelabelWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)  # 已处理 / 总数 / 当前文件名
    image_ready = QtCore.pyqtSignal(str, list)  # 单图模式：图像路径 + shape 字典
    file_done = QtCore.pyqtSignal(str, int, list)  # 批量模式：路径 / 形状数 / 标签
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        jobs: list[tuple[Path, Path]],
        weights: str,
        conf: float,
        *,
        device: str,
        write_to_disk: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._weights = weights
        self._conf = conf
        self._device = device
        self._write = write_to_disk
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self._jobs)
        try:
            for index, (image_path, label_path) in enumerate(self._jobs):
                if self._cancelled:
                    return
                shapes = predict_shapes(
                    str(image_path), self._weights, self._conf, device=self._device
                )
                if self._write:
                    if shapes:
                        _write_prelabel(str(image_path), str(label_path), shapes)
                        self.file_done.emit(
                            str(image_path),
                            len(shapes),
                            sorted({s["label"] for s in shapes if s.get("label")}),
                        )
                else:
                    self.image_ready.emit(str(image_path), shapes)
                self.progress.emit(index + 1, total, Path(image_path).name)
        except Exception as exc:  # 模型加载失败 / 权重损坏等
            self.failed.emit(str(exc))


def _write_prelabel(image_path: str, label_path: str, shapes: list[dict]) -> None:
    from PIL import Image

    label_path_obj = Path(label_path)
    label_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        width, height = image.size
    write_label_file(
        filename=str(label_path_obj),
        shapes=shapes,
        image_path=os.path.relpath(image_path, label_path_obj.parent),
        image_height=height,
        image_width=width,
        flags={},
    )


class PrelabelDialog(QtWidgets.QDialog):
    start_requested = QtCore.pyqtSignal(object)  # dict(weights, conf, jobs, write)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        weight_options: list[tuple[str, str]],
        current_image: str | None,
        unlabeled_count: int,
        job_provider,  # callable[[str], list[tuple[Path, Path]]]
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI 预标注")
        self.setMinimumWidth(420)
        self._job_provider = job_provider
        self.worker: PrelabelWorker | None = None
        self._current_image = current_image
        self._unlabeled_count = unlabeled_count

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)

        self._weights_combo = QtWidgets.QComboBox()
        for label_text, data in weight_options:
            self._weights_combo.addItem(label_text, data)
        form.addRow("模型权重", self._weights_combo)

        self._conf_spin = QtWidgets.QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setToolTip(
            "置信度越低候选越多（含误检），越高越保守；预标注建议 0.25 左右"
        )
        form.addRow("置信度阈值", self._conf_spin)

        self._device_combo = QtWidgets.QComboBox()
        for choice in DEVICE_CHOICES:
            self._device_combo.addItem(_DEVICE_LABELS.get(choice, choice), choice)
        self._device_combo.setCurrentIndex(0)
        form.addRow("计算设备", self._device_combo)
        layout.addLayout(form)

        self._current_radio = QtWidgets.QRadioButton(
            "仅当前图像" + (f"（{Path(current_image).name}）" if current_image else "")
        )
        self._current_radio.setChecked(current_image is not None)
        self._current_radio.setEnabled(current_image is not None)
        self._batch_radio = QtWidgets.QRadioButton(
            f"全部未标注图像（{unlabeled_count} 张）"
        )
        self._batch_radio.setEnabled(unlabeled_count > 0)
        if current_image is None:
            self._batch_radio.setChecked(unlabeled_count > 0)
        layout.addWidget(self._current_radio)
        layout.addWidget(self._batch_radio)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._status = QtWidgets.QLabel(
            "推理在后台进行，结束后候选标注需人工检查微调再保存。"
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("开始预标注")
        self._start_button.setProperty("accent", True)
        self._start_button.clicked.connect(self._on_start)
        self._cancel_button = QtWidgets.QPushButton("取消任务")
        self._cancel_button.setProperty("danger", True)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_worker)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(self._start_button, 1)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------- actions
    def _on_start(self) -> None:
        scope = "batch" if self._batch_radio.isChecked() else "current"
        jobs = self._job_provider(scope)
        if not jobs:
            QtWidgets.QMessageBox.information(
                self, "AI 预标注", "该范围内没有需要处理的图像。"
            )
            return
        device, warning = validate_device(self._device_combo.currentData())
        self.start_requested.emit(
            {
                "weights": self._weights_combo.currentData()
                or self._weights_combo.currentText(),
                "conf": float(self._conf_spin.value()),
                "device": device,
                "device_warning": warning,
                "jobs": jobs,
                "write": scope == "batch",
            }
        )

    def begin_run(self, total: int, device: str, warning: str | None = None) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.show()
        self._start_button.setEnabled(False)
        self._weights_combo.setEnabled(False)
        self._conf_spin.setEnabled(False)
        self._device_combo.setEnabled(False)
        self._current_radio.setEnabled(False)
        self._batch_radio.setEnabled(False)
        self._cancel_button.setEnabled(True)
        note = f"计算设备 {device}" + (f"；{warning}" if warning else "")
        self._status.setText(f"正在加载模型并推理（{note}）…")

    def on_progress(self, done: int, total: int, name: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        if name:
            self._status.setText(f"({done}/{total}) {name}")

    def on_finished(self) -> None:
        self._cancel_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._weights_combo.setEnabled(True)
        self._conf_spin.setEnabled(True)
        self._device_combo.setEnabled(True)
        self._current_radio.setEnabled(self._current_image is not None)
        self._batch_radio.setEnabled(self._unlabeled_count > 0)

    def mark_done(self, message: str) -> None:
        self._status.setText(message)

    def _cancel_worker(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
        self._cancel_button.setEnabled(False)
        self._status.setText("正在停止（当前这张完成后中断）…")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
        super().closeEvent(event)
