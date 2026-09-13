"""AI 预标注对话框：本地 YOLO / 大模型双引擎，后台线程推理，进度可视化。

推理跑在 QThread 里，界面不卡顿；单图模式把候选 shape 发回画布，批量模式
直接写标注 JSON（与手动保存同格式），逐张回报进度。大模型引擎走 OpenAI
兼容接口，服务商预设见 :mod:`trainhub.core.prelabel_vlm`；API Key 存在本机
QSettings，绝不写入项目目录（trainhub.yaml 会被整目录拷贝分享）。
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
from ..core.prelabel_vlm import ENV_KEYS
from ..core.prelabel_vlm import VLM_PROVIDERS
from ..core.prelabel_vlm import VLMProvider
from ..core.prelabel_vlm import provider_env_key
from ..core.prelabel_vlm import test_connection

_SETTINGS_ORG = "trainhub"
_SETTINGS_APP = "trainhub"

_DEVICE_LABELS = {
    "auto": "自动（按平台选择，优先 CUDA）",
    "cuda": "CUDA（NVIDIA 显卡）",
    "mps": "MPS（Apple 芯片）",
    "cpu": "CPU",
}


def _saved_api_key(provider_key: str) -> str:
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return str(settings.value(f"vlm/api_key/{provider_key}", "") or "")


def _store_api_key(provider_key: str, api_key: str) -> None:
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    settings.setValue(f"vlm/api_key/{provider_key}", api_key)


class ConnectionTestWorker(QtCore.QThread):
    """发一张内置小图验证 Key、端点与视觉能力。"""

    ok = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        provider: VLMProvider,
        api_key: str,
        model: str,
        base_url: str,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def run(self) -> None:
        try:
            test_connection(
                provider=self._provider,
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.ok.emit(f"连接成功：{self._provider.display_name} 可用")


class PrelabelWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)  # 已处理 / 总数 / 当前文件名
    image_ready = QtCore.pyqtSignal(str, list)  # 单图模式：图像路径 + shape 字典
    file_done = QtCore.pyqtSignal(str, int, list)  # 批量模式：路径 / 形状数 / 标签
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        jobs: list[tuple[Path, Path]],
        predict_fn,  # callable[[Path], list[dict]]
        *,
        write_to_disk: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._predict = predict_fn
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
                shapes = self._predict(image_path)
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
        except Exception as exc:  # 模型加载失败 / 权重损坏 / API 调用失败等
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
    start_requested = QtCore.pyqtSignal(object)  # 参数字典，含 engine 与引擎各自的字段

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
        self.setMinimumWidth(480)
        self._job_provider = job_provider
        self.worker: PrelabelWorker | None = None
        self._test_worker: ConnectionTestWorker | None = None
        self._current_image = current_image
        self._unlabeled_count = unlabeled_count

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # ------------------------------------------------------------ 引擎
        engine_row = QtWidgets.QHBoxLayout()
        self._yolo_engine_radio = QtWidgets.QRadioButton("本地 YOLO 权重（离线）")
        self._vlm_engine_radio = QtWidgets.QRadioButton("大模型 API（在线）")
        if weight_options:
            self._yolo_engine_radio.setChecked(True)
        else:
            self._vlm_engine_radio.setChecked(True)
        self._yolo_engine_radio.toggled.connect(self._on_engine_changed)
        engine_row.addWidget(self._yolo_engine_radio)
        engine_row.addWidget(self._vlm_engine_radio)
        engine_row.addStretch(1)
        layout.addLayout(engine_row)

        # ------------------------------------------------------ YOLO 分组
        self._yolo_group = QtWidgets.QGroupBox("YOLO 引擎")
        yolo_form = QtWidgets.QFormLayout(self._yolo_group)
        yolo_form.setHorizontalSpacing(12)
        yolo_form.setVerticalSpacing(9)

        self._weights_combo = QtWidgets.QComboBox()
        for label_text, data in weight_options:
            self._weights_combo.addItem(label_text, data)
        yolo_form.addRow("模型权重", self._weights_combo)

        self._conf_spin = QtWidgets.QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setToolTip(
            "置信度越低候选越多（含误检），越高越保守；预标注建议 0.25 左右"
        )
        yolo_form.addRow("置信度阈值", self._conf_spin)

        self._device_combo = QtWidgets.QComboBox()
        for choice in DEVICE_CHOICES:
            self._device_combo.addItem(_DEVICE_LABELS.get(choice, choice), choice)
        self._device_combo.setCurrentIndex(0)
        yolo_form.addRow("计算设备", self._device_combo)
        self._yolo_group.setVisible(self._yolo_engine_radio.isChecked())
        layout.addWidget(self._yolo_group)

        # ------------------------------------------------------ VLM 分组
        self._vlm_group = QtWidgets.QGroupBox("大模型引擎（OpenAI 兼容接口）")
        vlm_form = QtWidgets.QFormLayout(self._vlm_group)
        vlm_form.setHorizontalSpacing(12)
        vlm_form.setVerticalSpacing(9)

        self._provider_combo = QtWidgets.QComboBox()
        for provider in VLM_PROVIDERS.values():
            self._provider_combo.addItem(provider.display_name, provider.key)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        vlm_form.addRow("服务商", self._provider_combo)

        self._baseurl_edit = QtWidgets.QLineEdit()
        self._baseurl_edit.setToolTip("OpenAI 兼容接口地址，预设会自动填好，可按需修改")
        vlm_form.addRow("接口地址", self._baseurl_edit)

        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setToolTip("要调用的视觉模型名称")
        vlm_form.addRow("模型", self._model_combo)

        self._key_edit = QtWidgets.QLineEdit()
        self._key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("粘贴 API Key（保存在本机，不写入项目）")
        vlm_form.addRow("API Key", self._key_edit)

        self._hint_edit = QtWidgets.QLineEdit()
        self._hint_edit.setPlaceholderText("可选，如：只标完整可见的目标；忽略文字水印")
        vlm_form.addRow("补充要求", self._hint_edit)

        self._vlm_note = QtWidgets.QLabel()
        self._vlm_note.setWordWrap(True)
        vlm_form.addRow(self._vlm_note)

        test_row = QtWidgets.QHBoxLayout()
        self._test_button = QtWidgets.QPushButton("测试连接")
        self._test_button.clicked.connect(self._test_connection)
        test_row.addWidget(self._test_button)
        test_row.addStretch(1)
        vlm_form.addRow(test_row)
        self._vlm_group.setVisible(self._vlm_engine_radio.isChecked())
        layout.addWidget(self._vlm_group)

        # ------------------------------------------------------ 共用部分
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

        self._on_provider_changed()

    # ------------------------------------------------------------- engine
    def _on_engine_changed(self) -> None:
        yolo = self._yolo_engine_radio.isChecked()
        self._yolo_group.setVisible(yolo)
        self._vlm_group.setVisible(not yolo)

    def _current_provider(self) -> VLMProvider:
        key = self._provider_combo.currentData()
        return VLM_PROVIDERS.get(key, next(iter(VLM_PROVIDERS.values())))

    def _on_provider_changed(self) -> None:
        provider = self._current_provider()
        self._baseurl_edit.setText(provider.base_url)
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for model in provider.models:
            self._model_combo.addItem(model)
        if provider.default_model:
            self._model_combo.setCurrentText(provider.default_model)
        self._model_combo.blockSignals(False)
        if provider.models:
            self._model_combo.lineEdit().setPlaceholderText("")
        else:
            self._model_combo.lineEdit().setPlaceholderText("填写模型名称")
        key = _saved_api_key(provider.key) or provider_env_key(provider.key) or ""
        self._key_edit.setText(key)
        env_names = [
            name
            for name in ENV_KEYS.get(provider.key, ())
            if os.environ.get(name, "").strip()
        ]
        placeholder = "粘贴 API Key（保存在本机，不写入项目）"
        if env_names:
            placeholder = f"已从环境变量 {env_names[0]} 读取，也可粘贴其他 Key"
        self._key_edit.setPlaceholderText(placeholder)
        self._key_edit.setToolTip("API Key 保存在本机 QSettings 中，不会写入项目目录")
        notes = []
        if provider.grounding_verified:
            notes.append("✓ 该服务商官方支持检测框（grounding）输出。")
        else:
            notes.append("⚠ 该服务商未官方声明检测框输出能力，建议先对当前图像试标一张。")
        notes.append("批量模式会把图像内容上传到所选服务商。")
        self._vlm_note.setText(" ".join(notes))

    # ------------------------------------------------------------- actions
    def _test_connection(self) -> None:
        if self._test_worker is not None and self._test_worker.isRunning():
            return
        provider = self._current_provider()
        api_key = self._key_edit.text().strip()
        model = self._model_combo.currentText().strip()
        base_url = self._baseurl_edit.text().strip()
        if not api_key:
            QtWidgets.QMessageBox.warning(self, "测试连接", "请先填写 API Key。")
            return
        self._test_button.setEnabled(False)
        self._status.setText(f"正在测试 {provider.display_name} …")
        self._test_worker = ConnectionTestWorker(
            provider, api_key, model, base_url, parent=self
        )
        self._test_worker.ok.connect(self._on_test_ok)
        self._test_worker.failed.connect(self._on_test_failed)
        self._test_worker.finished.connect(
            lambda: self._test_button.setEnabled(True)
        )
        self._test_worker.start()

    def _on_test_ok(self, message: str) -> None:
        self._status.setText(message + " 可以开始预标注。")

    def _on_test_failed(self, message: str) -> None:
        self._status.setText("测试连接失败。")
        QtWidgets.QMessageBox.warning(self, "测试连接失败", message)

    def _on_start(self) -> None:
        scope = "batch" if self._batch_radio.isChecked() else "current"
        jobs = self._job_provider(scope)
        if not jobs:
            QtWidgets.QMessageBox.information(
                self, "AI 预标注", "该范围内没有需要处理的图像。"
            )
            return
        if self._vlm_engine_radio.isChecked():
            provider = self._current_provider()
            api_key = self._key_edit.text().strip()
            model = self._model_combo.currentText().strip()
            base_url = self._baseurl_edit.text().strip()
            if not api_key:
                QtWidgets.QMessageBox.warning(
                    self, "API Key", "请先填写 API Key，或点击「测试连接」验证。"
                )
                return
            if not base_url:
                QtWidgets.QMessageBox.warning(
                    self, "接口地址", "请填写服务商的 OpenAI 兼容接口地址。"
                )
                return
            if not model:
                QtWidgets.QMessageBox.warning(
                    self, "模型", "请填写要调用的视觉模型名称。"
                )
                return
            _store_api_key(provider.key, api_key)
            self.start_requested.emit(
                {
                    "engine": "vlm",
                    "provider": provider.key,
                    "api_key": api_key,
                    "model": model,
                    "base_url": base_url,
                    "hint": self._hint_edit.text().strip(),
                    "jobs": jobs,
                    "write": scope == "batch",
                }
            )
        else:
            device, warning = validate_device(self._device_combo.currentData())
            self.start_requested.emit(
                {
                    "engine": "yolo",
                    "weights": self._weights_combo.currentData()
                    or self._weights_combo.currentText(),
                    "conf": float(self._conf_spin.value()),
                    "device": device,
                    "device_warning": warning,
                    "jobs": jobs,
                    "write": scope == "batch",
                }
            )

    def begin_run(self, total: int, note: str) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.show()
        self._start_button.setEnabled(False)
        self._test_button.setEnabled(False)
        self._yolo_group.setEnabled(False)
        self._vlm_group.setEnabled(False)
        self._current_radio.setEnabled(False)
        self._batch_radio.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._status.setText(f"正在推理（{note}）…")

    def on_progress(self, done: int, total: int, name: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        if name:
            self._status.setText(f"({done}/{total}) {name}")

    def on_finished(self) -> None:
        self._cancel_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._test_button.setEnabled(True)
        self._yolo_group.setEnabled(True)
        self._vlm_group.setEnabled(True)
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
