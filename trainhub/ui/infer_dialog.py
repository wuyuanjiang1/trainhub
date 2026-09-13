"""推理预览：用项目训练产出的权重对新图跑一遍，画框结果另存可查。

轻量的"训练→应用"闭环：选择 runs/ 下的 .pt 权重，对项目图像（或任选
文件夹）逐张推理，画框结果存到该权重对应运行目录的 predictions/ 下，
列表点选即可在应用内预览，双击交给系统看图器。仅支持 YOLO 权重；
nnU-Net 的留出集评估暂未接入。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

from ..core.dataset import find_images


class InferWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    image_done = QtCore.pyqtSignal(str, str)  # 源图 / 结果图
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        weights: str,
        images: list[Path],
        out_dir: Path,
        conf: float,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._weights = weights
        self._images = images
        self._out_dir = out_dir
        self._conf = conf
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO(self._weights)
            total = len(self._images)
            for index, path in enumerate(self._images):
                if self._cancelled:
                    return
                result = model.predict(str(path), conf=self._conf, verbose=False)[0]
                plotted = result.plot()  # BGR ndarray，含框与类别标注
                out = self._out_dir / f"{path.stem}.jpg"
                Image.fromarray(plotted[:, :, ::-1]).save(out, quality=90)
                self.image_done.emit(str(path), str(out))
                self.progress.emit(index + 1, total, path.name)
        except Exception as exc:
            self.failed.emit(str(exc))


class InferDialog(QtWidgets.QDialog):
    start_requested = QtCore.pyqtSignal(object)  # dict(weights, conf, images, out_dir)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        weight_options: list[tuple[str, str]],
        images_dir: Path,
        out_dir: Path,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("推理预览")
        self.setMinimumWidth(640)
        self.worker: InferWorker | None = None
        self._images_dir = Path(images_dir)
        self._custom_dir: Path | None = None
        self._out_dir = Path(out_dir)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)

        self._weights_combo = QtWidgets.QComboBox()
        for label_text, data in weight_options:
            self._weights_combo.addItem(label_text, data)
        form.addRow("模型权重", self._weights_combo)

        self._conf_spin = QtWidgets.QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.25)
        form.addRow("置信度阈值", self._conf_spin)

        source_row = QtWidgets.QHBoxLayout()
        source_row.setSpacing(6)
        self._source_edit = QtWidgets.QLineEdit(str(self._images_dir))
        self._source_edit.setReadOnly(True)
        pick_button = QtWidgets.QPushButton("选择文件夹…")
        pick_button.clicked.connect(self._pick_folder)
        source_row.addWidget(self._source_edit, 1)
        source_row.addWidget(pick_button)
        form.addRow("推理来源", source_row)
        layout.addLayout(form)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._status = QtWidgets.QLabel(
            f"结果存到 {self._out_dir}，点选列表即可预览，双击交给系统看图器。"
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self._list = QtWidgets.QListWidget()
        self._list.setMinimumWidth(220)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._open_image)
        self._preview = QtWidgets.QLabel("（尚无结果）")
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(320, 280)
        split.addWidget(self._list)
        split.addWidget(self._preview)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        self._start_button = QtWidgets.QPushButton("开始推理")
        self._start_button.setProperty("accent", True)
        self._start_button.clicked.connect(self._on_start)
        self._cancel_button = QtWidgets.QPushButton("取消任务")
        self._cancel_button.setProperty("danger", True)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_worker)
        open_button = QtWidgets.QPushButton("打开结果目录")
        open_button.clicked.connect(self._open_out_dir)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(self._start_button, 1)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(open_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------- actions
    def _pick_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择推理图像目录")
        if folder:
            self._custom_dir = Path(folder)
            self._source_edit.setText(str(self._custom_dir))

    def _on_start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        source = self._custom_dir or self._images_dir
        images = find_images(source)
        if not images:
            QtWidgets.QMessageBox.information(
                self, "推理预览", f"{source} 里没有找到图像。"
            )
            return
        self.start_requested.emit(
            {
                "weights": self._weights_combo.currentData()
                or self._weights_combo.currentText(),
                "conf": float(self._conf_spin.value()),
                "images": images,
                "out_dir": str(self._out_dir),
            }
        )

    def begin_run(self, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.show()
        self._start_button.setEnabled(False)
        self._weights_combo.setEnabled(False)
        self._conf_spin.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._list.clear()
        self._preview.setText("（尚无结果）")
        self._status.setText("正在加载模型并推理…")

    def on_progress(self, done: int, total: int, name: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.setText(f"({done}/{total}) {name}")

    def on_image_done(self, src: str, out: str) -> None:
        self._list.addItem(Path(src).name)
        self._list.item(self._list.count() - 1).setData(
            QtCore.Qt.ItemDataRole.UserRole, out
        )
        if self._list.count() == 1:
            self._list.setCurrentRow(0)

    def on_failed(self, message: str) -> None:
        self._status.setText("推理失败。")
        QtWidgets.QMessageBox.critical(self, "推理预览失败", message)

    def on_finished(self) -> None:
        self._cancel_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._weights_combo.setEnabled(True)
        self._conf_spin.setEnabled(True)
        if self.worker is not None and not getattr(self.worker, "_cancelled", False):
            self._status.setText(
                f"完成，结果图在 {self._out_dir}（点选列表预览，双击交给系统看图器）。"
            )

    def mark_done(self, message: str) -> None:
        self._status.setText(message)

    def _cancel_worker(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
        self._cancel_button.setEnabled(False)
        self._status.setText("正在停止（当前这张完成后中断）…")

    def _on_row_changed(self, row: int) -> None:
        item = self._list.item(row)
        if item is None:
            return
        out = Path(item.data(QtCore.Qt.ItemDataRole.UserRole))
        pixmap = QtGui.QPixmap(str(out))
        if pixmap.isNull():
            self._preview.setText("无法加载该图片")
            return
        self._preview.setPixmap(
            pixmap.scaled(
                self._preview.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _open_image(self, item: QtWidgets.QListWidgetItem) -> None:
        out = Path(item.data(QtCore.Qt.ItemDataRole.UserRole))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out)))

    def _open_out_dir(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._out_dir)))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
        super().closeEvent(event)
