"""推理页：用训练产出的权重批量推理，并生成「再训练参考」报告。

左侧结果列表（含每张检出框数与最高置信度），中间画框预览，右侧为
汇总 / 每类检出 / 与真值对比 / 再训练建议报告。对比逻辑在
:mod:`trainhub.core.evaluate`（无 Qt，可单测）。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

from ..core import evaluate
from ..core.dataset import load_annotation
from ..core.geometry import bbox_of
from ..core.prelabel import find_project_weights
from ..core.recorder import read_metrics  # noqa: F401  (训练信息由 results.csv 提供)


class InferWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    image_done = QtCore.pyqtSignal(dict)  # 单张摘要
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        project,
        weights: str,
        images: list[Path],
        out_dir: Path,
        conf: float,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._weights = weights
        self._images = images
        self._out_dir = out_dir
        self._conf = conf
        self._cancelled = False
        self.stats: dict = {}

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from PIL import Image
            from ultralytics import YOLO

            self._out_dir.mkdir(parents=True, exist_ok=True)
            model = YOLO(self._weights)
            total = len(self._images)
            all_confs: list[float] = []
            per_class_pred: dict[str, dict] = {}
            per_class_gt: dict[str, dict[str, int]] = {}
            gt_images = 0
            zero_images = 0
            low_conf_boxes = 0
            boxes_total = 0

            for index, path in enumerate(self._images):
                if self._cancelled:
                    return
                result = model.predict(str(path), conf=self._conf, verbose=False)[0]
                names = result.names or {}
                boxes = result.boxes
                n_boxes = 0 if boxes is None else len(boxes)
                confs = boxes.conf.tolist() if boxes is not None and n_boxes else []
                max_conf = max(confs) if confs else None
                preds = []
                class_counts: dict[str, int] = {}
                if boxes is not None and n_boxes:
                    for box in boxes:
                        label = names.get(int(box.cls), str(int(box.cls)))
                        conf = float(box.conf)
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                        preds.append((label, (x1, y1, x2, y2), conf))
                        class_counts[label] = class_counts.get(label, 0) + 1
                        all_confs.append(conf)
                        if conf < 0.15:
                            low_conf_boxes += 1
                    for label, count in class_counts.items():
                        info = per_class_pred.setdefault(label, {"count": 0, "confs": []})
                        info["count"] += count
                        info["confs"].extend(
                            c for lbl, _, c in preds if lbl == label
                        )
                boxes_total += n_boxes
                if n_boxes == 0:
                    zero_images += 1

                # 与真值对比（该图有标注时）
                gt_item = load_annotation(path, self._project.annotations_dir)
                gts = []
                for shape in gt_item.shapes:
                    box = bbox_of(shape)
                    if box is not None:
                        gts.append((shape.label, box))
                if gts:
                    gt_images += 1
                    matched = evaluate.match_detections(preds, gts)
                    for label, info in matched.items():
                        acc = per_class_gt.setdefault(
                            label, {"tp": 0, "fp": 0, "fn": 0}
                        )
                        for key in acc:
                            acc[key] += info[key]

                plotted = result.plot()
                out = self._out_dir / f"{path.stem}.jpg"
                Image.fromarray(plotted[:, :, ::-1]).save(out, quality=90)

                self.image_done.emit(
                    {
                        "src": str(path),
                        "out": str(out),
                        "boxes": n_boxes,
                        "max_conf": max_conf,
                        "classes": class_counts,
                        "gt_boxes": len(gts),
                    }
                )
                self.progress.emit(index + 1, total, path.name)

            avg_conf = sum(all_confs) / len(all_confs) if all_confs else None
            for info in per_class_pred.values():
                confs = info.pop("confs", [])
                info["avg_conf"] = sum(confs) / len(confs) if confs else None
            self.stats = {
                "images": total,
                "boxes": boxes_total,
                "zero_images": zero_images,
                "avg_conf": avg_conf,
                "low_conf_boxes": low_conf_boxes,
                "per_class_pred": per_class_pred,
                "gt_images": gt_images,
                "per_class_gt": per_class_gt,
            }
        except Exception as exc:
            self.failed.emit(str(exc))


class InferTab(QtWidgets.QWidget):
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, project, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._project = project
        self._worker: InferWorker | None = None
        self._custom_dir: Path | None = None
        self._out_dir: Path | None = None
        self._training_info: dict | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)

        weights_row = QtWidgets.QHBoxLayout()
        weights_row.setSpacing(6)
        self._weights_combo = QtWidgets.QComboBox()
        self._weights_combo.currentIndexChanged.connect(self._on_weights_changed)
        refresh_button = QtWidgets.QPushButton("刷新")
        refresh_button.setToolTip("重新扫描 runs/*/weights/*.pt")
        refresh_button.clicked.connect(self.refresh_weights)
        weights_row.addWidget(self._weights_combo, 1)
        weights_row.addWidget(refresh_button)
        form.addRow("模型权重", weights_row)

        self._conf_spin = QtWidgets.QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.25)
        form.addRow("置信度阈值", self._conf_spin)

        source_row = QtWidgets.QHBoxLayout()
        source_row.setSpacing(6)
        self._source_edit = QtWidgets.QLineEdit()
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText("默认使用项目的 images/ 目录")
        pick_button = QtWidgets.QPushButton("选择文件夹…")
        pick_button.clicked.connect(self._pick_folder)
        source_row.addWidget(self._source_edit, 1)
        source_row.addWidget(pick_button)
        form.addRow("推理来源", source_row)
        layout.addLayout(form)

        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(8)
        self._start_button = QtWidgets.QPushButton("开始推理")
        self._start_button.setProperty("accent", True)
        self._start_button.clicked.connect(self._on_start)
        self._cancel_button = QtWidgets.QPushButton("取消任务")
        self._cancel_button.setProperty("danger", True)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_worker)
        open_button = QtWidgets.QPushButton("打开结果目录")
        open_button.clicked.connect(self._open_out_dir)
        run_row.addWidget(self._start_button, 1)
        run_row.addWidget(self._cancel_button)
        run_row.addWidget(open_button)
        layout.addLayout(run_row)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        layout.addWidget(self._progress)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self._list = QtWidgets.QListWidget()
        self._list.setMinimumWidth(230)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._open_image)
        self._preview = QtWidgets.QLabel("（尚无结果）")
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(320, 280)
        self._report = QtWidgets.QTextBrowser()
        self._report.setMinimumWidth(340)
        self._report.setPlaceholderText("推理完成后，这里给出汇总与再训练参考。")
        split.addWidget(self._list)
        split.addWidget(self._preview)
        split.addWidget(self._report)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        layout.addWidget(split, 1)

        self._status = QtWidgets.QLabel(
            "选择权重后开始推理；有标注的图像会自动与真值对比并给出再训练建议。"
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self.refresh_weights()

    # ------------------------------------------------------------- project
    def set_project(self, project) -> None:
        self._project = project
        self._custom_dir = None
        self._source_edit.clear()
        self.refresh_weights()

    def refresh_weights(self) -> None:
        self._weights_combo.blockSignals(True)
        self._weights_combo.clear()
        weights = find_project_weights(self._project.root, limit=12)
        for w in weights:
            self._weights_combo.addItem(f"{w.parent.parent.name} / {w.name}", str(w))
        self._weights_combo.blockSignals(False)
        if weights:
            self._on_weights_changed()
        else:
            self._training_info = None
            self._status.setText("项目里还没有训练产出的权重（runs/*/weights/*.pt），先训练一次。")

    def _on_weights_changed(self) -> None:
        path = self._weights_combo.currentData()
        if not path:
            return
        self._out_dir = Path(path).parent.parent / "predictions"
        self._training_info = self._read_training_info(path)

    def _read_training_info(self, weights_path: str) -> dict | None:
        run_dir = Path(weights_path).parent.parent
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return None
        try:
            import csv as _csv

            with open(csv_path, encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
        except (OSError, UnicodeDecodeError):
            return None
        if not rows:
            return None
        key = next((k for k in rows[0] if "mAP50(B)" in k), None)
        return {
            "run": run_dir.name,
            "epochs": len(rows),
            "last_map50": float(rows[-1][key]) if key else None,
        }

    # ------------------------------------------------------------- actions
    def _pick_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择推理图像目录")
        if folder:
            self._custom_dir = Path(folder)
            self._source_edit.setText(str(self._custom_dir))

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._weights_combo.count() == 0:
            QtWidgets.QMessageBox.information(
                self, "推理", "项目里还没有训练产出的权重，先训练一次。"
            )
            return
        source = self._custom_dir or self._project.images_dir
        from ..core.dataset import find_images

        images = find_images(source)
        if not images:
            QtWidgets.QMessageBox.information(
                self, "推理", f"{source} 里没有找到图像。"
            )
            return
        self._start_worker(images)

    def _start_worker(self, images: list[Path]) -> None:
        assert self._out_dir is not None
        weights = self._weights_combo.currentData() or self._weights_combo.currentText()
        self._worker = InferWorker(
            project=self._project,
            weights=weights,
            images=images,
            out_dir=self._out_dir,
            conf=float(self._conf_spin.value()),
            parent=self,
        )
        w = self._worker
        w.progress.connect(self._on_progress)
        w.image_done.connect(self._on_image_done)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        self._set_running(True)
        self._list.clear()
        self._preview.setText("（尚无结果）")
        self._status.setText("正在加载模型并推理…")
        w.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._cancel_button.setEnabled(False)
        self._status.setText("正在停止（当前这张完成后中断）…")

    def _open_out_dir(self) -> None:
        if self._out_dir is None:
            return
        self._out_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._out_dir)))

    def _open_image(self, item: QtWidgets.QListWidgetItem) -> None:
        out = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if out:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out)))

    # ------------------------------------------------------------- 回调
    def _set_running(self, running: bool) -> None:
        self._start_button.setEnabled(not running)
        self._weights_combo.setEnabled(not running)
        self._conf_spin.setEnabled(not running)
        self._cancel_button.setEnabled(running)

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.setText(f"({done}/{total}) {name}")

    def _on_image_done(self, info: dict) -> None:
        n = info["boxes"]
        conf_text = (
            f"，最高置信度 {info['max_conf']:.2f}" if info.get("max_conf") is not None else ""
        )
        item = QtWidgets.QListWidgetItem(f"{Path(info['src']).name}（{n} 框{conf_text}）")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, info["out"])
        self._list.addItem(item)
        if self._list.count() == 1:
            self._list.setCurrentRow(0)

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

    def _on_failed(self, message: str) -> None:
        self._status.setText("推理失败。")
        QtWidgets.QMessageBox.critical(self, "推理失败", message)

    def _on_finished(self) -> None:
        self._set_running(False)
        worker = self._worker
        if worker is not None and getattr(worker, "_cancelled", False):
            self._status.setText("已停止。")
            return
        stats = dict(worker.stats if worker is not None else {})
        stats["training"] = self._training_info
        self._report.setPlainText(evaluate.build_retrain_report(stats))
        self._status.setText(
            f"完成，结果图在 {self._out_dir}。右侧为汇总与再训练参考。"
        )
        self.statusMessage.emit("推理完成")

    # -------------------------------------------------------------- 杂项
    def has_running_worker(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def shutdown(self, timeout_ms: int = 8000) -> None:
        """窗口关闭时给 worker 时间收尾，避免销毁运行中的线程。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(timeout_ms)
