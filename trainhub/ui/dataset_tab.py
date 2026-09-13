"""数据集页：查看项目里有多少图像、标注了多少、各类别分布。"""

from __future__ import annotations

from PyQt6 import QtCore
from PyQt6 import QtWidgets

from ..core.dataset import label_histogram
from ..core.dataset import scan_dataset
from ..core.project import Project
from ..trainers.yolo.converter import export_yolo_dataset
from .theme import section_label


class DatasetTab(QtWidgets.QWidget):
    requestImportFiles = QtCore.pyqtSignal()
    requestImportFolder = QtCore.pyqtSignal()

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project

        self._summary = QtWidgets.QLabel()
        self._summary.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setTextVisible(True)

        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["类别", "对象数量", "占比"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        import_files = QtWidgets.QPushButton("导入图像文件…")
        import_files.setProperty("accent", True)
        import_files.clicked.connect(self.requestImportFiles)
        import_folder = QtWidgets.QPushButton("导入图像文件夹…")
        import_folder.setProperty("accent", True)
        import_folder.clicked.connect(self.requestImportFolder)
        export_yolo = QtWidgets.QPushButton("labelme2yolo 导出")
        export_yolo.clicked.connect(self._export_yolo)
        refresh = QtWidgets.QPushButton("刷新统计")
        refresh.clicked.connect(self.refresh)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(import_files)
        buttons.addWidget(import_folder)
        buttons.addWidget(export_yolo)
        buttons.addStretch(1)
        buttons.addWidget(refresh)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        layout.addLayout(buttons)
        layout.addWidget(self._summary)
        layout.addWidget(self._progress)
        layout.addWidget(section_label("类别分布"))
        layout.addWidget(self._table, 1)

        self.refresh()

    def set_project(self, project: Project) -> None:
        self._project = project
        self.refresh()

    def refresh(self) -> None:
        images_dir = self._project.images_dir
        annotations_dir = self._project.annotations_dir
        items = scan_dataset(images_dir, annotations_dir)

        annotated = sum(1 for item in items if item.json_path is not None)
        with_shapes = sum(1 for item in items if item.shapes)
        total_objects = sum(len(item.shapes) for item in items)

        self._summary.setText(
            f"<b>项目：</b>{self._project.name}<br>"
            f"<b>目录：</b>{self._project.root}<br>"
            f"<b>图像总数：</b>{len(items)} &nbsp;&nbsp;"
            f"<b>已有标注文件：</b>{annotated} &nbsp;&nbsp;"
            f"<b>含对象：</b>{with_shapes} &nbsp;&nbsp;"
            f"<b>对象总数：</b>{total_objects}"
        )

        self._progress.setMaximum(max(len(items), 1))
        self._progress.setValue(annotated)
        self._progress.setFormat(f"标注进度 {annotated}/{len(items)}")

        histogram = label_histogram(items)
        self._table.setRowCount(len(histogram))
        for row, (label, count) in enumerate(histogram.items()):
            share = f"{count / total_objects * 100:.1f}%" if total_objects else "-"
            for column, text in enumerate((label, str(count), share)):
                item = QtWidgets.QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row, column, item)

    def _export_yolo(self) -> None:
        items = scan_dataset(
            self._project.images_dir, self._project.annotations_dir
        )
        annotated = [item for item in items if item.shapes]
        if not annotated:
            QtWidgets.QMessageBox.warning(
                self, "没有标注", "还没有可导出的标注，请先在标注页完成标注。"
            )
            return
        out_dir = self._project.datasets_dir / "yolo_detect"
        try:
            result = export_yolo_dataset(
                annotated,
                task="detect",
                classes=list(self._project.labels),
                out_dir=out_dir,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))
            return

        detail = (
            f"输出目录：{result.dataset_dir}\n"
            f"类别：{', '.join(result.classes) or '（无）'}\n"
            f"训练集：{result.train_count} 张   验证集：{result.val_count} 张"
        )
        if result.skipped:
            detail += f"\n跳过对象：{result.skipped}"
        QtWidgets.QMessageBox.information(self, "labelme2yolo 导出完成", detail)
