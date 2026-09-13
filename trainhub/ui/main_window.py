"""主窗口：顶栏（项目操作 + 主题切换）+ 数据集 / 标注 / 训练 / 推理 四页签。"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtGui
from PyQt6 import QtWidgets

from ..core.project import Project
from .annotate_tab import AnnotateTab
from .dataset_tab import DatasetTab
from .infer_tab import InferTab
from .top_bar import TopBar
from .train_tab import TrainTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        project: Project | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("trainhub")
        self.resize(1560, 960)
        self.setMinimumSize(1120, 700)
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.move(
            screen.x() + max((screen.width() - self.width()) // 2, 0),
            screen.y() + max((screen.height() - self.height()) // 2, 0),
        )

        self._project = project or Project.create(
            Path.cwd() / "trainhub_project", name="新项目"
        )

        self._annotate = AnnotateTab(self._project)
        self._dataset = DatasetTab(self._project)
        self._train = TrainTab(self._project)
        self._infer = InferTab(self._project)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._dataset, "数据集")
        self._tabs.addTab(self._annotate, "标注")
        self._tabs.addTab(self._train, "训练")
        self._tabs.addTab(self._infer, "推理")

        self._top_bar = TopBar()
        self._top_bar.new_project_requested.connect(self.new_project)
        self._top_bar.open_project_requested.connect(self.open_project)
        self._top_bar.save_project_requested.connect(self.save_project)
        self._top_bar.about_requested.connect(self._show_about)

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._top_bar)
        central_layout.addWidget(self._tabs, 1)
        self.setCentralWidget(central)

        self._annotate.statusMessage.connect(self.statusBar().showMessage)
        self._train.statusMessage.connect(self.statusBar().showMessage)
        self._infer.statusMessage.connect(self.statusBar().showMessage)
        self._annotate.labelsChanged.connect(lambda _: self._dataset.refresh())
        self._annotate.imagesChanged.connect(lambda: self._dataset.refresh())
        self._annotate.dirtyChanged.connect(self._on_dirty_changed)
        self._train.runFinished.connect(lambda _: self._dataset.refresh())

        self._dataset.requestImportFiles.connect(self._annotate.import_images)
        self._dataset.requestImportFolder.connect(self._annotate.import_folder)
        self._top_bar.set_project(self._project.name)

        # 快捷键（原菜单项迁移；action 挂在主窗口上即可全局生效）
        for text, seq, slot in (
            ("新建项目", "Ctrl+N", self.new_project),
            ("打开项目", "Ctrl+Shift+O", self.open_project),
            ("保存项目", "Ctrl+Shift+S", self.save_project),
            ("刷新统计", "F5", self._dataset.refresh),
        ):
            action = QtGui.QAction(text, self)
            action.setShortcut(seq)
            action.triggered.connect(slot)
            self.addAction(action)

        self._update_title()
        self.statusBar().showMessage(f"项目目录：{self._project.root}")

    def _action(self, text: str, slot, shortcut: str | None = None) -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    # -------------------------------------------------------------- project
    def new_project(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择（或新建）项目目录"
        )
        if not root:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "项目名称", "名称：", text=Path(root).name
        )
        if not ok:
            return
        self._switch_project(Project.create(root, name=name or Path(root).name))

    def open_project(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "选择项目目录")
        if not root:
            return
        try:
            project = Project.open_or_create(root)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "打开项目失败", str(exc))
            return
        self._switch_project(project)

    def save_project(self) -> None:
        self._annotate.save_current()
        self._project.save()
        self.statusBar().showMessage(f"项目已保存：{self._project.config_path}")

    def _switch_project(self, project: Project) -> None:
        self._project = project
        self._annotate.set_project(project)
        self._dataset.set_project(project)
        self._train.set_project(project)
        self._infer.set_project(project)
        self._update_title()
        self.statusBar().showMessage(f"项目目录：{project.root}")

    def _on_dirty_changed(self, dirty: bool) -> None:
        self._update_title()
        if dirty:
            self.statusBar().showMessage("有未保存的标注修改")

    def _update_title(self) -> None:
        self.setWindowTitle(f"trainhub · {self._project.name}")
        self._top_bar.set_project(self._project.name)

    # ---------------------------------------------------------------- misc
    def _show_about(self) -> None:
        from .. import __version__

        QtWidgets.QMessageBox.about(
            self,
            "关于 trainhub",
            f"<b>trainhub {__version__}</b><br>"
            "基于 PyQt6 的一站式标注与训练工作站。<br><br>"
            "标注引擎移植自 labelme 6.3.0（GPL-3.0），"
            "标注格式与 labelme 完全兼容。<br>"
            "训练后端：Ultralytics YOLO、nnU-Net。",
        )

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._train.is_training:
            answer = QtWidgets.QMessageBox.question(
                self,
                "训练进行中",
                "训练仍在进行，退出会中断训练。确定退出吗？",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                a0.ignore()
                return
            self._train.shutdown()
        if self._infer.has_running_worker():
            self._infer.shutdown()
        if self._annotate._dirty:
            answer = QtWidgets.QMessageBox.question(
                self,
                "未保存的标注",
                "当前标注尚未保存，退出前保存吗？",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                a0.ignore()
                return
            if answer == QtWidgets.QMessageBox.StandardButton.Save:
                self._annotate.save_current()
        self._project.save()
        a0.accept()
