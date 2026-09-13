"""主窗口：数据集 / 标注 / 训练 三个页签 + 项目管理。"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtGui
from PyQt6 import QtWidgets

from ..core.project import Project
from .annotate_tab import AnnotateTab
from .dataset_tab import DatasetTab
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

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._dataset, "数据集")
        self._tabs.addTab(self._annotate, "标注")
        self._tabs.addTab(self._train, "训练")
        self.setCentralWidget(self._tabs)

        self._annotate.statusMessage.connect(self.statusBar().showMessage)
        self._train.statusMessage.connect(self.statusBar().showMessage)
        self._annotate.labelsChanged.connect(lambda _: self._dataset.refresh())
        self._annotate.imagesChanged.connect(lambda: self._dataset.refresh())
        self._annotate.dirtyChanged.connect(self._on_dirty_changed)
        self._train.runFinished.connect(lambda _: self._dataset.refresh())

        self._dataset.requestImportFiles.connect(self._annotate.import_images)
        self._dataset.requestImportFolder.connect(self._annotate.import_folder)

        self._build_menu()
        self._build_toolbar()
        self._update_title()
        self.statusBar().showMessage(f"项目目录：{self._project.root}")

    # ----------------------------------------------------------------- menu
    def _build_menu(self) -> None:
        project_menu = self.menuBar().addMenu("项目")
        project_menu.addAction(self._action("新建项目…", self.new_project, "Ctrl+N"))
        project_menu.addAction(self._action("打开项目…", self.open_project, "Ctrl+Shift+O"))
        project_menu.addAction(self._action("保存项目", self.save_project, "Ctrl+Shift+S"))
        project_menu.addSeparator()
        project_menu.addAction(self._action("退出", self.close, "Ctrl+Q"))

        data_menu = self.menuBar().addMenu("数据")
        data_menu.addAction(self._action("导入图像文件…", self._annotate.import_images))
        data_menu.addAction(
            self._action("导入图像文件夹…", self._annotate.import_folder)
        )
        data_menu.addAction(self._action("刷新统计", self._dataset.refresh, "F5"))

        help_menu = self.menuBar().addMenu("帮助")
        help_menu.addAction(self._action("关于 trainhub", self._show_about))

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.addAction(self._action("打开项目…", self.open_project))
        toolbar.addSeparator()
        toolbar.addAction(self._action("导入文件", self._annotate.import_images))
        toolbar.addAction(self._action("导入文件夹", self._annotate.import_folder))
        toolbar.addSeparator()
        toolbar.addAction(self._action("保存标注", self._annotate.save_current))
        self.addToolBar(toolbar)

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
        self._update_title()
        self.statusBar().showMessage(f"项目目录：{project.root}")

    def _on_dirty_changed(self, dirty: bool) -> None:
        self._update_title()
        if dirty:
            self.statusBar().showMessage("有未保存的标注修改")

    def _update_title(self) -> None:
        # 标题栏保持简洁；项目名/目录见状态栏，未保存修改会提示在状态栏。
        self.setWindowTitle("trainhub")

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
