"""训练页：选训练器/任务 → 调参 → 训练 → 实时看 loss 与指标曲线。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

from ..core.project import Project
from ..core.recorder import read_metrics
from ..core.registry import all_trainers
from ..core.registry import get_trainer
from ..core.registry import load_builtin_trainers
from ..core.trainer import TrainJob
from ..core.trainer import TrainResult
from .metric_chart import MetricChart
from .param_form import ParamForm
from .runner import TrainRunner
from .theme import LOG_COLORS
from .theme import section_label


class TrainTab(QtWidgets.QWidget):
    statusMessage = QtCore.pyqtSignal(str)
    runFinished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._runner: TrainRunner | None = None
        self._form: ParamForm | None = None

        load_builtin_trainers()

        self._trainer_combo = QtWidgets.QComboBox()
        self._task_combo = QtWidgets.QComboBox()
        self._trainer_combo.currentIndexChanged.connect(self._on_trainer_changed)
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)

        self._run_name = QtWidgets.QLineEdit()
        self._run_name.setPlaceholderText("留空则按时间自动命名")

        self._params_host = QtWidgets.QVBoxLayout()
        self._params_host.setContentsMargins(0, 0, 0, 0)
        self._params_scroll = QtWidgets.QScrollArea()
        self._params_scroll.setWidgetResizable(True)

        self._start_button = QtWidgets.QPushButton("开始训练")
        self._start_button.setProperty("accent", True)
        self._start_button.clicked.connect(self.start_training)
        self._stop_button = QtWidgets.QPushButton("停止")
        self._stop_button.setProperty("danger", True)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop_training)

        self._phase_label = QtWidgets.QLabel("尚未开始")
        self._phase_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._chart = MetricChart()
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        self._log.setProperty("log", True)
        log_font = QtGui.QFont()
        # Menlo is macOS-only; keep an explicit monospace stack so per-epoch
        # summary lines align on Windows / Linux too.
        log_font.setFamilies(
            ["Consolas", "Menlo", "Cascadia Mono", "Courier New"]
        )
        log_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        log_font.setPointSize(10)
        self._log.setFont(log_font)

        self._artifacts = QtWidgets.QListWidget()
        self._artifacts.itemDoubleClicked.connect(self._open_artifact)

        self._history_combo = QtWidgets.QComboBox()
        self._history_combo.currentIndexChanged.connect(self._on_history_changed)

        self._build_layout()
        self._populate_trainers()
        self.set_project(project)

    # --------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        selector = QtWidgets.QFormLayout()
        selector.addRow("训练框架", self._trainer_combo)
        selector.addRow("任务类型", self._task_combo)
        selector.addRow("运行名称", self._run_name)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 6, 10)
        left_layout.setSpacing(8)
        left_layout.addLayout(selector)
        left_layout.addWidget(self._params_scroll, 1)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self._start_button, 1)
        actions.addWidget(self._stop_button)
        left_layout.addLayout(actions)
        left_layout.addWidget(self._phase_label)
        left_layout.addWidget(self._progress)
        left_layout.addWidget(section_label("历史运行"))
        left_layout.addWidget(self._history_combo)

        right = QtWidgets.QTabWidget()
        right.addTab(self._chart, "训练曲线")
        right.addTab(self._log, "训练日志")
        right.addTab(self._artifacts, "产物")

        # Draggable split: the form keeps a readable minimum width, the charts
        # get the lion's share on wide screens.
        panes = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        panes.setChildrenCollapsible(False)
        panes.setHandleWidth(6)
        panes.addWidget(left)
        panes.addWidget(right)
        panes.setStretchFactor(0, 5)
        panes.setStretchFactor(1, 6)
        left.setMinimumWidth(380)
        right.setMinimumWidth(440)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(panes)

    # ------------------------------------------------------------- trainers
    def _populate_trainers(self) -> None:
        self._trainer_combo.blockSignals(True)
        self._trainer_combo.clear()
        for trainer in all_trainers():
            suffix = "" if trainer.is_available() else "  ⚠ 依赖未安装"
            self._trainer_combo.addItem(
                f"{trainer.display_name}{suffix}", trainer.key
            )
        self._trainer_combo.blockSignals(False)
        if self._trainer_combo.count():
            self._on_trainer_changed()

    def _current_trainer(self):
        key = self._trainer_combo.currentData()
        return get_trainer(key) if key else None

    def _on_trainer_changed(self) -> None:
        trainer = self._current_trainer()
        self._task_combo.blockSignals(True)
        self._task_combo.clear()
        if trainer is not None:
            for spec in trainer.tasks:
                self._task_combo.addItem(spec.display_name, spec.key)
        self._task_combo.blockSignals(False)
        self._on_task_changed()

    def _on_task_changed(self) -> None:
        trainer = self._current_trainer()
        task_key = self._task_combo.currentData()
        if trainer is None or not task_key:
            return
        specs = trainer.all_params(str(task_key))
        self._form = ParamForm(specs)
        self._form.set_values(self._saved_params(trainer.key, str(task_key)))
        container = QtWidgets.QWidget()
        wrapper = QtWidgets.QVBoxLayout(container)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(self._form)
        wrapper.addStretch(1)
        self._params_scroll.setWidget(container)
        self._apply_task_hint(trainer, str(task_key))

    def _apply_task_hint(self, trainer, task_key: str) -> None:
        try:
            spec = trainer.task(task_key)
        except KeyError:
            return
        annotated = (
            "需要矩形框" if spec.annotation == "bbox" else "需要多边形/笔刷"
        )
        self._phase_label.setText(
            f"{trainer.display_name} · {spec.display_name}（{annotated}）"
        )

    # --------------------------------------------------------------- params
    def _saved_params(self, trainer_key: str, task_key: str) -> dict:
        try:
            return dict(
                self._project.params.get("trainers", {})
                .get(trainer_key, {})
                .get(task_key, {})
            )
        except (AttributeError, TypeError):
            return {}

    def _store_params(self, trainer_key: str, task_key: str, params: dict) -> None:
        try:
            trainers = self._project.params.setdefault("trainers", {})
            trainers.setdefault(trainer_key, {})[task_key] = params
            self._project.save()
        except (AttributeError, TypeError, OSError):
            pass

    # -------------------------------------------------------------- project
    def set_project(self, project: Project) -> None:
        self._project = project
        if self._form is not None:
            trainer = self._current_trainer()
            task_key = self._task_combo.currentData()
            if trainer is not None and task_key:
                self._form.set_values(self._saved_params(trainer.key, str(task_key)))
        self.refresh_history()

    def refresh_history(self) -> None:
        self._history_combo.blockSignals(True)
        self._history_combo.clear()
        self._history_combo.addItem("（当前训练）", "")
        runs_dir = self._project.runs_dir
        if runs_dir.exists():
            for path in sorted(runs_dir.iterdir(), reverse=True):
                if path.is_dir() and (path / "metrics.jsonl").exists():
                    self._history_combo.addItem(path.name, str(path))
        self._history_combo.blockSignals(False)

    def _on_history_changed(self) -> None:
        path = self._history_combo.currentData()
        if not path:
            return
        events = read_metrics(Path(path))
        self._chart.load_events(events)
        self._log.appendPlainText(f"—— 载入历史运行 {Path(path).name} ——")
        self._artifacts.clear()
        for file in sorted(Path(path).rglob("*")):
            if file.is_file():
                item = QtWidgets.QListWidgetItem(str(file.relative_to(path)))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(file))
                self._artifacts.addItem(item)

    # -------------------------------------------------------------- running
    def start_training(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            return
        trainer = self._current_trainer()
        task_key = self._task_combo.currentData()
        if trainer is None or not task_key or self._form is None:
            return
        if not trainer.is_available():
            QtWidgets.QMessageBox.warning(
                self,
                "依赖未安装",
                f"{trainer.display_name} 的依赖未安装。\n\n"
                "请在 trainhub 环境中安装后再试：\n"
                "  pip install ultralytics nnunetv2",
            )
            return

        params = self._form.values()
        self._store_params(trainer.key, str(task_key), params)

        run_name = self._run_name.text().strip() or (
            f"{trainer.key}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        run_dir = self._project.run_dir(run_name)
        if run_dir.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "运行已存在",
                f"运行目录 {run_name} 已存在，继续会覆盖其中的产物。是否继续？",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        job = TrainJob(
            project=self._project,
            task_key=str(task_key),
            params=params,
            run_dir=run_dir,
        )

        self._chart.reset()
        self._log.clear()
        self._artifacts.clear()
        self._set_running(True)

        self._runner = TrainRunner(trainer, job, self)
        self._runner.logMessage.connect(self._append_log)
        self._runner.progressChanged.connect(self._on_progress)
        self._runner.metricReceived.connect(self._on_metric)
        self._runner.artifactCreated.connect(self._on_artifact)
        self._runner.succeeded.connect(self._on_succeeded)
        self._runner.failed.connect(self._on_failed)
        self._runner.finished.connect(self._on_runner_finished)
        self._runner.start()

    def stop_training(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            self._runner.cancel()
            self._stop_button.setEnabled(False)
            self._phase_label.setText("正在停止 ...")

    @property
    def is_training(self) -> bool:
        return self._runner is not None and self._runner.isRunning()

    def shutdown(self, timeout_ms: int = 8000) -> None:
        """Cancel training and give the worker thread time to unwind.

        Destroying a running QThread aborts the process, so on exit we must not
        let the window outlive it.
        """
        if self._runner is not None and self._runner.isRunning():
            self._runner.cancel()
            self._runner.wait(timeout_ms)

    def _set_running(self, running: bool) -> None:
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._params_scroll.setEnabled(not running)
        self._trainer_combo.setEnabled(not running)
        self._task_combo.setEnabled(not running)

    # ------------------------------------------------------------- signals
    def _append_log(self, message: str, level: str) -> None:
        color = LOG_COLORS.get(level, LOG_COLORS["info"])
        escaped = (
            message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        self._log.appendHtml(f'<span style="color:{color}">{escaped}</span>')

    def _on_progress(self, phase: str, current: int, total: int, message: str) -> None:
        self._phase_label.setText(f"{phase} — {message}" if message else phase)
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(min(current, total))
        else:
            self._progress.setRange(0, 0)

    def _on_metric(self, epoch: int, total: int, metrics: dict, phase: str) -> None:
        self._chart.add_point(epoch, metrics)
        if total:
            self._progress.setRange(0, total)
            self._progress.setValue(min(epoch, total))
        summary = "  ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:6])
        self._phase_label.setText(f"Epoch {epoch}/{total}  {summary}")

    def _on_artifact(self, path: str, kind: str, label: str) -> None:
        item = QtWidgets.QListWidgetItem(f"[{kind}] {label or Path(path).name}")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
        self._artifacts.addItem(item)

    def _on_succeeded(self, result: TrainResult) -> None:
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        best = ""
        if result.best_metrics:
            best = "  最佳指标: " + ", ".join(
                f"{k}={v:.4f}" for k, v in result.best_metrics.items()
            )
        self._phase_label.setText(f"训练完成{best}")
        self._append_log(f"训练完成，产物目录: {result.run_dir}", "info")
        self.refresh_history()
        self.runFinished.emit(result)

    def _on_failed(self, message: str) -> None:
        self._phase_label.setText("训练失败")
        self._append_log(f"训练失败: {message}", "error")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        QtWidgets.QMessageBox.critical(self, "训练失败", message)
        self.statusMessage.emit(f"训练失败: {message}")

    def _on_runner_finished(self) -> None:
        self._set_running(False)

    def _open_artifact(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(Path(path).parent))
            )
