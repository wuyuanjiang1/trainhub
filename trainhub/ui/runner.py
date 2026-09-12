"""Runs a trainer on a worker thread and re-emits its events as Qt signals."""

from __future__ import annotations

import traceback

from PyQt6 import QtCore

from ..core.events import EventSink
from ..core.trainer import BaseTrainer
from ..core.trainer import TrainJob
from ..core.trainer import TrainResult


class TrainRunner(QtCore.QThread):
    logMessage = QtCore.pyqtSignal(str, str)
    progressChanged = QtCore.pyqtSignal(str, int, int, str)
    metricReceived = QtCore.pyqtSignal(int, int, dict, str)
    artifactCreated = QtCore.pyqtSignal(str, str, str)
    preparedReady = QtCore.pyqtSignal(object)
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        trainer: BaseTrainer,
        job: TrainJob,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._trainer = trainer
        self._job = job
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def job(self) -> TrainJob:
        return self._job

    def run(self) -> None:  # noqa: D102 - QThread entry point
        sink = EventSink(
            on_log=lambda event: self.logMessage.emit(event.message, event.level),
            on_progress=lambda event: self.progressChanged.emit(
                event.phase, event.current, event.total, event.message
            ),
            on_metric=lambda event: self.metricReceived.emit(
                event.epoch, event.total_epochs, event.metrics, event.phase
            ),
            on_artifact=lambda event: self.artifactCreated.emit(
                event.path, event.kind, event.label
            ),
            is_cancelled=lambda: self._cancelled,
        )
        try:
            prepared = self._trainer.prepare(self._job, sink)
            self.preparedReady.emit(prepared)
            if self._cancelled:
                raise RuntimeError("训练在数据准备阶段被取消")
            result: TrainResult = self._trainer.train(self._job, prepared, sink)
            self.succeeded.emit(result)
        except Exception as exc:  # surface any failure in the GUI
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)
