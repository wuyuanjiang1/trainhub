"""应用入口：创建 QApplication 并显示主窗口。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fix_null_streams() -> None:
    # Under a windowed interpreter (pythonw.exe, no console) sys.stdout /
    # sys.stderr are None; third-party code that writes to them directly
    # (Ultralytics' tqdm progress bar) then dies with
    # "'NoneType' object has no attribute 'write'".  Swap in null sinks.
    # Spawn-based worker processes re-run this via the package __main__.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _prepare_environment() -> None:
    _fix_null_streams()
    # torch and opencv each bundle their own OpenMP runtime (macOS, and conda
    # setups on Windows too); without this the interpreter can abort on import
    # with a duplicate-libomp error.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # matplotlib must know which Qt binding to use before it is imported.
    os.environ.setdefault("QT_API", "pyqt6")
    # nnU-Net calls into torch ops that MPS (Apple Silicon) has not
    # implemented yet; the variable is ignored on CUDA / CPU platforms.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def run(argv: list[str] | None = None) -> int:
    _prepare_environment()

    from PyQt6 import QtWidgets

    from .core.project import Project
    from .ui.main_window import MainWindow
    from .ui.theme import apply_theme

    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("trainhub")
    app.setApplicationDisplayName("trainhub")
    apply_theme(app)

    project = None
    if len(argv) > 1:
        candidate = Path(argv[1]).expanduser()
        if candidate.exists():
            project = Project.open_or_create(candidate)

    window = MainWindow(project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
