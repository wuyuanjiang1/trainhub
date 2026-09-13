"""产物预览面板：点选文件即预览——图片等比缩放，文本直接查看，其余给出提示。"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

from .theme import mono_font

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
_TEXT_SUFFIXES = {".txt", ".csv", ".yaml", ".yml", ".json", ".log", ".py"}
_MAX_TEXT_BYTES = 512 * 1024
_MAX_TEXT_LINES = 400

_PAGE_EMPTY, _PAGE_IMAGE, _PAGE_TEXT, _PAGE_INFO = range(4)


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{int(num)} B" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


class _FitImageLabel(QtWidgets.QLabel):
    """等比缩放到可用空间；比可视区小的图不放大，保持原始清晰度。"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self._source: QtGui.QPixmap | None = None
        self._rendered_size = QtCore.QSize()

    def set_source(self, pixmap: QtGui.QPixmap | None) -> None:
        self._source = pixmap
        self._rendered_size = QtCore.QSize()
        self._rescale()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            super().setPixmap(QtGui.QPixmap())
            return
        area = self.size()
        if self._rendered_size == area or area.width() < 8 or area.height() < 8:
            return
        pixmap = self._source
        if pixmap.width() > area.width() or pixmap.height() > area.height():
            pixmap = pixmap.scaled(
                area,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        self._rendered_size = QtCore.QSize(area)
        super().setPixmap(pixmap)


class ArtifactPreview(QtWidgets.QWidget):
    """产物预览：头部显示文件名与大小，正文按类型切换图片 / 文本 / 提示页。"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._empty = QtWidgets.QLabel("选择左侧文件即可预览\n（双击打开所在文件夹）")
        self._empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty.setProperty("dim", True)

        self._image_area = QtWidgets.QScrollArea()
        self._image_area.setWidgetResizable(True)
        self._image_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._image_label = _FitImageLabel()
        self._image_area.setWidget(self._image_label)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setProperty("log", True)
        self._text.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._text.setFont(mono_font())

        self._info = QtWidgets.QLabel()
        self._info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._info.setWordWrap(True)
        self._info.setProperty("dim", True)

        self._stack = QtWidgets.QStackedWidget()
        for page in (self._empty, self._image_area, self._text, self._info):
            self._stack.addWidget(page)

        self._header = QtWidgets.QLabel()
        self._header.setProperty("dim", True)
        self._header.setContentsMargins(4, 0, 4, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._header)
        layout.addWidget(self._stack, 1)

        self._show_page(_PAGE_EMPTY)

    # ------------------------------------------------------------------ api
    def clear(self) -> None:
        self._header.setText("")
        self._show_page(_PAGE_EMPTY)

    def preview(self, path_str: str) -> None:
        path = Path(path_str)
        if not path.is_file():
            self.clear()
            return
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        self._header.setText(f"{path.name} · {_human_size(size)}")
        suffix = path.suffix.lower()

        if suffix in _IMAGE_SUFFIXES:
            pixmap = QtGui.QPixmap(str(path))
            if pixmap.isNull():
                self._info.setText("无法加载该图片")
                self._show_page(_PAGE_INFO)
            else:
                self._image_label.set_source(pixmap)
                self._show_page(_PAGE_IMAGE)
            return

        if suffix in _TEXT_SUFFIXES:
            try:
                raw = path.read_bytes()[:_MAX_TEXT_BYTES]
            except OSError as exc:
                self._info.setText(f"读取失败：{exc}")
                self._show_page(_PAGE_INFO)
                return
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > _MAX_TEXT_LINES:
                text = (
                    "\n".join(lines[:_MAX_TEXT_LINES])
                    + f"\n…（已截断，完整内容共 {len(lines)} 行）"
                )
            self._text.setPlainText(text or "（空文件）")
            self._show_page(_PAGE_TEXT)
            return

        self._info.setText(
            f"{path.suffix or '（无后缀）'} 文件暂不支持应用内预览，\n双击可打开所在文件夹。"
        )
        self._show_page(_PAGE_INFO)

    # ------------------------------------------------------------ internals
    def _show_page(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
