"""Centralized warm charcoal dark theme.

Deep warm gray surfaces, a terracotta accent, and pill-shaped buttons — the
"claude.ai dark" look. Every widget picks up its colors from here so the app
stays consistent.
"""

from __future__ import annotations

from PyQt6 import QtGui
from PyQt6 import QtWidgets

# ---- palette ---------------------------------------------------------------
BG = "#1F1E1D"              # window / page — warm charcoal
SURFACE = "#2A2926"         # cards, inputs
SURFACE_RAISED = "#34322E"  # buttons, menus
HOVER = "#3D3A34"           # button hover
PRESSED = "#262522"         # button pressed
BORDER = "#37342F"
BORDER_STRONG = "#4A463E"
TEXT = "#F4F2EC"
TEXT_DIM = "#B3AFA3"
TEXT_FAINT = "#7A766B"

ACCENT = "#D97757"          # terracotta (Claude coral)
ACCENT_HOVER = "#E28B6D"
ACCENT_STRONG = "#C15F3C"   # pressed
ACCENT_SOFT = "rgba(217, 119, 87, 0.22)"

SUCCESS = "#8FAE6A"
WARNING = "#D98E2B"
ERROR = "#E06C55"

LOG_COLORS = {
    "info": TEXT_DIM,
    "warning": WARNING,
    "error": ERROR,
}

_TOKENS = {
    "BG": BG,
    "SURFACE": SURFACE,
    "SURFACE_RAISED": SURFACE_RAISED,
    "HOVER": HOVER,
    "PRESSED": PRESSED,
    "BORDER": BORDER,
    "BORDER_STRONG": BORDER_STRONG,
    "TEXT": TEXT,
    "TEXT_DIM": TEXT_DIM,
    "TEXT_FAINT": TEXT_FAINT,
    "ACCENT": ACCENT,
    "ACCENT_HOVER": ACCENT_HOVER,
    "ACCENT_STRONG": ACCENT_STRONG,
    "ACCENT_SOFT": ACCENT_SOFT,
    "SUCCESS": SUCCESS,
    "WARNING": WARNING,
    "ERROR": ERROR,
}

_QSS = """
QWidget {
    color: __TEXT__;
}

QMainWindow, QDialog {
    background-color: __BG__;
}

QToolTip {
    background-color: __SURFACE_RAISED__;
    color: __TEXT__;
    border: 1px solid __BORDER_STRONG__;
    padding: 4px 6px;
}

QLabel {
    background: transparent;
}
QLabel[dim="true"] {
    color: __TEXT_DIM__;
    font-weight: 600;
}

QTabWidget::pane {
    background-color: __BG__;
    border: none;
}
QTabBar::tab {
    background: transparent;
    color: __TEXT_DIM__;
    padding: 8px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:hover:!selected {
    color: __TEXT__;
}
QTabBar::tab:selected {
    color: __TEXT__;
    border-bottom: 2px solid __ACCENT__;
}

QPushButton {
    background-color: __SURFACE_RAISED__;
    color: __TEXT__;
    border: 1px solid __BORDER__;
    border-radius: 10px;
    padding: 7px 18px;
}
QPushButton:hover {
    background-color: __HOVER__;
    border-color: __BORDER_STRONG__;
}
QPushButton:pressed {
    background-color: __PRESSED__;
}
QPushButton:disabled {
    color: __TEXT_FAINT__;
    background-color: __SURFACE__;
    border-color: __BORDER__;
}
QPushButton[accent="true"] {
    background-color: __ACCENT__;
    border-color: __ACCENT__;
    color: #ffffff;
}
QPushButton[accent="true"]:hover {
    background-color: __ACCENT_HOVER__;
    border-color: __ACCENT_HOVER__;
}
QPushButton[accent="true"]:pressed {
    background-color: __ACCENT_STRONG__;
}
QPushButton[accent="true"]:disabled {
    background-color: __SURFACE__;
    border-color: __BORDER__;
    color: __TEXT_FAINT__;
}
QPushButton[danger="true"]:!disabled {
    color: __ERROR__;
}
QPushButton[danger="true"]:hover:!disabled {
    background-color: rgba(224, 108, 85, 0.16);
    border-color: __ERROR__;
}
QPushButton[danger="true"]:pressed:!disabled {
    background-color: rgba(224, 108, 85, 0.28);
    border-color: __ERROR__;
}

QToolButton {
    background-color: __SURFACE_RAISED__;
    color: __TEXT__;
    border: 1px solid __BORDER__;
    border-radius: 10px;
    padding: 6px 12px;
}
QToolButton:hover {
    background-color: __HOVER__;
    border-color: __BORDER_STRONG__;
}
QToolButton:pressed {
    background-color: __PRESSED__;
}
QToolButton:checked {
    background-color: __ACCENT__;
    color: #ffffff;
    border-color: __ACCENT__;
}
QToolButton:disabled {
    color: __TEXT_FAINT__;
    background-color: __SURFACE__;
    border-color: __BORDER__;
}

QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox, QComboBox {
    background-color: __SURFACE__;
    color: __TEXT__;
    border: 1px solid __BORDER__;
    border-radius: 10px;
    padding: 5px 10px;
    selection-background-color: __ACCENT__;
    selection-color: #ffffff;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QAbstractSpinBox:focus, QComboBox:focus {
    border-color: __ACCENT__;
}
QPlainTextEdit[log="true"] {
    padding: 8px 12px;
    font-variant-ligatures: none;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QAbstractSpinBox:disabled, QComboBox:disabled {
    color: __TEXT_FAINT__;
    background-color: __SURFACE__;
}

QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: __SURFACE_RAISED__;
    color: __TEXT__;
    border: 1px solid __BORDER_STRONG__;
    selection-background-color: __ACCENT__;
    selection-color: #ffffff;
}

QAbstractItemView {
    background-color: __SURFACE__;
    color: __TEXT__;
    border: 1px solid __BORDER__;
    border-radius: 10px;
    alternate-background-color: __SURFACE_RAISED__;
    selection-background-color: __ACCENT_SOFT__;
    selection-color: __TEXT__;
    outline: none;
}
QAbstractItemView::item {
    padding: 4px 6px;
}
QAbstractItemView::item:hover {
    background-color: __SURFACE_RAISED__;
}
QAbstractItemView::item:selected {
    background-color: __ACCENT_SOFT__;
    color: __TEXT__;
}
QTableView {
    gridline-color: __BORDER__;
}

QHeaderView::section {
    background-color: __SURFACE__;
    color: __TEXT_DIM__;
    border: none;
    border-bottom: 1px solid __BORDER__;
    padding: 6px 8px;
}
QTableCornerButton::section {
    background-color: __SURFACE__;
    border: none;
}

QProgressBar {
    background-color: __SURFACE__;
    color: __TEXT__;
    border: 1px solid __BORDER__;
    border-radius: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: __ACCENT__;
    border-radius: 8px;
}

QGroupBox {
    border: 1px solid __BORDER__;
    border-radius: 12px;
    margin-top: 12px;
    padding-top: 6px;
    font-weight: 600;
    background-color: __SURFACE__;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: __ACCENT__;
}

QCheckBox, QRadioButton {
    color: __TEXT__;
    spacing: 6px;
    background: transparent;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid __BORDER_STRONG__;
    background-color: __SURFACE__;
    border-radius: 6px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: __ACCENT__;
}
QCheckBox::indicator:checked {
    background-color: __ACCENT__;
    border-color: __ACCENT__;
}
QRadioButton::indicator {
    border-radius: 9px;
}
QRadioButton::indicator:checked {
    background-color: __ACCENT__;
    border-color: __ACCENT__;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: __BORDER_STRONG__;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: __TEXT_FAINT__;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: __BORDER_STRONG__;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: __TEXT_FAINT__;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
    background: transparent;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}

QMenuBar {
    background-color: __BG__;
    color: __TEXT__;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: __SURFACE_RAISED__;
}
QMenu {
    background-color: __SURFACE_RAISED__;
    color: __TEXT__;
    border: 1px solid __BORDER_STRONG__;
    padding: 4px;
}
QMenu::item {
    padding: 6px 22px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: __ACCENT_SOFT__;
}
QMenu::separator {
    height: 1px;
    background-color: __BORDER__;
    margin: 4px 8px;
}

QToolBar {
    background-color: __BG__;
    border: none;
    border-bottom: 1px solid __BORDER__;
    padding: 4px;
    spacing: 4px;
}
QToolBar::separator {
    background-color: __BORDER__;
    width: 1px;
    margin: 4px 6px;
}

QStatusBar {
    background-color: __BG__;
    color: __TEXT_DIM__;
    border-top: 1px solid __BORDER__;
}
QStatusBar::item {
    border: none;
}

QSplitter::handle {
    background-color: __BORDER__;
    border-radius: 2px;
}
QSplitter::handle:hover {
    background-color: __BORDER_STRONG__;
}
"""


def _stylesheet() -> str:
    css = _QSS
    for key, value in _TOKENS.items():
        css = css.replace(f"__{key}__", value)
    return css


def _palette() -> QtGui.QPalette:
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(BG))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(SURFACE))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(SURFACE_RAISED))
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(SURFACE_RAISED))
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(SURFACE_RAISED))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(ACCENT))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(ACCENT))
    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(TEXT_FAINT))

    disabled = QtGui.QPalette.ColorGroup.Disabled
    pal.setColor(disabled, QtGui.QPalette.ColorRole.Text, QtGui.QColor(TEXT_FAINT))
    pal.setColor(disabled, QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(TEXT_FAINT))
    pal.setColor(disabled, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(TEXT_FAINT))
    pal.setColor(disabled, QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(BORDER_STRONG))
    pal.setColor(disabled, QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(TEXT_DIM))
    return pal


def apply_theme(app: QtWidgets.QApplication) -> None:
    """Apply the Claude-inspired warm light theme to an already-created QApplication."""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setStyleSheet(_stylesheet())


def mono_font() -> QtGui.QFont:
    """跨平台等宽字体（Windows: Consolas / macOS: Menlo / 兜底: Courier New）。"""
    font = QtGui.QFont()
    font.setFamilies(["Consolas", "Menlo", "Cascadia Mono", "Courier New"])
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setPointSize(10)
    return font


def section_label(text: str) -> QtWidgets.QLabel:
    """A dim, small-caps-styled section header for the panel lists."""
    label = QtWidgets.QLabel(text)
    label.setProperty("dim", True)
    return label
