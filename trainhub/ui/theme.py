"""「暖纸工作台」主题：明暗双主题、暖调、珊瑚橙唯一强调色。

设计规范要点：
- 暖米色底 + 4 档底色分层 + 1px 发丝线；禁投影 / 渐变 / 胶囊 / 蓝灰
- 珊瑚橙 #D97757 是唯一强调色，只用于主按钮实底、选中态、输入焦点边
- 小节标题衬线、数值等宽、正文无衬线；字号阶梯 17/14/13/12.5/11.5/11
- 圆角：控件 9px、卡片 12px；语义色低饱和

对外 API：
- ``apply_theme(app)`` —— 按 QSettings 记忆的主题应用到 QApplication
- ``set_dark(app, dark)`` —— 切换浅色/深色并记忆
- ``mono_font() / section_label(text)`` —— 等宽字体与衬线小节标题
- ``LOG_COLORS / SUCCESS / WARNING / ERROR`` —— 语义色（LOG_COLORS 随主题更新）

注意：``ACCENT / BG / SURFACE / BORDER / TEXT / TEXT_DIM`` 是模块级静态
常量（供 MetricChart 等在导入时取色）；运行时切换主题不会改变已导入的
绑定，颜色完全跟随主题需要重启或动态读取。
"""

from __future__ import annotations

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

_SETTINGS_ORG = "trainhub"
_SETTINGS_APP = "trainhub"

# ---- 调色板（明暗对等，全部暖调） ------------------------------------------

LIGHT = {
    "BG": "#F4F3EE",
    "SIDEBAR": "#EDEBE3",
    "PANEL": "#FAF9F5",
    "PANEL_ALT": "#F1EFE7",
    "HAIRLINE": "#E4E1D6",
    "HAIRLINE_STRONG": "#CFCAB9",
    "INK": "#2D2A26",
    "INK2": "#6E6A5E",
    "MUTED": "#9B9588",
    "FAINT": "#B3AEA1",
    "ACCENT": "#D97757",
    "ACCENT_HOVER": "#C96A4D",
    "ACCENT_PRESS": "#B85C42",
    "ACCENT_INK": "#B85C42",
    "ACCENT_SOFT": "#F6E9E2",
    "ACCENT_SOFT2": "#F0DDD3",
    "ACCENT_BORDER": "#E5C4B4",
    "DANGER": "#D9483B",
    "DANGER_SOFT": "#F9EDE9",
    "DANGER_BORDER": "#E0AFA3",
    "DANGER_INK": "#B04A38",
    "SUCCESS": "#5E8C56",
    "WARN": "#C2913D",
    "VIEWPORT": "#E7E3D8",
    "SCROLL": "#D6D2C4",
    "SCROLL_HOVER": "#C1BCAB",
    "TOOLTIP_BG": "#2D2A26",
    "TOOLTIP_FG": "#F4F3EE",
}

DARK = {
    "BG": "#262624",
    "SIDEBAR": "#1F1E1D",
    "PANEL": "#30302E",
    "PANEL_ALT": "#2A2A28",
    "HAIRLINE": "#3E3E3A",
    "HAIRLINE_STRONG": "#54544E",
    "INK": "#F0EEE7",
    "INK2": "#B0AEA5",
    "MUTED": "#8A877D",
    "FAINT": "#6A6862",
    "ACCENT": "#D97757",
    "ACCENT_HOVER": "#E08A6D",
    "ACCENT_PRESS": "#C96A4D",
    "ACCENT_INK": "#E08A6D",
    "ACCENT_SOFT": "#3A2A23",
    "ACCENT_SOFT2": "#452F26",
    "ACCENT_BORDER": "#5C3B2E",
    "DANGER": "#E5484D",
    "DANGER_SOFT": "#3A2325",
    "DANGER_BORDER": "#5C3236",
    "DANGER_INK": "#E88A8D",
    "SUCCESS": "#7FB074",
    "WARN": "#D9A94D",
    "VIEWPORT": "#1A1915",
    "SCROLL": "#4A4A45",
    "SCROLL_HOVER": "#5C5C56",
    "TOOLTIP_BG": "#F0EEE7",
    "TOOLTIP_FG": "#262624",
}

# 语义色取明暗两套的中间调，两套主题下都可读
SUCCESS = "#6FA05E"
WARNING = "#C2913D"
ERROR = "#D4524A"

# 视口（标注画布留白、预览区 letterbox）专用底色
VIEWPORT_LIGHT = LIGHT["VIEWPORT"]
VIEWPORT_DARK = DARK["VIEWPORT"]

# 主题变更回调：fn(dark)。组件在构造时订阅，切换主题时被通知自行重刷
# （matplotlib 图表、自绘视口等 QSS 管不到的部分）。
_listeners: list = []


def subscribe_theme_listener(fn) -> None:
    """注册主题变更回调 fn(dark)。"""
    _listeners.append(fn)


def _notify_listeners(dark: bool) -> None:
    for fn in list(_listeners):
        try:
            fn(dark)
        except Exception:
            pass

# ---- 主题状态 ----------------------------------------------------------------


def _saved_dark() -> bool:
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return settings.value("ui/dark", False, type=bool)


_cur = DARK if _saved_dark() else LIGHT
_is_dark = _cur is DARK

# 静态常量（MetricChart 在导入时取色）；set_dark 会同步更新这些模块级名字
ACCENT = _cur["ACCENT"]
BG = _cur["BG"]
SURFACE = _cur["PANEL"]
BORDER = _cur["HAIRLINE"]
TEXT = _cur["INK"]
TEXT_DIM = _cur["INK2"]

LOG_COLORS = {
    "info": _cur["MUTED"],
    "warning": _cur["WARN"],
    "error": "#D4524A",
}


def is_dark() -> bool:
    return _is_dark


def set_dark(app: QtWidgets.QApplication, dark: bool) -> None:
    """切换浅色/深色主题并记忆；全局 QSS 与调色板即时重刷。"""
    global _cur, _is_dark, ACCENT, BG, SURFACE, BORDER, TEXT, TEXT_DIM
    _is_dark = bool(dark)
    _cur = DARK if _is_dark else LIGHT
    ACCENT = _cur["ACCENT"]
    BG = _cur["BG"]
    SURFACE = _cur["PANEL"]
    BORDER = _cur["HAIRLINE"]
    TEXT = _cur["INK"]
    TEXT_DIM = _cur["INK2"]
    LOG_COLORS.update(
        {"info": _cur["MUTED"], "warning": _cur["WARN"], "error": "#D4524A"}
    )
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    settings.setValue("ui/dark", _is_dark)
    app.setStyleSheet(_stylesheet(_cur))
    app.setPalette(_palette(_cur))
    # Qt 已知行为：热切换主题时，部分容器内的控件不会因 setStyleSheet
    # 自动重新 polish（如 QGroupBox 背景残留旧主题），强制刷一遍。
    for widget in app.allWidgets():
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    _notify_listeners(_is_dark)


def apply_theme(app: QtWidgets.QApplication) -> None:
    """按记忆的主题（默认浅色）为已创建的 QApplication 应用整套主题。"""
    app.setStyle("Fusion")
    app.setStyleSheet(_stylesheet(_cur))
    app.setPalette(_palette(_cur))
    font = QtGui.QFont()
    font.setFamilies(["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "sans-serif"])
    font.setPointSize(10)
    app.setFont(font)


def mono_font() -> QtGui.QFont:
    """跨平台等宽字体（JetBrains Mono 优先，macOS/Windows 各有回退）。"""
    font = QtGui.QFont()
    font.setFamilies(["JetBrains Mono", "Consolas", "Menlo", "Courier New"])
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setPointSize(10)
    return font


def section_label(text: str) -> QtWidgets.QLabel:
    """衬线小节标题（粗体 + 发丝线下划线），对应风格里的 section。"""
    label = QtWidgets.QLabel(text)
    label.setProperty("section", True)
    return label


# ---- QSS 与调色板 ------------------------------------------------------------


def _stylesheet(pal: dict) -> str:
    css = _QSS
    for key, value in pal.items():
        css = css.replace(f"__{key}__", value)
    return css


def _palette(pal: dict) -> QtGui.QPalette:
    p = QtGui.QPalette()
    qcolor = QtGui.QColor
    p.setColor(QtGui.QPalette.ColorRole.Window, qcolor(pal["BG"]))
    p.setColor(QtGui.QPalette.ColorRole.WindowText, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.Base, qcolor(pal["PANEL"]))
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase, qcolor(pal["PANEL_ALT"]))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipBase, qcolor(pal["TOOLTIP_BG"]))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipText, qcolor(pal["TOOLTIP_FG"]))
    p.setColor(QtGui.QPalette.ColorRole.Text, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.Button, qcolor(pal["PANEL"]))
    p.setColor(QtGui.QPalette.ColorRole.ButtonText, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.BrightText, qcolor("#ffffff"))
    p.setColor(QtGui.QPalette.ColorRole.Highlight, qcolor(pal["ACCENT_SOFT"]))
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.Link, qcolor(pal["ACCENT"]))
    p.setColor(QtGui.QPalette.ColorRole.PlaceholderText, qcolor(pal["FAINT"]))

    disabled = QtGui.QPalette.ColorGroup.Disabled
    p.setColor(disabled, QtGui.QPalette.ColorRole.Text, qcolor(pal["FAINT"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.ButtonText, qcolor(pal["FAINT"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.WindowText, qcolor(pal["FAINT"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.Highlight, qcolor(pal["HAIRLINE_STRONG"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.HighlightedText, qcolor(pal["INK2"]))
    return p


_QSS = """
* {
    color: __INK__;
}

QMainWindow, QDialog {
    background-color: __BG__;
}

QToolTip {
    background-color: __TOOLTIP_BG__;
    color: __TOOLTIP_FG__;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}

QLabel { background: transparent; }
QLabel[dim="true"] {
    color: __MUTED__;
    font-size: 11px;
}
/* 衬线小节标题：风格里最有辨识度的元素 */
QLabel[section="true"] {
    font-family: "Noto Serif CJK SC", "Songti SC", "SimSun", serif;
    font-weight: 700;
    font-size: 12px;
    color: __INK2__;
    border-bottom: 1px solid __HAIRLINE__;
    padding-bottom: 2px;
}

/* 主内容页签：下划线式 */
QTabWidget::pane {
    background-color: __BG__;
    border: none;
}
QTabBar::tab {
    background: transparent;
    color: __MUTED__;
    padding: 7px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
}
QTabBar::tab:hover:!selected {
    color: __INK2__;
}
QTabBar::tab:selected {
    color: __ACCENT_INK__;
    font-weight: 600;
    border-bottom: 2px solid __ACCENT__;
}

/* 按钮：radius 9、min-height 24、hover 只变底/边一档 */
QPushButton {
    background-color: __PANEL__;
    color: __INK__;
    border: 1px solid __HAIRLINE_STRONG__;
    border-radius: 9px;
    padding: 5px 14px;
    min-height: 24px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: __BG__;
    border-color: __INK2__;
}
QPushButton:pressed {
    background-color: __PANEL_ALT__;
}
QPushButton:focus {
    border-color: __ACCENT__;
}
QPushButton:disabled {
    color: __FAINT__;
    background-color: __PANEL_ALT__;
    border-color: __HAIRLINE__;
}
QPushButton[accent="true"] {
    background-color: __ACCENT__;
    border-color: __ACCENT__;
    color: #ffffff;
    font-weight: 600;
}
QPushButton[accent="true"]:hover {
    background-color: __ACCENT_HOVER__;
    border-color: __ACCENT_HOVER__;
}
QPushButton[accent="true"]:pressed {
    background-color: __ACCENT_PRESS__;
}
QPushButton[accent="true"]:disabled {
    background-color: __PANEL_ALT__;
    border-color: __HAIRLINE__;
    color: __FAINT__;
}
/* Danger：非实底——panel 底 + danger 边 + danger 字，hover 才上 danger-soft */
QPushButton[danger="true"]:!disabled {
    background-color: __PANEL__;
    color: __DANGER_INK__;
    border-color: __DANGER_BORDER__;
    font-weight: 600;
}
QPushButton[danger="true"]:hover:!disabled {
    background-color: __DANGER_SOFT__;
    border-color: __DANGER__;
}
QPushButton[danger="true"]:pressed:!disabled {
    background-color: __DANGER_SOFT__;
    border-color: __DANGER__;
}

QToolButton {
    background: transparent;
    color: __INK2__;
    border: 1px solid __HAIRLINE_STRONG__;
    border-radius: 9px;
    padding: 5px 12px;
    font-weight: 500;
}
QToolButton:hover {
    background-color: __BG__;
    border-color: __INK2__;
}
QToolButton:pressed {
    background-color: __PANEL_ALT__;
}
QToolButton:checked {
    background-color: __ACCENT_SOFT__;
    color: __ACCENT_INK__;
    border-color: __ACCENT_BORDER__;
}
QToolButton:disabled {
    color: __FAINT__;
    border-color: __HAIRLINE__;
}

/* 输入类：panel 底 + hairline-strong 边，focus 边变 accent */
QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox, QComboBox {
    background-color: __PANEL__;
    color: __INK__;
    border: 1px solid __HAIRLINE_STRONG__;
    border-radius: 9px;
    padding: 5px 10px;
    selection-background-color: __ACCENT_SOFT2__;
    selection-color: __INK__;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QAbstractSpinBox:focus, QComboBox:focus {
    border-color: __ACCENT__;
}
QPlainTextEdit[log="true"] {
    padding: 10px;
    font-family: "JetBrains Mono", "Consolas", "Menlo", monospace;
    font-size: 12px;
    color: __INK2__;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QAbstractSpinBox:disabled, QComboBox:disabled {
    color: __FAINT__;
    background-color: __PANEL_ALT__;
}

QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: __PANEL__;
    color: __INK__;
    border: 1px solid __HAIRLINE__;
    border-radius: 9px;
    padding: 4px;
    selection-background-color: __ACCENT_SOFT__;
    selection-color: __ACCENT_INK__;
}
QComboBox QAbstractItemView::item { border-radius: 6px; padding: 4px 8px; }

QAbstractItemView {
    background-color: __PANEL__;
    color: __INK__;
    border: 1px solid __HAIRLINE__;
    border-radius: 10px;
    selection-background-color: __ACCENT_SOFT__;
    selection-color: __INK__;
    outline: none;
}
QAbstractItemView::item { padding: 4px 6px; }
QAbstractItemView::item:hover { background-color: __PANEL_ALT__; }
QAbstractItemView::item:selected { background-color: __ACCENT_SOFT__; color: __INK__; }

/* 表格：斑马纹 panel-alt、表头 panel-alt、行间发丝线 */
QTableView {
    background-color: __PANEL__;
    alternate-background-color: __PANEL_ALT__;
    gridline-color: transparent;
    border: 1px solid __HAIRLINE__;
    border-radius: 10px;
    selection-background-color: __ACCENT_SOFT__;
    selection-color: __INK__;
}
QTableView::item { border-bottom: 1px solid __HAIRLINE__; }
QHeaderView::section {
    background-color: __PANEL_ALT__;
    color: __INK2__;
    border: none;
    border-bottom: 1px solid __HAIRLINE_STRONG__;
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
}
QTableCornerButton::section {
    background-color: __PANEL_ALT__;
    border: none;
}

QProgressBar {
    background-color: __PANEL_ALT__;
    color: __INK2__;
    border: 1px solid __HAIRLINE__;
    border-radius: 9px;
    text-align: center;
    font-size: 12px;
}
QProgressBar::chunk {
    background-color: __ACCENT__;
    border-radius: 8px;
}

/* 卡片：panel 底 + hairline 边 + radius 12；标题衬线 ink-2 */
QGroupBox {
    background-color: __PANEL__;
    border: 1px solid __HAIRLINE__;
    border-radius: 12px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
    font-weight: 500;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    font-family: "Noto Serif CJK SC", "Songti SC", "SimSun", serif;
    font-weight: 700;
    font-size: 12px;
    color: __INK2__;
}

QCheckBox, QRadioButton {
    color: __INK__;
    spacing: 6px;
    background: transparent;
    font-size: 12px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid __HAIRLINE_STRONG__;
    background-color: __PANEL__;
    border-radius: 5px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: __ACCENT__;
}
QCheckBox::indicator:checked {
    background-color: __ACCENT__;
    border-color: __ACCENT__;
}
QRadioButton::indicator { border-radius: 8px; }
QRadioButton::indicator:checked {
    background-color: __ACCENT__;
    border-color: __ACCENT__;
}

QScrollArea { background: transparent; border: none; }

/* 滚动条：8px、无箭头、圆角 handle */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: __SCROLL__;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background-color: __SCROLL_HOVER__; }
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: __SCROLL__;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background-color: __SCROLL_HOVER__; }
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0; height: 0; background: transparent;
}
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* 顶栏：等宽粗标题 + muted 副标题 + 项目操作 + 主题分段切换 */
QFrame#topBar {
    background-color: __BG__;
    border-bottom: 1px solid __HAIRLINE__;
}
QFrame#topBar QLabel#topTitle {
    font-family: "JetBrains Mono", "Consolas", "Menlo", monospace;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 1px;
    color: __INK__;
}
QFrame#topBar QLabel#topSubtitle {
    color: __MUTED__;
    font-size: 11px;
}
QFrame#topBar QToolButton {
    background: transparent;
    color: __INK2__;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 4px 10px;
    font-size: 12px;
}
QFrame#topBar QToolButton:hover {
    color: __INK__;
    border-color: __HAIRLINE_STRONG__;
}

/* 主题分段切换：外层 panel-alt 容器，选中项 panel 底浮起一档 */
QFrame#segWrap {
    background-color: __PANEL_ALT__;
    border: 1px solid __HAIRLINE__;
    border-radius: 9px;
}
QFrame#segWrap QToolButton {
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 4px 12px;
    color: __INK2__;
    font-size: 12px;
}
QFrame#segWrap QToolButton:checked {
    background-color: __PANEL__;
    color: __ACCENT_INK__;
    font-weight: 600;
}

/* 侧栏卡片（训练页表单等）：固定宽、内部滚动 */
QFrame#sideCard {
    background-color: __PANEL__;
    border: 1px solid __HAIRLINE__;
    border-radius: 12px;
}

/* 菜单 / 工具栏 / 状态栏：bg 同页面底，发丝线分隔 */
QMenuBar {
    background-color: __BG__;
    color: __INK__;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 13px;
}
QMenuBar::item:selected { background-color: __PANEL_ALT__; }
QMenu {
    background-color: __PANEL__;
    color: __INK__;
    border: 1px solid __HAIRLINE__;
    border-radius: 9px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 22px;
    border-radius: 6px;
}
QMenu::item:selected { background-color: __ACCENT_SOFT__; color: __ACCENT_INK__; }
QMenu::separator {
    height: 1px;
    background-color: __HAIRLINE__;
    margin: 4px 8px;
}

QToolBar {
    background-color: __BG__;
    border: none;
    border-bottom: 1px solid __HAIRLINE__;
    padding: 4px 8px;
    spacing: 6px;
}
QToolBar::separator {
    background-color: __HAIRLINE__;
    width: 1px;
    margin: 4px 6px;
}

QStatusBar {
    background-color: __SIDEBAR__;
    color: __INK2__;
    border-top: 1px solid __HAIRLINE__;
    font-size: 12px;
}
QStatusBar::item { border: none; }

/* 分割条：透明留白，hover 才显示发丝线 */
QSplitter::handle {
    background-color: transparent;
}
QSplitter::handle:hover {
    background-color: __HAIRLINE__;
}
"""


def _palette(pal: dict) -> QtGui.QPalette:
    p = QtGui.QPalette()
    qcolor = QtGui.QColor
    p.setColor(QtGui.QPalette.ColorRole.Window, qcolor(pal["BG"]))
    p.setColor(QtGui.QPalette.ColorRole.WindowText, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.Base, qcolor(pal["PANEL"]))
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase, qcolor(pal["PANEL_ALT"]))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipBase, qcolor(pal["TOOLTIP_BG"]))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipText, qcolor(pal["TOOLTIP_FG"]))
    p.setColor(QtGui.QPalette.ColorRole.Text, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.Button, qcolor(pal["PANEL"]))
    p.setColor(QtGui.QPalette.ColorRole.ButtonText, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.BrightText, qcolor("#ffffff"))
    p.setColor(QtGui.QPalette.ColorRole.Highlight, qcolor(pal["ACCENT_SOFT"]))
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, qcolor(pal["INK"]))
    p.setColor(QtGui.QPalette.ColorRole.Link, qcolor(pal["ACCENT"]))
    p.setColor(QtGui.QPalette.ColorRole.PlaceholderText, qcolor(pal["FAINT"]))

    disabled = QtGui.QPalette.ColorGroup.Disabled
    p.setColor(disabled, QtGui.QPalette.ColorRole.Text, qcolor(pal["FAINT"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.ButtonText, qcolor(pal["FAINT"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.WindowText, qcolor(pal["FAINT"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.Highlight, qcolor(pal["HAIRLINE_STRONG"]))
    p.setColor(disabled, QtGui.QPalette.ColorRole.HighlightedText, qcolor(pal["INK2"]))
    return p
