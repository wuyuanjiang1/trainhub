"""顶栏：等宽工具名 + 项目信息 + 项目操作按钮 + 主题分段切换。

替代传统菜单栏/工具栏——简约科技风：动作平铺在顶栏，主题切换用分段控件。
"""

from __future__ import annotations

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

from . import theme


class TopBar(QtWidgets.QFrame):
    new_project_requested = QtCore.pyqtSignal()
    open_project_requested = QtCore.pyqtSignal()
    save_project_requested = QtCore.pyqtSignal()
    about_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("topBar")

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(16, 8, 16, 8)
        row.setSpacing(12)

        title = QtWidgets.QLabel("trainhub")
        title.setObjectName("topTitle")
        self._subtitle = QtWidgets.QLabel("标注 · 训练 · 推理")
        self._subtitle.setObjectName("topSubtitle")
        row.addWidget(title)
        row.addWidget(self._subtitle)
        row.addStretch(1)

        def small(text: str, tip: str | None, slot) -> QtWidgets.QToolButton:
            button = QtWidgets.QToolButton()
            button.setText(text)
            if tip:
                button.setToolTip(tip)
            button.clicked.connect(slot)
            return button

        row.addWidget(small("新建项目", "Ctrl+N", self.new_project_requested.emit))
        row.addWidget(small("打开项目", "Ctrl+Shift+O", self.open_project_requested.emit))
        row.addWidget(small("保存项目", "Ctrl+Shift+S", self.save_project_requested.emit))
        about = small("?", "关于 trainhub", self.about_requested.emit)
        about.clicked.connect(lambda: self.about_requested.emit())
        row.addWidget(about)
        row.addSpacing(4)

        # 主题分段切换（规范：外层 panel-alt 圆角容器 + 互斥按钮）
        seg = QtWidgets.QFrame()
        seg.setObjectName("segWrap")
        seg_row = QtWidgets.QHBoxLayout(seg)
        seg_row.setContentsMargins(2, 2, 2, 2)
        seg_row.setSpacing(2)
        self._light_button = QtWidgets.QToolButton()
        self._light_button.setText("浅色")
        self._light_button.setCheckable(True)
        self._dark_button = QtWidgets.QToolButton()
        self._dark_button.setText("深色")
        self._dark_button.setCheckable(True)
        group = QtWidgets.QButtonGroup(self)
        group.addButton(self._light_button)
        group.addButton(self._dark_button)
        self._light_button.toggled.connect(self._on_toggle)
        seg_row.addWidget(self._light_button)
        seg_row.addWidget(self._dark_button)
        row.addWidget(seg)

        self.set_theme_state(theme.is_dark())
        theme.subscribe_theme_listener(self.set_theme_state)

    def _on_toggle(self, checked: bool) -> None:
        # checked 的是浅色按钮 → dark = not checked
        theme.set_dark(QtWidgets.QApplication.instance(), not checked)

    def set_theme_state(self, dark: bool) -> None:
        self._dark_button.setChecked(dark)
        self._light_button.setChecked(not dark)

    def set_project(self, name: str) -> None:
        self._subtitle.setText(f"{name} · 标注 · 训练 · 推理")
