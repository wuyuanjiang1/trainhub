"""Builds a training-parameter form from a trainer's :class:`ParamSpec` list.

This is what makes the app extensible: a new trainer ships a new list of specs
and gets a working, validated form without any GUI changes.
"""

from __future__ import annotations

from typing import Any

from PyQt6 import QtCore
from PyQt6 import QtWidgets

from ..core.params import ParamSpec
from ..core.params import default_params
from ..core.params import grouped

_SPIN_LIMIT = 1_000_000_000


class ParamForm(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        specs: list[ParamSpec],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._specs = specs
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._getters: dict[str, Any] = {}
        self._setters: dict[str, Any] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for group_name, group_specs in grouped(specs).items():
            box = QtWidgets.QGroupBox(group_name)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(9)
            form.setContentsMargins(12, 10, 12, 12)
            for spec in group_specs:
                widget = self._build(spec)
                if spec.help:
                    widget.setToolTip(spec.help)
                label = QtWidgets.QLabel(spec.label)
                label.setProperty("dim", True)
                label.setToolTip(spec.help)
                form.addRow(label, widget)
            layout.addWidget(box)

        layout.addStretch(1)

    # ------------------------------------------------------------- building
    def _build(self, spec: ParamSpec) -> QtWidgets.QWidget:
        widget: QtWidgets.QWidget
        if spec.type == "int":
            spin = QtWidgets.QSpinBox()
            low = int(spec.minimum) if spec.minimum is not None else -_SPIN_LIMIT
            high = int(spec.maximum) if spec.maximum is not None else _SPIN_LIMIT
            spin.setRange(max(low, -_SPIN_LIMIT), min(high, _SPIN_LIMIT))
            if spec.step:
                spin.setSingleStep(int(spec.step))
            spin.setValue(int(spec.default))
            spin.setMinimumHeight(30)
            spin.valueChanged.connect(self.valueChanged)
            self._getters[spec.key] = spin.value
            self._setters[spec.key] = spin.setValue
            widget = spin

        elif spec.type == "float":
            spin = QtWidgets.QDoubleSpinBox()
            low = spec.minimum if spec.minimum is not None else -_SPIN_LIMIT
            high = spec.maximum if spec.maximum is not None else _SPIN_LIMIT
            spin.setRange(low, high)
            spin.setDecimals(spec.decimals)
            spin.setSingleStep(spec.step if spec.step else 10 ** (-spec.decimals))
            spin.setValue(float(spec.default))
            spin.setMinimumHeight(30)
            spin.valueChanged.connect(self.valueChanged)
            self._getters[spec.key] = spin.value
            self._setters[spec.key] = spin.setValue
            widget = spin

        elif spec.type == "bool":
            check = QtWidgets.QCheckBox()
            check.setChecked(bool(spec.default))
            check.toggled.connect(self.valueChanged)
            self._getters[spec.key] = check.isChecked
            self._setters[spec.key] = check.setChecked
            widget = check

        elif spec.type == "choice":
            combo = QtWidgets.QComboBox()
            combo.addItems(list(spec.choices))
            combo.setEditable(spec.allow_custom)
            if spec.allow_custom:
                combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            combo.setCurrentText(str(spec.default))
            combo.setMinimumHeight(30)
            combo.currentTextChanged.connect(self.valueChanged)
            self._getters[spec.key] = combo.currentText
            self._setters[spec.key] = combo.setCurrentText
            widget = combo

        elif spec.type == "path":
            container = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            edit = QtWidgets.QLineEdit(str(spec.default))
            edit.setMinimumHeight(30)
            button = QtWidgets.QPushButton("浏览…")
            button.setFixedWidth(70)
            button.clicked.connect(lambda: self._browse(edit))
            edit.textChanged.connect(self.valueChanged)
            row.addWidget(edit)
            row.addWidget(button)
            self._getters[spec.key] = edit.text
            self._setters[spec.key] = edit.setText
            widget = container

        else:  # str
            edit = QtWidgets.QLineEdit(str(spec.default))
            edit.setMinimumHeight(30)
            edit.textChanged.connect(self.valueChanged)
            self._getters[spec.key] = edit.text
            self._setters[spec.key] = edit.setText
            widget = edit

        self._editors[spec.key] = widget
        return widget

    def _browse(self, edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件")
        if path:
            edit.setText(path)

    # ---------------------------------------------------------------- values
    def values(self) -> dict[str, Any]:
        raw = {key: getter() for key, getter in self._getters.items()}
        # Coerce through the specs so a free-text combo entry becomes a number
        # where the spec says so.
        out: dict[str, Any] = dict(default_params(self._specs))
        for spec in self._specs:
            value = raw.get(spec.key, spec.default)
            try:
                out[spec.key] = spec.clamp(value)
            except (TypeError, ValueError):
                out[spec.key] = spec.default
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        for spec in self._specs:
            if spec.key not in values:
                continue
            try:
                self._setters[spec.key](values[spec.key])
            except (TypeError, ValueError):
                continue

    def reset(self) -> None:
        self.set_values(default_params(self._specs))
