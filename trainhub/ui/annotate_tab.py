"""标注页：把移植过来的 labelme 画布接进项目工作流，交互忠实复刻 labelme。

Images live in ``<project>/images`` and labelme-format JSON in
``<project>/annotations``; saving keeps the labelme schema untouched so the
files can be opened by labelme itself.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import imgviz
import numpy as np
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets

from ..annotator import Canvas
from ..annotator import LabelDialog
from ..annotator import LabelListWidget
from ..annotator import Shape
from ..annotator import ShapeClipboard
from ..annotator import UniqueLabelQListWidget
from ..annotator import ZoomWidget
from ..annotator import read_label_file
from ..annotator import utils
from ..annotator import write_label_file
from ..annotator._qt import add_actions
from ..annotator.widgets.label_list_widget import LabelListWidgetItem
from ..annotator.widgets.label_list_widget import format_shape_label
from ..core.dataset import IMAGE_SUFFIXES
from ..core.prelabel import find_project_weights
from ..core.prelabel import predict_shapes
from ..core.prelabel_vlm import VLM_PROVIDERS
from ..core.prelabel_vlm import predict_shapes_vlm
from ..core.project import Project
from ..trainers.yolo.converter import import_yolo_dataset
from .prelabel_dialog import PrelabelDialog
from .prelabel_dialog import PrelabelWorker
from .theme import section_label

LABEL_COLORMAP = imgviz.label_colormap()

# 形状模式及其中文名与快捷键。方向与 labelme 的左侧竖排工具栏一致。
_SHAPE_MODES: tuple[tuple[str, str, str | None], ...] = (
    ("polygon", "多边形", "Ctrl+P"),
    ("rectangle", "矩形", "Ctrl+R"),
    ("oriented_rectangle", "定向矩形", None),
    ("circle", "圆形", None),
    ("point", "点", None),
    ("line", "直线", None),
    ("linestrip", "折线", None),
)


def _rgb_from_colormap_id(*, label_id: int) -> tuple[int, int, int]:
    r, g, b = LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)].tolist()
    return r, g, b


def _new_action(
    parent: QtWidgets.QWidget,
    text: str,
    slot=None,
    shortcut: str | list[str] | tuple[str, ...] | None = None,
    tip: str | None = None,
    enabled: bool = True,
    checkable: bool = False,
    checked: bool = False,
) -> QtGui.QAction:
    action = QtGui.QAction(text, parent)
    if shortcut:
        if isinstance(shortcut, (list, tuple)):
            action.setShortcuts(list(shortcut))
        else:
            action.setShortcut(shortcut)
    if tip:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        action.triggered.connect(slot)
    action.setEnabled(enabled)
    if checkable:
        action.setCheckable(True)
        action.setChecked(checked)
    # 只在标注页内部生效，避免抢占其它页签文本输入框的 Ctrl+C/V/Z 等。
    action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
    return action


def shape_to_dict(shape: Shape) -> dict:
    data = dict(shape.other_data)
    data.update(
        label=shape.label,
        points=[[float(p.x()), float(p.y())] for p in shape.points],
        group_id=shape.group_id,
        description=shape.description,
        shape_type=shape.shape_type,
        flags=shape.flags,
        mask=None
        if shape.mask is None
        else utils.img_arr_to_b64(shape.mask.astype(np.uint8)),
    )
    return data


def dict_to_shape(raw: dict) -> Shape:
    shape = Shape(
        label=raw.get("label"),
        shape_type=raw.get("shape_type", "polygon"),
        flags=dict(raw.get("flags") or {}),
        group_id=raw.get("group_id"),
        description=raw.get("description") or "",
    )
    # 保留 VLM 预标注回传的 other_data（如 label_cn 中英对照）
    shape.other_data = dict(raw.get("other_data") or {})
    shape.points = [
        QtCore.QPointF(float(x), float(y)) for x, y in (raw.get("points") or [])
    ]
    mask_b64 = raw.get("mask")
    if mask_b64:
        shape.mask = utils.img_b64_to_arr(mask_b64).astype(bool)
    return shape


class AnnotateTab(QtWidgets.QWidget):
    labelsChanged = QtCore.pyqtSignal(list)
    imagesChanged = QtCore.pyqtSignal()
    statusMessage = QtCore.pyqtSignal(str)
    dirtyChanged = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._items: list[Path] = []
        self._index = -1
        self._image_path: Path | None = None
        self._dirty = False
        self._zoom_mode = "fit"
        self._shape_clipboard = ShapeClipboard(self)
        self._prelabel_dialog: PrelabelDialog | None = None
        self._prelabel_worker: PrelabelWorker | None = None
        self._prelabel_new_labels: set[str] = set()
        self._prelabel_written = 0
        self._prelabel_applied = 0

        self._label_dialog = LabelDialog(
            text="输入或选择标签",
            parent=self,
            labels=list(project.labels),
        )

        self._canvas = Canvas()
        self._canvas.set_fill_drawing(True)
        self._canvas.zoom_request.connect(self._zoom_requested)
        self._canvas.scroll_request.connect(self._on_scroll_request)
        self._canvas.pan_request.connect(self._on_pan_request)
        self._canvas.new_shape.connect(self._on_new_shape)
        self._canvas.selection_changed.connect(self._on_shape_selection_changed)
        self._canvas.shape_moved.connect(self.mark_dirty)
        self._canvas.drawing_polygon.connect(self._on_drawing_polygon_changed)
        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.status_updated.connect(self.statusMessage)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self._zoom_widget = ZoomWidget()
        self._zoom_widget.valueChanged.connect(self._paint_canvas)

        self._unique_label_list = UniqueLabelQListWidget()
        self._unique_label_list.setToolTip("选择标签后即可绘制；按 Esc 取消选择")
        self._unique_label_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._unique_label_list.customContextMenuRequested.connect(
            self._show_unique_label_menu
        )

        self._label_list = LabelListWidget()
        self._label_list.item_selection_changed.connect(self._label_selection_changed)
        self._label_list.item_double_clicked.connect(self._edit_label)
        self._label_list.item_changed.connect(self._on_label_item_changed)
        self._label_list.item_dropped.connect(self._on_label_order_changed)
        self._label_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._label_list.customContextMenuRequested.connect(self._show_label_list_menu)

        self._file_list = QtWidgets.QListWidget()
        self._file_list.currentRowChanged.connect(self._on_file_row_changed)

        self._coords_label = QtWidgets.QLabel(" ")

        self._build_actions()
        self._build_toolbar()
        self._build_layout()
        self._populate_canvas_context_menu()

        self._switch_canvas_mode(edit=True)
        self.set_project(project)

    # -------------------------------------------------------------- actions
    def _build_actions(self) -> None:
        self._edit_mode_action = _new_action(
            self,
            "编辑标注",
            lambda: self._switch_canvas_mode(edit=True),
            "Ctrl+J",
            "选中并拖动已有标注（Ctrl+J）",
            enabled=False,
        )
        self._undo_action = _new_action(
            self,
            "撤销",
            self.undo_shape_edit,
            "Ctrl+Z",
            "撤销上一次添加或编辑",
            enabled=False,
        )
        self._undo_last_point_action = _new_action(
            self,
            "撤销上一点",
            self._canvas.undo_last_point,
            "Ctrl+Z",
            "撤销绘制中的上一个点",
            enabled=False,
        )
        self._delete_action = _new_action(
            self,
            "删除标注",
            self.delete_selected_shapes,
            "Delete",
            "删除选中的标注",
            enabled=False,
        )
        self._prelabel_action = _new_action(
            self,
            "AI 预标注",
            self.open_prelabel_dialog,
            "Ctrl+Shift+P",
            "用 YOLO 模型自动生成候选标注，可单张或批量（Ctrl+Shift+P）",
            enabled=False,
        )
        self._edit_label_action = _new_action(
            self,
            "编辑标签",
            self._edit_label,
            "Ctrl+E",
            "修改选中标注的标签",
            enabled=False,
        )
        self._duplicate_action = _new_action(
            self,
            "复制标注",
            self.duplicate_selected_shapes,
            "Ctrl+D",
            "复制选中的标注",
            enabled=False,
        )
        self._copy_action = _new_action(
            self,
            "复制形状",
            self.copy_selected_shapes,
            "Ctrl+C",
            "复制选中的形状到剪贴板",
            enabled=False,
        )
        self._paste_action = _new_action(
            self,
            "粘贴形状",
            self.paste_shapes,
            "Ctrl+V",
            "粘贴剪贴板中的形状",
            enabled=False,
        )
        self._remove_point_action = _new_action(
            self,
            "删除选中点",
            self.remove_selected_point,
            "Backspace",
            "从多边形删除选中点",
            enabled=False,
        )
        self._add_point_to_edge_action = _new_action(
            self,
            "在边上加点",
            self._canvas.add_point_to_edge,
            tip="在悬停的边上插入一个新点",
            enabled=False,
        )

        self._draw_actions: list[tuple[str, QtGui.QAction]] = []
        for mode, label, shortcut in _SHAPE_MODES:
            action = _new_action(
                self,
                label,
                lambda checked=False, m=mode: self._switch_canvas_mode(
                    edit=False, create_mode=m
                ),
                shortcut,
                f"开始绘制{label}" if shortcut is None else f"开始绘制{label}（{shortcut}）",
                enabled=False,
            )
            self._draw_actions.append((mode, action))

        self._save_action = _new_action(self, "保存", self.save_current, "Ctrl+S", "保存当前标注")
        self._delete_image_action = _new_action(
            self, "删除当前图像", self.delete_current_image, tip="删除当前图像及其标注文件"
        )
        self._open_prev_action = _new_action(
            self, "上一张", self.prev_image, "Ctrl+Shift+A", "上一张图像"
        )
        self._open_next_action = _new_action(
            self, "下一张", self.next_image, "Ctrl+Shift+D", "下一张图像"
        )
        self._import_files_action = _new_action(self, "导入图像…", self.import_images, tip="导入图像文件")
        self._import_folder_action = _new_action(
            self, "导入文件夹…", self.import_folder, tip="导入图像文件夹"
        )

        self._zoom_in_action = _new_action(
            self, "放大", lambda: self._add_zoom(1.1), ["Ctrl++", "Ctrl+="], "放大", enabled=False
        )
        self._zoom_out_action = _new_action(
            self, "缩小", lambda: self._add_zoom(0.9), "Ctrl+-", "缩小", enabled=False
        )
        self._zoom_org_action = _new_action(
            self, "原始大小", self._set_zoom_to_original, "Ctrl+0", "缩放到原始大小", enabled=False
        )
        self._fit_action = _new_action(
            self,
            "适应窗口",
            self.set_fit_window_mode,
            "Ctrl+F",
            "缩放跟随窗口大小",
            enabled=False,
            checkable=True,
            checked=True,
        )

        self._context_menu_actions: tuple[QtGui.QAction, ...] = (
            *[action for _, action in self._draw_actions],
            self._edit_mode_action,
            self._edit_label_action,
            self._duplicate_action,
            self._copy_action,
            self._paste_action,
            self._delete_action,
            self._undo_action,
            self._undo_last_point_action,
            self._add_point_to_edge_action,
            self._remove_point_action,
        )

    # -------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        self._toolbar = QtWidgets.QToolBar("标注")
        self._toolbar.setMovable(False)
        for action in (
            self._import_files_action,
            self._import_folder_action,
            None,
            self._open_prev_action,
            self._open_next_action,
            None,
            self._save_action,
            self._delete_image_action,
            None,
            self._prelabel_action,
            None,
            self._edit_mode_action,
            self._duplicate_action,
            self._delete_action,
            self._undo_action,
            None,
            self._fit_action,
            self._zoom_out_action,
            self._zoom_in_action,
            self._zoom_org_action,
        ):
            if action is None:
                self._toolbar.addSeparator()
            else:
                self._toolbar.addAction(action)
        self._toolbar.addWidget(self._zoom_widget)

        self._shape_toolbar = QtWidgets.QToolBar("形状工具")
        self._shape_toolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self._shape_toolbar.setMovable(False)
        for _, action in self._draw_actions:
            self._shape_toolbar.addAction(action)

    def _build_layout(self) -> None:
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(section_label("文件列表"))
        left_layout.addWidget(self._file_list, 1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(section_label("标签列表"))
        right_layout.addWidget(self._unique_label_list, 1)
        right_layout.addWidget(section_label("标注列表"))
        right_layout.addWidget(self._label_list, 2)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._scroll)
        splitter.addWidget(right)
        splitter.setSizes([200, 800, 280])

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self._shape_toolbar)
        body.addWidget(splitter, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addLayout(body, 1)
        layout.addWidget(self._coords_label)

    def _populate_canvas_context_menu(self) -> None:
        self._canvas.menus[0].clear()
        add_actions(self._canvas.menus[0], self._context_menu_actions)
        add_actions(
            self._canvas.menus[1],
            (
                _new_action(self, "复制到此处", self.copy_shape),
                _new_action(self, "移动到此处", self.move_shape),
            ),
        )

    def _show_label_list_menu(self, point: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu()
        menu.addAction(self._edit_label_action)
        menu.addAction(self._delete_action)
        menu.exec(self._label_list.mapToGlobal(point))

    def _show_unique_label_menu(self, point: QtCore.QPoint) -> None:
        item = self._unique_label_list.itemAt(point)
        menu = QtWidgets.QMenu()
        if item is not None:
            label = item.data(QtCore.Qt.ItemDataRole.UserRole)
            delete_action = menu.addAction(f"删除标签「{label}」")
            delete_action.triggered.connect(lambda: self._delete_label(label))
        menu.exec(self._unique_label_list.mapToGlobal(point))

    def _delete_label(self, label: str) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "删除标签",
            f"确定从项目标签列表中删除「{label}」吗？\n\n"
            "已有标注对象上的该标签不会被修改。",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        item = self._unique_label_list.find_label_item(label)
        if item is not None:
            self._unique_label_list.takeItem(self._unique_label_list.row(item))

        for i in range(self._label_dialog.label_list.count() - 1, -1, -1):
            if self._label_dialog.label_list.item(i).text() == label:
                self._label_dialog.label_list.takeItem(i)

        if self._project.remove_label(label):
            self._project.save()
            self.labelsChanged.emit(list(self._project.labels))
        self.statusMessage.emit(f"已删除标签「{label}」")

    # -------------------------------------------------------------- project
    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def set_project(self, project: Project) -> None:
        if self._dirty and not self._can_continue():
            return
        self._project = project
        self._label_dialog = LabelDialog(
            text="输入或选择标签",
            parent=self,
            labels=list(project.labels),
        )
        self._rebuild_unique_label_list()
        self._reset_state()
        self.refresh_file_list()

    def _rebuild_unique_label_list(self) -> None:
        self._unique_label_list.clear()
        for label in self._project.labels:
            self._unique_label_list.add_label_item(
                label=label,
                color=self._get_rgb_by_label(label, self._unique_label_list),
            )

    def _reset_state(self) -> None:
        self._label_list.clear()
        self._image_path = None
        self._index = -1
        self._canvas.reset_state()

    def refresh_file_list(self) -> None:
        self._items = sorted(
            path
            for path in self._project.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self._file_list.blockSignals(True)
        self._file_list.clear()
        for path in self._items:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
            if self._label_path(path).exists():
                item.setToolTip("已标注")
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            else:
                item.setToolTip("未标注")
            self._file_list.addItem(item)
        self._file_list.blockSignals(False)

        if self._items:
            self._file_list.setCurrentRow(0)
        else:
            self._reset_state()
            self.statusMessage.emit("还没有图像，点击「导入图像…」导入")

    def _label_path(self, image_path: Path) -> Path:
        return self._project.annotations_dir / f"{image_path.stem}.json"

    # ------------------------------------------------------------ load/save
    def _on_file_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._items):
            return
        if not self._can_continue():
            self._file_list.blockSignals(True)
            self._file_list.setCurrentRow(self._index)
            self._file_list.blockSignals(False)
            return
        self.load_image(row)

    def load_image(self, row: int) -> None:
        path = self._items[row]
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            self.statusMessage.emit(f"无法读取图像: {path.name}")
            return

        self._reset_state()
        self._index = row
        self._image_path = path
        self._canvas.load_pixmap(pixmap, clear_shapes=True)

        shapes: list[Shape] = []
        label_path = self._label_path(path)
        if label_path.exists():
            try:
                data = read_label_file(str(label_path))
            except Exception as exc:
                self.statusMessage.emit(f"读取标注失败: {exc}")
            else:
                shapes = [dict_to_shape(raw) for raw in data.shapes]
        self._load_shapes(shapes, replace=True)

        self._zoom_mode = "fit"
        self._adjust_scale()
        self._enable_loaded_actions()
        self.set_dirty(False)
        self.statusMessage.emit(
            f"{path.name} — {pixmap.width()}x{pixmap.height()}，{len(shapes)} 个对象"
        )

    def _enable_loaded_actions(self) -> None:
        for _, draw_action in self._draw_actions:
            draw_action.setEnabled(True)
        self._zoom_in_action.setEnabled(True)
        self._zoom_out_action.setEnabled(True)
        self._zoom_org_action.setEnabled(True)
        self._fit_action.setEnabled(True)
        self._prelabel_action.setEnabled(True)

    def save_current(self) -> bool:
        if self._image_path is None:
            return False
        image_path = self._image_path
        label_path = self._label_path(image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        shapes = [
            shape_to_dict(s)
            for item in self._label_list
            if (s := item.shape()) is not None
        ]
        pixmap = self._canvas.pixmap
        try:
            write_label_file(
                filename=str(label_path),
                shapes=shapes,
                image_path=os.path.relpath(image_path, label_path.parent),
                image_height=pixmap.height() if not pixmap.isNull() else None,
                image_width=pixmap.width() if not pixmap.isNull() else None,
                flags={},
            )
        except Exception as exc:
            self.statusMessage.emit(f"保存失败: {exc}")
            return False

        labels = [
            s.label
            for item in self._label_list
            if (s := item.shape()) is not None and s.label
        ]
        if self._project.add_labels(labels):
            self._project.save()
            self.labelsChanged.emit(list(self._project.labels))

        item = self._file_list.item(self._index)
        if item is not None:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setToolTip("已标注")

        self.set_dirty(False)
        self.statusMessage.emit(f"已保存 {label_path.name}")
        return True

    # ------------------------------------------------------------ AI 预标注
    def open_prelabel_dialog(self) -> None:
        if not self._items:
            QtWidgets.QMessageBox.information(self, "AI 预标注", "请先导入图像。")
            return
        if self._prelabel_worker is not None and self._prelabel_worker.isRunning():
            return

        weight_options: list[tuple[str, str]] = [
            (f"{w.parent.parent.name} / {w.name}（项目训练）", str(w))
            for w in find_project_weights(self._project.root)
        ]
        weight_options.append(("yolo11n.pt（通用预训练）", "yolo11n.pt"))
        current_image = str(self._image_path) if self._image_path else None
        unlabeled = sum(1 for p in self._items if not self._label_path(p).exists())

        def job_provider(scope: str) -> list[tuple[Path, Path]]:
            if scope == "current":
                if self._image_path is None:
                    return []
                return [(self._image_path, self._label_path(self._image_path))]
            return [
                (p, self._label_path(p))
                for p in self._items
                if not self._label_path(p).exists()
            ]

        self._prelabel_dialog = PrelabelDialog(
            self,
            weight_options=weight_options,
            current_image=current_image,
            unlabeled_count=unlabeled,
            job_provider=job_provider,
        )
        self._prelabel_dialog.start_requested.connect(self._start_prelabel)
        self._prelabel_dialog.show()

    def _start_prelabel(self, params: dict) -> None:
        engine = params.get("engine", "yolo")
        if engine == "vlm":
            provider = VLM_PROVIDERS[params["provider"]]
            labels = list(self._project.labels)

            def predict_fn(image_path: Path) -> list[dict]:
                return predict_shapes_vlm(
                    str(image_path),
                    provider=provider,
                    api_key=params["api_key"],
                    model=params["model"],
                    base_url=params["base_url"],
                    labels=labels,
                    hint=params.get("hint", ""),
                    prompt=params.get("prompt", ""),
                )

            note = provider.display_name
            if params["write"]:
                note += "；图像将上传至该服务商"
        else:
            weights, conf, device = params["weights"], params["conf"], params["device"]

            def predict_fn(image_path: Path) -> list[dict]:
                return predict_shapes(str(image_path), weights, conf, device=device)

            note = f"计算设备 {params['device']}"
            if params.get("device_warning"):
                note += f"；{params['device_warning']}"

        worker = PrelabelWorker(
            jobs=params["jobs"],
            predict_fn=predict_fn,
            write_to_disk=params["write"],
            parent=self,
        )
        worker.progress.connect(self._prelabel_dialog.on_progress)
        worker.image_ready.connect(self._on_prelabel_image_ready)
        worker.file_done.connect(self._on_prelabel_file_done)
        worker.failed.connect(self._on_prelabel_failed)
        worker.finished.connect(self._on_prelabel_finished)
        self._prelabel_worker = worker
        self._prelabel_dialog.worker = worker
        self._prelabel_new_labels = set()
        self._prelabel_written = 0
        self._prelabel_applied = 0
        self._prelabel_dialog.begin_run(len(params["jobs"]), note)
        if params["write"]:
            self.statusMessage.emit("AI 预标注：批量推理中 …")
        worker.start()

    def _on_prelabel_image_ready(self, image_path: str, shape_dicts: list) -> None:
        if self._image_path is None or str(self._image_path) != image_path:
            return
        new_shapes = [dict_to_shape(raw) for raw in shape_dicts]
        if not new_shapes:
            self.statusMessage.emit("AI 预标注：本图未检出候选对象，可尝试调低置信度")
            return
        self._canvas.load_shapes(shapes=new_shapes, replace=False)
        self._label_list.clear()
        self._load_shapes(self._canvas.shapes)
        self.set_dirty(True)
        self._undo_action.setEnabled(self._canvas.can_restore_shape)
        self._prelabel_applied += len(new_shapes)
        self._register_prelabel_labels({s.label for s in new_shapes if s.label})
        self.statusMessage.emit(
            f"AI 预标注：当前图新增 {len(new_shapes)} 个候选，请检查微调后保存"
        )

    def _on_prelabel_file_done(self, image_path: str, count: int, labels: list) -> None:
        self._prelabel_written += 1
        self._prelabel_new_labels.update(label for label in labels if label)
        try:
            row = self._items.index(Path(image_path))
        except ValueError:
            return
        item = self._file_list.item(row)
        if item is not None:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setToolTip("已标注（AI 预标注）")

    def _on_prelabel_failed(self, message: str) -> None:
        self.statusMessage.emit(f"AI 预标注失败: {message}")
        QtWidgets.QMessageBox.critical(self, "AI 预标注失败", message)

    def _on_prelabel_finished(self) -> None:
        worker = self._prelabel_worker
        cancelled = bool(worker is not None and getattr(worker, "_cancelled", False))
        self._register_prelabel_labels(self._prelabel_new_labels)
        dialog = self._prelabel_dialog
        if dialog is not None:
            dialog.on_finished()
            if cancelled:
                dialog.mark_done("已停止。已处理的图像保持有效。")
            elif worker is not None and getattr(worker, "_write", False):
                dialog.mark_done(
                    f"完成，共为 {self._prelabel_written} 张图像写入候选标注。\n"
                    "候选可能有漏检/误检，请逐张检查微调后保存。"
                )
            else:
                dialog.mark_done(
                    f"完成，已在当前图像添加 {self._prelabel_applied} 个候选标注。\n"
                    "请检查微调后保存；不满意可点撤销。"
                )
        self.statusMessage.emit(
            "AI 预标注已停止" if cancelled else f"AI 预标注完成：{self._prelabel_written} 张"
        )
        self._prelabel_worker = None

    def _register_prelabel_labels(self, labels: set[str]) -> None:
        if labels and self._project.add_labels(sorted(labels)):
            self._project.save()
            self.labelsChanged.emit(list(self._project.labels))

    def import_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择图像", "", "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        self._project.images_dir.mkdir(parents=True, exist_ok=True)
        imported_labels: set[str] = set()
        imported = 0
        for path in paths:
            src = Path(path)
            target = self._project.images_dir / src.name
            if target.exists():
                continue
            imported_labels |= self._copy_image_with_annotation(src, target)
            imported += 1
        self._merge_imported_labels(imported_labels)
        self.refresh_file_list()
        self.statusMessage.emit(f"已导入 {imported} 张图像")
        self.imagesChanged.emit()

    def import_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图像目录")
        if not folder:
            return
        folder = Path(folder)
        if self._is_yolo_dataset(folder):
            self._import_yolo_folder(folder)
            return
        if not self._confirm_and_clear():
            return
        self._reset_state()
        self.set_dirty(False)

        imported_labels: set[str] = set()
        copied = 0
        self._project.images_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                target = self._project.images_dir / path.name
                if target.exists():
                    continue
                imported_labels |= self._copy_image_with_annotation(path, target)
                copied += 1
        self._merge_imported_labels(imported_labels)
        self.refresh_file_list()
        self.statusMessage.emit(f"从文件夹导入 {copied} 张图像")
        self.imagesChanged.emit()

    def _is_yolo_dataset(self, folder: Path) -> bool:
        return (folder / "images").is_dir() and (folder / "labels").is_dir()

    def _confirm_and_clear(self) -> bool:
        if not self._has_existing_data():
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "替换数据集",
            "导入新数据集会清空当前项目的所有图像与标注，是否继续？",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        self._clear_dataset()
        return True

    def _import_yolo_folder(self, folder: Path) -> None:
        if not self._confirm_and_clear():
            return
        self._reset_state()
        self.set_dirty(False)
        result = import_yolo_dataset(
            folder, self._project.images_dir, self._project.annotations_dir
        )
        if result.labels and self._project.add_labels(result.labels):
            self._project.save()
            self.labelsChanged.emit(list(self._project.labels))
        self.refresh_file_list()
        self.statusMessage.emit(
            f"已导入 YOLO 数据集 {result.imported} 张（{result.annotated} 张已标注）"
        )
        self.imagesChanged.emit()

    def _has_existing_data(self) -> bool:
        for directory in (self._project.images_dir, self._project.annotations_dir):
            if directory.exists() and any(directory.iterdir()):
                return True
        return False

    def _clear_dataset(self) -> None:
        for directory in (self._project.images_dir, self._project.annotations_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def _copy_image_with_annotation(self, src: Path, target: Path) -> set[str]:
        shutil.copy2(src, target)
        labels: set[str] = set()
        json_src = src.with_suffix(".json")
        if json_src.exists():
            json_target = self._project.annotations_dir / f"{target.stem}.json"
            json_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_src, json_target)
            labels = self._labels_from_json(json_src)
        return labels

    def _labels_from_json(self, json_path: Path) -> set[str]:
        labels: set[str] = set()
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            for shape in data.get("shapes") or []:
                label = shape.get("label")
                if label:
                    labels.add(str(label))
        except (OSError, ValueError, TypeError):
            pass
        return labels

    def _merge_imported_labels(self, labels: set[str]) -> None:
        if labels and self._project.add_labels(sorted(labels)):
            self._project.save()
            self.labelsChanged.emit(list(self._project.labels))

    def delete_current_image(self) -> None:
        if self._image_path is None or self._index < 0:
            return
        image_path = self._image_path
        old_index = self._index
        answer = QtWidgets.QMessageBox.question(
            self,
            "删除当前图像",
            f"确定删除当前图像「{image_path.name}」吗？\n\n"
            "图像文件与对应的标注文件都会被永久删除。",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        for path in (self._label_path(image_path), image_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")

        self._items.pop(old_index)
        self._file_list.blockSignals(True)
        self._file_list.takeItem(old_index)
        self._file_list.blockSignals(False)

        self._reset_state()
        self.set_dirty(False)
        self.imagesChanged.emit()

        if self._items:
            new_row = min(old_index, len(self._items) - 1)
            self.load_image(new_row)
            self._file_list.blockSignals(True)
            self._file_list.setCurrentRow(new_row)
            self._file_list.blockSignals(False)
        else:
            self.statusMessage.emit("已删除当前图像，项目中没有图像了")

        if errors:
            self.statusMessage.emit("删除时部分文件失败: " + "；".join(errors))
        else:
            self.statusMessage.emit(f"已删除 {image_path.name}")

    # -------------------------------------------------------------- editing
    def _switch_canvas_mode(
        self, edit: bool = True, create_mode: str | None = None
    ) -> None:
        self._canvas.set_editing(edit)
        if create_mode is not None:
            self._canvas.create_mode = create_mode
        if edit:
            for _, draw_action in self._draw_actions:
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in self._draw_actions:
                draw_action.setEnabled(create_mode != draw_mode)
        self._edit_mode_action.setEnabled(not edit)

    def _on_new_shape(self) -> None:
        items = self._unique_label_list.selectedItems()
        text = None
        if items:
            text = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        flags: dict[str, bool] = {}
        group_id = None
        description = ""
        previous_text = self._label_dialog.edit.text()
        text, flags, group_id, description = self._label_dialog.popup(text)
        if not text:
            self._label_dialog.edit.setText(previous_text)

        if text:
            self._label_list.clearSelection()
            shapes = self._canvas.set_last_label(text, flags)
            for shape in shapes:
                shape.group_id = group_id
                shape.description = description
                self.add_label(shape)
            self._edit_mode_action.setEnabled(True)
            self._undo_last_point_action.setEnabled(False)
            self._undo_action.setEnabled(True)
            self.mark_dirty()
        else:
            self._canvas.undo_last_line()
            self._canvas.shape_backups.pop()

    def add_label(self, shape: Shape) -> None:
        assert shape.label is not None
        label_list_item = LabelListWidgetItem(shape=shape)
        self._label_list.add_item(label_list_item)
        if self._unique_label_list.find_label_item(shape.label) is None:
            self._unique_label_list.add_label_item(
                label=shape.label,
                color=self._get_rgb_by_label(
                    label=shape.label,
                    unique_label_list=self._unique_label_list,
                ),
            )
        self._label_dialog.add_label_history(shape.label)
        self._update_shape_color(shape)
        label_list_item.setText(format_shape_label(shape))

    def _update_shape_color(self, shape: Shape) -> None:
        assert shape.label is not None
        r, g, b = self._get_rgb_by_label(
            shape.label, unique_label_list=self._unique_label_list
        )
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(
        self,
        label: str,
        unique_label_list: UniqueLabelQListWidget,
    ) -> tuple[int, int, int]:
        item = unique_label_list.find_label_item(label)
        item_index: int = (
            unique_label_list.indexFromItem(item).row()
            if item
            else unique_label_list.count()
        )
        label_id = 1 + item_index  # skip black color by default
        return _rgb_from_colormap_id(label_id=label_id)

    def remove_labels(self, shapes: list[Shape]) -> None:
        self._label_list.item_dropped.disconnect(self._on_label_order_changed)
        for shape in shapes:
            item = self._label_list.find_item_by_shape(shape)
            self._label_list.remove_item(item)
        self._label_list.item_dropped.connect(self._on_label_order_changed)

    def _load_shapes(self, shapes: list[Shape], replace: bool = True) -> None:
        self._label_list.item_selection_changed.disconnect(
            self._label_selection_changed
        )
        for shape in shapes:
            self.add_label(shape)
        self._label_list.clearSelection()
        self._label_list.item_selection_changed.connect(self._label_selection_changed)
        self._canvas.load_shapes(shapes=shapes, replace=replace)

    def undo_shape_edit(self) -> None:
        self._canvas.restore_last_shape()
        self._label_list.clear()
        self._load_shapes(self._canvas.shapes)
        self._undo_action.setEnabled(self._canvas.can_restore_shape)

    def delete_selected_shapes(self) -> None:
        selected = self._canvas.selected_shapes
        if not selected:
            return
        yes = QtWidgets.QMessageBox.StandardButton.Yes
        no = QtWidgets.QMessageBox.StandardButton.No
        msg = f"确定永久删除 {len(selected)} 个标注吗？此操作无法撤销。"
        if QtWidgets.QMessageBox.warning(self, "注意", msg, yes | no, yes) == yes:
            self.remove_labels(self._canvas.delete_selected())
            self.mark_dirty()

    def duplicate_selected_shapes(self) -> None:
        self._insert_shapes([s.copy() for s in self._canvas.selected_shapes])

    def copy_selected_shapes(self) -> None:
        self._shape_clipboard.store(self._canvas.selected_shapes)

    def paste_shapes(self) -> None:
        self._insert_shapes(self._shape_clipboard.paste())

    def _insert_shapes(self, shapes: list[Shape]) -> None:
        if not shapes:
            return
        self._load_shapes(shapes=shapes, replace=False)
        self._canvas.select_shapes(shapes)
        self.mark_dirty()

    def copy_shape(self) -> None:
        self._canvas.end_move(copy=True)
        for shape in self._canvas.selected_shapes:
            self.add_label(shape)
        self._label_list.clearSelection()
        self.mark_dirty()

    def move_shape(self) -> None:
        self._canvas.end_move(copy=False)
        self.mark_dirty()

    def remove_selected_point(self) -> None:
        self._canvas.remove_selected_point()
        self._canvas.update()
        if (
            self._canvas.hovered_shape
            and not self._canvas.hovered_shape.points
        ):
            self._canvas.delete_shape(self._canvas.hovered_shape)
            self.remove_labels([self._canvas.hovered_shape])
        self.mark_dirty()

    def _edit_label(self, value: object | None = None) -> None:
        items = self._label_list.selected_items()
        if not items:
            return

        shapes = [item.shape() for item in items if item.shape() is not None]
        if not shapes:
            return
        first_shape = shapes[0]

        if len(items) == 1:
            edit_text = edit_flags = edit_group_id = edit_description = True
        else:
            edit_text = all(s.label == first_shape.label for s in shapes[1:])
            edit_flags = all(s.flags == first_shape.flags for s in shapes[1:])
            edit_group_id = all(
                s.group_id == first_shape.group_id for s in shapes[1:]
            )
            edit_description = all(
                s.description == first_shape.description for s in shapes[1:]
            )

        if not edit_text:
            self._label_dialog.edit.setDisabled(True)
            self._label_dialog.label_list.setDisabled(True)
        if not edit_group_id:
            self._label_dialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self._label_dialog.edit_description.setDisabled(True)

        text, flags, group_id, description = self._label_dialog.popup(
            text=first_shape.label if edit_text else "",
            flags=first_shape.flags if edit_flags else None,
            group_id=first_shape.group_id if edit_group_id else None,
            description=first_shape.description if edit_description else None,
            flags_disabled=not edit_flags,
        )

        if not edit_text:
            self._label_dialog.edit.setDisabled(False)
            self._label_dialog.label_list.setDisabled(False)
        if not edit_group_id:
            self._label_dialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self._label_dialog.edit_description.setDisabled(False)

        if text is None:
            return

        self._canvas.backup_shapes()
        for item in items:
            shape = item.shape()
            assert shape is not None
            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            assert shape.label is not None
            item.setText(format_shape_label(shape))
            self.mark_dirty()
            if self._unique_label_list.find_label_item(shape.label) is None:
                self._unique_label_list.add_label_item(
                    label=shape.label,
                    color=self._get_rgb_by_label(
                        label=shape.label,
                        unique_label_list=self._unique_label_list,
                    ),
                )

    # -------------------------------------------------------------- signals
    def _label_selection_changed(self) -> None:
        selected_shapes: list[Shape] = []
        for item in self._label_list.selected_items():
            shape = item.shape()
            if shape is not None:
                selected_shapes.append(shape)
        if selected_shapes:
            self._canvas.select_shapes(selected_shapes)
        elif self._canvas.deselect_shape():
            self._canvas.update()

    def _on_shape_selection_changed(self, selected_shapes: list[Shape]) -> None:
        self._label_list.item_selection_changed.disconnect(
            self._label_selection_changed
        )
        for shape in self._canvas.selected_shapes:
            shape.selected = False
        self._label_list.clearSelection()
        self._canvas.selected_shapes = selected_shapes
        for shape in self._canvas.selected_shapes:
            shape.selected = True
            item = self._label_list.find_item_by_shape(shape)
            self._label_list.select_item(item)
            self._label_list.scroll_to_item(item)
        self._label_list.item_selection_changed.connect(self._label_selection_changed)

        n_selected = len(selected_shapes) > 0
        self._delete_action.setEnabled(n_selected)
        self._duplicate_action.setEnabled(n_selected)
        self._copy_action.setEnabled(n_selected)
        self._edit_label_action.setEnabled(n_selected)

    def _on_label_item_changed(self, item: LabelListWidgetItem) -> None:
        is_visible_new = item.checkState() == QtCore.Qt.CheckState.Checked

        selected_group = (
            self._label_list.selection_at_press()
            or self._label_list.selected_items()
        )
        items_to_toggle = (
            selected_group
            if item in selected_group and len(selected_group) > 1
            else [item]
        )
        items_to_change = [
            it
            for it in items_to_toggle
            if (sh := it.shape()) is not None and sh.visible != is_visible_new
        ]
        if not items_to_change:
            return

        new_check_state = (
            QtCore.Qt.CheckState.Checked
            if is_visible_new
            else QtCore.Qt.CheckState.Unchecked
        )
        with QtCore.QSignalBlocker(self._label_list._model):
            for item_to_toggle in items_to_change:
                shape_to_toggle = item_to_toggle.shape()
                assert shape_to_toggle is not None
                item_to_toggle.setCheckState(new_check_state)
                self._canvas.set_shape_visible(shape_to_toggle, is_visible_new)

        self._canvas.backup_shapes()
        self._undo_action.setEnabled(self._canvas.can_restore_shape)

    def _on_label_order_changed(self) -> None:
        self.mark_dirty()
        shapes = [
            s for item in self._label_list if (s := item.shape()) is not None
        ]
        self._canvas.load_shapes(shapes)

    def _on_drawing_polygon_changed(self, drawing: bool = True) -> None:
        self._edit_mode_action.setEnabled(not drawing)
        self._undo_last_point_action.setEnabled(drawing)
        self._undo_action.setEnabled(not drawing)
        self._delete_action.setEnabled(not drawing)

    def _on_mouse_moved(self, pos: QtCore.QPointF) -> None:
        if self._canvas.pixmap.isNull():
            return
        self._coords_label.setText(f"x: {pos.x():.0f}   y: {pos.y():.0f}")

    # ----------------------------------------------------------------- zoom
    def _paint_canvas(self, _value: int | None = None) -> None:
        self._canvas.scale = 0.01 * self._zoom_widget.value()
        self._canvas.adjustSize()
        self._canvas.update()

    def _set_zoom(self, value: int, pos: QtCore.QPointF | None = None) -> None:
        value = max(self._zoom_widget.minimum(), min(self._zoom_widget.maximum(), value))
        old_width = self._canvas.width()
        self._zoom_widget.setValue(value)
        new_width = self._canvas.width()
        if pos is None or old_width == 0 or new_width == old_width:
            return
        factor = new_width / old_width
        h_bar = self._scroll.horizontalScrollBar()
        v_bar = self._scroll.verticalScrollBar()
        h_bar.setValue(int(h_bar.value() + pos.x() * (factor - 1)))
        v_bar.setValue(int(v_bar.value() + pos.y() * (factor - 1)))

    def _add_zoom(
        self, increment: float, pos: QtCore.QPointF | None = None
    ) -> None:
        current = self._zoom_widget.value()
        value = (
            math.ceil(current * increment)
            if increment > 1
            else math.floor(current * increment)
        )
        self._zoom_mode = "manual"
        self._set_zoom(value, pos)
        self._fit_action.setChecked(False)

    def _set_zoom_to_original(self) -> None:
        self._zoom_mode = "manual"
        self._set_zoom(100)
        self._fit_action.setChecked(False)

    def _zoom_requested(self, delta: int, pos: QtCore.QPointF) -> None:
        self._add_zoom(1.1 if delta > 0 else 0.9, pos)

    def set_fit_window_mode(self, value: bool = True) -> None:
        self._zoom_mode = "fit" if value else "manual"
        self._adjust_scale()

    def _adjust_scale(self) -> None:
        if self._zoom_mode == "fit":
            scale = self._fit_window_scale()
        else:
            scale = 1.0
        self._set_zoom(int(scale * 100))
        self._fit_action.setChecked(self._zoom_mode == "fit")

    def _fit_window_scale(self) -> float:
        FIT_WINDOW_SCROLLBAR_MARGIN = 2.0
        pixmap = self._canvas.pixmap
        viewport = self._scroll.viewport()
        if pixmap.isNull() or viewport is None:
            return 1.0
        if pixmap.width() == 0 or pixmap.height() == 0:
            return 1.0
        available_w = viewport.width() - FIT_WINDOW_SCROLLBAR_MARGIN
        available_h = viewport.height() - FIT_WINDOW_SCROLLBAR_MARGIN
        return min(available_w / pixmap.width(), available_h / pixmap.height())

    def _on_scroll_request(self, delta: int, orientation: int) -> None:
        bar = (
            self._scroll.horizontalScrollBar()
            if orientation == QtCore.Qt.Orientation.Horizontal
            else self._scroll.verticalScrollBar()
        )
        bar.setValue(int(bar.value() - delta * 0.1 * bar.singleStep()))

    def _on_pan_request(self, step: QtCore.QPoint) -> None:
        self._scroll.horizontalScrollBar().setValue(
            self._scroll.horizontalScrollBar().value() - step.x()
        )
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().value() - step.y()
        )

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(a0)
        if self._zoom_mode == "fit" and not self._canvas.pixmap.isNull():
            self._adjust_scale()

    # ---------------------------------------------------------------- misc
    def prev_image(self) -> None:
        if self._index > 0:
            self._file_list.setCurrentRow(self._index - 1)

    def next_image(self) -> None:
        if 0 <= self._index < len(self._items) - 1:
            self._file_list.setCurrentRow(self._index + 1)

    def mark_dirty(self) -> None:
        self._undo_action.setEnabled(self._canvas.can_restore_shape)
        self.set_dirty(True)

    def set_dirty(self, value: bool) -> None:
        if self._dirty != value:
            self._dirty = value
            self.dirtyChanged.emit(value)

    def _can_continue(self) -> bool:
        if not self._dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "未保存的标注",
            "当前图像的修改尚未保存，是否保存？",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_current()
        self.set_dirty(False)
        return True
