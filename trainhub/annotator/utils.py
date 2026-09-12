from __future__ import annotations

import warnings

import imgviz.io
import numpy

from ._qt import add_actions
from ._qt import direction_angle
from ._qt import distance
from ._qt import distance_to_line
from ._qt import format_shortcut
from ._qt import label_validator
from ._qt import new_action
from ._qt import new_button
from ._qt import new_icon
from ._qt import project_point_on_line
from ._qt import project_point_on_perpendicular_line
from .image import apply_exif_orientation
from .image import img_arr_to_b64
from .image import img_arr_to_data
from .image import img_b64_to_arr
from .image import img_data_to_arr
from .image import img_data_to_pil
from .image import img_data_to_png_data
from .image import img_pil_to_data
from .image import img_qt_to_arr
from .shape_utils import masks_to_bboxes
from .shape_utils import shape_to_mask
from .shape_utils import shapes_to_label

__all__ = [
    "add_actions",
    "apply_exif_orientation",
    "direction_angle",
    "distance",
    "distance_to_line",
    "format_shortcut",
    "img_arr_to_b64",
    "img_arr_to_data",
    "img_b64_to_arr",
    "img_data_to_arr",
    "img_data_to_pil",
    "img_data_to_png_data",
    "img_pil_to_data",
    "img_qt_to_arr",
    "label_validator",
    "lblsave",
    "masks_to_bboxes",
    "new_action",
    "new_button",
    "new_icon",
    "project_point_on_line",
    "project_point_on_perpendicular_line",
    "shape_to_mask",
    "shapes_to_label",
]


def lblsave(filename: str, lbl: numpy.ndarray) -> None:
    warnings.warn(
        "annotator.utils.lblsave is deprecated; use imgviz.io.lblsave",
        DeprecationWarning,
        stacklevel=2,
    )
    imgviz.io.lblsave(filename, lbl)
