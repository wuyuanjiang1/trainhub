"""Self-contained PyQt6 port of the labelme 6.3.0 annotation engine.

Ported from labelme 6.3.0, which is Copyright (C) 2005-2024 labelme
contributors (wkentaro and friends) and licensed under GPL-3.0-only.
Being a derivative work, this package -- and therefore trainhub as a
whole -- is distributed under GPL-3.0-only; see the LICENSE file at the
repository root.

This package never imports ``labelme``, ``osam`` or ``onnxruntime``. The
labelme JSON annotation format is kept byte-compatible so files can be
exchanged with labelme and fed directly to YOLO / nnUNet pipelines.
"""

# NOTE: utils is imported first on purpose. It pulls in label_file through
# shape_utils, which resolves the utils<->label_file circular import cleanly.
from . import utils  # noqa: F401  isort:skip

from ._version import __appname__
from ._version import __version__
from .canvas import Canvas
from .config import load_config
from .label_file import LabelFile
from .label_file import LabelFileError
from .label_file import LabelFileReadError
from .label_file import LabelFileWriteError
from .label_file import read_label_file
from .label_file import write_label_file
from .shape import POLYLINE_SHAPE_TYPES
from .shape import Shape
from .shape import ShapeType
from .shape_clipboard import ShapeClipboard
from .widgets.label_dialog import LabelDialog
from .widgets.label_list_widget import LabelListWidget
from .widgets.unique_label_qlist_widget import UniqueLabelQListWidget
from .widgets.zoom_widget import ZoomWidget

__all__ = [
    "Canvas",
    "LabelDialog",
    "LabelFile",
    "LabelFileError",
    "LabelFileReadError",
    "LabelFileWriteError",
    "LabelListWidget",
    "POLYLINE_SHAPE_TYPES",
    "Shape",
    "ShapeClipboard",
    "ShapeType",
    "UniqueLabelQListWidget",
    "ZoomWidget",
    "__appname__",
    "__version__",
    "load_config",
    "read_label_file",
    "utils",
    "write_label_file",
]
