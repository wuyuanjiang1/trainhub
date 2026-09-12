"""Shape geometry helpers shared by the dataset converters."""

from __future__ import annotations

import math

from .dataset import Shape

CIRCLE_SEGMENTS = 24


def circle_points(shape: Shape) -> list[tuple[float, float]]:
    (cx, cy), (ex, ey) = shape.points[0], shape.points[1]
    radius = math.hypot(ex - cx, ey - cy)
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
            cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS),
        )
        for i in range(CIRCLE_SEGMENTS)
    ]


def bbox_of(shape: Shape) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounding box, for any shape that encloses an area."""
    points = shape.points
    if shape.shape_type == "circle" and len(points) >= 2:
        (cx, cy), (ex, ey) = points[0], points[1]
        radius = math.hypot(ex - cx, ey - cy)
        return cx - radius, cy - radius, cx + radius, cy + radius
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_of(shape: Shape) -> list[tuple[float, float]]:
    """Outer contour, approximating circles with a regular polygon."""
    if shape.shape_type == "circle" and len(shape.points) >= 2:
        return circle_points(shape)
    if shape.shape_type in ("rectangle", "oriented_rectangle") and len(shape.points) >= 2:
        xs = [p[0] for p in shape.points]
        ys = [p[1] for p in shape.points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return list(shape.points)


def area_of(shape: Shape) -> float:
    """Shoelace area, used to paint smaller objects on top of larger ones."""
    polygon = polygon_of(shape)
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0
