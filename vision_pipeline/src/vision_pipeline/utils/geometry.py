"""Geometry helpers for ROIs and line velocity."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return True if point is inside polygon using ray casting."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance(point_a: Point, point_b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def path_length(path: Sequence[Point]) -> float:
    """Total length of a polyline path."""
    return sum(distance(path[i - 1], path[i]) for i in range(1, len(path)))
