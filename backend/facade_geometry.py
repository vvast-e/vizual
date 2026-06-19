"""
Facade polygon helpers: gable split, signed area (no imports from main.py).
"""
from __future__ import annotations

import numpy as np


def polygon_signed_area(pts: np.ndarray) -> float:
    """Shoelace signed area; pts (N,2)."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_area_abs(pts: np.ndarray) -> float:
    return abs(polygon_signed_area(pts))


def split_gable_roof_polygon(
    polygon: list[list[float | int]],
) -> tuple[list[list[int]], list[list[int]]] | None:
    """
    Split a simple roof-gable outline into lower polygon + upper triangle.

    Assumes simplified polygon (e.g. approxPolyDP) with >= 5 vertices.
    Picks the vertex with minimal y (image top) as apex; neighbors must be lower.

    Returns:
        (polygon_lower, polygon_upper_triangle) as lists of [int,int], or None.
    """
    pts = np.array(polygon, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 5:
        return None
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    n = pts.shape[0]
    if n < 5:
        return None

    y = pts[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = y_max - y_min
    if span < 4.0:
        return None

    delta_y = max(2.0, 0.01 * span)
    apex_idx = int(np.argmin(y))
    y_apex = float(y[apex_idx])
    prev_i = (apex_idx - 1) % n
    next_i = (apex_idx + 1) % n
    if not (float(y[prev_i]) > y_apex + delta_y and float(y[next_i]) > y_apex + delta_y):
        return None

    upper = pts[[prev_i, apex_idx, next_i]]
    if _polygon_area_abs(upper) < 20.0:
        return None

    lower_idx = [i for i in range(n) if i != apex_idx]
    lower = pts[lower_idx]
    if _polygon_area_abs(lower) < 50.0:
        return None

    poly_upper = [[int(round(x)), int(round(y))] for x, y in upper]
    poly_lower = [[int(round(x)), int(round(y))] for x, y in lower]
    return poly_lower, poly_upper
