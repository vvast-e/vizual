"""
FastAPI backend for wall detection (room-wall-visualizer).
Run from backend directory: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio
import os
import tempfile
import json
import time
import hashlib
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure we run in backend directory so relative paths in submodules work
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(BACKEND_DIR, "..")),
)
WARP_DEBUG_DIR = os.path.join(PROJECT_ROOT, "warp-debug")
WARP_DEBUG_SAVE = True
DETECT_DEBUG_DIR = os.path.join(PROJECT_ROOT, "detect-debug")
DETECT_DEBUG_SAVE = os.environ.get("DETECT_DEBUG_SAVE", "1") == "1"
# Постпроцесс: размер ядра CLOSE для слияния фрагментов (0 = авто по ширине кадра)
INTERIOR_OPENING_CLOSE_KSIZE = int(os.environ.get("INTERIOR_OPENING_CLOSE_KSIZE", "0"))
# Постпроцесс: минимальная площадь компоненты проёма в пикселях (0 = авто)
INTERIOR_MIN_OPENING_AREA_PX = int(os.environ.get("INTERIOR_MIN_OPENING_AREA_PX", "0"))
# Детекция проёмов по нормалям DSINE: порог углового отклонения от доминантной плоскости стены (градусы)
INTERIOR_OPENING_NORMAL_DEG = float(os.environ.get("INTERIOR_OPENING_NORMAL_DEG", "30"))
# ADE20K класс-индексы для интерьерных проёмов (Mask2Former /segformer)
ADE_WINDOW_CLASS = 8   # windowpane
ADE_DOOR_CLASS = 14    # door
ADE_WALL_CLASS = 0     # wall
# Включить семантическую составляющую (Mask2Former) в консенсус проёмов интерьера
INTERIOR_OPENING_SEMANTIC_ENABLE = os.environ.get("INTERIOR_OPENING_SEMANTIC_ENABLE", "1") == "1"
# Минимальная доля wall-пикселей (ADE20K) внутри layout-полигонов стен, ниже которой
# семантика считается недостоверной для этого кадра и игнорируется (fallback на нормали)
INTERIOR_SEMANTIC_MIN_WALL_FRACTION = float(os.environ.get("INTERIOR_SEMANTIC_MIN_WALL_FRACTION", "0.15"))
# Уточнять плоскости стен по семантической wall-маске (вырезать мебель/объекты из wall_minus_holes)
INTERIOR_WALL_REFINE_ENABLE = os.environ.get("INTERIOR_WALL_REFINE_ENABLE", "1") == "1"
TEXTURES_DIR = os.path.join(PROJECT_ROOT, "public", "textures")
ALLOWED_TEX_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

from texture_mapping import get_wall_corners, get_ceiling_corners, image_resize, wall_polygon_centroid

# Lazy imports after chdir: wall_segmentation and wall_estimation use relative paths
segmentation_model = None


def _load_models():
    global segmentation_model
    from wall_segmentation.segmenation import build_model
    segmentation_model = build_model()


def _detect_walls_sync(image_path: str) -> dict:
    """Run wall segmentation + estimation, return walls and image_size. Blocking."""
    from wall_segmentation.segmenation import wall_segmenting
    from wall_estimation.estimation import wall_estimation

    path_image = os.path.abspath(image_path)
    # Wall segmentation: binary mask
    mask1 = wall_segmenting(segmentation_model, path_image)
    # Wall estimation: layout map (left/center/right wall by color)
    estimation_map = wall_estimation(path_image)
    corners_list = get_wall_corners(estimation_map)

    # Intersect with segmentation mask (as in original app)
    mask2 = np.zeros(mask1.shape, dtype=np.uint8)
    for pts in corners_list:
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask2, [pts_arr], color=255)
    mask2_bool = mask2.astype(bool)
    mask1_bool = mask1.astype(bool)
    _ = mask1_bool & mask2_bool  # combined mask not needed for response

    h, w = mask1.shape[:2]
    walls = []
    for i, pts in enumerate(corners_list):
        center = wall_polygon_centroid(pts)
        corners = [[int(p[0]), int(p[1])] for p in pts]
        walls.append({
            "id": i + 1,
            "corners": corners,
            "center": [round(center[0], 2), round(center[1], 2)],
            "surface": "wall",
        })

    ceiling_pts = get_ceiling_corners(estimation_map)
    if ceiling_pts is not None:
        center = wall_polygon_centroid(ceiling_pts)
        corners = [[int(p[0]), int(p[1])] for p in ceiling_pts]
        walls.append({
            "id": len(walls) + 1,
            "corners": corners,
            "center": [round(center[0], 2), round(center[1], 2)],
            "surface": "ceiling",
        })

    return {
        "walls": walls,
        "image_size": {"width": w, "height": h},
    }

def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки как [TL, TR, BR, BL] для любого выпуклого четырехугольника."""
    pts = pts.astype(np.float32)
    
    # 1. Центроид
    cx, cy = np.mean(pts, axis=0)
    
    # 2. Углы относительно центроида
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    
    # 3. Сортировка по углу
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # 4. Найти TL (минимальная сумма x+y)
    sums = sorted_pts[:, 0] + sorted_pts[:, 1]
    tl_idx = np.argmin(sums)
    
    # 5. Сдвинуть массив
    ordered = np.roll(sorted_pts, -tl_idx, axis=0)
    
    # 6. Проверить направление и инвертировать если нужно
    v1 = ordered[1] - ordered[0]
    v2 = ordered[-1] - ordered[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        ordered[[1, 3]] = ordered[[3, 1]]
    
    return ordered


def _tile_texture_rgba(texture_rgba: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    """Create tiled RGBA texture image of size (height,width)."""
    safe = float(max(0.05, min(1.0, scale)))
    th, tw = texture_rgba.shape[:2]
    tile_w = max(1, int(round(tw * safe)))
    tile_h = max(1, int(round(th * safe)))
    tile = cv2.resize(texture_rgba, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            y2 = min(height, y + tile_h)
            x2 = min(width, x + tile_w)
            out[y:y2, x:x2] = tile[0 : y2 - y, 0 : x2 - x]
    return out


def _polygon_area_signed(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _sanitize_polygon_points(poly_pts: np.ndarray) -> np.ndarray:
    pts = np.array(poly_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-3:
        pts = pts[:-1]
    out: list[np.ndarray] = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) >= 1.0:
            out.append(p)
    if len(out) > 2 and np.linalg.norm(out[0] - out[-1]) < 1.0:
        out.pop()
    return np.array(out, dtype=np.float32)


def _is_convex(prev_p: np.ndarray, cur_p: np.ndarray, next_p: np.ndarray, ccw: bool) -> bool:
    v1 = cur_p - prev_p
    v2 = next_p - cur_p
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    return cross > 1e-6 if ccw else cross < -1e-6


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-8:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-6 and v >= -1e-6 and (u + v) <= 1.0 + 1e-6


def _triangulate_polygon_ear_clip(poly_pts: np.ndarray) -> list[tuple[int, int, int]]:
    n = len(poly_pts)
    if n < 3:
        return []
    indices = list(range(n))
    ccw = _polygon_area_signed(poly_pts) > 0
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < n * n:
        guard += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_cur = indices[i]
            i_next = indices[(i + 1) % m]
            a = poly_pts[i_prev]
            b = poly_pts[i_cur]
            c = poly_pts[i_next]
            if not _is_convex(a, b, c, ccw):
                continue
            has_inside = False
            for j in indices:
                if j in (i_prev, i_cur, i_next):
                    continue
                if _point_in_triangle(poly_pts[j], a, b, c):
                    has_inside = True
                    break
            if has_inside:
                continue
            tris.append((i_prev, i_cur, i_next))
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:
            return []
    if len(indices) == 3:
        tris.append((indices[0], indices[1], indices[2]))
    return tris


def _sample_rect_perimeter_points(width: int, height: int, count: int, ccw: bool) -> np.ndarray:
    w = float(max(1, width - 1))
    h = float(max(1, height - 1))
    per = 2.0 * (w + h)
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts: list[list[float]] = []
    for i in range(count):
        t = (i / count) * per
        if t < w:
            x, y = t, 0.0
        elif t < w + h:
            x, y = w, t - w
        elif t < 2 * w + h:
            x, y = w - (t - (w + h)), h
        else:
            x, y = 0.0, h - (t - (2 * w + h))
        pts.append([x, y])
    arr = np.array(pts, dtype=np.float32)
    if not ccw:
        arr = arr[::-1].copy()
    return arr


def _warp_triangle_affine(
    src_rgba: np.ndarray,
    dst_rgba: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
) -> None:
    src_tri = src_tri.astype(np.float32)
    dst_tri = dst_tri.astype(np.float32)
    r1 = cv2.boundingRect(src_tri)
    r2 = cv2.boundingRect(dst_tri)
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    src_crop = src_rgba[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    if src_crop.size == 0:
        return

    src_rect = np.array([[p[0] - r1[0], p[1] - r1[1]] for p in src_tri], dtype=np.float32)
    dst_rect = np.array([[p[0] - r2[0], p[1] - r2[1]] for p in dst_tri], dtype=np.float32)
    M = cv2.getAffineTransform(src_rect, dst_rect)
    warped = cv2.warpAffine(
        src_crop,
        M,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_rect), 1.0, lineType=cv2.LINE_AA)
    mask3 = mask[..., None]
    roi = dst_rgba[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]].astype(np.float32)
    blended = roi * (1.0 - mask3) + warped.astype(np.float32) * mask3
    dst_rgba[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = np.clip(blended, 0, 255).astype(np.uint8)


def _visvalingam_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Area of triangle formed by three 2D points (for Visvalingam-Whyatt)."""
    return 0.5 * abs(
        float((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    )


def _visvalingam_reduce(pts: np.ndarray, target_n: int) -> np.ndarray:
    """
    Reduce a closed polygon to *target_n* vertices using Visvalingam-Whyatt.
    Iteratively removes the vertex whose removal changes shape area the least
    (i.e. the one that forms the smallest triangle with its two neighbours).
    """
    pts_list: list[np.ndarray] = [p.copy() for p in pts]
    while len(pts_list) > target_n:
        n = len(pts_list)
        min_area = float("inf")
        min_idx = 0
        for i in range(n):
            a = pts_list[(i - 1) % n]
            b = pts_list[i]
            c = pts_list[(i + 1) % n]
            area = _visvalingam_triangle_area(a, b, c)
            if area < min_area:
                min_area = area
                min_idx = i
        pts_list.pop(min_idx)
    return np.array(pts_list, dtype=np.float32)


def _expand_triangle_to_quad(
    hull_pts: np.ndarray, original_pts: np.ndarray
) -> np.ndarray:
    """
    Turn a 3-vertex convex hull (triangle) into a wall quad.

    Strategy
    --------
    1. Peak = topmost vertex (smallest Y).
    2. Look for original polygon points in the upper 15-50 % height band that are
       NOT the peak — these are the "eaves" points where wall meets roof.
    3. If two suitable eaves points exist, use them as TL/TR.
    4. Otherwise, intersect hull slopes at 30 % height from peak.
    """
    sorted_idx = np.argsort(hull_pts[:, 1])
    peak = hull_pts[sorted_idx[0]]
    base = hull_pts[sorted_idx[1:]].copy()
    if base[0][0] > base[1][0]:
        base = base[::-1]
    bl, br = base[0], base[1]

    y_peak = float(peak[1])
    y_base = float(max(bl[1], br[1]))
    total_h = y_base - y_peak
    if total_h < 3:
        mid = (bl + br) / 2.0
        return np.array(
            [[mid[0] - 10, y_peak], [mid[0] + 10, y_peak], br, bl],
            dtype=np.float32,
        )

    # --- try to find eaves from original polygon points ---
    upper: list[np.ndarray] = []
    for p in original_pts:
        y = float(p[1])
        if y_peak + 0.10 * total_h < y < y_peak + 0.50 * total_h:
            upper.append(p)
    if len(upper) >= 2:
        arr = np.array(upper, dtype=np.float32)
        tl = arr[int(np.argmin(arr[:, 0]))]
        tr = arr[int(np.argmax(arr[:, 0]))]
        if np.linalg.norm(tl - tr) >= 5:
            return np.array([tl, tr, br, bl], dtype=np.float32)

    # --- fallback: intersect slopes at 30 % from peak ---
    y_eaves = y_peak + 0.30 * total_h
    t_l = float(np.clip((y_eaves - peak[1]) / (bl[1] - peak[1] + 1e-9), 0, 1))
    t_r = float(np.clip((y_eaves - peak[1]) / (br[1] - peak[1] + 1e-9), 0, 1))
    tl = np.array(
        [peak[0] + t_l * (bl[0] - peak[0]), y_eaves], dtype=np.float32
    )
    tr = np.array(
        [peak[0] + t_r * (br[0] - peak[0]), y_eaves], dtype=np.float32
    )
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _corners_from_polygon_auto(poly_pts: list[list[float]] | np.ndarray) -> np.ndarray:
    """
    Derive 4 perspective corners (wall trapezoid) from an N-point polygon.

    v3 — hull-vertex approach (preserves perspective):

    * hull == 4 → use vertices directly  (most walls; keeps real Y-coords)
    * hull >  4 → Visvalingam-Whyatt reduce to 4  (gable/hip roofs)
    * hull == 3 → expand triangle: find eaves from original points
    """
    pts = np.array(poly_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("Polygon must have at least 3 points")

    hull = cv2.convexHull(pts.astype(np.int32))
    hull_pts = hull[:, 0].astype(np.float32)
    n_hull = len(hull_pts)
    if n_hull < 3:
        raise ValueError("Degenerate convex hull")

    method = ""

    if n_hull == 4:
        quad = hull_pts.copy()
        method = "hull4"
    elif n_hull > 4:
        # Smart Roof Decapitation (Gable/Hip multi-point removal):
        xs = hull_pts[:, 0]
        ys = hull_pts[:, 1]
        min_x, max_x = np.min(xs), np.max(xs)
        width = max_x - min_x
        
        # Find Eaves candidates
        left_mask = xs <= min_x + 0.15 * width
        right_mask = xs >= max_x - 0.15 * width
        
        apex_y = float(np.min(ys))
        quad = None
        
        if np.any(left_mask) and np.any(right_mask):
            eave_l_idx = np.where(left_mask)[0][np.argmin(ys[left_mask])]
            eave_r_idx = np.where(right_mask)[0][np.argmin(ys[right_mask])]
            
            eave_l = hull_pts[eave_l_idx]
            eave_r = hull_pts[eave_r_idx]
            
            # A true gable/hip roof apex must be significantly higher than BOTH eaves
            if apex_y < eave_l[1] - 10.0 and apex_y < eave_r[1] - 10.0:
                dx = eave_r[0] - eave_l[0]
                dy = eave_r[1] - eave_l[1]
                
                new_hull = []
                for pt in hull_pts:
                    px, py = float(pt[0]), float(pt[1])
                    if np.array_equal(pt, eave_l) or np.array_equal(pt, eave_r):
                        new_hull.append(pt)
                        continue
                        
                    if min(eave_l[0], eave_r[0]) < px < max(eave_l[0], eave_r[0]):
                        if abs(dx) > 1e-6:
                            line_y = eave_l[1] + (px - eave_l[0]) * dy / dx
                        else:
                            line_y = min(eave_l[1], eave_r[1])
                            
                        if py < line_y - 2.0:
                            continue
                            
                    new_hull.append(pt)
                    
                hull_pts = np.array(new_hull, dtype=np.float32)
                method = f"visvalingam({n_hull}->4, decapitated)"
            else:
                method = f"visvalingam({n_hull}->4)"
        else:
            method = f"visvalingam({n_hull}->4)"
            
        if len(hull_pts) < 4:
            # Fallback if decapitation removed too many points
            hull_pts = hull[:, 0].astype(np.float32)
            
        quad = _visvalingam_reduce(hull_pts, 4)
    else:
        # n_hull == 3
        quad = _expand_triangle_to_quad(hull_pts, pts)
        method = "triangle_expand"

    ordered = _order_points_tl_tr_br_bl(quad)
    area = abs(_polygon_area_signed(ordered))

    if area < 200:
        # Fallback to axis-aligned bounding box
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        ordered = _order_points_tl_tr_br_bl(
            np.array(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                dtype=np.float32,
            )
        )
        area = abs(_polygon_area_signed(ordered))
        method += "+bbox_fallback"

    print(
        f"[warp-debug] _corners_from_polygon_auto v3: "
        f"input={len(pts)}pts hull={n_hull}pts method={method} "
        f"corners={np.round(ordered, 1).tolist()} "
        f"area={area:.0f}"
    )
    return ordered


def _warp_region_inv_uv_rgba(
    tiled: np.ndarray,
    W: int,
    H: int,
    src_quad_tl_tr_br_bl: np.ndarray,
    dst_pts_raw: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    """
    Inverse perspective sample from tiled texture into warped RGBA.
    src_quad_tl_tr_br_bl: (4,2) float32 — TL, TR, BR, BL in texture pixel coords.
    dst_pts_raw: (4,2) destination quad (any order, reordered internally).
    Writes only pixels where region_mask > 0.
    """
    warped = np.zeros((H, W, 4), dtype=np.uint8)
    dst_pts = _order_points_tl_tr_br_bl(dst_pts_raw.reshape(4, 2).astype(np.float32))
    src_pts = src_quad_tl_tr_br_bl.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M)
    ys, xs = np.where(region_mask > 0)
    if xs.size == 0:
        return warped
    ones = np.ones_like(xs, dtype=np.float64)
    pts_h = np.stack([xs.astype(np.float64), ys.astype(np.float64), ones], axis=0)
    src = M_inv @ pts_h
    wv = src[2]
    valid = np.abs(wv) > 1e-8
    sx = np.zeros_like(wv)
    sy = np.zeros_like(wv)
    sx[valid] = src[0][valid] / wv[valid]
    sy[valid] = src[1][valid] / wv[valid]
    sx = np.mod(sx, float(W))
    sy = np.mod(sy, float(H))
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)
    wa = ((1.0 - fx) * (1.0 - fy))[:, None]
    wb = (fx * (1.0 - fy))[:, None]
    wc = ((1.0 - fx) * fy)[:, None]
    wd = (fx * fy)[:, None]
    s00 = tiled[y0, x0].astype(np.float32)
    s10 = tiled[y0, x1].astype(np.float32)
    s01 = tiled[y1, x0].astype(np.float32)
    s11 = tiled[y1, x1].astype(np.float32)
    sampled = (wa * s00 + wb * s10 + wc * s01 + wd * s11).clip(0, 255).astype(np.uint8)
    warped[ys, xs] = sampled
    return warped


def _warp_texture_overlay_multi_regions_sync(
    tiled: np.ndarray,
    W: int,
    H: int,
    regions: list[dict],
    polygon_full: list[list[float]] | None,
    opacity: float,
    exterior_mask: np.ndarray | None = None,
) -> bytes:
    """Two-region gable warp: lower strip then upper strip of tiled texture."""
    if len(regions) != 2:
        raise ValueError("regions must contain exactly 2 items")

    region_polys: list[np.ndarray] = []
    region_masks: list[np.ndarray] = []
    dst_raw_list: list[np.ndarray] = []

    for ri, reg in enumerate(regions):
        c = reg.get("corners")
        p = reg.get("polygon")
        if not isinstance(c, list) or len(c) != 4:
            raise ValueError(f"regions[{ri}].corners must have 4 points")
        if not isinstance(p, list) or len(p) < 3:
            raise ValueError(f"regions[{ri}].polygon must have at least 3 points")
        poly_pts = np.array(p, dtype=np.float32).reshape(-1, 2)
        poly_pts[:, 0] = np.clip(poly_pts[:, 0], 0, W - 1)
        poly_pts[:, 1] = np.clip(poly_pts[:, 1], 0, H - 1)
        poly_pts = _sanitize_polygon_points(poly_pts)
        if len(poly_pts) < 3:
            raise ValueError(f"regions[{ri}].polygon degenerate after sanitize")
        rm = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(rm, [poly_pts.astype(np.int32)], 255)
        region_polys.append(poly_pts)
        region_masks.append(rm)
        # Always re-derive corners from region polygon (frontend corners may have duplicates)
        try:
            dc = _corners_from_polygon_auto(poly_pts)
            print(f"[warp-debug] region[{ri}] auto-derived corners from polygon ({len(poly_pts)} pts)")
        except Exception as e:
            print(f"[warp-debug] region[{ri}] auto-derive failed ({e}), using frontend corners")
            dc = np.array(c, dtype=np.float32).reshape(4, 2)
        dc[:, 0] = np.clip(dc[:, 0], 0, W - 1)
        dc[:, 1] = np.clip(dc[:, 1], 0, H - 1)
        dst_raw_list.append(dc)

    mask = np.zeros((H, W), dtype=np.uint8)
    if polygon_full and len(polygon_full) >= 3:
        poly_full = np.array(polygon_full, dtype=np.float32).reshape(-1, 2)
        poly_full[:, 0] = np.clip(poly_full[:, 0], 0, W - 1)
        poly_full[:, 1] = np.clip(poly_full[:, 1], 0, H - 1)
        poly_full = _sanitize_polygon_points(poly_full)
        if len(poly_full) >= 3:
            cv2.fillPoly(mask, [poly_full.astype(np.int32)], 255)
    if not np.any(mask > 0):
        mask = cv2.bitwise_or(region_masks[0], region_masks[1])

    if exterior_mask is not None:
        mask = cv2.bitwise_and(mask, exterior_mask)

    h0 = float(np.max(region_polys[0][:, 1]) - np.min(region_polys[0][:, 1]))
    h1 = float(np.max(region_polys[1][:, 1]) - np.min(region_polys[1][:, 1]))
    t_raw = h1 / (h0 + h1 + 1e-6)
    t_split = float(np.clip(t_raw, 0.05, 0.45))
    hm1 = float(max(H - 1, 1))
    y_split = float(np.clip(t_split * hm1, 1.0, float(max(H - 2, 1))))

    src_lower = np.array(
        [[0.0, y_split], [float(W - 1), y_split], [float(W - 1), float(H - 1)], [0.0, float(H - 1)]],
        dtype=np.float32,
    )
    src_upper = np.array(
        [[0.0, 0.0], [float(W - 1), 0.0], [float(W - 1), y_split], [0.0, y_split]],
        dtype=np.float32,
    )

    warped_accum = np.zeros((H, W, 4), dtype=np.uint8)
    src_quads = [src_lower, src_upper]
    for i, (rm, dst_raw, src_quad) in enumerate(zip(region_masks, dst_raw_list, src_quads)):
        try:
            warped_r = _warp_region_inv_uv_rgba(tiled, W, H, src_quad, dst_raw, rm)
        except Exception as e:
            print(f"[warp-debug] region {i} inv_uv failed: {e}")
            warped_r = np.zeros((H, W, 4), dtype=np.uint8)
        m = rm > 0
        warped_accum[m] = warped_r[m]
        print(
            f"[warp-debug] mode=multi_gable region={i} t_split={t_split:.4f} y_split={y_split:.2f} "
            f"pixels={int(np.count_nonzero(m))}"
        )

    safe_opacity = float(max(0.0, min(1.0, opacity)))
    alpha = (mask.astype(np.float32) / 255.0) * safe_opacity
    warped_alpha = warped_accum[:, :, 3].astype(np.float32) / 255.0
    out_alpha = (warped_alpha * alpha * 255.0).clip(0, 255).astype(np.uint8)
    warped_accum[:, :, 3] = out_alpha
    if exterior_mask is not None:
        warped_accum[:, :, 3] = cv2.bitwise_and(warped_accum[:, :, 3], exterior_mask)

    warped_bgra = cv2.cvtColor(warped_accum, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", warped_bgra)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()


def _warp_texture_overlay_sync(
    texture_bytes: bytes,
    corners: list[list[float]],
    image_width: int,
    image_height: int,
    texture_scale: float,
    opacity: float,
    polygon: list[list[float]] | None = None,
    regions: list[dict] | None = None,
    mask_base64: str | None = None,
) -> bytes:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be > 0")

    # Decode the optional raster mask if provided
    exterior_mask: np.ndarray | None = None
    if mask_base64:
        import base64
        try:
            arr = np.frombuffer(base64.b64decode(mask_base64), dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if decoded is not None and decoded.shape == (image_height, image_width):
                exterior_mask = decoded
        except Exception as e:
            print(f"[warp-debug] failed to decode mask_base64: {e}")

    # ALWAYS re-derive 4-corner perspective quad from polygon when available.
    # Frontend corners can contain duplicates from old broken algorithms.
    use_polygon_early = bool(polygon) and isinstance(polygon, list) and len(polygon) >= 3
    if use_polygon_early:
        poly_tmp = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        old_corners = corners
        corners = _corners_from_polygon_auto(poly_tmp).tolist()
        print(
            f"[warp-debug] auto-derived 4 corners from polygon ({len(polygon)} pts): "
            f"old={old_corners} new={[[round(c[0],1),round(c[1],1)] for c in corners]}"
        )
    if len(corners) != 4:
        raise ValueError("corners must have 4 points (provide 4 corners or polygon with >=3 pts to auto-derive)")

    tex_arr = np.frombuffer(texture_bytes, dtype=np.uint8)
    tex = cv2.imdecode(tex_arr, cv2.IMREAD_UNCHANGED)
    if tex is None:
        raise ValueError("Failed to decode texture image")

    # Ensure RGBA
    if tex.ndim == 2:
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGBA)
    elif tex.shape[2] == 3:
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGBA)
    elif tex.shape[2] == 4:
        # OpenCV loads as BGRA
        tex = cv2.cvtColor(tex, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unsupported texture channels")

    W = int(image_width)
    H = int(image_height)
    tiled = _tile_texture_rgba(tex, W, H, texture_scale)

    if regions is not None and isinstance(regions, list) and len(regions) == 2:
        return _warp_texture_overlay_multi_regions_sync(
            tiled, W, H, regions, polygon, opacity, exterior_mask
        )
    use_polygon = bool(polygon) and len(polygon) >= 3
    poly_pts: np.ndarray | None = None
    if use_polygon:
        poly_pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        poly_pts[:, 0] = np.clip(poly_pts[:, 0], 0, W - 1)
        poly_pts[:, 1] = np.clip(poly_pts[:, 1], 0, H - 1)
        poly_pts = _sanitize_polygon_points(poly_pts)

    dst_pts_raw = np.array(corners, dtype=np.float32).reshape(4, 2)
    dst_pts = _order_points_tl_tr_br_bl(dst_pts_raw)
    
    # Проверка на вырожденный четырёхугольник (формула Гаусса)
    x = dst_pts[:, 0]
    y = dst_pts[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if area < 100:
        raise ValueError(f"Degenerate quadrilateral detected: area={area:.1f}")

    src_pts = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    try:
        det2 = float(np.linalg.det(M[:2, :2]))
    except Exception:
        det2 = float("nan")
    print(
        "[warp-debug] perspective "
        f"src={src_pts.tolist()} "
        f"dst_raw={np.round(dst_pts_raw, 2).tolist()} "
        f"dst_ord={np.round(dst_pts, 2).tolist()} "
        f"det2={det2:.6f}"
    )
    mask = np.zeros((H, W), dtype=np.uint8)
    if use_polygon and poly_pts is not None and len(poly_pts) >= 3:
        cv2.fillPoly(mask, [poly_pts.astype(np.int32)], 255)
        pminx = float(np.min(poly_pts[:, 0])) if poly_pts.size else 0.0
        pmaxx = float(np.max(poly_pts[:, 0])) if poly_pts.size else 0.0
        pminy = float(np.min(poly_pts[:, 1])) if poly_pts.size else 0.0
        pmaxy = float(np.max(poly_pts[:, 1])) if poly_pts.size else 0.0
        print(
            "[warp-debug] polygon "
            f"len={len(poly_pts)} "
            f"bbox=({pminx:.1f},{pminy:.1f})-({pmaxx:.1f},{pmaxy:.1f})"
        )
    else:
        cv2.fillPoly(mask, [dst_pts.astype(np.int32)], 255)
        dminx = float(np.min(dst_pts[:, 0])) if dst_pts.size else 0.0
        dmaxx = float(np.max(dst_pts[:, 0])) if dst_pts.size else 0.0
        dminy = float(np.min(dst_pts[:, 1])) if dst_pts.size else 0.0
        dmaxy = float(np.max(dst_pts[:, 1])) if dst_pts.size else 0.0
        print(
            "[warp-debug] quad-mask "
            f"bbox=({dminx:.1f},{dminy:.1f})-({dmaxx:.1f},{dmaxy:.1f})"
        )

    # Intersect with the raster mask (holes) if provided
    if exterior_mask is not None:
        mask = cv2.bitwise_and(mask, exterior_mask)
        print(f"[warp-debug] applied exterior_mask, new nonzero: {int(np.count_nonzero(mask))}")

    warped: np.ndarray
    if use_polygon and poly_pts is not None and len(poly_pts) >= 3:
        # Use built-in OpenCV warp with BORDER_WRAP to seamlessly tile the texture 
        # beyond the blue perspective quad, filling the entire green polygon mask.
        warped = cv2.warpPerspective(
            tiled,
            M,
            dsize=(W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
    else:
        warped = cv2.warpPerspective(
            tiled,
            M,
            dsize=(W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    safe_opacity = float(max(0.0, min(1.0, opacity)))
    alpha = (mask.astype(np.float32) / 255.0) * safe_opacity
    warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0
    out_alpha = (warped_alpha * alpha * 255.0).clip(0, 255).astype(np.uint8)
    warped[:, :, 3] = out_alpha
    if exterior_mask is not None:
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], exterior_mask)
    try:
        mask_nonzero = int(np.count_nonzero(mask))
        alpha_nonzero = int(np.count_nonzero(out_alpha))
        alpha_mean = float(out_alpha[out_alpha > 0].mean()) if alpha_nonzero > 0 else 0.0
    except Exception:
        mask_nonzero, alpha_nonzero, alpha_mean = 0, 0, 0.0
    print(
        "[warp-debug] alpha "
        f"mask_nonzero={mask_nonzero} "
        f"alpha_nonzero={alpha_nonzero} "
        f"alpha_mean={alpha_mean:.2f} "
        f"opacity={safe_opacity:.2f}"
    )

    warped_bgra = cv2.cvtColor(warped, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", warped_bgra)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_load_models)
    yield
    # no cleanup needed


app = FastAPI(title="Wall detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Catalog (PostgreSQL): materials, colors, admins
import catalog_routes  # noqa: E402

app.include_router(catalog_routes.router)

# Uploaded textures/colors (dev: also served by Vite from /public; Docker: proxy to backend)
os.makedirs(os.path.join(PROJECT_ROOT, "public", "textures"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "public", "colors"), exist_ok=True)
app.mount(
    "/textures",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "public", "textures")),
    name="textures",
)
app.mount(
    "/colors",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "public", "colors")),
    name="colors",
)


@app.post("/api/detect-walls")
async def detect_walls(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (JPEG/PNG)")
    suffix = ".jpg" if "jpeg" in file.content_type else ".png"
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Resize if too large (as in original: height 600)
        img = Image.open(tmp_path)
        img_np = np.asarray(img)
        if img_np.shape[0] > 600:
            img_np = image_resize(img_np, height=600)
            Image.fromarray(img_np).save(tmp_path)

        # Capture resized bytes for downstream window/door detection
        with open(tmp_path, "rb") as f:
            resized_bytes = f.read()

        result = await asyncio.to_thread(_detect_walls_sync, tmp_path)

        # Detect windows/doors and build wall_minus_holes mask so the interior
        # texturing pipeline can carve openings out the same way exterior does.
        try:
            result["masks"] = await _build_interior_wall_minus_holes(resized_bytes, result)
        except Exception as e:
            print(f"[detect-walls] openings detection failed: {e!r}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _consolidate_opening_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Слияние фрагментов → заполнение дыр → фильтр по площади."""
    if not np.any(mask):
        return mask

    # 1. Убрать точечный шум
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # 2. Слить близкие фрагменты в единый проём
    close_k = INTERIOR_OPENING_CLOSE_KSIZE if INTERIOR_OPENING_CLOSE_KSIZE > 0 else max(7, width // 120)
    close_k = close_k | 1  # нечётное
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 3. Заполнить внутренние дыры контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # 4. Отбросить мелкие компоненты (шум от лестниц, теней и т.п.)
    min_area = INTERIOR_MIN_OPENING_AREA_PX if INTERIOR_MIN_OPENING_AREA_PX > 0 \
        else max(200, (width * height) // 4000)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    result = np.zeros_like(filled)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            result[labels == lbl] = 255
    return result


def _detect_openings_by_normals(normals: np.ndarray, walls: list, width: int, height: int):
    """Геометрическая детекция проёмов: попиксельное отклонение нормали от доминантной
    плоскости каждой стены/потолка. Открытый проём (видна другая геометрия под другим
    углом) даёт большое угловое отклонение; плоская стена — малое.

    Возвращает (openings_raw, deviation_deg):
    - openings_raw: (H, W) uint8 маска 0/255 сырых кандидатов-проёмов, обрезанная по стенам
    - deviation_deg: (H, W) float32 карта углового отклонения (для debug-heatmap)
    """
    openings_raw = np.zeros((height, width), dtype=np.uint8)
    deviation_deg = np.zeros((height, width), dtype=np.float32)

    for wall in walls:
        pts = np.array(wall.get("corners") or [], dtype=np.int32)
        if pts.size < 6:
            continue

        wall_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(wall_mask, [pts], color=255)
        mask_bool = wall_mask > 0
        if not np.any(mask_bool):
            continue

        wall_normals = normals[mask_bool]  # (N, 3)
        dominant = np.median(wall_normals, axis=0)
        dom_norm = np.linalg.norm(dominant)
        if dom_norm < 1e-6:
            continue
        dominant = dominant / dom_norm

        dot = np.abs(np.clip(wall_normals @ dominant, -1.0, 1.0))
        ang_deg = np.degrees(np.arccos(dot))

        ys, xs = np.where(mask_bool)
        deviation_deg[ys, xs] = ang_deg
        wall_openings = ang_deg > INTERIOR_OPENING_NORMAL_DEG
        openings_raw[ys[wall_openings], xs[wall_openings]] = 255

    return openings_raw, deviation_deg


async def _build_interior_wall_minus_holes(image_bytes: bytes, walls_result: dict) -> dict:
    """Detect openings (doors/passages) via a consensus of Mask2Former ADE20K
    semantics and DSINE surface-normal geometry.

    Returns a base64 PNG mask where wall pixels are 255 and opening pixels are 0.

    Семантика (windowpane=8, door=14) — основной сигнал для дверей/окон с рамкой.
    Нормали DSINE ловят безрамочные проходы, где семантика молчит (комната за
    проёмом просто не размечена как door/window). Итог — объединение обоих масок,
    обрезанное по layout-полигонам стен. Семантика best-effort: если доля
    wall-пикселей внутри layout-стен слишком мала (модель "промазала" по кадру),
    семантическая маска отбрасывается и остаются только нормали.
    """
    import hashlib
    import time as _time
    from plane_split import request_normals
    from exterior_pipeline import segment_ade20k, encode_mask_png_base64

    image_size = walls_result.get("image_size") or {}
    width = int(image_size.get("width") or 0)
    height = int(image_size.get("height") or 0)
    walls = walls_result.get("walls") or []
    if width <= 0 or height <= 0 or not walls:
        return {}

    # ── Маска объединения стен (строим ДО морфологии, чтобы обрезать проёмы по стенам) ──
    wall_union = np.zeros((height, width), dtype=np.uint8)
    for wall in walls:
        pts = np.array(wall.get("corners") or [], dtype=np.int32)
        if pts.size >= 6:
            cv2.fillPoly(wall_union, [pts], color=255)

    # ── Семантика ADE20K (Mask2Former) — best-effort ─────────────────────────
    windows_raw = np.zeros((height, width), dtype=np.uint8)
    doors_raw = np.zeros((height, width), dtype=np.uint8)
    furniture_raw = np.zeros((height, width), dtype=np.uint8)
    seg_vis = None
    if INTERIOR_OPENING_SEMANTIC_ENABLE:
        seg = await segment_ade20k(image_bytes)  # (H, W) uint8, значения 0-149
        if seg.size > 0:
            if seg.shape != (height, width):
                seg = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)

            wall_union_bool = wall_union > 0
            wall_pixels_total = int(np.count_nonzero(wall_union_bool))
            wall_class_hits = int(np.count_nonzero((seg == ADE_WALL_CLASS) & wall_union_bool))
            wall_fraction = (wall_class_hits / wall_pixels_total) if wall_pixels_total > 0 else 0.0

            if wall_fraction >= INTERIOR_SEMANTIC_MIN_WALL_FRACTION:
                windows_raw = ((seg == ADE_WINDOW_CLASS).astype(np.uint8)) * 255
                doors_raw = ((seg == ADE_DOOR_CLASS).astype(np.uint8)) * 255
                windows_raw = cv2.bitwise_and(windows_raw, wall_union)
                doors_raw = cv2.bitwise_and(doors_raw, wall_union)
                seg_vis = np.zeros((height, width), dtype=np.uint8)
                seg_vis[seg == ADE_WINDOW_CLASS] = 128
                seg_vis[seg == ADE_DOOR_CLASS] = 255

                # Часть B: уточнение плоскости стены — всё, что внутри layout-полигона
                # стены НЕ относится к классу wall/window/door (мебель, картины, шум
                # переразметки в углах), считаем не-стеной и вырежем из wall_minus_holes.
                if INTERIOR_WALL_REFINE_ENABLE:
                    non_wall = ~np.isin(seg, (ADE_WALL_CLASS, ADE_WINDOW_CLASS, ADE_DOOR_CLASS))
                    furniture_raw = (non_wall & wall_union_bool).astype(np.uint8) * 255
            else:
                print(
                    f"[detect-debug] semantic wall_fraction={wall_fraction:.3f} "
                    f"< {INTERIOR_SEMANTIC_MIN_WALL_FRACTION}, discarding semantic mask for this frame"
                )
        else:
            print("[detect-debug] Mask2Former /segformer unavailable, skipping semantic openings")
    semantic_raw = cv2.bitwise_or(windows_raw, doors_raw)

    # ── Карта нормалей поверхности (DSINE) ───────────────────────────────────
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    normals = await asyncio.to_thread(request_normals, image_bgr) if image_bgr is not None else None

    if normals is None:
        print("[detect-debug] DSINE normals unavailable, using semantic-only openings")
        openings_raw = semantic_raw
        deviation_deg = np.zeros((height, width), dtype=np.float32)
    else:
        if normals.shape[:2] != (height, width):
            normals = cv2.resize(normals, (width, height), interpolation=cv2.INTER_LINEAR)
            norms = np.linalg.norm(normals, axis=2, keepdims=True)
            normals = normals / (norms + 1e-8)

        normals_raw, deviation_deg = _detect_openings_by_normals(normals, walls, width, height)
        normals_raw = cv2.bitwise_and(normals_raw, wall_union)
        # Консенсус: семантика (двери/окна с рамкой) ∪ нормали (безрамочные проходы)
        openings_raw = cv2.bitwise_or(semantic_raw, normals_raw)

    # Consolidation: OPEN → CLOSE → fill holes → area filter
    openings_mask = _consolidate_opening_mask(openings_raw.copy(), width, height)

    # Часть B: консолидированная маска мебели/объектов на стене (та же морфология,
    # чтобы отсечь точечный шум переразметки и не выедать тонкие полоски у краёв стены)
    furniture_mask = (
        _consolidate_opening_mask(furniture_raw.copy(), width, height)
        if np.any(furniture_raw) else furniture_raw
    )
    excluded_mask = cv2.bitwise_or(openings_mask, furniture_mask)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    holes_dilated = cv2.dilate(excluded_mask, k_dilate, iterations=1)

    wall_minus_holes = cv2.bitwise_and(wall_union, cv2.bitwise_not(holes_dilated))

    # ── Пер-стеновая геометрия: polygon (сглаженный контур с вырезанными
    # краевыми проёмами) + скорректированные corners через _corners_from_polygon_auto.
    # Best-effort: при ошибке/вырожденном контуре стена остаётся с исходными corners.
    for wall in walls:
        try:
            surface = wall.get("surface", "wall")
            base_pts = np.array(wall.get("corners") or [], dtype=np.int32)
            if base_pts.size < 6:
                continue
            wall_fill = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(wall_fill, [base_pts], color=255)
            # Для стен вычитаем проёмы; потолок оставляем целым (балки/шум нормалей
            # не должны прорезать контур потолка).
            region = (
                cv2.bitwise_and(wall_fill, wall_minus_holes)
                if surface != "ceiling"
                else wall_fill
            )
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < (width * height) / 200:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            poly_pts = approx.reshape(-1, 2)
            if len(poly_pts) < 3:
                continue
            wall["polygon"] = [[int(p[0]), int(p[1])] for p in poly_pts]

            quad = _corners_from_polygon_auto(poly_pts)  # [TL, TR, BR, BL]
            # Ремап под порядок фронта [TL, BL, BR, TR]
            remapped = [quad[0], quad[3], quad[2], quad[1]]
            wall["corners"] = [[int(p[0]), int(p[1])] for p in remapped]
            center = wall_polygon_centroid(poly_pts.tolist())
            wall["center"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
        except Exception as _geo_err:
            print(f"[detect-debug] per-wall geometry refine failed for wall {wall.get('id')}: {_geo_err}")

    # ── Debug-вывод по подпапкам (как в exterior-debug) ─────────────────────
    if DETECT_DEBUG_SAVE:
        try:
            call_ts = int(_time.time() * 1000)
            call_digest = hashlib.md5(image_bytes).hexdigest()[:10]
            call_dir = os.path.join(DETECT_DEBUG_DIR, f"call_{call_ts}_{call_digest}")
            os.makedirs(call_dir, exist_ok=True)

            def _save(name: str, img: np.ndarray):
                p = os.path.join(call_dir, name)
                cv2.imwrite(p, img)
                print(f"[detect-debug] saved: {p}")

            # Стадийные PNG для диагностики
            _save("wall_union.png", wall_union)
            _save("openings_raw.png", openings_raw)
            if np.any(furniture_mask):
                _save("furniture_mask.png", furniture_mask)
            _save("holes_dilated.png", holes_dilated)

            # Heatmap углового отклонения нормали от доминантной плоскости стены (0..90° -> 0..255)
            normal_dev_vis = np.clip(deviation_deg / 90.0 * 255.0, 0, 255).astype(np.uint8)
            normal_dev_vis = cv2.bitwise_and(normal_dev_vis, wall_union)
            _save("normal_dev.png", normal_dev_vis)

            if seg_vis is not None:
                _save("seg_raw.png", seg_vis)

            _save("wall_minus_holes.png", wall_minus_holes)

            # Пер-стеновые polygon (сглаженный контур, вырезанные проёмы) поверх фото
            polys_vis = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if polys_vis is not None:
                for wall in walls:
                    poly = wall.get("polygon")
                    if not poly or len(poly) < 3:
                        continue
                    pts = np.array(poly, dtype=np.int32)
                    surface = wall.get("surface", "wall")
                    color = (0, 165, 255) if surface == "ceiling" else (255, 0, 255)
                    cv2.polylines(polys_vis, [pts], isClosed=True, color=color, thickness=2)
                _save("wall_polys.png", polys_vis)

            # Overlay: стены/потолок + консолидированные контуры проёмов
            overlay = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if overlay is not None:
                for wall in walls:
                    pts = np.array(wall.get("corners") or [], dtype=np.int32)
                    surface = wall.get("surface", "wall")
                    if pts.size >= 6:
                        color = (0, 165, 255) if surface == "ceiling" else (0, 255, 0)
                        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
                        cx, cy = wall.get("center") or [pts[:, 0].mean(), pts[:, 1].mean()]
                        label = "ceiling" if surface == "ceiling" else f"wall {wall.get('id', '')}"
                        cv2.putText(overlay, label, (int(cx), int(cy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Контуры консолидированных проёмов (не сырых)
                contours_o, _ = cv2.findContours(openings_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours_o, -1, (0, 0, 255), 2)
                for c in contours_o:
                    x, y, w_, h_ = cv2.boundingRect(c)
                    cv2.putText(overlay, "opening", (x, max(0, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                _save("overlay.png", overlay)
        except Exception as _dbg_err:
            print(f"[detect-debug] save failed: {_dbg_err}")

    return {"wall_minus_holes": encode_mask_png_base64(wall_minus_holes)}


MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1920"))


def _resize_image_if_needed(img):
    """Resize image if any dimension exceeds MAX_IMAGE_SIZE."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


@app.post("/api/detect-exterior")
async def detect_exterior(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (JPEG/PNG)")
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    img_arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    
    # Resize large images for faster CPU inference
    orig_h, orig_w = img.shape[:2]
    img = _resize_image_if_needed(img)
    h, w = img.shape[:2]
    
    # Re-encode to bytes for pipeline
    _, buf = cv2.imencode(".jpg", img)
    resized_contents = buf.tobytes()

    from exterior_pipeline import run_exterior_pipeline

    try:
        result = await run_exterior_pipeline(resized_contents, w, h)
        try:
            walls = result.get("walls") if isinstance(result, dict) else None
            masks = result.get("masks") if isinstance(result, dict) else None
            if isinstance(masks, dict):
                print(
                    f"[detect-exterior] walls={len(walls or [])} "
                    f"wall_minus_holes_b64_len={len(masks.get('wall_minus_holes') or '')}"
                )
        except Exception:
            pass
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@app.post("/api/exterior/split")
async def exterior_split(
    mask_base64: str = Form(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),
    target_wall_id: Optional[int] = Form(None),
    walls_json: Optional[str] = Form(None),
):
    """Split exterior facade mask by user-drawn line. Returns new walls[]."""
    from exterior_pipeline import split_exterior_by_line

    if target_wall_id is not None and not walls_json:
        raise HTTPException(
            status_code=400,
            detail="walls_json is required when target_wall_id is set",
        )

    existing: list[dict] | None = None
    if walls_json is not None:
        try:
            parsed = json.loads(walls_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"invalid walls_json: {e}") from e
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail="walls_json must be a JSON array")
        if len(parsed) == 0:
            raise HTTPException(status_code=400, detail="walls_json must be non-empty")
        if not all(isinstance(x, dict) for x in parsed):
            raise HTTPException(status_code=400, detail="walls_json must contain only objects")
        existing = [x for x in parsed if isinstance(x, dict)]

    try:
        walls = split_exterior_by_line(
            mask_base64,
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            int(image_width),
            int(image_height),
            existing_walls=existing,
            target_wall_id=target_wall_id,
        )
        return {"walls": walls}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/estimate-homography")
async def estimate_homography(
    polygon: str = Form(...),
):
    """
    Вычислить гомографию (перспективу) из формы полигона.
    
    Принимает массив точек полигона, возвращает массив из 4 углов
    для правильной перспективной трансформации.
    """
    try:
        polygon_parsed = json.loads(polygon)
        
        if not isinstance(polygon_parsed, list) or len(polygon_parsed) < 3:
            raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
        
        # Вычисляем перспективу из формы
        corners = compute_perspective_from_polygon(polygon_parsed)
        
        return {"corners": corners, "method": "homography-estimation"}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in polygon")
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e)) from e


def compute_perspective_from_polygon(polygon: list) -> list:
    """
    Вычислить перспективные углы из формы полигона.
    Делегирует в _corners_from_polygon_auto (scan-line width analysis).
    """
    if len(polygon) < 3:
        return polygon

    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)

    if len(pts) == 3:
        # Для треугольника — расширяем до четырёхугольника:
        # самая длинная сторона = нижняя (ближе к камере)
        edges = [
            (dist(pts[0], pts[1]), 0, 1),
            (dist(pts[1], pts[2]), 1, 2),
            (dist(pts[2], pts[0]), 2, 0),
        ]
        edges.sort(key=lambda x: x[0], reverse=True)
        _, i1, i2 = edges[0]
        p1, p2 = pts[i1], pts[i2]
        # Третья точка — вершина (она же определяет высоту)
        other_idx = ({0, 1, 2} - {i1, i2}).pop()
        apex = pts[other_idx]
        # Проекция апекса на нижнюю сторону → два верхних угла
        mid_top = (apex + (p1 + p2) / 2) / 2
        half_w = dist(p1, p2) * 0.25
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = dist(p1, p2) + 1e-9
        ux, uy = dx / length, dy / length
        pts = np.array([
            [mid_top[0] - ux * half_w, mid_top[1] - uy * half_w],
            [mid_top[0] + ux * half_w, mid_top[1] + uy * half_w],
            [float(p2[0]), float(p2[1])],
            [float(p1[0]), float(p1[1])],
        ], dtype=np.float32)

    corners = _corners_from_polygon_auto(pts)
    return corners.tolist()


def dist(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


@app.post("/api/warp-wall-texture")
async def warp_wall_texture(
    texture: UploadFile = File(...),
    corners: str = Form(...),
    polygon: str | None = Form(None),
    regions: str | None = Form(None),
    image_width: int = Form(...),
    image_height: int = Form(...),
    texture_scale: float = Form(0.25),
    opacity: float = Form(1.0),
    mask_base64: str | None = Form(None),
):
    try:
        texture_bytes = await texture.read()
        corners_parsed = json.loads(corners)
        polygon_parsed = None
        if polygon:
            polygon_parsed = json.loads(polygon)
        regions_parsed: list[dict] | None = None
        if regions:
            raw_reg = json.loads(regions)
            if not isinstance(raw_reg, list) or len(raw_reg) != 2:
                raise ValueError("regions must be a JSON array of length 2")
            regions_parsed = []
            for ri, item in enumerate(raw_reg):
                if not isinstance(item, dict):
                    raise ValueError(f"regions[{ri}] must be an object")
                c = item.get("corners")
                p = item.get("polygon")
                if not isinstance(c, list) or len(c) != 4:
                    raise ValueError(f"regions[{ri}].corners must have 4 points")
                if not isinstance(p, list) or len(p) < 3:
                    raise ValueError(f"regions[{ri}].polygon must have at least 3 points")
                corners_r = [[float(c[i][0]), float(c[i][1])] for i in range(4) if isinstance(c[i], list) and len(c[i]) == 2]
                if len(corners_r) != 4:
                    raise ValueError(f"regions[{ri}].corners invalid point format")
                poly_r = [pt for pt in p if isinstance(pt, list) and len(pt) == 2]
                if len(poly_r) < 3:
                    raise ValueError(f"regions[{ri}].polygon invalid point format")
                regions_parsed.append({"corners": corners_r, "polygon": poly_r})
        print(
            f"[warp-wall-texture] corners_len={len(corners_parsed) if isinstance(corners_parsed, list) else '??'} "
            f"polygon_len={len(polygon_parsed) if isinstance(polygon_parsed, list) else 0} "
            f"regions={'2' if regions_parsed else '0'} "
            f"image=({image_width},{image_height}) scale={texture_scale} opacity={opacity}"
        )
        if isinstance(corners_parsed, list):
            print(f"[warp-wall-texture] corners={corners_parsed}")
        # Clip polygon to plausible structure early.
        if polygon_parsed and isinstance(polygon_parsed, list):
            polygon_parsed = [pt for pt in polygon_parsed if isinstance(pt, list) and len(pt) == 2]
            print(
                f"[warp-wall-texture] polygon_sample="
                f"{polygon_parsed[:6]} (len={len(polygon_parsed)})"
            )
        png_bytes = await asyncio.to_thread(
            _warp_texture_overlay_sync,
            texture_bytes,
            corners_parsed,
            int(image_width),
            int(image_height),
            float(texture_scale),
            float(opacity),
            polygon_parsed,
            regions_parsed,
            mask_base64,
        )
        if WARP_DEBUG_SAVE:
            try:
                os.makedirs(WARP_DEBUG_DIR, exist_ok=True)
                payload = {
                    "corners": corners_parsed,
                    "polygon": polygon_parsed,
                    "regions": regions_parsed,
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "texture_scale": float(texture_scale),
                    "opacity": float(opacity),
                    "texture_filename": getattr(texture, "filename", None),
                    "texture_content_type": getattr(texture, "content_type", None),
                }
                digest = hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
                ts = int(time.time() * 1000)
                out_name = f"warp_{ts}_{digest}.png"
                out_path = os.path.join(WARP_DEBUG_DIR, out_name)
                with open(out_path, "wb") as f:
                    f.write(png_bytes)
                print(f"[warp-debug] saved: {out_path}")

                if polygon_parsed and isinstance(polygon_parsed, list) and len(polygon_parsed) >= 3:
                    try:
                        W = int(image_width)
                        H = int(image_height)
                        mask = np.zeros((H, W), dtype=np.uint8)
                        poly_pts = np.array(polygon_parsed, dtype=np.float32).reshape(-1, 2)
                        poly_pts[:, 0] = np.clip(poly_pts[:, 0], 0, W - 1)
                        poly_pts[:, 1] = np.clip(poly_pts[:, 1], 0, H - 1)
                        cv2.fillPoly(mask, [poly_pts.astype(np.int32)], 255)
                        mask_name = f"warp_mask_polygon_{ts}_{digest}.png"
                        mask_path = os.path.join(WARP_DEBUG_DIR, mask_name)
                        cv2.imwrite(mask_path, mask)
                        print(f"[warp-debug] saved: {mask_path}")
                    except Exception as mask_err:
                        print(f"[warp-debug] mask save failed: {mask_err}")
            except Exception as save_err:
                print(f"[warp-debug] save failed: {save_err}")
        return Response(content=png_bytes, media_type="image/png")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid corners JSON")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


