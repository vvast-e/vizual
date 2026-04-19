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
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure we run in backend directory so relative paths in submodules work
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
WARP_DEBUG_DIR = os.path.join(PROJECT_ROOT, "warp-debug")
WARP_DEBUG_SAVE = True
TEXTURES_DIR = os.path.join(PROJECT_ROOT, "public", "textures")
ALLOWED_TEX_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

from texture_mapping import get_wall_corners, image_resize, wall_polygon_centroid

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

        result = await asyncio.to_thread(_detect_walls_sync, tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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


# ─── Materials CRUD API ───────────────────────────────────────────────

ADMIN_LOGIN = "admin"
ADMIN_PASSWORD = "1Qst2la#"


def _check_admin_auth(authorization: str | None) -> None:
    """Validate Basic auth header for admin endpoints."""
    if not authorization or not authorization.startswith("Basic "):
        raise HTTPException(status_code=401, detail="Authorization required")
    import base64
    try:
        decoded = base64.b64decode(authorization.split(" ", 1)[1]).decode("utf-8")
        login, password = decoded.split(":", 1)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if login != ADMIN_LOGIN or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/api/materials")
async def list_materials():
    """List all texture files from public/textures/ recursively."""
    os.makedirs(TEXTURES_DIR, exist_ok=True)
    materials: list[dict] = []
    tex_path = Path(TEXTURES_DIR)
    for f in sorted(tex_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in ALLOWED_TEX_EXTS:
            rel = f.relative_to(tex_path).as_posix()
            # Skip swatch thumbnails
            if "/swatches/" in rel or rel.startswith("swatches/"):
                continue
            name = f.stem.replace("-", " ").replace("_", " ").title()
            category = rel.split("/")[0] if "/" in rel else "other"
            materials.append({
                "id": rel.replace("/", "-").replace(".", "-"),
                "name": name,
                "filename": rel,
                "category": category,
                "url": f"/textures/{rel}",
            })
    return {"materials": materials}


@app.post("/api/materials")
async def upload_material(
    file: UploadFile = File(...),
    name: str = Form(""),
    category: str = Form("other"),
    authorization: str | None = Form(None),
):
    """Upload a new texture file. Requires admin auth."""
    _check_admin_auth(f"Basic {authorization}" if authorization else None)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")

    ext = Path(file.filename or "texture.png").suffix.lower()
    if ext not in ALLOWED_TEX_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Sanitize category directory
    safe_category = "".join(c for c in category if c.isalnum() or c in "-_").strip() or "other"
    cat_dir = os.path.join(TEXTURES_DIR, safe_category)
    os.makedirs(cat_dir, exist_ok=True)

    # Generate unique filename
    safe_name = "".join(c for c in (name or file.filename or "texture") if c.isalnum() or c in "-_ ").strip().replace(" ", "-").lower()
    if not safe_name:
        safe_name = uuid.uuid4().hex[:8]
    dest_name = f"{safe_name}{ext}"
    dest_path = os.path.join(cat_dir, dest_name)

    # Avoid overwrite
    counter = 1
    while os.path.exists(dest_path):
        dest_name = f"{safe_name}-{counter}{ext}"
        dest_path = os.path.join(cat_dir, dest_name)
        counter += 1

    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    rel = f"{safe_category}/{dest_name}"
    return {
        "id": rel.replace("/", "-").replace(".", "-"),
        "name": name or safe_name,
        "filename": rel,
        "category": safe_category,
        "url": f"/textures/{rel}",
    }


@app.delete("/api/materials/{filename:path}")
async def delete_material(filename: str, authorization: str | None = Header(None)):
    """Delete a texture file. Requires admin auth via Authorization header."""
    _check_admin_auth(authorization)
    safe = filename.replace("..", "").replace("\\", "/").strip("/")
    file_path = os.path.join(TEXTURES_DIR, safe)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(file_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": safe}
