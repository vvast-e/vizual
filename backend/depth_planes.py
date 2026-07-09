"""
depth_planes.py — деление маски стены фасада на плоскости через метрическую глубину.

Итерация 3 — метрическая глубина + последовательный RANSAC:
  - Метрическая модель (Depth-Anything-V2-Metric-Outdoor) возвращает реальные метры.
  - Back-projection: 2D-пиксели → 3D-облако точек (f ≈ W).
  - Sequential RANSAC: итеративно фитим плоскость по облаку (best-of-N через cross-product),
    уточняем SVD по инлайерам, удаляем их и повторяем → выделяем отдельные планарные регионы.
  - Нормаль берётся из фита плоскости (устойчива к попиксельному шуму).
  - Гейт ориентации: угол между нормалями двух плоскостей >= MIN_PLANE_ANGLE_DEG.
  - Debug-PNG сохраняются ВСЕГДА (до любых ранних return) — устраняет слепоту при отказах.

Предыдущие итерации провалились из-за аналитических нормалей через Sobel:
  - Итер.1: попиксельные нормали шумные → кластеры salt-and-pepper → после чистки 0 масок.
  - Итер.2: добавили пространство в KMeans → кластеры сплошные, но нормали одинаковые (cos_diff=0.008).
  Корень: из относительной глубины нормали ≈ nz=1 повсюду.
"""

import logging
import os
import time
from typing import Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── env-конфиги ────────────────────────────────────────────────────────────────────────────
DEPTH_URL = os.getenv("EXTERIOR_DEPTH_URL", "http://127.0.0.1:8001/depth")
DEPTH_TIMEOUT_S = float(os.getenv("EXTERIOR_DEPTH_TIMEOUT_MS", "60000")) / 1000.0

# Фокусное расстояние = FOCAL_RATIO × ширина изображения. 1.0 ≈ FOV 60°.
DEPTH_FOCAL_RATIO = float(os.getenv("EXTERIOR_DEPTH_FOCAL_RATIO", "1.0"))

# RANSAC: порог inlier = THRESH_RATIO × median_depth (масштабонезависимо).
DEPTH_RANSAC_THRESH_RATIO = float(os.getenv("EXTERIOR_DEPTH_RANSAC_THRESH_RATIO", "0.015"))

# Кол-во итераций RANSAC (случайных троек точек).
DEPTH_RANSAC_ITERS = int(os.getenv("EXTERIOR_DEPTH_RANSAC_ITERS", "300"))

# Минимальный угол (в градусах) между нормалями двух плоскостей → считаем их разными гранями.
DEPTH_MIN_PLANE_ANGLE_DEG = float(os.getenv("EXTERIOR_DEPTH_MIN_PLANE_ANGLE_DEG", "12"))

# Минимальная доля площади кадра для суб-маски.
DEPTH_MIN_PLANE_AREA_RATIO = float(os.getenv("EXTERIOR_DEPTH_MIN_PLANE_AREA_RATIO", "0.01"))

# Максимальное число плоскостей (итераций RANSAC).
DEPTH_MAX_K = int(os.getenv("EXTERIOR_DEPTH_MAX_K", "4"))


# ── 1. Запрос карты глубины ──────────────────────────────────────────────────────────────────

def request_depth(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    POST /depth → float32 карта глубины H×W (значения в метрах, метрическая модель).
    При ошибке возвращает None → пайплайн откатится на геометрию.
    """
    t0 = time.perf_counter()
    try:
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            logger.warning("[depth_planes] Failed to encode image")
            return None

        with httpx.Client(timeout=DEPTH_TIMEOUT_S) as client:
            resp = client.post(
                DEPTH_URL,
                files={"image": ("image.jpg", buf.tobytes(), "image/jpeg")},
            )

        if resp.status_code != 200:
            logger.warning("[depth_planes] /depth returned %d", resp.status_code)
            return None

        h = int(resp.headers.get("X-Depth-Height", 0))
        w = int(resp.headers.get("X-Depth-Width", 0))
        if h <= 0 or w <= 0:
            logger.warning("[depth_planes] /depth: missing size headers")
            return None

        depth = np.frombuffer(resp.content, dtype=np.float32).reshape(h, w)

        ih, iw = image_bgr.shape[:2]
        if (h, w) != (ih, iw):
            depth = cv2.resize(depth, (iw, ih), interpolation=cv2.INTER_LINEAR)

        logger.debug("[depth_planes] depth ok, shape %s, %.2fs", depth.shape, time.perf_counter() - t0)
        return depth

    except Exception as exc:
        logger.warning("[depth_planes] request_depth failed: %s", exc)
        return None


# ── 2. Back-projection: 2D → 3D ─────────────────────────────────────────────────────────────

def back_project(
    depth: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    focal_ratio: float = DEPTH_FOCAL_RATIO,
) -> np.ndarray:
    """
    Проецировать пиксели (ys, xs) с глубиной depth[ys, xs] в 3D-точки.

    Допущение: камера pinhole с f = W * focal_ratio, cx = W/2, cy = H/2.
    Возвращает N×3 float32 [X, Y, Z].
    """
    H, W = depth.shape
    f = W * focal_ratio
    cx, cy = W / 2.0, H / 2.0

    Z = depth[ys, xs].astype(np.float64)   # N, метры
    X = (xs.astype(np.float64) - cx) * Z / f
    Y = (ys.astype(np.float64) - cy) * Z / f

    return np.column_stack([X, Y, Z]).astype(np.float32)


# ── 3. Последовательный RANSAC: выделение планарных регионов ────────────────────────────────

def _fit_plane_ransac(
    pts: np.ndarray,
    iters: int,
    thresh: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    RANSAC для одной плоскости по облаку 3D-точек (N×3).
    Возвращает (normal_unit, point_on_plane, inlier_fraction).
    normal_unit — единичная нормаль (3,).
    """
    N = len(pts)
    if N < 3:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return n, pts[0], 0.0

    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    best_d = 0.0

    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1.astype(np.float64), v2.astype(np.float64))
        nlen = np.linalg.norm(n)
        if nlen < 1e-10:
            continue
        n /= nlen
        d = float(np.dot(n, p0.astype(np.float64)))
        dists = np.abs(pts.astype(np.float64) @ n - d)
        inliers = dists < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = n
            best_d = d

    # Уточнение: SVD по всем инлайерам.
    if best_count >= 3:
        inl_pts = pts[best_inliers].astype(np.float64)
        centroid = inl_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inl_pts - centroid)
        refined_n = Vt[-1]  # нормаль = наименьшее сингулярное значение
        refined_n /= np.linalg.norm(refined_n) + 1e-15
        # Проверяем, что не перевернули направление.
        if np.dot(refined_n, best_normal) < 0:
            refined_n = -refined_n
        best_normal = refined_n

    frac = best_count / N
    normal_f32 = best_normal.astype(np.float32)
    normal_f32 /= np.linalg.norm(normal_f32) + 1e-8
    return normal_f32, pts[best_inliers].mean(axis=0) if best_count > 0 else pts[0], frac


def sequential_ransac_planes(
    pts: np.ndarray,          # N×3
    ys: np.ndarray,           # N int (пиксельные координаты)
    xs: np.ndarray,           # N int
    image_height: int,
    image_width: int,
    min_area_px: int,
    max_planes: int,
    iters: int = DEPTH_RANSAC_ITERS,
    thresh_ratio: float = DEPTH_RANSAC_THRESH_RATIO,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Последовательный RANSAC: выделить до max_planes планарных регионов.
    На каждой итерации — найти лучшую плоскость, сохранить маску инлайеров, удалить их.
    Возвращает (list[binary_mask H×W], list[normal_vec 3]).
    """
    # Медианная глубина как опорный масштаб для порога inlier.
    median_z = float(np.median(pts[:, 2]))
    if median_z <= 0:
        median_z = 1.0
    thresh = thresh_ratio * median_z

    rng = np.random.default_rng(42)
    remaining = np.ones(len(pts), dtype=bool)   # активные точки

    submasks: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    for _ in range(max_planes):
        active_idx = np.where(remaining)[0]
        if len(active_idx) < max(50, min_area_px):
            break

        active_pts = pts[active_idx]
        normal, _, _ = _fit_plane_ransac(active_pts, iters, thresh, rng)

        # Ещё раз посчитать inliers по уточнённой нормали.
        d = float(np.dot(normal.astype(np.float64), active_pts.astype(np.float64).T).mean())
        # Пересчитаем d точно как median(n·p) для robustness.
        dot_vals = active_pts.astype(np.float64) @ normal.astype(np.float64)
        d = float(np.median(dot_vals))
        dists = np.abs(active_pts.astype(np.float64) @ normal.astype(np.float64) - d)
        inl_local = dists < thresh

        inl_count = int(inl_local.sum())
        if inl_count < min_area_px:
            # Слишком мало инлайеров — дальнейшие итерации бессмысленны.
            break

        # Построить бинарную маску.
        inl_global_idx = active_idx[inl_local]
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        mask[ys[inl_global_idx], xs[inl_global_idx]] = 255

        submasks.append(mask)
        normals.append(normal)

        # Удалить инлайеров из активного набора.
        remaining[inl_global_idx] = False

    return submasks, normals


# ── вспомогательная: морфочистка суб-маски ───────────────────────────────────────────────────

def _clean_submask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """
    CLOSE (21×21, iter=2) + OPEN (7×7) → сохранить ВСЕ связные компоненты >= min_area_px.
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), 8)
    if n <= 1:
        return out

    result = np.zeros_like(out)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            result[lbl == i] = 255
    return result


# ── 4. Главная функция: вызов RANSAC + фильтрация + debug ───────────────────────────────────

def cluster_planes(
    wall_mask: np.ndarray,
    depth: np.ndarray,
    image_width: int,
    image_height: int,
    min_area_ratio: float = DEPTH_MIN_PLANE_AREA_RATIO,
    max_walls: int = DEPTH_MAX_K,
    debug_dir: str = "",
) -> tuple[list[np.ndarray], dict]:
    """
    Разделить маску стены на планарные суб-маски через RANSAC по 3D-облаку точек.

    Возвращает (list[binary_mask], debug_dict).
    Итерация 3: убран аргумент normals= (нормали берутся из RANSAC-фита).
    """
    t0 = time.perf_counter()
    debug: dict = {
        "planes_found": 0,
        "plane_angle_deg": 0.0,
        "elapsed_s": 0.0,
        "reason": "init",
    }

    total_px = image_width * image_height
    min_area_px = max(200, int(total_px * min_area_ratio))

    ys, xs = np.where(wall_mask > 0)
    if len(ys) < 500:
        debug["reason"] = "too_few_pixels"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug_images(debug_dir, depth, None, [], ys, xs, image_height, image_width, wall_mask)
        return [], debug

    # Back-project в 3D.
    pts = back_project(depth, ys, xs)

    # Отбросить точки с нулевой/отрицательной глубиной (артефакты модели).
    # Порог относительный: > 1% медианной глубины — работает при любом масштабе модели.
    median_z = float(np.median(pts[:, 2]))
    z_thresh = max(median_z * 0.01, 1e-5) if median_z > 0 else 1e-5
    valid = pts[:, 2] > z_thresh
    if valid.sum() < 500:
        debug["reason"] = "too_few_valid_depth"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug_images(debug_dir, depth, None, [], ys, xs, image_height, image_width, wall_mask)
        return [], debug
    pts, ys, xs = pts[valid], ys[valid], xs[valid]

    # Downsample для скорости RANSAC (сохраняем представительный сэмпл).
    MAX_PTS = 50_000
    if len(pts) > MAX_PTS:
        rng_idx = np.random.RandomState(42).choice(len(pts), MAX_PTS, replace=False)
        pts_s, ys_s, xs_s = pts[rng_idx], ys[rng_idx], xs[rng_idx]
    else:
        pts_s, ys_s, xs_s = pts, ys, xs

    # Последовательный RANSAC.
    plane_masks_raw, plane_normals = sequential_ransac_planes(
        pts_s, ys_s, xs_s,
        image_height, image_width,
        min_area_px=min_area_px,
        max_planes=max_walls,
    )

    debug["planes_found"] = len(plane_normals)

    # Debug-визуализация — ВСЕГДА, до гейтов (чтобы видеть отказы).
    if debug_dir:
        try:
            label_indices = _make_label_indices(plane_masks_raw, image_height, image_width, wall_mask)
            _save_debug_images(
                debug_dir, depth, plane_normals,
                label_indices, ys, xs, image_height, image_width, wall_mask,
            )
        except Exception as e:
            logger.debug("[depth_planes] debug save failed: %s", e)

    if len(plane_normals) < 2:
        debug["reason"] = "only_one_plane_found"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return [], debug

    # Гейт ориентации: минимальный угол между нормалями пар плоскостей.
    min_angle_deg = float("inf")
    for i in range(len(plane_normals)):
        for j in range(i + 1, len(plane_normals)):
            ni = plane_normals[i].astype(np.float64)
            nj = plane_normals[j].astype(np.float64)
            cos_sim = abs(float(np.clip(np.dot(ni / (np.linalg.norm(ni) + 1e-8),
                                               nj / (np.linalg.norm(nj) + 1e-8)), -1.0, 1.0)))
            angle = float(np.degrees(np.arccos(min(cos_sim, 1.0))))
            min_angle_deg = min(min_angle_deg, angle)

    debug["plane_angle_deg"] = round(min_angle_deg, 2)

    if min_angle_deg < DEPTH_MIN_PLANE_ANGLE_DEG:
        debug["reason"] = f"planes_too_parallel_{min_angle_deg:.1f}deg"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return [], debug

    # Морфочистка и фильтрация по площади.
    submasks: list[np.ndarray] = []
    for raw in plane_masks_raw:
        cleaned = _clean_submask(raw, min_area_px)
        if int(np.count_nonzero(cleaned)) >= min_area_px:
            submasks.append(cleaned)

    if len(submasks) < 2:
        debug["reason"] = f"too_few_valid_submasks_after_filter ({len(submasks)}/need 2)"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return [], debug

    debug["reason"] = "ok"
    debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
    logger.debug(
        "[depth_planes] ok: planes=%d angle_deg=%.1f masks=%d elapsed=%.2fs",
        len(plane_normals), min_angle_deg, len(submasks), debug["elapsed_s"],
    )
    return submasks, debug


# ── вспомогательная: сборка label-карты для debug ───────────────────────────────────────────

def _make_label_indices(
    plane_masks: list[np.ndarray],
    H: int,
    W: int,
    wall_mask: np.ndarray,
) -> np.ndarray:
    """Возвращает H×W int32: -1=вне маски, k=индекс плоскости."""
    labels = np.full((H, W), -1, dtype=np.int32)
    labels[wall_mask > 0] = len(plane_masks)  # «неприписанные»
    for k, m in enumerate(plane_masks):
        labels[m > 0] = k
    return labels


# ── Debug-визуализация ───────────────────────────────────────────────────────────────────────

def _save_debug_images(
    debug_dir: str,
    depth: np.ndarray,
    plane_normals: Optional[list],
    label_indices,          # H×W int32 или [] если RANSAC не запускался
    ys: np.ndarray,
    xs: np.ndarray,
    image_height: int,
    image_width: int,
    wall_mask: np.ndarray,
) -> None:
    """Сохранить depth colormap и label map в debug_dir."""

    # 1. Depth — colormap (JET).
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max > d_min:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
    else:
        depth_norm = np.zeros(depth.shape, dtype=np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(debug_dir, "depth_planes_depth.png"), depth_color)

    # 2. Label map — каждая плоскость своим цветом.
    PALETTE = [
        (60, 120, 255),   # синий
        (50, 200, 50),    # зелёный
        (0, 80, 220),     # тёмно-синий
        (200, 50, 200),   # фиолетовый
        (30, 200, 200),   # циан
        (0, 165, 255),    # оранжевый
    ]
    label_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    label_img[:] = (40, 40, 40)  # серый фон

    if isinstance(label_indices, np.ndarray):
        # Раскрасить по индексам.
        n_planes = len(plane_normals) if plane_normals else 0
        for k in range(n_planes):
            color = PALETTE[k % len(PALETTE)]
            label_img[label_indices == k] = color
        # «Неприписанные» — серые (оставить фон).
    elif len(ys) > 0:
        # RANSAC не запускался — просто обозначить пиксели маски серым.
        label_img[ys, xs] = (100, 100, 100)

    # Контур маски стены.
    contours, _ = cv2.findContours(
        (wall_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(label_img, contours, -1, (255, 255, 255), 1)

    # Подписи нормалей плоскостей.
    if plane_normals:
        for k, n in enumerate(plane_normals):
            color = PALETTE[k % len(PALETTE)]
            text = f"P{k}: n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"
            cv2.putText(label_img, text, (10, 20 + k * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(debug_dir, "depth_planes_labels.png"), label_img)
