"""
plane_split.py — деление маски стены фасада на планарные грани через нормали поверхности.

Заменяет depth_planes.py (итерации 1-3), который провалился из-за того, что из карты глубины
не извлекается разница ориентаций граней при фронтальном/слабоугловом ракурсе.

Подход: нормали поверхности дают прямой сигнал ориентации плоскости. Грани с разной ориентацией =
разные кластеры нормалей. Деградирует честно: на фронтали нормали однородны → 1 кластер → пусто →
фолбэк на геометрию без ложного деления.

Гибрид:
  1. Нормали (основной сигнал): KMeans/агломеративная кластеризация азимутального угла нормалей.
  2. VP по LSD (валидация/снаппинг ребра): x-позиция границы смещается к ближайшему LSD-ребру
     при совпадении в пределах PLANE_VP_SNAP_PX.
  3. Геометрический каскад (фолбэк в exterior_pipeline.py, снаружи этого модуля).

Debug-PNG сохраняются ВСЕГДА (до любых return).
"""

import logging
import os
import time
from typing import Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── env-параметры ──────────────────────────────────────────────────────────────────────────────
NORMALS_URL = os.getenv("EXTERIOR_NORMALS_URL", "http://127.0.0.1:8001/normals")
NORMALS_TIMEOUT_S = float(os.getenv("EXTERIOR_NORMALS_TIMEOUT_MS", "60000")) / 1000.0

# Минимальный угол (°) между средними нормалями кластеров → считаем их разными гранями.
PLANE_MIN_ANGLE_DEG = float(os.getenv("EXTERIOR_DEPTH_MIN_PLANE_ANGLE_DEG", "12"))

# Минимальная доля площади кадра для суб-маски грани.
PLANE_MIN_AREA_RATIO = float(os.getenv("EXTERIOR_DEPTH_MIN_PLANE_AREA_RATIO", "0.01"))

# Максимальное число граней.
PLANE_MAX_K = int(os.getenv("EXTERIOR_DEPTH_MAX_K", "4"))

# Размер gaussian-blur для сглаживания нормалей (убирает шум кирпича/лепнины).
# 0 = отключить.
PLANE_NORMALS_BLUR = int(os.getenv("EXTERIOR_NORMALS_BLUR_PX", "21"))

# Снаппинг границы маски к ближайшему LSD-ребру: максимальное расстояние (px).
PLANE_VP_SNAP_PX = int(os.getenv("EXTERIOR_PLANE_VP_SNAP_PX", "30"))

# Допуск вертикальности угла здания (°): насколько ребро может отклоняться от строгой вертикали.
PLANE_SEAM_VERT_TOL_DEG = float(os.getenv("EXTERIOR_PLANE_SEAM_VERT_TOL_DEG", "20"))

# Минимальная доля локальной высоты стены, которую ребро должно покрывать по Y.
PLANE_SEAM_VEXTENT_FRAC = float(os.getenv("EXTERIOR_PLANE_SEAM_VEXTENT_FRAC", "0.45"))

# Мин. доля площади стены, чтобы регион считался ОТДЕЛЬНОЙ гранью, а не островком (окно/шум).
# Структурный факт «окно << стены»: грань фасада >= ~8% площади; окно/артефакт обычно < 5%.
PLANE_REGION_MIN_AREA_FRAC = float(os.getenv("EXTERIOR_PLANE_REGION_MIN_AREA_FRAC", "0.08"))

# Защита от битой маски проёмов: если вычитание окон оставляет < этой доли стены,
# holes_mask игнорируется (маска явно сломана). >50% выреза для нормального фасада невозможно.
PLANE_HOLES_KEEP_FRAC = float(os.getenv("EXTERIOR_PLANE_HOLES_KEEP_FRAC", "0.5"))


# ── 1. Запрос карты нормалей ───────────────────────────────────────────────────────────────────

def request_normals(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    POST /normals → float32 карта нормалей H×W×3 (единичные векторы, система координат камеры).
    При ошибке возвращает None → пайплайн откатится на геометрию.
    """
    t0 = time.perf_counter()
    try:
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            logger.warning("[plane_split] Failed to encode image")
            return None

        with httpx.Client(timeout=NORMALS_TIMEOUT_S) as client:
            resp = client.post(
                NORMALS_URL,
                files={"image": ("image.jpg", buf.tobytes(), "image/jpeg")},
            )

        if resp.status_code != 200:
            logger.warning("[plane_split] /normals returned %d", resp.status_code)
            return None

        h = int(resp.headers.get("X-Normals-Height", 0))
        w = int(resp.headers.get("X-Normals-Width", 0))
        c = int(resp.headers.get("X-Normals-Channels", 3))
        if h <= 0 or w <= 0:
            logger.warning("[plane_split] /normals: missing size headers")
            return None

        normals = np.frombuffer(resp.content, dtype=np.float32).reshape(h, w, c)

        ih, iw = image_bgr.shape[:2]
        if (h, w) != (ih, iw):
            # Ресайз нормалей через компоненты, потом перенормализация.
            normals = cv2.resize(normals, (iw, ih), interpolation=cv2.INTER_LINEAR)
            norms = np.linalg.norm(normals, axis=2, keepdims=True)
            normals = normals / (norms + 1e-8)

        logger.debug(
            "[plane_split] normals ok, shape %s, %.2fs",
            normals.shape, time.perf_counter() - t0,
        )
        return normals

    except Exception as exc:
        logger.warning("[plane_split] request_normals failed: %s", exc)
        return None


# ── 2. Сглаживание нормалей ────────────────────────────────────────────────────────────────────

def _smooth_normals(normals: np.ndarray, wall_mask: np.ndarray, blur_px: int) -> np.ndarray:
    """
    Сгладить карту нормалей в области маски стены через Gaussian blur по каждому каналу,
    затем перенормировать. Убирает шум кирпича/лепнины.
    """
    if blur_px <= 0:
        return normals

    ksize = blur_px | 1  # нечётное
    smoothed = np.zeros_like(normals)
    mask_f = (wall_mask > 0).astype(np.float32)

    for c in range(normals.shape[2]):
        ch = normals[:, :, c] * mask_f  # зануляем вне маски перед blur
        blurred = cv2.GaussianBlur(ch, (ksize, ksize), 0)
        # Нормировать blur вес-суммой (компенсация близости к краю маски).
        weight = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
        safe_w = np.where(weight > 1e-4, weight, 1.0)  # избегаем деления на ноль
        smoothed[:, :, c] = np.where(weight > 1e-4, blurred / safe_w, normals[:, :, c])

    norms = np.linalg.norm(smoothed, axis=2, keepdims=True)
    return smoothed / (norms + 1e-8)


# ── 3. Кластеризация нормалей ──────────────────────────────────────────────────────────────────

def _orient_to_hemisphere(normals_nx3: np.ndarray) -> np.ndarray:
    """
    Ориентировать нормали в одно полушарие: поверхности, обращённые от камеры (Nz>0 в
    OpenCV-системе, где Z — от камеры), переворачиваем. После этого все нормали стен
    здания смотрят примерно в сторону камеры (Nz<0) — wrap-around ±180° исчезает,
    антипараллельность «отражённых» нормалей устранена.
    """
    n = normals_nx3.copy().astype(np.float64)
    # В системе DSINE: нормаль фасадной стены имеет Nz < 0 (смотрит к камере).
    # Если Nz > 0, вектор указывает от камеры (крыша или задняя поверхность) — флип.
    flip = n[:, 2] > 0
    n[flip] = -n[flip]
    return n.astype(np.float32)


def _cluster_normals(
    normals_px: np.ndarray,    # N×3, уже ориентированные в полушарие
    ys: np.ndarray,
    xs: np.ndarray,
    image_height: int,
    image_width: int,
    max_k: int,
    min_area_px: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    KMeans по горизонтальной проекции нормалей (Nx, Nz) — 2D, без wrap-проблемы.
    Выбирает k=2..max_k по снижению inertia (>30% → продолжаем).
    Средняя нормаль кластера = нормализованное среднее реальных векторов.
    Возвращает (маски H×W, средние_нормали).
    """
    from sklearn.cluster import KMeans

    # Фичи: горизонтальная проекция (Nx, Nz), нормализованная по строке.
    feats = normals_px[:, [0, 2]].astype(np.float64)  # N×2
    norms_2d = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / (norms_2d + 1e-8)

    best_k = 1
    best_labels = np.zeros(len(normals_px), dtype=np.int32)
    prev_inertia = float("inf")

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=8, max_iter=150, random_state=42)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # подавить ConvergenceWarning при однородных данных
            km.fit(feats)
        inertia = float(km.inertia_)

        improvement = (prev_inertia - inertia) / (prev_inertia + 1e-8)
        if improvement < 0.25:
            break

        labels = km.labels_.astype(np.int32)
        sizes = [int((labels == i).sum()) for i in range(k)]
        if min(sizes) < min_area_px:
            break

        best_k = k
        best_labels = labels
        prev_inertia = inertia

    # Собрать маски и реальные средние нормали.
    masks: list[np.ndarray] = []
    avg_normals: list[np.ndarray] = []

    for k in range(best_k):
        idx = best_labels == k
        if idx.sum() < min_area_px:
            continue
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        mask[ys[idx], xs[idx]] = 255
        masks.append(mask)

        # Реальное среднее нормалей кластера (устойчиво к шуму, нет wrap).
        mean_n = normals_px[idx].astype(np.float64).mean(axis=0)
        mean_n /= np.linalg.norm(mean_n) + 1e-8
        avg_normals.append(mean_n.astype(np.float32))

    return masks, avg_normals


def _merge_similar_clusters(
    masks: list[np.ndarray],
    normals: list[np.ndarray],
    min_angle_deg: float,
    min_area_px: int,
    image_height: int,
    image_width: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Слить кластеры, чьи нормали ближе min_angle_deg (оверсегментация шума/крыши).
    После слияния отфильтровать по min_area_px.
    Возвращает (новые маски, новые нормали).
    """
    if len(masks) <= 1:
        return masks, normals

    # Жадное слияние: найти ближайшую пару, слить, повторить.
    merged_masks = [m.copy() for m in masks]
    merged_normals = [n.copy() for n in normals]

    changed = True
    while changed and len(merged_masks) > 1:
        changed = False
        best_cos = -2.0
        best_i, best_j = -1, -1
        for i in range(len(merged_normals)):
            for j in range(i + 1, len(merged_normals)):
                ni = merged_normals[i].astype(np.float64)
                nj = merged_normals[j].astype(np.float64)
                cos_v = float(np.clip(
                    np.dot(ni / (np.linalg.norm(ni) + 1e-8), nj / (np.linalg.norm(nj) + 1e-8)),
                    -1.0, 1.0,
                ))
                angle = float(np.degrees(np.arccos(cos_v)))
                # Слияние если угол < порога (нормали слишком похожи).
                if angle < min_angle_deg and cos_v > best_cos:
                    best_cos = cos_v
                    best_i, best_j = i, j

        if best_i >= 0:
            # Слить j в i.
            new_mask = cv2.bitwise_or(merged_masks[best_i], merged_masks[best_j])
            # Новая нормаль = нормализованная сумма (взвешенная по площади).
            area_i = float(np.count_nonzero(merged_masks[best_i]))
            area_j = float(np.count_nonzero(merged_masks[best_j]))
            total = area_i + area_j + 1e-8
            new_n = (merged_normals[best_i].astype(np.float64) * area_i +
                     merged_normals[best_j].astype(np.float64) * area_j) / total
            new_n /= np.linalg.norm(new_n) + 1e-8

            merged_masks[best_i] = new_mask
            merged_normals[best_i] = new_n.astype(np.float32)
            merged_masks.pop(best_j)
            merged_normals.pop(best_j)
            changed = True

    # Фильтр по площади.
    out_masks, out_normals = [], []
    for m, n in zip(merged_masks, merged_normals):
        if int(np.count_nonzero(m)) >= min_area_px:
            out_masks.append(m)
            out_normals.append(n)

    return out_masks, out_normals


# ── 4. VP-снаппинг ────────────────────────────────────────────────────────────────────────────

def snap_boundary_to_vp_edges(
    plane_masks: list[np.ndarray],
    lsd_vertical_xs: list[int],
    image_height: int,
    image_width: int,
    snap_px: int = PLANE_VP_SNAP_PX,
) -> tuple[list[np.ndarray], dict]:
    """
    Сдвинуть вертикальную границу между гранями (нормальный сплит) к ближайшему
    вертикальному LSD-ребру, если |Δx| <= snap_px.

    plane_masks: ровно 2 маски (left/right); при > 2 возвращает без изменений.
    lsd_vertical_xs: список x-координат вертикальных LSD-рёбер (от geom-каскада).
    Возвращает (новые маски, debug-словарь).
    """
    snap_debug: dict = {
        "vp_agreement": False,
        "dx": None,
        "snapped_x": None,
        "original_x": None,
        "final_x": None,
    }

    if len(plane_masks) != 2:
        return plane_masks, snap_debug

    # Найти вертикальную границу между масками: медиана x-координат пограничных пикселей.
    left_mask, right_mask = plane_masks[0], plane_masks[1]
    border = cv2.bitwise_and(
        cv2.dilate(left_mask, np.ones((3, 3), np.uint8), iterations=2),
        cv2.dilate(right_mask, np.ones((3, 3), np.uint8), iterations=2),
    )
    bys, bxs = np.where(border > 0)
    if len(bxs) == 0:
        return plane_masks, snap_debug

    boundary_x = int(np.median(bxs))
    snap_debug["original_x"] = boundary_x
    final_x = boundary_x

    # Снаппинг к ближайшему LSD-ребру (если есть и достаточно близко).
    if lsd_vertical_xs:
        xs_arr = np.array(lsd_vertical_xs)
        nearest_x = int(xs_arr[int(np.argmin(np.abs(xs_arr - boundary_x)))])
        dx = abs(boundary_x - nearest_x)
        snap_debug["dx"] = dx
        if dx <= snap_px:
            snap_debug["vp_agreement"] = True
            snap_debug["snapped_x"] = nearest_x
            final_x = nearest_x
            logger.debug(
                "[plane_split] VP snap: boundary_x=%d → lsd_x=%d (dx=%d)",
                boundary_x, nearest_x, dx,
            )

    snap_debug["final_x"] = final_x
    return plane_masks, snap_debug


# ── 5. Вспомогательные функции для профиля по столбцам ───────────────────────────────────────


def _snap_x_to_lsd(
    x: int,
    lsd_xs: list[int],
    snap_px: int = PLANE_VP_SNAP_PX,
) -> tuple[int, bool, Optional[int]]:
    """Снапнуть x к ближайшему LSD-ребру если |Δ| <= snap_px. Возвращает (final_x, snapped, dx)."""
    if not lsd_xs:
        return x, False, None
    xs_arr = np.array(lsd_xs)
    nearest = int(xs_arr[int(np.argmin(np.abs(xs_arr - x)))])
    dx = abs(x - nearest)
    if dx <= snap_px:
        return nearest, True, dx
    return x, False, dx


def _vertical_seams_from_normals(
    normals_smooth: np.ndarray,                       # H×W×3
    analysis_mask: np.ndarray,                        # H×W uint8
    dominant_normals: list[np.ndarray],               # K×3, доминирующие ориентации
    image_height: int,
    image_width: int,
    vert_tol_deg: float = PLANE_SEAM_VERT_TOL_DEG,
    vextent_frac: float = PLANE_SEAM_VEXTENT_FRAC,
    nms_px: int = 0,
    debug_dir: str = "",
) -> list[int]:
    """
    Найти x-координаты вертикальных рёбер между планарными гранями — через region-cleaning.

    Принцип («найти переход от розового к голубому и построить вертикаль», без порогов под кадр):
    1. Каждый пиксель analysis_mask → ближайшая доминирующая ориентация (region_map).
    2. Region-cleaning: мелкие связные островки каждого класса (окна/шум < PLANE_REGION_MIN_AREA_FRAC
       площади стены) → -1 (не присвоено). Пустоты заполняются ближайшим сохранённым регионом
       (евклидов nearest-label fill через distance_transform_edt).
       Структурный факт: окно ≪ стены → всегда островок, поглощается независимо от bbox.
    3. Граница = горизонтальный переход в очищенной карте (большие регионы, окна исчезли).
       Санити-гейты: вертикальность + охват высоты стены (факты об архитектурном угле).
    4. seam_x = медиана X компоненты, затем NMS.
    """
    if len(dominant_normals) < 2:
        return []

    # ── Region-map: ближайшая доминирующая ориентация ────────────────────────────────────────
    ay, ax_arr = np.where(analysis_mask > 0)
    if len(ay) == 0:
        return []

    an = normals_smooth[ay, ax_arr].astype(np.float64)
    norms_an = np.linalg.norm(an, axis=1, keepdims=True)
    an = an / (norms_an + 1e-8)
    # Ориентировать в полушарие (Nz > 0 → задняя поверхность → флип)
    flip = an[:, 2] > 0
    an[flip] = -an[flip]

    dn = np.stack(dominant_normals, axis=0).astype(np.float64)  # K×3
    dn_norms = np.linalg.norm(dn, axis=1, keepdims=True)
    dn = dn / (dn_norms + 1e-8)

    sims = an @ dn.T   # N×K
    labels = sims.argmax(axis=1).astype(np.int16)

    region_map = np.full((image_height, image_width), -1, dtype=np.int16)
    region_map[ay, ax_arr] = labels

    # ── Region-cleaning: поглощение мелких островков (окна/шум) ─────────────────────────────
    wall_area = int(np.count_nonzero(analysis_mask))
    min_region_px = max(200, int(wall_area * PLANE_REGION_MIN_AREA_FRAC))
    num_labels = len(dominant_normals)

    region_clean = region_map.copy()
    for lbl in range(num_labels):
        lbl_mask = (region_map == lbl).astype(np.uint8)
        n_cc, cc_map, cc_stats, _ = cv2.connectedComponentsWithStats(lbl_mask, connectivity=8)
        for cc_id in range(1, n_cc):
            if cc_stats[cc_id, cv2.CC_STAT_AREA] < min_region_px:
                region_clean[cc_map == cc_id] = -1   # мелкий островок → не присвоено

    # ── Nearest-label fill: пустоты → ближайший сохранённый регион ───────────────────────────
    from scipy.ndimage import distance_transform_edt

    seed = region_clean >= 0     # пиксели с валидной меткой
    gap = (~seed) & (analysis_mask > 0)   # пустоты внутри маски стены
    if gap.any() and seed.any():
        # Для каждого не-seed пикселя: координаты ближайшего seed-пикселя
        _, idx = distance_transform_edt(~seed, return_indices=True)
        region_filled = region_clean.copy()
        gy, gx = np.where(gap)
        region_filled[gy, gx] = region_clean[idx[0][gy, gx], idx[1][gy, gx]]
    else:
        region_filled = region_clean.copy()
    region_filled[analysis_mask == 0] = -1   # вне маски — не присвоено

    # ── Debug: сохранить очищенную карту регионов ────────────────────────────────────────────
    if debug_dir:
        try:
            dbg = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            dbg[:] = (20, 20, 20)
            for lbl in range(num_labels):
                dbg[region_filled == lbl] = _PALETTE[lbl % len(_PALETTE)]
            cv2.imwrite(os.path.join(debug_dir, "region_clean.png"), dbg)
        except Exception as _e:
            logger.debug("[plane_split] region_clean debug failed: %s", _e)

    # ── Граница = горизонтальный переход в очищенной карте ───────────────────────────────────
    left_lab = region_filled[:, :-1]   # H×(W-1)
    right_lab = region_filled[:, 1:]
    both_valid = (left_lab >= 0) & (right_lab >= 0)
    is_boundary = both_valid & (left_lab != right_lab)

    boundary_map = np.zeros((image_height, image_width), dtype=np.uint8)
    boundary_map[:, 1:][is_boundary] = 255

    if int(np.count_nonzero(boundary_map)) == 0:
        return []

    # ── Связные компоненты → санити-гейты ────────────────────────────────────────────────────
    num_cc, cc_labels, _, _ = cv2.connectedComponentsWithStats(boundary_map, connectivity=8)
    col_heights = (analysis_mask > 0).sum(axis=0).astype(np.float32)  # W

    seams: list[int] = []

    for cc_id in range(1, num_cc):
        cc_ys, cc_xs = np.where(cc_labels == cc_id)
        if len(cc_ys) < 3:
            continue

        y_min, y_max = int(cc_ys.min()), int(cc_ys.max())
        y_extent = y_max - y_min + 1

        x_min_cc, x_max_cc = int(cc_xs.min()), int(cc_xs.max())
        local_wall_h = float(col_heights[x_min_cc:x_max_cc + 1].max())
        if local_wall_h < 10:
            continue

        # Гейт по охвату высоты (факт: архитектурный угол идёт во всю высоту стены)
        if y_extent < vextent_frac * local_wall_h:
            continue

        # Гейт по вертикальности (факт: архитектурный угол вертикален)
        pts = np.stack([cc_xs, cc_ys], axis=1).astype(np.float32)
        if len(pts) >= 2:
            vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            angle_from_horiz = float(np.degrees(np.arctan2(abs(vy), abs(vx) + 1e-8)))
            if angle_from_horiz < (90.0 - vert_tol_deg):
                continue  # диагональ крыши или мусор
        else:
            continue

        seam_x = int(np.median(cc_xs))
        seams.append(seam_x)

    # ── NMS: убрать дубликаты ближе nms_px ───────────────────────────────────────────────────
    if nms_px > 0 and len(seams) > 1:
        seams_sorted = sorted(seams)
        merged: list[int] = [seams_sorted[0]]
        for sx in seams_sorted[1:]:
            if sx - merged[-1] >= nms_px:
                merged.append(sx)
            else:
                merged[-1] = (merged[-1] + sx) // 2
        seams = merged

    return sorted(seams)


# ── вспомогательная: морфочистка суб-маски ────────────────────────────────────────────────────

def _clean_submask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """CLOSE (21×21, iter=2) + OPEN (7×7) → сохранить компоненты >= min_area_px."""
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


# ── 5. Главная функция ────────────────────────────────────────────────────────────────────────

def cluster_planes_by_normals(
    wall_mask: np.ndarray,
    normals: np.ndarray,                              # H×W×3 float32, единичные векторы
    image_width: int,
    image_height: int,
    lsd_vertical_xs: Optional[list[int]] = None,     # VP-снаппинг (опционально)
    holes_mask: Optional[np.ndarray] = None,          # маска окон/дверей (исключить из анализа)
    openings_bboxes: Optional[list[list[float]]] = None,  # reserved — больше не форвардится в seam-функцию
    min_area_ratio: float = PLANE_MIN_AREA_RATIO,
    max_walls: int = PLANE_MAX_K,
    debug_dir: str = "",
) -> tuple[list[np.ndarray], dict]:
    """
    Разделить маску стены на планарные суб-маски через кластеризацию нормалей поверхности.

    Возвращает (list[binary_mask H×W], debug_dict).
    Деградирует честно: фронталь → нормали однородны → 1 кластер → [], reason='only_one_cluster'.
    """
    t0 = time.perf_counter()
    debug: dict = {
        "planes_found": 0,
        "plane_angle_deg": 0.0,
        "elapsed_s": 0.0,
        "reason": "init",
        "vp_snap": {},
    }

    total_px = image_width * image_height
    min_area_px = max(200, int(total_px * min_area_ratio))

    # analysis_mask = wall_mask без окон/дверей (нормали стёкол не участвуют в поиске рёбер).
    # Защита: если маска проёмов битая и вырезает > половины стены — игнорируем её
    # (вертикальный гейт всё равно отфильтрует реальные кромки окон).
    analysis_mask = wall_mask
    debug["holes_ignored"] = False
    if holes_mask is not None and holes_mask.shape == wall_mask.shape:
        candidate = np.where(holes_mask > 0, 0, wall_mask).astype(np.uint8)
        wall_area = int(np.count_nonzero(wall_mask))
        cand_area = int(np.count_nonzero(candidate))
        if wall_area > 0 and cand_area >= PLANE_HOLES_KEEP_FRAC * wall_area:
            analysis_mask = candidate
        else:
            debug["holes_ignored"] = True
            logger.warning(
                "[plane_split] holes_mask ignored: removes %.0f%% of wall (%d/%d) — broken openings",
                100.0 * (1.0 - cand_area / max(wall_area, 1)), cand_area, wall_area,
            )

    ys, xs = np.where(analysis_mask > 0)
    if len(ys) < 500:
        debug["reason"] = "too_few_pixels"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals, [], [], {})
        return [], debug

    # Сглаживание по analysis_mask (без окон/дверей).
    normals_smooth = _smooth_normals(normals, analysis_mask, PLANE_NORMALS_BLUR)

    # Извлечь нормали по пикселям analysis_mask.
    normals_px = normals_smooth[ys, xs]  # N×3

    # Фильтр невалидных нормалей (нулевая/почти нулевая норма).
    norms = np.linalg.norm(normals_px, axis=1)
    valid = norms > 0.5
    if valid.sum() < 500:
        debug["reason"] = "too_few_valid_normals"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals_smooth, [], [], {})
        return [], debug
    normals_px = normals_px[valid]
    ys_v, xs_v = ys[valid], xs[valid]

    # Ориентировать нормали в одно полушарие (устраняет wrap-проблему и антипараллельность).
    normals_oriented = _orient_to_hemisphere(normals_px)

    # Кластеризация по 2D горизонтальной проекции нормалей (без 1D-азимут wrap).
    plane_masks_raw, plane_normals = _cluster_normals(
        normals_oriented, ys_v, xs_v, image_height, image_width, max_walls, min_area_px,
    )

    debug["planes_found"] = len(plane_normals)
    debug["plane_normals"] = [
        [round(float(n[0]), 3), round(float(n[1]), 3), round(float(n[2]), 3)]
        for n in plane_normals
    ]

    if len(plane_normals) < 2:
        debug["reason"] = "only_one_cluster"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals_smooth, plane_masks_raw, plane_normals, {})
        return [], debug

    # Слияние похожих кластеров (оверсегментация крыши/шума).
    plane_masks_merged, plane_normals = _merge_similar_clusters(
        plane_masks_raw, plane_normals, PLANE_MIN_ANGLE_DEG, min_area_px,
        image_height, image_width,
    )
    debug["planes_after_merge"] = len(plane_normals)

    if len(plane_normals) < 2:
        debug["reason"] = "only_one_cluster_after_merge"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals_smooth, plane_masks_merged, plane_normals, {})
        return [], debug

    # Гейт ориентации: угол между двумя КРУПНЕЙШИМИ кластерами (по площади).
    # Без abs(): нормали ориентированы в одно полушарие → истинный угол корректен.
    areas = [int(np.count_nonzero(m)) for m in plane_masks_merged]
    top2_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)[:2]
    i0, i1 = top2_idx[0], top2_idx[1]
    ni = plane_normals[i0].astype(np.float64)
    nj = plane_normals[i1].astype(np.float64)
    cos_v = float(np.clip(
        np.dot(ni / (np.linalg.norm(ni) + 1e-8), nj / (np.linalg.norm(nj) + 1e-8)),
        -1.0, 1.0,
    ))
    best_pair_angle = float(np.degrees(np.arccos(cos_v)))
    debug["plane_angle_deg"] = round(best_pair_angle, 2)
    debug["chosen_pair"] = [int(i0), int(i1)]

    if best_pair_angle < PLANE_MIN_ANGLE_DEG:
        debug["reason"] = f"planes_too_parallel_{best_pair_angle:.1f}deg"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals_smooth, plane_masks_merged, plane_normals, {})
        return [], debug

    # ── Детекция вертикальных рёбер через region-cleaning ─────────────────────────────────────
    nms_px = max(40, int(image_width * 0.04))
    seams_raw = _vertical_seams_from_normals(
        normals_smooth, analysis_mask, plane_normals,
        image_height, image_width, nms_px=nms_px,
        debug_dir=debug_dir,
    )

    # ── Снап каждого стыка к ближайшему LSD-ребру ──────────────────────────────────────────────
    lsd_xs = lsd_vertical_xs or []
    seams: list[int] = []
    vp_snaps: list[dict] = []
    any_vp_agreement = False
    for sx in seams_raw:
        fx, snapped, dx = _snap_x_to_lsd(sx, lsd_xs)
        seams.append(fx)
        vp_snaps.append({"original_x": sx, "final_x": fx, "vp_agreement": snapped, "dx": dx})
        if snapped:
            any_vp_agreement = True
            logger.debug("[plane_split] seam VP snap: %d → %d (dx=%d)", sx, fx, dx)

    debug["seams_x"] = seams
    debug["vp_snaps"] = vp_snaps
    debug["vp_snap"] = {"vp_agreement": any_vp_agreement}  # совместимость с exterior_pipeline

    if not seams:
        debug["reason"] = "no_vertical_seam"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals_smooth, [], plane_normals, {})
        return [], debug

    # ── Резать wall_mask на полосы по стыкам ───────────────────────────────────────────────────
    boundaries = [0] + sorted(seams) + [image_width]
    submasks: list[np.ndarray] = []
    for i in range(len(boundaries) - 1):
        x_start, x_end = boundaries[i], boundaries[i + 1]
        band = wall_mask.copy()
        band[:, :x_start] = 0
        band[:, x_end:] = 0
        if int(np.count_nonzero(band)) >= min_area_px:
            submasks.append(band)

    debug["bands"] = len(submasks)

    if len(submasks) < 2:
        debug["reason"] = f"too_few_bands ({len(submasks)}/need 2)"
        debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if debug_dir:
            _save_debug(debug_dir, normals_smooth, submasks, plane_normals, {})
        return [], debug

    if debug_dir:
        _save_debug(debug_dir, normals_smooth, submasks, plane_normals, {"vp_agreement": any_vp_agreement})

    debug["reason"] = "ok"
    debug["elapsed_s"] = round(time.perf_counter() - t0, 3)
    logger.debug(
        "[plane_split] ok: planes=%d angle_deg=%.1f seams=%s bands=%d vp=%s elapsed=%.2fs",
        len(plane_normals), best_pair_angle, seams, len(submasks),
        any_vp_agreement, debug["elapsed_s"],
    )
    return submasks, debug


# ── Debug-визуализация ────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (60, 120, 255),   # синий
    (50, 200, 50),    # зелёный
    (0, 80, 220),     # тёмно-синий
    (200, 50, 200),   # фиолетовый
    (30, 200, 200),   # циан
    (0, 165, 255),    # оранжевый
]


def _save_debug(
    debug_dir: str,
    normals: np.ndarray,           # H×W×3
    plane_masks: list[np.ndarray],
    plane_normals: list[np.ndarray],
    snap_debug: dict,
) -> None:
    """Сохранить normals_rgb.png и plane_labels.png в debug_dir."""
    try:
        H, W = normals.shape[:2]

        # 1. Нормали как RGB: X→R, Y→G, Z→B ([-1,1]→[0,255]).
        normals_rgb = ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        # OpenCV хранит BGR → нормали (R=X, G=Y, B=Z) конвертим X→B, Z→R для красивого вида.
        normals_bgr = normals_rgb[:, :, ::-1].copy()
        cv2.imwrite(os.path.join(debug_dir, "normals_rgb.png"), normals_bgr)

        # 2. Label map.
        label_img = np.zeros((H, W, 3), dtype=np.uint8)
        label_img[:] = (40, 40, 40)

        for k, mask in enumerate(plane_masks):
            color = _PALETTE[k % len(_PALETTE)]
            label_img[mask > 0] = color

        # Граница маски стены (белый контур).
        # Собираем объединение всех масок как "маска стены" для контура.
        union = np.zeros((H, W), dtype=np.uint8)
        for m in plane_masks:
            union = cv2.bitwise_or(union, m)
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(label_img, contours, -1, (255, 255, 255), 1)

        # Линия снаппинга.
        snapped_x = snap_debug.get("snapped_x")
        orig_x = snap_debug.get("original_x")
        if orig_x is not None:
            cv2.line(label_img, (orig_x, 0), (orig_x, H - 1), (100, 100, 255), 1)
        if snapped_x is not None and snap_debug.get("vp_agreement"):
            cv2.line(label_img, (snapped_x, 0), (snapped_x, H - 1), (0, 255, 255), 2)

        # Подписи нормалей.
        for k, n in enumerate(plane_normals):
            color = _PALETTE[k % len(_PALETTE)]
            text = f"P{k}: n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"
            cv2.putText(label_img, text, (10, 20 + k * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if snap_debug.get("vp_agreement"):
            cv2.putText(label_img, f"VP snap dx={snap_debug.get('dx')}px",
                        (10, 20 + len(plane_normals) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(debug_dir, "plane_labels.png"), label_img)

    except Exception as e:
        logger.debug("[plane_split] debug save failed: %s", e)
