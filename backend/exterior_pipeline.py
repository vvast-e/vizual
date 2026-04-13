"""
Exterior detection pipeline.
Calls local model_service (GroundingDINO + SAM) via HTTP,
postprocesses masks, returns wall/holes/wall_minus_holes.
"""
import base64
import io
import hashlib
import os
import time

import cv2
import httpx
import numpy as np

from facade_geometry import split_gable_roof_polygon

GDINO_URL = os.getenv("EXTERIOR_GDINO_URL", "http://127.0.0.1:8001/gdino")
SAM_URL = os.getenv("EXTERIOR_SAM_URL", "http://127.0.0.1:8001/sam")
TIMEOUT_S = int(os.getenv("EXTERIOR_TIMEOUT_MS", "30000")) / 1000.0
SCORE_THRESH_BUILDING = float(os.getenv("EXTERIOR_SCORE_THRESH_BUILDING", "0.3"))
SCORE_THRESH_OPENINGS = float(os.getenv("EXTERIOR_SCORE_THRESH_OPENINGS", "0.22"))

# Debug output for investigating mask->front rendering.
# Saves intermediate PNGs + prints summary to stdout.
EXTERIOR_DEBUG_SAVE = os.getenv("EXTERIOR_DEBUG_SAVE", "1") in {"1", "true", "TRUE", "yes", "YES"}
EXTERIOR_POLYGON_EPS_RATIO = float(os.getenv("EXTERIOR_POLYGON_EPS_RATIO", "0.004"))

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
EXTERIOR_DEBUG_DIR = os.path.join(PROJECT_ROOT, "exterior-debug")

# Optional: gradient+k-means plane hints vs geometry split (see exterior_depth_planes.py)
EXTERIOR_PLANE_DEPTH_ENABLE = os.getenv("EXTERIOR_PLANE_DEPTH_ENABLE", "0") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_ARBITER_DEPTH_MARGIN = float(os.getenv("EXTERIOR_ARBITER_DEPTH_MARGIN", "0.08"))
# На сколько «стен» больше у геометрии, чем у depth, чтобы принудительно взять depth (при достаточном score_depth)
EXTERIOR_ARBITER_FRAGMENT_GAP = int(os.getenv("EXTERIOR_ARBITER_FRAGMENT_GAP", "2"))
# Если геометрия/шов дали мало стен, а k-means по глубине — заметно больше (типично 2 vs 4)
EXTERIOR_ARBITER_UNDERSEG_GAP = int(os.getenv("EXTERIOR_ARBITER_UNDERSEG_GAP", "2"))
EXTERIOR_ARBITER_UNDERSEG_MIN_SCORE = float(os.getenv("EXTERIOR_ARBITER_UNDERSEG_MIN_SCORE", "0.52"))
# Сравнение с depth при 1–2 гео-стенах даже без EXTERIOR_PLANE_DEPTH_ENABLE (дороже по CPU)
EXTERIOR_PLANE_COMPARE_FEW_WALLS = os.getenv("EXTERIOR_PLANE_COMPARE_FEW_WALLS", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}


def _bbox_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_inside(inner: list[float], outer: list[float], threshold: float = 0.6) -> bool:
    """Check if inner bbox is mostly inside outer bbox (by intersection/inner_area ratio)."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = _bbox_area(inner)
    if inner_area < 1:
        return False
    return (inter / inner_area) >= threshold


async def detect_building_bbox(image_bytes: bytes) -> list[float] | None:
    """Call GroundingDINO to find the union of all wall/facade bboxes."""
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(
            GDINO_URL,
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={
                "prompt": "wall . external wall . facade",
                "score_threshold": str(SCORE_THRESH_BUILDING),
            },
        )
    resp.raise_for_status()
    data = resp.json()
    bboxes = data.get("bboxes", [])
    if not bboxes:
        return None
    
    # Calculate the union of all detected building parts
    min_x = min(b["bbox"][0] for b in bboxes)
    min_y = min(b["bbox"][1] for b in bboxes)
    max_x = max(b["bbox"][2] for b in bboxes)
    max_y = max(b["bbox"][3] for b in bboxes)
    
    return [min_x, min_y, max_x, max_y]


async def detect_openings_bboxes(
    image_bytes: bytes,
    building_bbox: list[float],
) -> dict[str, list[list[float]]]:
    """Call GroundingDINO to find windows/doors inside building_bbox."""
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(
            GDINO_URL,
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={
                "prompt": "window . door",
                "score_threshold": str(SCORE_THRESH_OPENINGS),
            },
        )
    resp.raise_for_status()
    data = resp.json()

    windows: list[list[float]] = []
    doors: list[list[float]] = []

    for item in data.get("bboxes", []):
        bbox = item["bbox"]
        label = item.get("label", "").lower()
        if not _bbox_inside(bbox, building_bbox):
            continue
        if "door" in label:
            doors.append(bbox)
        else:
            windows.append(bbox)

    return {"windows": windows, "doors": doors}


async def sam_mask_from_bbox(image_bytes: bytes, bbox: list[float]) -> np.ndarray:
    """Call SAM to get a binary mask (H,W uint8 0/255) for a given bbox."""
    import json

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(
            SAM_URL,
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={"bbox": json.dumps(bbox)},
        )
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to decode SAM mask PNG")
    return mask


def postprocess_masks(
    wall_mask: np.ndarray,
    holes_mask: np.ndarray,
    dilate_px: int = 4,
) -> np.ndarray:
    """Subtract dilated holes from wall mask, clean small components and smooth edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    holes_dilated = cv2.dilate(holes_mask, kernel, iterations=1)
    result = cv2.bitwise_and(wall_mask, cv2.bitwise_not(holes_dilated))

    # Apply morphological CLOSE and OPEN to smooth jagged edges from SAM
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, smooth_kernel, iterations=2)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, smooth_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    total_area = result.shape[0] * result.shape[1]
    min_area = total_area * 0.005
    clean = np.zeros_like(result)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def encode_mask_png_base64(mask: np.ndarray) -> str:
    """Encode uint8 mask (0/255) to base64 PNG string."""
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Failed to encode mask PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


EXTERIOR_MIN_WALL_AREA_RATIO = 0.008
EXTERIOR_MAX_WALLS = 6
EXTERIOR_WATERSHED_DIST_RATIO = 0.35
EXTERIOR_CONTOUR_EPS_RATIO = 0.012
# Доп. масштабы approxPolyDP для устойчивых углов (доля периметра).
_eps_frac_raw = os.getenv("EXTERIOR_CONTOUR_EPS_FRACS", "0.006,0.009,0.012,0.018,0.024")
_eps_frac_list: list[float] = []
for _part in _eps_frac_raw.split(","):
    _p = _part.strip()
    if not _p:
        continue
    try:
        _eps_frac_list.append(float(_p))
    except ValueError:
        pass
EXTERIOR_CONTOUR_EPS_FRACS = tuple(_eps_frac_list) if _eps_frac_list else (0.006, 0.009, 0.012, 0.018, 0.024)
EXTERIOR_CORNER_ANGLE_DEG = 50
EXTERIOR_CONTOUR_BISECTOR_SPLIT = os.getenv("EXTERIOR_CONTOUR_BISECTOR_SPLIT", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_MIN_SPLIT_RATIO = 0.12
# При разрезе L-образного фасада: линия между двумя плоскостями в фото обычно близка к вертикали.
EXTERIOR_PREFER_VERTICAL_SPLIT = float(os.getenv("EXTERIOR_PREFER_VERTICAL_SPLIT", "0.78"))
EXTERIOR_SEAM_ENABLE = os.getenv("EXTERIOR_SEAM_ENABLE", "1") in {"1", "true", "TRUE", "yes", "YES"}
EXTERIOR_SEAM_ANGLE_THR_DEG = float(os.getenv("EXTERIOR_SEAM_ANGLE_THR_DEG", "12"))
EXTERIOR_SEAM_MIN_LINE_RATIO = float(os.getenv("EXTERIOR_SEAM_MIN_LINE_RATIO", "0.40"))
EXTERIOR_SEAM_CENTER_MARGIN_RATIO = float(os.getenv("EXTERIOR_SEAM_CENTER_MARGIN_RATIO", "0.10"))
EXTERIOR_SEAM_W_BALANCE = float(os.getenv("EXTERIOR_SEAM_W_BALANCE", "0.45"))
EXTERIOR_SEAM_W_EDGE = float(os.getenv("EXTERIOR_SEAM_W_EDGE", "0.40"))
EXTERIOR_SEAM_W_CENTER = float(os.getenv("EXTERIOR_SEAM_W_CENTER", "0.15"))
EXTERIOR_SEAM_MIN_SCORE = float(os.getenv("EXTERIOR_SEAM_MIN_SCORE", "0.00"))
EXTERIOR_SEAM_TOP_K = int(os.getenv("EXTERIOR_SEAM_TOP_K", "4"))
EXTERIOR_SEAM_NMS_X_GAP_RATIO = float(os.getenv("EXTERIOR_SEAM_NMS_X_GAP_RATIO", "0.08"))
EXTERIOR_SEAM_CONTOUR_ENABLE = os.getenv("EXTERIOR_SEAM_CONTOUR_ENABLE", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_SEAM_CONTOUR_TOP_Y_MAX = float(os.getenv("EXTERIOR_SEAM_CONTOUR_TOP_Y_MAX", "0.42"))
EXTERIOR_SEAM_CONTOUR_BOTTOM_Y_MIN = float(os.getenv("EXTERIOR_SEAM_CONTOUR_BOTTOM_Y_MIN", "0.58"))
EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG = float(os.getenv("EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG", "24.0"))
EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO = float(os.getenv("EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO", "0.08"))
EXTERIOR_SEAM_CONTOUR_MAX_K = int(os.getenv("EXTERIOR_SEAM_CONTOUR_MAX_K", "6"))
EXTERIOR_SEAM_CONTOUR_BONUS_BOTTOM = float(os.getenv("EXTERIOR_SEAM_CONTOUR_BONUS_BOTTOM", "0.14"))
EXTERIOR_SEAM_CONTOUR_BONUS_TOP = float(os.getenv("EXTERIOR_SEAM_CONTOUR_BONUS_TOP", "0.06"))
EXTERIOR_SEAM_CONTOUR_X_TOL_RATIO = float(os.getenv("EXTERIOR_SEAM_CONTOUR_X_TOL_RATIO", "0.03"))
EXTERIOR_SEAM_SECOND_PASS = os.getenv("EXTERIOR_SEAM_SECOND_PASS", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
# Если геометрия дала слишком много «стен», один раз посчитать k-means плоскости (как у арбитра) и сравнить
EXTERIOR_FRAGMENT_RESCUE = os.getenv("EXTERIOR_FRAGMENT_RESCUE", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_ARBITER_MIN_DEPTH_SCORE = float(os.getenv("EXTERIOR_ARBITER_MIN_DEPTH_SCORE", "0.42"))


def _line_verticality(pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """1.0 — почти вертикальная линия, 0.0 — почти горизонтальная."""
    dx = float(pt_b[0] - pt_a[0])
    dy = float(pt_b[1] - pt_a[1])
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return 0.0
    return abs(dy) / n


def _split_mask_by_line(
    mask: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split mask by line ax+by+c=0. Returns (mask_left, mask_right) or None if invalid."""
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    vals = a * xx + b * yy + c
    nonzero = (mask > 0) & (vals != 0)
    left = mask.copy()
    left[nonzero & (vals > 0)] = 0
    right = mask.copy()
    right[nonzero & (vals < 0)] = 0
    if cv2.countNonZero(left) < 100 or cv2.countNonZero(right) < 100:
        return None
    return (left, right)


def _find_corner_indices(contour: np.ndarray, angle_deg: float = EXTERIOR_CORNER_ANGLE_DEG) -> list[int]:
    """Find contour vertex indices where angle changes significantly (corner candidates)."""
    pts = contour.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 5:
        return []
    indices: list[int] = []
    for i in range(n):
        v1 = pts[(i - 1) % n] - pts[i]
        v2 = pts[(i + 1) % n] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        if angle < 180 - angle_deg:
            indices.append(i)
    return indices


def _vertex_salience_poly(pts: np.ndarray, i: int) -> float:
    """Острота угла и относительная длина смежных рёбер (для ранжирования кандидатов)."""
    n = len(pts)
    v1 = pts[i] - pts[(i - 1) % n]
    v2 = pts[(i + 1) % n] - pts[i]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1.0 or n2 < 1.0:
        return 0.0
    u1 = v1 / n1
    u2 = v2 / n2
    cos_a = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    sharp = max(0.0, (180.0 - angle) / 180.0)
    edge_strength = min(n1, n2) / (n1 + n2)
    return float(sharp * (0.5 + 0.5 * edge_strength))


def _collect_contour_corner_candidates(
    contour: np.ndarray,
    angle_deg: float = EXTERIOR_CORNER_ANGLE_DEG,
) -> list[tuple[float, float, float, np.ndarray, int]]:
    """
    Углы с нескольких масштабов approxPolyDP; слияние близких точек по сетке.
    Возвращает (px, py, salience, approx_pts, idx) по убыванию salience.
    """
    per = cv2.arcLength(contour, True)
    if per < 10:
        return []
    best: dict[tuple[int, int], tuple[float, float, float, np.ndarray, int]] = {}
    for frac in EXTERIOR_CONTOUR_EPS_FRACS:
        eps = float(frac) * per
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) < 4:
            continue
        idxs = _find_corner_indices(approx, angle_deg=angle_deg)
        pts = approx.reshape(-1, 2).astype(np.float64)
        for idx in idxs:
            sal = _vertex_salience_poly(pts, idx)
            if sal <= 0:
                continue
            px_f = float(pts[idx, 0])
            py_f = float(pts[idx, 1])
            key = (int(round(px_f / 12.0)), int(round(py_f / 12.0)))
            if key not in best or sal > best[key][2]:
                best[key] = (px_f, py_f, sal, pts.copy(), idx)
    return sorted(best.values(), key=lambda t: -t[2])


def _split_line_candidates_for_corner(
    px: float,
    py: float,
    approx_pts: np.ndarray,
    approx_idx: int,
    image_width: int,
    image_height: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Вертикаль x=px и (опционально) линия ⟂ биссектрисе рёбер многоугольника в вершине."""
    H = image_height
    W = image_width
    va = np.array([px, 0.0], dtype=np.float64)
    vb = np.array([px, float(H - 1)], dtype=np.float64)
    out: list[tuple[np.ndarray, np.ndarray]] = [(va, vb)]
    if not EXTERIOR_CONTOUR_BISECTOR_SPLIT:
        return out
    n = len(approx_pts)
    if n < 3 or approx_idx < 0 or approx_idx >= n:
        return out
    p = approx_pts[approx_idx].astype(np.float64)
    prev_p = approx_pts[(approx_idx - 1) % n].astype(np.float64)
    next_p = approx_pts[(approx_idx + 1) % n].astype(np.float64)
    d1 = p - prev_p
    d2 = next_p - p
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < 2.0 or n2 < 2.0:
        return out
    d1 /= n1
    d2 /= n2
    bis = d1 + d2
    nb = float(np.linalg.norm(bis))
    if nb < 1e-3:
        perp = np.array([-d1[1], d1[0]], dtype=np.float64)
    else:
        bis /= nb
        perp = np.array([-bis[1], bis[0]], dtype=np.float64)
    pn = float(np.linalg.norm(perp))
    if pn < 1e-6:
        return out
    perp /= pn
    L = float(max(W, H) * 3)
    out.append((p - perp * L, p + perp * L))
    return out


def _contour_split_component(
    component_mask: np.ndarray,
    min_area: int,
    max_walls: int,
) -> list[np.ndarray]:
    """Try to split a single component by contour corners. Returns list of sub-masks."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [component_mask]
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area * 2:
        return [component_mask]

    H, W = component_mask.shape[:2]
    candidates = _collect_contour_corner_candidates(contour)
    if not candidates:
        eps = EXTERIOR_CONTOUR_EPS_RATIO * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) < 4:
            return [component_mask]
        idxs = _find_corner_indices(approx)
        pts_fb = approx.reshape(-1, 2).astype(np.float64)
        candidates = [
            (float(pts_fb[i, 0]), float(pts_fb[i, 1]), _vertex_salience_poly(pts_fb, i), pts_fb.copy(), i)
            for i in idxs
        ]

    if not candidates:
        return [component_mask]

    w_vert = EXTERIOR_PREFER_VERTICAL_SPLIT
    w_bal = 1.0 - w_vert
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score = -1.0

    for px, py, _sal, approx_pts, aidx in candidates:
        for pt_a, pt_b in _split_line_candidates_for_corner(px, py, approx_pts, aidx, W, H):
            line_res = _split_mask_by_line(
                component_mask,
                float(pt_a[0]),
                float(pt_a[1]),
                float(pt_b[0]),
                float(pt_b[1]),
            )
            if line_res is None:
                continue
            left, right = line_res
            a1, a2 = cv2.countNonZero(left), cv2.countNonZero(right)
            total = a1 + a2
            if total < 100:
                continue
            r1, r2 = a1 / total, a2 / total
            if r1 < EXTERIOR_MIN_SPLIT_RATIO or r2 < EXTERIOR_MIN_SPLIT_RATIO:
                continue
            balance = min(r1, r2) / max(r1, r2)
            vert = _line_verticality(pt_a, pt_b)
            score = w_bal * balance + w_vert * vert
            if score > best_score:
                best_score = score
                best_split = (left, right)

    if best_split is None:
        return [component_mask]

    left, right = best_split
    result: list[np.ndarray] = []

    def add_rec(m: np.ndarray) -> None:
        if len(result) >= max_walls:
            return
        area = cv2.countNonZero(m)
        if area < min_area:
            return
        sub = _contour_split_component(m, min_area, max_walls - len(result))
        for s in sub:
            if cv2.countNonZero(s) >= min_area and len(result) < max_walls:
                result.append(s)

    add_rec(left)
    add_rec(right)
    return result if result else [component_mask]


def _inner_corner_split_component(
    component_mask: np.ndarray,
    min_area: int,
    max_walls: int,
) -> list[np.ndarray]:
    """
    Split L-shaped single component into 2+ parts by its inner corner.
    Перебирает несколько глубоких convexity defects и линии: вертикаль через pivot,
    биссектриса и перпендикуляр. Скоринг: баланс площадей + предпочтение вертикали
    (горизонтальный «верх/низ» часто выигрывает только по balance).
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [component_mask]
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return [component_mask]

    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return [component_mask]

    defects = cv2.convexityDefects(contour, hull)
    if defects is None or len(defects) == 0:
        return [component_mask]

    contour_pts = contour.reshape(-1, 2)
    H = component_mask.shape[0]
    defect_depths: list[tuple[float, int]] = []
    for i in range(defects.shape[0]):
        s_idx, e_idx, f_idx, depth = defects[i, 0]
        d = float(depth) / 256.0
        defect_depths.append((d, i))
    defect_depths.sort(key=lambda t: t[0], reverse=True)
    tried_idx = [idx for _, idx in defect_depths[: min(4, len(defect_depths))]]

    w_vert = EXTERIOR_PREFER_VERTICAL_SPLIT
    w_bal = 1.0 - w_vert

    def candidates_for_defect(def_i: int) -> list[tuple[np.ndarray, np.ndarray]]:
        s_idx, e_idx, f_idx, _ = defects[def_i, 0]
        s_idx, e_idx, f_idx = int(s_idx), int(e_idx), int(f_idx)
        p = contour_pts[f_idx].astype(float)
        s = contour_pts[s_idx].astype(float)
        e = contour_pts[e_idx].astype(float)
        px = float(p[0])
        cands: list[tuple[np.ndarray, np.ndarray]] = [
            (np.array([px, 0.0], dtype=np.float64), np.array([px, float(H - 1)], dtype=np.float64)),
        ]
        v1 = s - p
        v2 = e - p
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            return cands
        u1 = v1 / n1
        u2 = v2 / n2
        bis = u1 + u2
        if np.linalg.norm(bis) < 1e-6:
            bis = np.array([-u1[1], u1[0]], dtype=np.float64)
        else:
            bis = bis / np.linalg.norm(bis)
        perp = np.array([-bis[1], bis[0]], dtype=np.float64)
        L = 1000.0
        cands.append((p - bis * L, p + bis * L))
        cands.append((p - perp * L, p + perp * L))
        return cands

    best_score = -1.0
    best_parts: list[np.ndarray] | None = None

    for def_i in tried_idx:
        for ptA, ptB in candidates_for_defect(def_i):
            split_res = _split_mask_by_line(
                component_mask,
                float(ptA[0]),
                float(ptA[1]),
                float(ptB[0]),
                float(ptB[1]),
            )
            if split_res is None:
                continue
            left, right = split_res

            area_l = cv2.countNonZero(left)
            area_r = cv2.countNonZero(right)
            if area_l < min_area or area_r < min_area:
                continue

            balance = min(area_l, area_r) / max(area_l, area_r)
            vert = _line_verticality(ptA, ptB)
            score = w_bal * balance + w_vert * vert
            if score > best_score:
                best_score = score
                best_parts = [left, right]

    if best_parts is None:
        return [component_mask]

    out: list[np.ndarray] = []
    for part in best_parts:
        if len(out) >= max_walls:
            break
        out.append(part)
        if len(out) < max_walls:
            sub = _contour_split_component(part, min_area, max_walls - len(out))
            if len(sub) > 1:
                out.pop()
                out.extend(sub[: max_walls - len(out)])

    return out if out else [component_mask]


def _order_rect_corners_tl_bl_br_tr(box_pts: np.ndarray) -> list[list[int]]:
    """Sort 4 points of minAreaRect into [TL, BL, BR, TR] order."""
    pts = box_pts.reshape(4, 2).astype(float)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    left_pts = pts[pts[:, 0] <= cx]
    right_pts = pts[pts[:, 0] > cx]

    if len(left_pts) < 2:
        left_pts = pts[np.argsort(pts[:, 0])[:2]]
        right_pts = pts[np.argsort(pts[:, 0])[2:]]
    if len(right_pts) < 2:
        right_pts = pts[np.argsort(pts[:, 0])[2:]]
        left_pts = pts[np.argsort(pts[:, 0])[:2]]

    tl = left_pts[np.argmin(left_pts[:, 1])]
    bl = left_pts[np.argmax(left_pts[:, 1])]
    tr = right_pts[np.argmin(right_pts[:, 1])]
    br = right_pts[np.argmax(right_pts[:, 1])]

    return [
        [int(round(tl[0])), int(round(tl[1]))],
        [int(round(bl[0])), int(round(bl[1]))],
        [int(round(br[0])), int(round(br[1]))],
        [int(round(tr[0])), int(round(tr[1]))],
    ]


def _corners_from_polygon_perspective(polygon: list[list[int]]) -> list[list[int]] | None:
    """
    Build perspective-like quad [TL, BL, BR, TR] from polygon points.
    Uses top/bottom horizontal bands and left/right extremes in each band.
    """
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None

    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    y_span = y_max - y_min
    if y_span < 4.0:
        return None

    band = max(4.0, y_span * 0.30)
    top_pts = pts[pts[:, 1] <= (y_min + band)]
    bot_pts = pts[pts[:, 1] >= (y_max - band)]
    if top_pts.shape[0] < 2 or bot_pts.shape[0] < 2:
        return None

    tl = top_pts[np.argmin(top_pts[:, 0])]
    tr = top_pts[np.argmax(top_pts[:, 0])]
    bl = bot_pts[np.argmin(bot_pts[:, 0])]
    br = bot_pts[np.argmax(bot_pts[:, 0])]

    # Validate that left is really left of right and height is non-trivial.
    if not (tl[0] < tr[0] and bl[0] < br[0]):
        return None
    if abs(float(bl[1] - tl[1])) < 4.0 and abs(float(br[1] - tr[1])) < 4.0:
        return None

    quad = np.array([tl, bl, br, tr], dtype=np.float32)
    # Polygon area check for degeneracy
    x = quad[:, 0]
    y = quad[:, 1]
    area2 = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area2 < 50.0:
        return None

    return [
        [int(round(tl[0])), int(round(tl[1]))],
        [int(round(bl[0])), int(round(bl[1]))],
        [int(round(br[0])), int(round(br[1]))],
        [int(round(tr[0])), int(round(tr[1]))],
    ]


def _extract_wall_from_component(component_mask: np.ndarray) -> dict | None:
    """Extract corners + center from a single binary component mask."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    # Базовый fallback-quad (если перспективный из polygon невалиден).
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect)
    corners_fallback = _order_rect_corners_tl_bl_br_tr(box_pts)

    # Контур компоненты для точного оверлея/клипа (маска области стены).
    # Возвращаем упрощённую ломаную через approxPolyDP.
    perimeter = cv2.arcLength(contour, True)
    # Для polygon нужно меньшее упрощение, иначе получится почти quad.
    eps = EXTERIOR_POLYGON_EPS_RATIO * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    polygon = approx.reshape(-1, 2).astype(int).tolist()
    if len(polygon) < 3:
        # Fallback: хотя бы quad из minAreaRect.
        polygon = corners_fallback

    corners = _corners_from_polygon_perspective(polygon) or corners_fallback

    regions: list[dict] | None = None
    if len(polygon) >= 5:
        split_res = split_gable_roof_polygon(polygon)
        if split_res is not None:
            poly_lo, poly_up = split_res
            corners_lo = _corners_from_polygon_perspective(poly_lo)
            if corners_lo is None:
                lp = np.array(poly_lo, dtype=np.float32).reshape(-1, 1, 2)
                rect = cv2.minAreaRect(lp)
                box = cv2.boxPoints(rect)
                corners_lo = _order_rect_corners_tl_bl_br_tr(box)
            tri = np.array(poly_up, dtype=np.float32).reshape(-1, 1, 2)
            rect_u = cv2.minAreaRect(tri)
            box_u = cv2.boxPoints(rect_u)
            corners_up = _order_rect_corners_tl_bl_br_tr(box_u)
            regions = [
                {"corners": corners_lo, "polygon": poly_lo},
                {"corners": corners_up, "polygon": poly_up},
            ]

    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = float(np.mean([c[0] for c in corners]))
        cy = float(np.mean([c[1] for c in corners]))

    out: dict = {
        "corners": corners,
        "polygon": polygon,
        "center": [round(cx, 2), round(cy, 2)],
    }
    if regions is not None:
        out["regions"] = regions
    return out


def _watershed_split(mask: np.ndarray, max_segments: int = EXTERIOR_MAX_WALLS) -> list[np.ndarray]:
    """Split a single connected mask into multiple sub-masks via watershed."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_max = dist.max()
    if dist_max < 1:
        return [mask]

    threshold = EXTERIOR_WATERSHED_DIST_RATIO * dist_max
    _, sure_fg = cv2.threshold(dist, threshold, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 2:
        return [mask]

    markers = markers + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0

    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_mask, markers)

    sub_masks: list[np.ndarray] = []
    unique_labels = set(markers.flat)
    for label in sorted(unique_labels):
        if label <= 1:
            continue
        sub = np.zeros_like(mask)
        sub[markers == label] = 255
        area = cv2.countNonZero(sub)
        if area > 200:
            sub_masks.append(sub)
        if len(sub_masks) >= max_segments:
            break

    return sub_masks if sub_masks else [mask]


def split_walls_from_mask(
    wall_minus_holes: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Split wall_minus_holes mask into separate wall dicts:
    [{ id, corners: [[x,y]*4], center: [x,y] }, ...]
    """
    mask_bin = (wall_minus_holes > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    total_area = image_width * image_height
    min_area = total_area * EXTERIOR_MIN_WALL_AREA_RATIO

    component_masks: list[np.ndarray] = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            comp = np.zeros_like(mask_clean)
            comp[labels == i] = 255
            component_masks.append(comp)

    if len(component_masks) == 1:
        inner = _inner_corner_split_component(component_masks[0], int(min_area), EXTERIOR_MAX_WALLS)
        if len(inner) > 1:
            component_masks = inner
        else:
            sub = _contour_split_component(component_masks[0], int(min_area), EXTERIOR_MAX_WALLS)
            if len(sub) > 1:
                component_masks = sub
            else:
                sub_w = _watershed_split(component_masks[0])
                if len(sub_w) > 1:
                    component_masks = sub_w

    if not component_masks:
        return []

    raw_walls: list[dict] = []
    for comp_mask in component_masks:
        wall_data = _extract_wall_from_component(comp_mask)
        if wall_data:
            raw_walls.append(wall_data)

    raw_walls.sort(key=lambda w: w["center"][0])

    walls: list[dict] = []
    for idx, w in enumerate(raw_walls):
        item: dict = {
            "id": idx + 1,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        }
        if w.get("regions"):
            item["regions"] = w["regions"]
        walls.append(item)

    return walls


def _wall_polygon_pts_for_fill(wall: dict) -> np.ndarray | None:
    """Точки контура стены для fillPoly, shape (N,1,2) int32."""
    pts_out: list[list[int]] = []

    def _append_from(seq: object) -> None:
        if not isinstance(seq, list):
            return
        for p in seq:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts_out.append([int(round(float(p[0]))), int(round(float(p[1])))])

    poly = wall.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        pts_out.clear()
        _append_from(poly)
    if len(pts_out) < 3:
        pts_out.clear()
        _append_from(wall.get("corners"))
    if len(pts_out) < 3:
        return None
    arr = np.array(pts_out, dtype=np.int32)
    return arr.reshape(-1, 1, 2)


def _target_wall_mask(wall: dict, wall_minus_holes: np.ndarray) -> np.ndarray | None:
    pts = _wall_polygon_pts_for_fill(wall)
    if pts is None:
        return None
    h, w = wall_minus_holes.shape[:2]
    fill = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fill, [pts], 255)
    return cv2.bitwise_and(fill, wall_minus_holes)


def _split_exterior_one_wall_by_line(
    wall_minus_holes: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    existing_walls: list[dict],
    target_wall_id: int,
) -> list[dict]:
    tid = int(target_wall_id)
    target_entry: dict | None = None
    for w in existing_walls:
        if int(w.get("id", -1)) == tid:
            target_entry = w
            break
    if target_entry is None:
        raise ValueError(f"wall id {tid} not found in walls_json")

    target_mask = _target_wall_mask(target_entry, wall_minus_holes)
    if target_mask is None or cv2.countNonZero(target_mask) < 200:
        raise ValueError("target wall polygon invalid or empty after intersect with facade mask")

    res = _split_mask_by_line(target_mask, x1, y1, x2, y2)
    if res is None:
        raise ValueError("line does not split selected wall (both parts need minimum area)")

    left, right = res
    min_area = int(image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO)
    component_masks: list[np.ndarray] = []
    for part in (left, right):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        for i in range(1, num_labels):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                comp = np.zeros_like(wall_minus_holes)
                comp[labels == i] = 255
                component_masks.append(comp)

    if len(component_masks) < 2:
        raise ValueError("split produced fewer than 2 parts inside selected wall")

    if len(component_masks) > 2:
        component_masks.sort(key=lambda m: int(cv2.countNonZero(m)), reverse=True)
        component_masks = component_masks[:2]

    new_extracted: list[dict] = []
    for comp_mask in component_masks:
        wd = _extract_wall_from_component(comp_mask)
        if not wd:
            raise ValueError("failed to extract geometry from split part")
        wd.pop("regions", None)
        new_extracted.append(wd)

    others: list[dict] = []
    used_ids: set[int] = set()
    for w in existing_walls:
        wid = int(w.get("id", -1))
        if wid == tid:
            continue
        if wid > 0:
            used_ids.add(wid)
        others.append(dict(w))

    next_id = (max(used_ids) + 1) if used_ids else 1
    split_walls: list[dict] = []
    for wd in new_extracted:
        item: dict = {
            "id": next_id,
            "corners": wd["corners"],
            "polygon": wd.get("polygon"),
            "center": wd["center"],
        }
        next_id += 1
        split_walls.append(item)

    merged: list[dict] = others + split_walls
    merged.sort(key=lambda w: float(w["center"][0]))

    walls: list[dict] = []
    for w in merged:
        wid = int(w.get("id", 0))
        if wid <= 0:
            wid = next_id
            next_id += 1
        item: dict = {
            "id": wid,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        }
        if w.get("regions"):
            item["regions"] = w["regions"]
        walls.append(item)
    return walls


def split_exterior_by_line(
    mask_base64: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    existing_walls: list[dict] | None = None,
    target_wall_id: int | None = None,
) -> list[dict]:
    """
    Split wall_minus_holes mask by user-drawn line. Returns new walls[].
    If target_wall_id and existing_walls are set, only that wall's region is split; others unchanged.
    """
    arr = np.frombuffer(base64.b64decode(mask_base64), dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (image_height, image_width):
        return []

    if target_wall_id is not None:
        if not existing_walls:
            raise ValueError("existing_walls is required and must be non-empty when target_wall_id is set")
        return _split_exterior_one_wall_by_line(
            mask,
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
            existing_walls,
            target_wall_id,
        )

    res = _split_mask_by_line(mask, x1, y1, x2, y2)
    if res is None:
        return split_walls_from_mask(mask, image_width, image_height)

    left, right = res
    total_area = image_width * image_height
    min_area = int(total_area * EXTERIOR_MIN_WALL_AREA_RATIO)

    component_masks: list[np.ndarray] = []
    for part in (left, right):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                comp = np.zeros_like(mask)
                comp[labels == i] = 255
                component_masks.append(comp)

    if not component_masks:
        return split_walls_from_mask(mask, image_width, image_height)

    raw_walls: list[dict] = []
    for comp_mask in component_masks:
        wall_data = _extract_wall_from_component(comp_mask)
        if wall_data:
            raw_walls.append(wall_data)

    raw_walls.sort(key=lambda w: w["center"][0])
    walls: list[dict] = []
    for idx, w in enumerate(raw_walls):
        item: dict = {
            "id": idx + 1,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        }
        if w.get("regions"):
            item["regions"] = w["regions"]
        walls.append(item)
    return walls


def _decode_image_bgr(image_bytes: bytes, image_width: int, image_height: int) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != (image_height, image_width):
        img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    return img


def _count_large_components(mask: np.ndarray, min_area: int) -> int:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    count = 0
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            count += 1
    return count


def _build_walls_from_component_masks(component_masks: list[np.ndarray]) -> list[dict]:
    raw_walls: list[dict] = []
    for comp_mask in component_masks:
        wall_data = _extract_wall_from_component(comp_mask)
        if wall_data:
            raw_walls.append(wall_data)

    raw_walls.sort(key=lambda w: w["center"][0])
    walls: list[dict] = []
    for idx, w in enumerate(raw_walls):
        item: dict = {
            "id": idx + 1,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        }
        if w.get("regions"):
            item["regions"] = w["regions"]
        walls.append(item)
    return walls


def _geometry_split_score(walls: list[dict], image_width: int) -> float:
    """Heuristic confidence [0,1] for geometry-based wall split (штраф за лишнее дробление)."""
    if not walls:
        return 0.2
    n = len(walls)
    if n == 1:
        return 0.38
    xs = sorted(float(w["center"][0]) for w in walls)
    spread = (xs[-1] - xs[0]) / max(1.0, float(image_width))
    # Бонус за «несколько» стен ограничиваем; каждая лишняя после 2-х сильно снижает уверенность
    step_bonus = 0.12 * min(n - 1, 2)
    base = 0.42 + step_bonus + 0.35 * min(1.0, spread * 2.5)
    over = max(0, n - 2)
    penalty = 1.0 / (1.0 + 0.48 * over)
    return float(np.clip(base * penalty, 0.12, 0.92))


def _choose_final_walls_split(
    wall_minus_holes: np.ndarray,
    image_bgr: np.ndarray | None,
    walls_geo: list[dict],
    split_method_geo: str,
    seam_debug: dict,
    image_width: int,
    image_height: int,
) -> tuple[list[dict], str, dict, dict]:
    """
    Optionally replace geometry walls with depth-proxy plane clusters when scores favor it.
    Returns (walls, split_method, seam_debug, split_arbiter).
    """
    score_geo = _geometry_split_score(walls_geo, image_width)
    n_geo = len(walls_geo)
    run_plane = image_bgr is not None and (
        bool(EXTERIOR_PLANE_DEPTH_ENABLE)
        or (bool(EXTERIOR_FRAGMENT_RESCUE) and n_geo > 3)
        or (bool(EXTERIOR_PLANE_COMPARE_FEW_WALLS) and n_geo <= 2)
    )
    arb: dict = {
        "score_geo": round(score_geo, 4),
        "method_geo": split_method_geo,
        "depth_enabled": bool(EXTERIOR_PLANE_DEPTH_ENABLE),
        "few_walls_plane_compare": bool(EXTERIOR_PLANE_COMPARE_FEW_WALLS and n_geo <= 2),
        "fragment_rescue": bool(EXTERIOR_FRAGMENT_RESCUE and n_geo > 3 and not EXTERIOR_PLANE_DEPTH_ENABLE),
        "plane_compare_ran": False,
    }

    if not run_plane:
        arb["chosen"] = "geometry"
        return walls_geo, split_method_geo, seam_debug, arb

    from exterior_depth_planes import plane_cluster_masks

    min_area = int(image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO)
    masks_d, score_depth, dd = plane_cluster_masks(
        image_bgr, wall_minus_holes, min_area, EXTERIOR_MAX_WALLS
    )
    arb["plane_compare_ran"] = True
    arb["score_depth"] = round(float(score_depth), 4)
    arb["depth_debug"] = dd

    if len(masks_d) < 2:
        arb["chosen"] = "geometry"
        return walls_geo, split_method_geo, seam_debug, arb

    walls_depth = _build_walls_from_component_masks(masks_d)
    arb["walls_depth_count"] = len(walls_depth)
    n_depth = len(walls_depth)

    if n_depth < 2:
        arb["chosen"] = "geometry"
        return walls_geo, split_method_geo, seam_debug, arb

    gap = n_geo - n_depth
    # Явно: геометрия раздробила сильнее, чем k-means — берём более гладкое разбиение
    if (
        gap >= EXTERIOR_ARBITER_FRAGMENT_GAP
        and score_depth >= EXTERIOR_ARBITER_MIN_DEPTH_SCORE
    ):
        arb["chosen"] = "depth_planes"
        arb["pick_reason"] = "fragmentation_gap"
        return walls_depth, "depth_planes", seam_debug, arb

    gap_depth = n_depth - n_geo
    if (
        gap_depth >= EXTERIOR_ARBITER_UNDERSEG_GAP
        and score_depth >= EXTERIOR_ARBITER_UNDERSEG_MIN_SCORE
    ):
        arb["chosen"] = "depth_planes"
        arb["pick_reason"] = "undersegmentation_gap"
        arb["undersegmentation"] = {
            "n_geo": n_geo,
            "n_depth": n_depth,
            "gap": gap_depth,
        }
        return walls_depth, "depth_planes", seam_debug, arb

    if score_depth >= score_geo + EXTERIOR_ARBITER_DEPTH_MARGIN:
        arb["chosen"] = "depth_planes"
        arb["pick_reason"] = "score_margin"
        return walls_depth, "depth_planes", seam_debug, arb

    arb["chosen"] = "geometry"
    return walls_geo, split_method_geo, seam_debug, arb


def _line_edge_strength(gray: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray, width: int = 7) -> float:
    """Оценка силы вертикального градиента (шва) вдоль отрезка."""
    H, W = gray.shape[:2]
    g_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    stripe_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(
        stripe_mask,
        (int(round(pt_a[0])), int(round(pt_a[1]))),
        (int(round(pt_b[0])), int(round(pt_b[1]))),
        255,
        thickness=max(3, int(width)),
    )
    vals = np.abs(g_x[stripe_mask > 0])
    if vals.size == 0:
        return 0.0
    # Sobel может давать значения >255, нормализуем в [0..1].
    return float(np.clip(vals.mean() / 255.0, 0.0, 1.0))


def _component_masks_from_split_line(
    region_mask: np.ndarray,
    pt_a: np.ndarray,
    pt_b: np.ndarray,
    min_area: int,
) -> list[np.ndarray] | None:
    split_global = _split_mask_by_line(
        region_mask,
        float(pt_a[0]),
        float(pt_a[1]),
        float(pt_b[0]),
        float(pt_b[1]),
    )
    if split_global is None:
        return None
    left_g, right_g = split_global
    component_masks: list[np.ndarray] = []
    for part in (left_g, right_g):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        for i in range(1, num_labels):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                comp = np.zeros_like(region_mask)
                comp[labels == i] = 255
                component_masks.append(comp)
    return component_masks if len(component_masks) >= 2 else None


def _contour_vertical_seam_candidates(mask_crop: np.ndarray, bw: int, bh: int) -> list[dict]:
    """
    Кандидаты вертикального шва из изломов верхнего/нижнего контура.
    Нижний контур приоритетнее (крыша меньше мешает).
    """
    if not EXTERIOR_SEAM_CONTOUR_ENABLE:
        return []
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(contour, True)
    if per < 20.0:
        return []

    min_edge = max(6.0, EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO * float(min(bw, bh)))
    raw_bottom: list[dict] = []
    raw_top: list[dict] = []

    eps_fracs: list[float] = []
    for frac in EXTERIOR_CONTOUR_EPS_FRACS:
        f = float(frac)
        if 0.005 <= f <= 0.04:
            eps_fracs.append(f)
    if not eps_fracs:
        eps_fracs = [0.01, 0.016, 0.024]

    for frac in eps_fracs:
        approx = cv2.approxPolyDP(contour, float(frac) * per, True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < 5:
            continue
        for i in range(n):
            p_prev = pts[(i - 1) % n]
            p = pts[i]
            p_next = pts[(i + 1) % n]
            v1 = p_prev - p
            v2 = p_next - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < min_edge or n2 < min_edge:
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_a)))
            delta = 180.0 - angle
            if delta < EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG:
                continue

            px = float(p[0])
            py = float(p[1])
            if px <= 2.0 or px >= float(bw - 3):
                continue
            y_norm = py / max(1.0, float(bh - 1))
            zone = None
            bonus = 0.0
            if y_norm >= EXTERIOR_SEAM_CONTOUR_BOTTOM_Y_MIN:
                zone = "bottom"
                bonus = EXTERIOR_SEAM_CONTOUR_BONUS_BOTTOM
            elif y_norm <= EXTERIOR_SEAM_CONTOUR_TOP_Y_MAX:
                zone = "top"
                bonus = EXTERIOR_SEAM_CONTOUR_BONUS_TOP
            if zone is None:
                continue

            delta_norm = float(np.clip(delta / 90.0, 0.0, 1.0))
            score_hint = bonus * (0.65 + 0.35 * delta_norm)
            rec = {
                "mid_x_local": px,
                "pt_a_local": np.array([px, 0.0], dtype=np.float64),
                "pt_b_local": np.array([px, float(bh - 1)], dtype=np.float64),
                "source_bonus": float(score_hint),
                "source": f"contour_{zone}",
            }
            if zone == "bottom":
                raw_bottom.append(rec)
            else:
                raw_top.append(rec)

    # Оставляем уникальные bottom-кандидаты по X (основа для seam)
    out: list[dict] = []
    seen_bottom: set[int] = set()
    for rec in sorted(raw_bottom, key=lambda c: float(c["source_bonus"]), reverse=True):
        x = float(rec["mid_x_local"])
        x_bin = int(round(x / max(1.0, float(bw) * 0.03)))
        if x_bin in seen_bottom:
            continue
        seen_bottom.add(x_bin)
        out.append(rec)

    # Верхний излом учитываем только при подтверждении по нижнему X (иначе это, скорее всего, крыша).
    if out:
        x_tol = max(6.0, EXTERIOR_SEAM_CONTOUR_X_TOL_RATIO * float(bw))
        seen_top: set[int] = set()
        for rec in sorted(raw_top, key=lambda c: float(c["source_bonus"]), reverse=True):
            x = float(rec["mid_x_local"])
            if not any(abs(x - float(b["mid_x_local"])) <= x_tol for b in out):
                continue
            x_bin = int(round(x / max(1.0, float(bw) * 0.03)))
            if x_bin in seen_top:
                continue
            seen_top.add(x_bin)
            out.append(rec)

    out.sort(key=lambda c: float(c["source_bonus"]), reverse=True)
    return out[: max(1, EXTERIOR_SEAM_CONTOUR_MAX_K)]


def _debug_contour_corner_points(mask: np.ndarray) -> dict:
    """
    Диагностика углов контура фасада: кандидаты изломов в нижней/верхней зоне.
    Нужна для анализа, почему seam не выбирает ожидаемый вертикальный разрез.
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"bottom": [], "top": []}
    contour = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(contour, True)
    if per < 20.0:
        return {"bottom": [], "top": []}

    min_edge = max(6.0, EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO * float(min(w, h)))
    out_bottom: list[dict] = []
    out_top: list[dict] = []
    seen: set[tuple[int, str]] = set()

    eps_fracs: list[float] = []
    for frac in EXTERIOR_CONTOUR_EPS_FRACS:
        f = float(frac)
        if 0.005 <= f <= 0.04:
            eps_fracs.append(f)
    if not eps_fracs:
        eps_fracs = [0.01, 0.016, 0.024]

    for frac in eps_fracs:
        approx = cv2.approxPolyDP(contour, float(frac) * per, True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < 5:
            continue
        for i in range(n):
            p_prev = pts[(i - 1) % n]
            p = pts[i]
            p_next = pts[(i + 1) % n]
            v1 = p_prev - p
            v2 = p_next - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < min_edge or n2 < min_edge:
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_a)))
            delta = 180.0 - angle
            if delta < EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG:
                continue

            px = float(p[0])
            py = float(p[1])
            y_norm = py / max(1.0, float(h - 1))
            zone = ""
            if y_norm >= EXTERIOR_SEAM_CONTOUR_BOTTOM_Y_MIN:
                zone = "bottom"
            elif y_norm <= EXTERIOR_SEAM_CONTOUR_TOP_Y_MAX:
                zone = "top"
            if not zone:
                continue

            x_bin = int(round(px / max(1.0, float(w) * 0.03)))
            uniq = (x_bin, zone)
            if uniq in seen:
                continue
            seen.add(uniq)
            rec = {
                "x": round(px, 2),
                "y": round(py, 2),
                "delta_deg": round(delta, 2),
                "edge_min": round(min(n1, n2), 2),
                "eps_frac": round(float(frac), 4),
            }
            if zone == "bottom":
                out_bottom.append(rec)
            else:
                out_top.append(rec)

    out_bottom.sort(key=lambda d: (float(d["x"])))
    out_top.sort(key=lambda d: (float(d["x"])))
    return {
        "bottom": out_bottom[:20],
        "top": out_top[:20],
    }


def _split_by_bottom_contour_kink_once(
    wall_minus_holes: np.ndarray,
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[dict] | None:
    """
    Консервативный split по одному лучшему излому нижнего контура.
    Нужен как стабильный fallback, чтобы не «накликивать» много стен при одном явном изломе.
    """
    dbg = _debug_contour_corner_points(wall_minus_holes)
    bottom = dbg.get("bottom")
    if not isinstance(bottom, list) or not bottom:
        return None

    best_x: float | None = None
    best_s = -1.0
    for rec in bottom:
        if not isinstance(rec, dict):
            continue
        x = float(rec.get("x", -1))
        d = float(rec.get("delta_deg", 0.0))
        x_norm = x / max(1.0, float(image_width - 1))
        if x_norm < EXTERIOR_SEAM_CENTER_MARGIN_RATIO or x_norm > (1.0 - EXTERIOR_SEAM_CENTER_MARGIN_RATIO):
            continue
        # Предпочитаем резкий излом и точку не у самого края.
        center_q = 1.0 - abs(x_norm - 0.5) / 0.5
        s = 0.7 * (d / 180.0) + 0.3 * max(0.0, center_q)
        if s > best_s:
            best_s = s
            best_x = x

    if best_x is None:
        return None

    pt_a = np.array([best_x, 0.0], dtype=np.float64)
    pt_b = np.array([best_x, float(image_height - 1)], dtype=np.float64)
    component_masks = _component_masks_from_split_line(wall_minus_holes, pt_a, pt_b, min_area)
    if component_masks is None:
        return None
    return _build_walls_from_component_masks(component_masks)


def _seam_split_single_region(
    region_mask: np.ndarray,
    image_bgr: np.ndarray,
    image_width: int,
    image_height: int,
    min_area: int,
) -> tuple[list[np.ndarray] | None, dict]:
    """
    Один вертикальный шов внутри region_mask (полный кадр HxW). Возвращает список масок компонент или None.
    """
    dbg: dict = {
        "candidates_count": 0,
        "best_score": 0.0,
        "best_line": None,
        "top_lines": [],
        "contour_candidates_count": 0,
        "used": False,
    }
    x, y, w, h = cv2.boundingRect(region_mask)
    if w < 18 or h < 18:
        return None, dbg
    pad = max(2, int(0.02 * max(w, h)))
    x1i = max(0, x - pad)
    y1i = max(0, y - pad)
    x2i = min(image_width - 1, x + w + pad)
    y2i = min(image_height - 1, y + h + pad)
    bw = x2i - x1i + 1
    bh = y2i - y1i + 1
    if bw < 20 or bh < 20:
        return None, dbg

    crop = image_bgr[y1i : y2i + 1, x1i : x2i + 1]
    mask_crop = region_mask[y1i : y2i + 1, x1i : x2i + 1]
    if crop.size == 0 or mask_crop.size == 0 or cv2.countNonZero(mask_crop) < 200:
        return None, dbg

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=(mask_crop > 0).astype(np.uint8) * 255)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(40, int(0.15 * bh)),
        minLineLength=max(30, int(EXTERIOR_SEAM_MIN_LINE_RATIO * bh)),
        maxLineGap=max(10, int(0.06 * bh)),
    )

    center_margin = EXTERIOR_SEAM_CENTER_MARGIN_RATIO
    best_score = -1.0
    best_line_local: tuple[np.ndarray, np.ndarray] | None = None
    candidate_lines: list[dict] = []
    candidates_count = 0

    seam_candidates: list[dict] = []
    if lines is not None:
        for seg in lines.reshape(-1, 4):
            lx1, ly1, lx2, ly2 = [float(v) for v in seg.tolist()]
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < EXTERIOR_SEAM_MIN_LINE_RATIO * bh:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if abs(angle - 90.0) > EXTERIOR_SEAM_ANGLE_THR_DEG:
                continue
            mid_x = (lx1 + lx2) * 0.5
            seam_candidates.append(
                {
                    "mid_x_local": float(mid_x),
                    "pt_a_local": np.array([mid_x, 0.0], dtype=np.float64),
                    "pt_b_local": np.array([mid_x, float(bh - 1)], dtype=np.float64),
                    "source_bonus": 0.0,
                    "source": "hough",
                }
            )

    contour_candidates = _contour_vertical_seam_candidates(mask_crop, bw, bh)
    dbg["contour_candidates_count"] = len(contour_candidates)
    seam_candidates.extend(contour_candidates)
    if not seam_candidates:
        return None, dbg

    for cand in seam_candidates:
        mid_x = float(cand["mid_x_local"])
        mid_x_norm = mid_x / max(1.0, float(bw - 1))
        if mid_x_norm < center_margin or mid_x_norm > (1.0 - center_margin):
            continue
        mid_y = max(0, min(bh - 1, int(round(0.5 * (bh - 1)))))
        mid_xi = int(round(mid_x))
        mid_xi = max(0, min(bw - 1, mid_xi))
        if mask_crop[mid_y, mid_xi] == 0:
            ys_nz = np.where(mask_crop[:, mid_xi] > 0)[0]
            if ys_nz.size == 0:
                continue
            mid_y = int(ys_nz[ys_nz.size // 2])

        pt_a = cand["pt_a_local"]
        pt_b = cand["pt_b_local"]
        split_res = _split_mask_by_line(mask_crop, float(pt_a[0]), float(pt_a[1]), float(pt_b[0]), float(pt_b[1]))
        if split_res is None:
            continue
        left, right = split_res
        area_l = cv2.countNonZero(left)
        area_r = cv2.countNonZero(right)
        if area_l < min_area or area_r < min_area:
            continue
        balance = min(area_l, area_r) / max(area_l, area_r)
        edge_score = _line_edge_strength(gray, pt_a, pt_b, width=7)
        center_score = 1.0 - abs(mid_x_norm - 0.5) / 0.5
        score = (
            EXTERIOR_SEAM_W_BALANCE * balance
            + EXTERIOR_SEAM_W_EDGE * edge_score
            + EXTERIOR_SEAM_W_CENTER * max(0.0, center_score)
        )
        score += float(cand.get("source_bonus", 0.0))
        candidates_count += 1
        candidate_lines.append(
            {
                "score": float(score),
                "mid_x_local": float(mid_x),
                "line_local": (pt_a.copy(), pt_b.copy()),
                "source": str(cand.get("source", "")),
            }
        )
        if score > best_score:
            best_score = score
            best_line_local = (pt_a, pt_b)

    dbg["candidates_count"] = candidates_count
    dbg["best_score"] = float(max(0.0, best_score))
    if not candidate_lines or best_line_local is None or best_score < EXTERIOR_SEAM_MIN_SCORE:
        return None, dbg

    # NMS по X: оставляем топ линии, чтобы auto-split мог выбрать лучший вариант после 2-го прохода.
    candidate_lines.sort(key=lambda c: float(c["score"]), reverse=True)
    min_gap_px = max(8.0, float(EXTERIOR_SEAM_NMS_X_GAP_RATIO) * float(bw))
    selected: list[dict] = []
    for cand in candidate_lines:
        x_cur = float(cand["mid_x_local"])
        if any(abs(x_cur - float(s["mid_x_local"])) < min_gap_px for s in selected):
            continue
        selected.append(cand)
        if len(selected) >= max(1, EXTERIOR_SEAM_TOP_K):
            break

    top_lines_dbg: list[dict] = []
    for cand in selected:
        pt_a_l, pt_b_l = cand["line_local"]
        pt_a_g = np.array([pt_a_l[0] + x1i, pt_a_l[1] + y1i], dtype=np.float64)
        pt_b_g = np.array([pt_b_l[0] + x1i, pt_b_l[1] + y1i], dtype=np.float64)
        top_lines_dbg.append(
            {
                "score": round(float(cand["score"]), 4),
                "source": str(cand.get("source", "")),
                "line": [
                    round(float(pt_a_g[0]), 2),
                    round(float(pt_a_g[1]), 2),
                    round(float(pt_b_g[0]), 2),
                    round(float(pt_b_g[1]), 2),
                ],
            }
        )
    dbg["top_lines"] = top_lines_dbg

    pt_a_local, pt_b_local = best_line_local
    pt_a_global = np.array([pt_a_local[0] + x1i, pt_a_local[1] + y1i], dtype=np.float64)
    pt_b_global = np.array([pt_b_local[0] + x1i, pt_b_local[1] + y1i], dtype=np.float64)
    dbg["best_line"] = [
        round(float(pt_a_global[0]), 2),
        round(float(pt_a_global[1]), 2),
        round(float(pt_b_global[0]), 2),
        round(float(pt_b_global[1]), 2),
    ]

    component_masks = _component_masks_from_split_line(region_mask, pt_a_global, pt_b_global, min_area)
    if component_masks is None:
        return None, dbg
    dbg["used"] = True
    return component_masks, dbg


def _second_pass_seam_on_components(
    component_masks: list[np.ndarray],
    image_bgr: np.ndarray,
    image_width: int,
    image_height: int,
    min_area: int,
    max_walls: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Второй вертикальный шов на каждой крупной компоненте (до max_walls стен)."""
    wall_union = np.zeros((image_height, image_width), dtype=np.uint8)
    for m in component_masks:
        wall_union = cv2.bitwise_or(wall_union, m)
    total_a = max(1, int(cv2.countNonZero(wall_union)))

    by_area = sorted(component_masks, key=lambda m: int(cv2.countNonZero(m)), reverse=True)

    extra: list[dict] = []
    out: list[np.ndarray] = []
    for pm in by_area:
        a = int(cv2.countNonZero(pm))
        _, _, bw, _ = cv2.boundingRect(pm)
        wide_enough = bw >= int(0.15 * image_width)
        large_enough = a >= int(0.13 * total_a)
        room = max_walls - len(out)
        if EXTERIOR_SEAM_SECOND_PASS and room >= 2 and wide_enough and large_enough:
            sub, sdbg = _seam_split_single_region(pm, image_bgr, image_width, image_height, min_area)
            sdbg["parent_area_ratio"] = round(a / total_a, 3)
            if sub is not None and len(sub) >= 2 and len(sub) <= room:
                out.extend(sub)
                extra.append(sdbg)
                continue
        if len(out) < max_walls:
            out.append(pm)
    if len(out) < 2:
        return component_masks, []
    return out, extra


def _auto_split_by_corner_seam(
    wall_minus_holes: np.ndarray,
    image_bgr: np.ndarray | None,
    building_bbox: list[float] | None,
    image_width: int,
    image_height: int,
) -> tuple[list[dict] | None, dict]:
    seam_debug: dict = {
        "enabled": bool(EXTERIOR_SEAM_ENABLE),
        "used": False,
        "candidates_count": 0,
        "best_score": 0.0,
        "best_line": None,
        "second_pass": [],
    }
    if not EXTERIOR_SEAM_ENABLE or image_bgr is None or building_bbox is None or len(building_bbox) != 4:
        return None, seam_debug

    min_area = int(image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO)
    component_masks, inner_dbg = _seam_split_single_region(
        wall_minus_holes, image_bgr, image_width, image_height, min_area
    )
    seam_debug["candidates_count"] = int(inner_dbg.get("candidates_count", 0))
    seam_debug["best_score"] = float(inner_dbg.get("best_score", 0.0))
    seam_debug["best_line"] = inner_dbg.get("best_line")
    seam_debug["top_lines"] = inner_dbg.get("top_lines", [])
    if component_masks is None:
        return None, seam_debug

    # Выбираем лучший шов из top-K кандидатов: сначала по числу финальных стен, затем по score линии.
    trials: list[dict] = []
    top_lines = inner_dbg.get("top_lines", [])
    if isinstance(top_lines, list):
        for rec in top_lines[: max(1, EXTERIOR_SEAM_TOP_K)]:
            line = rec.get("line") if isinstance(rec, dict) else None
            if not isinstance(line, list) or len(line) != 4:
                continue
            pt_a = np.array([float(line[0]), float(line[1])], dtype=np.float64)
            pt_b = np.array([float(line[2]), float(line[3])], dtype=np.float64)
            cm = _component_masks_from_split_line(wall_minus_holes, pt_a, pt_b, min_area)
            if cm is None:
                continue
            refined_cm, second_dbg = _second_pass_seam_on_components(
                cm,
                image_bgr,
                image_width,
                image_height,
                min_area,
                EXTERIOR_MAX_WALLS,
            )
            walls_trial = _build_walls_from_component_masks(refined_cm)
            trials.append(
                {
                    "walls": walls_trial,
                    "walls_count": len(walls_trial),
                    "score": float(rec.get("score", 0.0)) if isinstance(rec, dict) else 0.0,
                    "line": line,
                    "second_pass": second_dbg,
                }
            )

    if trials:
        trials.sort(key=lambda t: (int(t["walls_count"]), float(t["score"])), reverse=True)
        best = trials[0]
        walls = best["walls"]
        seam_debug["best_line"] = best["line"]
        seam_debug["best_score"] = float(best["score"])
        seam_debug["second_pass"] = best["second_pass"]
    else:
        refined, second_dbg = _second_pass_seam_on_components(
            component_masks,
            image_bgr,
            image_width,
            image_height,
            min_area,
            EXTERIOR_MAX_WALLS,
        )
        seam_debug["second_pass"] = second_dbg
        walls = _build_walls_from_component_masks(refined)

    if len(walls) < 2:
        return None, seam_debug
    seam_debug["used"] = True
    return walls, seam_debug


async def run_exterior_pipeline(image_bytes: bytes, image_width: int, image_height: int) -> dict:
    """
    Full exterior detection pipeline:
    1) GroundingDINO → building bbox
    2) SAM → wall mask
    3) GroundingDINO → window/door bboxes
    4) SAM → individual opening masks → combined holes mask
    5) Postprocess → wall_minus_holes
    """
    call_ts = int(time.time() * 1000)
    call_digest = hashlib.md5(image_bytes).hexdigest()[:10]
    call_dir = os.path.join(EXTERIOR_DEBUG_DIR, f"call_{call_ts}_{call_digest}")
    os.makedirs(call_dir, exist_ok=True)

    building_bbox = await detect_building_bbox(image_bytes)
    if building_bbox is None:
        empty = np.zeros((image_height, image_width), dtype=np.uint8)
        empty_b64 = encode_mask_png_base64(empty)
        return {
            "walls": [],
            "image_size": {"width": image_width, "height": image_height},
            "masks": {
                "wall": empty_b64,
                "holes": empty_b64,
                "wall_minus_holes": empty_b64,
            },
            "debug": {"building_bbox": None, "windows_bboxes": [], "doors_bboxes": [], "walls_count": 0},
        }

    wall_mask = await sam_mask_from_bbox(image_bytes, building_bbox)

    openings = await detect_openings_bboxes(image_bytes, building_bbox)

    holes_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    all_opening_bboxes = openings["windows"] + openings["doors"]
    for ob in all_opening_bboxes:
        try:
            opening_mask = await sam_mask_from_bbox(image_bytes, ob)
            if opening_mask.shape == holes_mask.shape:
                holes_mask = cv2.bitwise_or(holes_mask, opening_mask)
        except Exception:
            continue

    if wall_mask.shape != (image_height, image_width):
        wall_mask = cv2.resize(wall_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    if holes_mask.shape != (image_height, image_width):
        holes_mask = cv2.resize(holes_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    wall_minus_holes = postprocess_masks(wall_mask, holes_mask)

    image_bgr = _decode_image_bgr(image_bytes, image_width, image_height)
    min_area = int(image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO)
    comp_count = _count_large_components(wall_minus_holes, min_area)
    split_method_geo = "geometry_fallback"
    walls_geo: list[dict]
    seam_debug: dict = {}
    if comp_count <= 1:
        walls_seam, seam_debug = _auto_split_by_corner_seam(
            wall_minus_holes, image_bgr, building_bbox, image_width, image_height
        )
        if walls_seam and len(walls_seam) >= 2:
            walls_geo = walls_seam
            split_method_geo = "seam"
        else:
            walls_kink = _split_by_bottom_contour_kink_once(
                wall_minus_holes, image_width, image_height, min_area
            )
            if walls_kink and len(walls_kink) >= 2:
                walls_geo = walls_kink
                split_method_geo = "bottom_kink"
            else:
                walls_geo = split_walls_from_mask(wall_minus_holes, image_width, image_height)
                split_method_geo = "geometry_mask"
    else:
        walls_geo = split_walls_from_mask(wall_minus_holes, image_width, image_height)
        split_method_geo = "geometry_cc"

    # Упрощённый режим: для отладки seam/deometry НЕ переключаемся на depth-арбитр.
    # Финальный результат = геометрия/seam, чтобы поведение было предсказуемым.
    walls = walls_geo
    split_method = split_method_geo
    split_arbiter = {
        "chosen": "geometry_only",
        "plane_compare_ran": False,
        "reason": "depth_arbiter_disabled",
    }

    if EXTERIOR_DEBUG_SAVE:
        try:
            # Save masks as PNGs for manual inspection.
            cv2.imwrite(os.path.join(call_dir, "wall.png"), wall_mask)
            cv2.imwrite(os.path.join(call_dir, "holes.png"), holes_mask)
            cv2.imwrite(os.path.join(call_dir, "wall_minus_holes.png"), wall_minus_holes)

            # Save an overlay visualization (polygons drawn on top of mask).
            overlay = cv2.cvtColor(wall_minus_holes, cv2.COLOR_GRAY2BGR)
            for w in walls:
                poly = w.get("polygon") or []
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(call_dir, "polygons_overlay.png"), overlay)

            # Save response summary (no base64 to keep files small).
            import json as _json

            summary = {
                "image_size": {"width": image_width, "height": image_height},
                "debug": {
                    "building_bbox": building_bbox,
                    "windows_bboxes": openings["windows"],
                    "doors_bboxes": openings["doors"],
                    "walls_count": len(walls),
                    "split_method": split_method,
                    "seam_debug": seam_debug,
                    "split_arbiter": split_arbiter,
                    "contour_corner_debug": _debug_contour_corner_points(wall_minus_holes),
                },
                "walls": [
                    {
                        "id": w.get("id"),
                        "center": w.get("center"),
                        "corners_len": len(w.get("corners") or []),
                        "polygon_len": len(w.get("polygon") or []),
                    }
                    for w in walls
                ],
            }
            with open(os.path.join(call_dir, "response_summary.json"), "w", encoding="utf-8") as f:
                f.write(_json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as dbg_err:
            print(f"[exterior-debug] save failed: {dbg_err}")

    # Always log a minimal summary to console (useful even if debug save off).
    try:
        polygon_lens = [len(w.get("polygon") or []) for w in walls]
        print(
            f"[detect-exterior] walls={len(walls)} polygon_lens={polygon_lens[:10]} "
            f"wall_minus_holes_nonzero={int(np.count_nonzero(wall_minus_holes))} "
            f"split={split_method} arb={split_arbiter.get('chosen', '')} call={call_ts}_{call_digest}"
        )
    except Exception:
        pass

    return {
        "walls": walls,
        "image_size": {"width": image_width, "height": image_height},
        "masks": {
            "wall": encode_mask_png_base64(wall_mask),
            "holes": encode_mask_png_base64(holes_mask),
            "wall_minus_holes": encode_mask_png_base64(wall_minus_holes),
        },
        "debug": {
            "building_bbox": building_bbox,
            "windows_bboxes": openings["windows"],
            "doors_bboxes": openings["doors"],
            "walls_count": len(walls),
            "split_method": split_method,
            "seam_debug": seam_debug,
            "split_arbiter": split_arbiter,
        },
    }
