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

GDINO_URL = os.getenv("EXTERIOR_GDINO_URL", "http://127.0.0.1:8001/gdino")
SAM_URL = os.getenv("EXTERIOR_SAM_URL", "http://127.0.0.1:8001/sam")
TIMEOUT_S = int(os.getenv("EXTERIOR_TIMEOUT_MS", "30000")) / 1000.0
SCORE_THRESH_BUILDING = float(os.getenv("EXTERIOR_SCORE_THRESH_BUILDING", "0.3"))
SCORE_THRESH_OPENINGS = float(os.getenv("EXTERIOR_SCORE_THRESH_OPENINGS", "0.35"))

# Debug output for investigating mask->front rendering.
# Saves intermediate PNGs + prints summary to stdout.
EXTERIOR_DEBUG_SAVE = os.getenv("EXTERIOR_DEBUG_SAVE", "1") in {"1", "true", "TRUE", "yes", "YES"}
EXTERIOR_POLYGON_EPS_RATIO = float(os.getenv("EXTERIOR_POLYGON_EPS_RATIO", "0.004"))

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
EXTERIOR_DEBUG_DIR = os.path.join(PROJECT_ROOT, "exterior-debug")


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
    """Call GroundingDINO to find the largest building/house bbox."""
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(
            GDINO_URL,
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
            data={
                "prompt": "house . building . facade",
                "score_threshold": str(SCORE_THRESH_BUILDING),
            },
        )
    resp.raise_for_status()
    data = resp.json()
    bboxes = data.get("bboxes", [])
    if not bboxes:
        return None
    largest = max(bboxes, key=lambda b: _bbox_area(b["bbox"]))
    return largest["bbox"]


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
    """Subtract dilated holes from wall mask, clean small components."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    holes_dilated = cv2.dilate(holes_mask, kernel, iterations=1)
    result = cv2.bitwise_and(wall_mask, cv2.bitwise_not(holes_dilated))

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
EXTERIOR_CORNER_ANGLE_DEG = 50
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
EXTERIOR_SEAM_MIN_SCORE = float(os.getenv("EXTERIOR_SEAM_MIN_SCORE", "0.42"))


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

    eps = EXTERIOR_CONTOUR_EPS_RATIO * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 4:
        return [component_mask]

    corners = _find_corner_indices(approx)
    if not corners:
        return [component_mask]

    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_ratio = 0.0

    pts = approx.reshape(-1, 2)
    for idx in corners:
        p = pts[idx]
        px, py = float(p[0]), float(p[1])
        y_top = max(0, int(py - 20))
        y_bot = min(component_mask.shape[0] - 1, int(py + 20))
        line_res = _split_mask_by_line(component_mask, px, y_top, px, y_bot)
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
        ratio = min(r1, r2)
        if ratio > best_ratio:
            best_ratio = ratio
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


def _extract_wall_from_component(component_mask: np.ndarray) -> dict | None:
    """Extract corners + center from a single binary component mask."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    # Стабильный quad для гомографии (перспективного warp).
    # Он не задаёт точную форму фасада, а только направление/перспективу наложения.
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect)
    corners = _order_rect_corners_tl_bl_br_tr(box_pts)

    # Контур компоненты для точного оверлея/клипа (маска области стены).
    # Возвращаем упрощённую ломаную через approxPolyDP.
    perimeter = cv2.arcLength(contour, True)
    # Для polygon нужно меньшее упрощение, иначе получится почти quad.
    eps = EXTERIOR_POLYGON_EPS_RATIO * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    polygon = approx.reshape(-1, 2).astype(int).tolist()
    if len(polygon) < 3:
        # Fallback: хотя бы quad из minAreaRect.
        polygon = corners

    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = float(np.mean([c[0] for c in corners]))
        cy = float(np.mean([c[1] for c in corners]))

    return {
        "corners": corners,
        "polygon": polygon,
        "center": [round(cx, 2), round(cy, 2)],
    }


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
        # 1) Сначала пробуем разбить L-образный фасад по внутренней вогнутости (угловой перелом).
        inner = _inner_corner_split_component(component_masks[0], int(min_area), EXTERIOR_MAX_WALLS)
        if len(inner) > 1:
            component_masks = inner
        else:
            # 2) fallback: контурные углы
            sub = _contour_split_component(component_masks[0], int(min_area), EXTERIOR_MAX_WALLS)
            if len(sub) > 1:
                component_masks = sub
            else:
                # 3) ultimate fallback: watershed
                sub = _watershed_split(component_masks[0])
                if len(sub) > 1:
                    component_masks = sub

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
        walls.append({
            "id": idx + 1,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        })

    return walls


def split_exterior_by_line(
    mask_base64: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Split wall_minus_holes mask by user-drawn line. Returns new walls[].
    """
    arr = np.frombuffer(base64.b64decode(mask_base64), dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (image_height, image_width):
        return []

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
        walls.append({
            "id": idx + 1,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        })
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
        walls.append({
            "id": idx + 1,
            "corners": w["corners"],
            "polygon": w.get("polygon"),
            "center": w["center"],
        })
    return walls


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
    }
    if not EXTERIOR_SEAM_ENABLE or image_bgr is None or building_bbox is None or len(building_bbox) != 4:
        return None, seam_debug

    x1, y1, x2, y2 = [float(v) for v in building_bbox]
    x1i = int(max(0, min(image_width - 2, round(x1))))
    y1i = int(max(0, min(image_height - 2, round(y1))))
    x2i = int(max(x1i + 1, min(image_width - 1, round(x2))))
    y2i = int(max(y1i + 1, min(image_height - 1, round(y2))))
    bw = x2i - x1i + 1
    bh = y2i - y1i + 1
    if bw < 20 or bh < 20:
        return None, seam_debug

    crop = image_bgr[y1i : y2i + 1, x1i : x2i + 1]
    mask_crop = wall_minus_holes[y1i : y2i + 1, x1i : x2i + 1]
    if crop.size == 0 or mask_crop.size == 0 or cv2.countNonZero(mask_crop) < 300:
        return None, seam_debug

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
    if lines is None:
        return None, seam_debug

    min_area = int(image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO)
    center_margin = EXTERIOR_SEAM_CENTER_MARGIN_RATIO
    best_score = -1.0
    best_line_local: tuple[np.ndarray, np.ndarray] | None = None
    candidates_count = 0

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
        mid_x_norm = mid_x / max(1.0, float(bw - 1))
        if mid_x_norm < center_margin or mid_x_norm > (1.0 - center_margin):
            continue
        mid_y = int(round((ly1 + ly2) * 0.5))
        mid_y = max(0, min(bh - 1, mid_y))
        mid_xi = int(round(mid_x))
        mid_xi = max(0, min(bw - 1, mid_xi))
        if mask_crop[mid_y, mid_xi] == 0:
            continue

        pt_a = np.array([mid_x, 0.0], dtype=np.float64)
        pt_b = np.array([mid_x, float(bh - 1)], dtype=np.float64)
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
        candidates_count += 1
        if score > best_score:
            best_score = score
            best_line_local = (pt_a, pt_b)

    seam_debug["candidates_count"] = candidates_count
    seam_debug["best_score"] = float(max(0.0, best_score))
    if best_line_local is None or best_score < EXTERIOR_SEAM_MIN_SCORE:
        return None, seam_debug

    pt_a_local, pt_b_local = best_line_local
    pt_a_global = np.array([pt_a_local[0] + x1i, pt_a_local[1] + y1i], dtype=np.float64)
    pt_b_global = np.array([pt_b_local[0] + x1i, pt_b_local[1] + y1i], dtype=np.float64)
    seam_debug["best_line"] = [
        round(float(pt_a_global[0]), 2),
        round(float(pt_a_global[1]), 2),
        round(float(pt_b_global[0]), 2),
        round(float(pt_b_global[1]), 2),
    ]

    split_global = _split_mask_by_line(
        wall_minus_holes,
        float(pt_a_global[0]),
        float(pt_a_global[1]),
        float(pt_b_global[0]),
        float(pt_b_global[1]),
    )
    if split_global is None:
        return None, seam_debug

    left_g, right_g = split_global
    component_masks: list[np.ndarray] = []
    for part in (left_g, right_g):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        for i in range(1, num_labels):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                comp = np.zeros_like(wall_minus_holes)
                comp[labels == i] = 255
                component_masks.append(comp)

    walls = _build_walls_from_component_masks(component_masks)
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
    split_method = "geometry_fallback"
    walls: list[dict]
    seam_debug: dict = {}
    if comp_count <= 1:
        walls_seam, seam_debug = _auto_split_by_corner_seam(
            wall_minus_holes, image_bgr, building_bbox, image_width, image_height
        )
        if walls_seam and len(walls_seam) >= 2:
            walls = walls_seam
            split_method = "seam"
        else:
            walls = split_walls_from_mask(wall_minus_holes, image_width, image_height)
    else:
        walls = split_walls_from_mask(wall_minus_holes, image_width, image_height)

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
            f"split={split_method} call={call_ts}_{call_digest}"
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
        },
    }
