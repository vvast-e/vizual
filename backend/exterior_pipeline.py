"""
Exterior detection pipeline.
Calls local model_service (GroundingDINO + SAM) via HTTP,
postprocesses masks, returns wall/holes/wall_minus_holes.
"""
import asyncio
import base64
import io
import hashlib
import logging
import os
import time
import json

logger = logging.getLogger(__name__)

import cv2
import httpx
import numpy as np
from sklearn.cluster import DBSCAN

from facade_geometry import split_gable_roof_polygon
from vp_detection import extract_manhattan_vps, project_to_horizon, project_verticals

GDINO_URL = os.getenv("EXTERIOR_GDINO_URL", "http://127.0.0.1:8001/gdino")
SAM_URL = os.getenv("EXTERIOR_SAM_URL", "http://127.0.0.1:8001/sam")
SAM_BATCH_URL = os.getenv("EXTERIOR_SAM_BATCH_URL", "http://127.0.0.1:8001/sam_batch")
TIMEOUT_S = int(os.getenv("EXTERIOR_TIMEOUT_MS", "120000")) / 1000.0
SCORE_THRESH_BUILDING = float(os.getenv("EXTERIOR_SCORE_THRESH_BUILDING", "0.3"))
SCORE_THRESH_OPENINGS = float(os.getenv("EXTERIOR_SCORE_THRESH_OPENINGS", "0.22"))
EXTERIOR_SAM_BATCH_ENABLE = os.getenv("EXTERIOR_SAM_BATCH_ENABLE", "1") in {"1", "true", "TRUE", "yes", "YES"}
EXTERIOR_SAM_BATCH_SIZE = max(1, int(os.getenv("EXTERIOR_SAM_BATCH_SIZE", "8")))
EXTERIOR_OPENINGS_NMS_IOU = float(os.getenv("EXTERIOR_OPENINGS_NMS_IOU", "0.5"))
EXTERIOR_OPENINGS_MIN_AREA_PX = int(os.getenv("EXTERIOR_OPENINGS_MIN_AREA_PX", "200"))

# Depth-planes: деление фасада по 3D-геометрии (Depth Anything V2 + кластеризация нормалей).
# По умолчанию выключено — включить EXTERIOR_DEPTH_SPLIT_ENABLE=1 для тестирования.
EXTERIOR_DEPTH_SPLIT_ENABLE = os.getenv("EXTERIOR_DEPTH_SPLIT_ENABLE", "0") in {
    "1", "true", "TRUE", "yes", "YES"
}
EXTERIOR_DEPTH_URL = os.getenv("EXTERIOR_DEPTH_URL", "http://127.0.0.1:8001/depth")

# Debug output for investigating mask->front rendering.
# Saves intermediate PNGs + prints summary to stdout.
EXTERIOR_DEBUG_SAVE = os.getenv("EXTERIOR_DEBUG_SAVE", "1") in {"1", "true", "TRUE", "yes", "YES"}
EXTERIOR_POLYGON_EPS_RATIO = float(os.getenv("EXTERIOR_POLYGON_EPS_RATIO", "0.004"))
EXTERIOR_GEOM_CASCADE_ENABLE = os.getenv("EXTERIOR_GEOM_CASCADE_ENABLE", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_GEOM_DEBUG_LEVEL = str(os.getenv("EXTERIOR_GEOM_DEBUG_LEVEL", "verbose")).strip().lower()
EXTERIOR_GEOM_DEBUG_SAVE_IMAGES = os.getenv("EXTERIOR_GEOM_DEBUG_SAVE_IMAGES", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_GEOM_DEBUG_SAVE_JSON = os.getenv("EXTERIOR_GEOM_DEBUG_SAVE_JSON", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_GEOM_DEBUG_TOPN = max(5, int(os.getenv("EXTERIOR_GEOM_DEBUG_TOPN", "50")))

EXTERIOR_LSD_MIN_SEG_LEN_PX = int(os.getenv("EXTERIOR_LSD_MIN_SEG_LEN_PX", "40"))
EXTERIOR_LSD_SAMPLING_POINTS = int(os.getenv("EXTERIOR_LSD_SAMPLING_POINTS", "5"))
EXTERIOR_LSD_INTERSECTION_NEIGHBOR_RADIUS_PX = int(os.getenv("EXTERIOR_LSD_INTERSECTION_NEIGHBOR_RADIUS_PX", "5"))
EXTERIOR_LSD_NMS_GAP_RATIO = float(os.getenv("EXTERIOR_LSD_NMS_GAP_RATIO", "0.03"))

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
EXTERIOR_DEBUG_DIR = os.path.join(PROJECT_ROOT, "exterior-debug")

# ---------------------------------------------------------------------------
# Retry helper: model_service (port 8001) may still be loading models at
# container startup.  Without retry the very first request gets
# "All connection attempts failed" (httpx.ConnectError) and returns 500.
# We retry up to MODEL_SERVICE_RETRY_ATTEMPTS times with exponential backoff.
# ---------------------------------------------------------------------------
MODEL_SERVICE_RETRY_ATTEMPTS = int(os.getenv("MODEL_SERVICE_RETRY_ATTEMPTS", "10"))
MODEL_SERVICE_RETRY_DELAY_S = float(os.getenv("MODEL_SERVICE_RETRY_DELAY_S", "3.0"))


async def _post_with_retry(url: str, **kwargs) -> httpx.Response:
    """httpx.AsyncClient.post with retry on ConnectError.

    Retries up to MODEL_SERVICE_RETRY_ATTEMPTS times when the target
    service is not yet ready (ConnectError / ConnectTimeout).  All other
    errors are raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MODEL_SERVICE_RETRY_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                resp = await client.post(url, **kwargs)
            return resp
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_exc = exc
            if attempt < MODEL_SERVICE_RETRY_ATTEMPTS:
                delay = MODEL_SERVICE_RETRY_DELAY_S * (1.5 ** (attempt - 1))
                print(
                    f"[exterior_pipeline] model_service not ready at {url}, "
                    f"retry {attempt}/{MODEL_SERVICE_RETRY_ATTEMPTS} in {delay:.1f}s …"
                )
                await asyncio.sleep(delay)
            else:
                print(
                    f"[exterior_pipeline] model_service at {url} unreachable "
                    f"after {MODEL_SERVICE_RETRY_ATTEMPTS} attempts"
                )
    raise last_exc  # type: ignore[misc]


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
    resp = await _post_with_retry(
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
    resp = await _post_with_retry(
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

    resp = await _post_with_retry(
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


async def sam_masks_from_bboxes_batch(image_bytes: bytes, bboxes: list[list[float]]) -> list[np.ndarray]:
    """Call SAM batch endpoint and return masks in bbox order."""
    if not bboxes:
        return []
    resp = await _post_with_retry(
        SAM_BATCH_URL,
        files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
        data={"bboxes": json.dumps(bboxes), "multimask_output": "false"},
    )
    resp.raise_for_status()
    data = resp.json()
    masks_b64 = data.get("masks", [])
    if not isinstance(masks_b64, list):
        raise ValueError("Invalid sam_batch response: masks must be list")
    out: list[np.ndarray] = []
    for i, mb64 in enumerate(masks_b64):
        arr = np.frombuffer(base64.b64decode(str(mb64)), dtype=np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to decode SAM batch mask at index {i}")
        out.append(mask)
    return out


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    if den <= 1e-6:
        return 0.0
    return float(inter / den)


def _filter_opening_bboxes(bboxes: list[list[float]]) -> list[list[float]]:
    """Remove tiny and highly-overlapping opening boxes before SAM."""
    if not bboxes:
        return []
    # 1) area filter
    filtered = [b for b in bboxes if _bbox_area(b) >= float(EXTERIOR_OPENINGS_MIN_AREA_PX)]
    if not filtered:
        return []
    # 2) greedy NMS by area desc (proxy for stability without per-box score)
    ordered = sorted(filtered, key=_bbox_area, reverse=True)
    kept: list[list[float]] = []
    for b in ordered:
        if any(_bbox_iou(b, k) >= EXTERIOR_OPENINGS_NMS_IOU for k in kept):
            continue
        kept.append(b)
    # keep deterministic left-to-right order for downstream debug
    kept.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return kept


def _clean_mask_for_geometry(mask: np.ndarray) -> np.ndarray:
    """Light denoise for wall geometry preserving macro-boundaries."""
    out = mask.copy()
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=1)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return out
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return out
    keep_idx = int(np.argmax(areas)) + 1
    clean = np.zeros_like(out)
    clean[lbl == keep_idx] = 255
    return clean


def _resample_contour_arc_length(contour: np.ndarray, step_px: float = 3.0) -> np.ndarray:
    """Uniform contour resampling by arc length."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return pts.astype(np.float32)
    d = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    per = float(np.sum(d))
    if per <= 1e-6:
        return pts.astype(np.float32)
    step = max(1.0, float(step_px))
    n_samples = max(16, int(round(per / step)))
    cum = np.cumsum(np.r_[0.0, d])
    target = np.linspace(0.0, per, n_samples, endpoint=False)
    out: list[np.ndarray] = []
    j = 0
    for t in target:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        a = pts[j % len(pts)]
        b = pts[(j + 1) % len(pts)]
        seg = max(1e-6, cum[j + 1] - cum[j])
        u = float((t - cum[j]) / seg)
        out.append((1.0 - u) * a + u * b)
    return np.array(out, dtype=np.float32)


def _multi_scale_corner_candidates(mask: np.ndarray) -> list[dict]:
    """
    Relative (in-image) corner candidates from multiple approx scales.
    No fixed tiers: score is rank-based inside current image.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.arcLength(contour, True) < 32:
        return []
    rs = _resample_contour_arc_length(contour, step_px=3.0).reshape(-1, 1, 2)
    per = float(cv2.arcLength(rs, True))
    h, w = mask.shape[:2]
    scale_fracs = [0.004, 0.007, 0.011, 0.016, 0.022]
    grouped: dict[int, dict] = {}
    for sf in scale_fracs:
        approx = cv2.approxPolyDP(rs, sf * per, True).reshape(-1, 2).astype(np.float64)
        n = len(approx)
        if n < 4:
            continue
        for i in range(n):
            p_prev = approx[(i - 1) % n]
            p = approx[i]
            p_next = approx[(i + 1) % n]
            v1 = p_prev - p
            v2 = p_next - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if min(n1, n2) < 5.0:
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_a)))
            delta = 180.0 - angle
            x = float(p[0])
            y = float(p[1])
            # very edge points are unstable for split
            if x < 4 or x > (w - 5) or y < 2 or y > (h - 3):
                continue
            key = int(round(x / max(6.0, 0.02 * w)))
            rec = grouped.get(key)
            if rec is None:
                grouped[key] = {
                    "x": x,
                    "y": y,
                    "delta_sum": max(0.0, delta),
                    "scale_hits": 1,
                    "seg_support": min(n1, n2),
                }
            else:
                rec["x"] = 0.5 * float(rec["x"]) + 0.5 * x
                rec["y"] = 0.5 * float(rec["y"]) + 0.5 * y
                rec["delta_sum"] = float(rec["delta_sum"]) + max(0.0, delta)
                rec["scale_hits"] = int(rec["scale_hits"]) + 1
                rec["seg_support"] = max(float(rec["seg_support"]), min(n1, n2))
    out: list[dict] = []
    if not grouped:
        return out
    for g in grouped.values():
        # Relative score; no absolute thresholding.
        score = float(g["delta_sum"]) * (1.0 + 0.35 * float(g["scale_hits"])) * (1.0 + 0.01 * float(g["seg_support"]))
        out.append(
            {
                "x": round(float(g["x"]), 2),
                "y": round(float(g["y"]), 2),
                "scale_hits": int(g["scale_hits"]),
                "seg_support": round(float(g["seg_support"]), 2),
                "score": round(score, 3),
                "source": "multi_scale_contour",
            }
        )
    out.sort(key=lambda r: float(r["score"]), reverse=True)
    return out


EXTERIOR_LSD_BOTTOM_BAND_RATIO = float(os.getenv("EXTERIOR_LSD_BOTTOM_BAND_RATIO", "0.35"))
EXTERIOR_LSD_BOTTOM_NEAR_RATIO = float(os.getenv("EXTERIOR_LSD_BOTTOM_NEAR_RATIO", "0.45"))
EXTERIOR_LSD_XACC_SIGMA_RATIO = float(os.getenv("EXTERIOR_LSD_XACC_SIGMA_RATIO", "0.025"))
EXTERIOR_LSD_XACC_MIN_PEAK_DIST_RATIO = float(os.getenv("EXTERIOR_LSD_XACC_MIN_PEAK_DIST_RATIO", "0.08"))
EXTERIOR_LSD_XACC_MIN_PROM_FRAC = float(os.getenv("EXTERIOR_LSD_XACC_MIN_PROM_FRAC", "0.35"))
EXTERIOR_LSD_XACC_MAX_SPLITS = max(1, int(os.getenv("EXTERIOR_LSD_XACC_MAX_SPLITS", "6")))


# ---------------------------------------------------------------------------
# RGB + LSD corner-detection cascade focused on bottom contour band
# ---------------------------------------------------------------------------

def _extract_bottom_profile(mask_bin: np.ndarray) -> dict:
    """Extract top/bottom facade profile per X from a binary mask."""
    h, w = mask_bin.shape[:2]
    y_bottom = np.full(w, -1, dtype=np.int32)
    y_top = np.full(w, -1, dtype=np.int32)
    valid_xs: list[int] = []
    for x in range(w):
        ys = np.where(mask_bin[:, x] > 0)[0]
        if ys.size == 0:
            continue
        valid_xs.append(x)
        y_top[x] = int(ys[0])
        y_bottom[x] = int(ys[-1])
    if not valid_xs:
        return {"valid": False, "y_bottom": y_bottom, "y_top": y_top}
    x_min = int(valid_xs[0])
    x_max = int(valid_xs[-1])
    facade_h = int(max(1, np.max(y_bottom[valid_xs] - y_top[valid_xs] + 1)))
    return {
        "valid": True,
        "x_min": x_min,
        "x_max": x_max,
        "y_bottom": y_bottom,
        "y_top": y_top,
        "facade_height": facade_h,
        "valid_xs": valid_xs,
    }


def _build_bottom_band_mask(mask_bin: np.ndarray, profile: dict, band_ratio: float) -> np.ndarray:
    """Adaptive lower band: for every X keep only bottom 30-40% of the facade column."""
    h, w = mask_bin.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if not bool(profile.get("valid")):
        return out
    y_bottom = profile["y_bottom"]
    y_top = profile["y_top"]
    ratio = float(np.clip(band_ratio, 0.1, 0.8))
    for x in profile.get("valid_xs", []):
        yb = int(y_bottom[x])
        yt = int(y_top[x])
        col_h = max(1, yb - yt + 1)
        band_h = max(12, int(round(col_h * ratio)))
        y0 = max(yt, yb - band_h + 1)
        out[y0 : yb + 1, x] = 255
    return cv2.bitwise_and(out, mask_bin)


def _lsd_segments_in_bottom_band(
    image_gray: np.ndarray,
    band_mask_bin: np.ndarray,
    profile: dict,
    near_ratio: float,
    wall_minus_holes: np.ndarray = None,
) -> list[dict]:
    """Detect smart LSD segments: full-height for verticals, bottom-band for horizontals."""
    h, w = image_gray.shape[:2]
    import cv2
    import numpy as np
    
    mask_bin = wall_minus_holes if wall_minus_holes is not None else band_mask_bin
    
    k_size = int(max(11, 0.015 * w))
    k_size += 1 if k_size % 2 == 0 else 0
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    smoothed_mask = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k_smooth)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, k_smooth)
    
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines_raw, _, _, nfa = lsd.detect(image_gray)
    if lines_raw is None or len(lines_raw) == 0:
        return []
        
    y_bottom = profile["y_bottom"]
    y_top = profile["y_top"]
    n_sample = max(3, EXTERIOR_LSD_SAMPLING_POINTS)
    near_ratio = float(np.clip(near_ratio, 0.1, 1.0))
    segments: list[dict] = []
    
    for i in range(len(lines_raw)):
        x1, y1, x2, y2 = lines_raw[i][0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < max(20, EXTERIOR_LSD_MIN_SEG_LEN_PX * 0.5):
            continue
            
        theta = float(np.arctan2(y2 - y1, x2 - x1))
        deg = np.degrees(theta) % 180
        is_vert = 40 <= deg <= 140
        
        inside_count = 0
        near_bottom_count = 0
        bottom_dists: list[float] = []
        ys_local: list[float] = []
        
        for t in np.linspace(0.0, 1.0, n_sample):
            sx_f = float(x1 + t * (x2 - x1))
            sy_f = float(y1 + t * (y2 - y1))
            sx = max(0, min(w - 1, int(round(sx_f))))
            sy = max(0, min(h - 1, int(round(sy_f))))
            
            if y_bottom[sx] < 0 or y_top[sx] < 0:
                continue
                
            if smoothed_mask[sy, sx] > 0:
                inside_count += 1
                
            ys_local.append(sy_f)
            col_h = max(1, int(y_bottom[sx] - y_top[sx] + 1))
            band_h = max(12, int(round(col_h * EXTERIOR_LSD_BOTTOM_BAND_RATIO)))
            dist_to_bottom = float(y_bottom[sx] - sy_f)
            bottom_dists.append(dist_to_bottom)
            
            if dist_to_bottom <= near_ratio * band_h:
                near_bottom_count += 1
                
        if inside_count < max(2, n_sample - 2):
            continue
            
        if not is_vert:
            if near_bottom_count < max(2, int(np.ceil(0.5 * inside_count))):
                continue
                
        segments.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "length": length, "theta": theta, "nfa": float(nfa[i][0]) if nfa is not None else 0.0,
            "inside_count": int(inside_count), "near_bottom_count": int(near_bottom_count),
            "bottom_dist_mean": round(float(np.mean(bottom_dists)) if bottom_dists else 0.0, 2),
            "y_mean": round(float(np.mean(ys_local)) if ys_local else 0.0, 2),
        })
    return segments


def _cluster_orientation_labels(segments: list[dict], eps_angle_deg: float) -> list[list[dict]]:
    """Cluster bottom-band segments by orientation using doubled-angle features."""
    if len(segments) < 2:
        return []
    thetas = np.array([s["theta"] for s in segments], dtype=np.float64)
    features = np.column_stack([np.cos(2.0 * thetas), np.sin(2.0 * thetas)])
    eps_dist = 2.0 * np.sin(np.radians(eps_angle_deg))
    clustering = DBSCAN(eps=eps_dist, min_samples=2, metric="euclidean").fit(features)
    labels = clustering.labels_
    unique_labels = sorted(set(labels) - {-1})
    clusters: list[list[dict]] = []
    for lbl in unique_labels:
        group = [segments[i] for i in range(len(segments)) if labels[i] == lbl]
        if len(group) >= 2:
            clusters.append(group)
    clusters.sort(key=lambda grp: sum(float(s["length"]) for s in grp), reverse=True)
    return clusters


def _segment_line_abc(seg: dict) -> tuple[float, float, float] | None:
    x1 = float(seg["x1"])
    y1 = float(seg["y1"])
    x2 = float(seg["x2"])
    y2 = float(seg["y2"])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = float(np.sqrt(a * a + b * b))
    if norm < 1e-8:
        return None
    return a / norm, b / norm, c / norm


def _split_segments_horizontal_vertical(segments: list[dict]) -> tuple[list[dict], list[dict], dict]:
    vp_vert, vert_segments, vp_h1, h1_segments, vp_h2, h2_segments = extract_manhattan_vps(segments)
    horizontals = h1_segments + h2_segments
    return horizontals, vert_segments, {"vp_vert": vp_vert, "vp_h1": vp_h1, "vp_h2": vp_h2}


def _build_vertical_x_accumulator(vert_segments: list[dict], image_width: int, vps: dict = None) -> dict:
    width = int(max(1, image_width))
    raw_acc = np.zeros((width,), dtype=np.float64)
    support_by_x = np.zeros((width,), dtype=np.float64)
    total_support = 0.0
    vp_v = vps.get("vp_vert") if vps else None
    vp_h1 = vps.get("vp_h1") if vps else None
    vp_h2 = vps.get("vp_h2") if vps else None
    horizon_line = project_to_horizon(vp_h1, vp_h2, image_width, image_width)
    projected = project_verticals(vert_segments, vp_v, horizon_line, image_width)
    for proj in projected:
        seg = proj["seg"]
        px = proj["x_proj"]
        w = max(1.0, float(seg.get("length", 0.0)))
        total_support += w
        center_x = int(round(px))
        lx = int(np.clip(center_x - 1, 0, width - 1))
        rx = int(np.clip(center_x + 1, 0, width - 1))
        if lx <= rx:
            raw_acc[lx : rx + 1] += w / (rx - lx + 1)
            support_by_x[lx : rx + 1] += 1.0
    return {
        "raw_acc": raw_acc,
        "support_by_x": support_by_x,
        "total_support": float(total_support),
    }


def _smooth_x_accumulator(raw_acc: np.ndarray, sigma_px: float) -> np.ndarray:
    """Smooth 1D x-accumulator with Gaussian kernel."""
    if raw_acc.size == 0:
        return raw_acc
    sigma = float(max(1.0, sigma_px))
    # cv2 Gaussian expects odd kernel size.
    k = int(max(3, 2 * round(3.0 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    arr = raw_acc.reshape(1, -1).astype(np.float32)
    out = cv2.GaussianBlur(arr, (k, 1), sigmaX=sigma, sigmaY=0)
    return out.reshape(-1).astype(np.float64)


def _detect_x_peaks(acc_smooth: np.ndarray, min_dist_px: int, min_prom_abs: float) -> list[dict]:
    """Detect local maxima in smoothed x-accumulator with prominence filter."""
    n = int(acc_smooth.size)
    if n < 3:
        return []
    peaks: list[dict] = []
    min_dist = max(1, int(min_dist_px))
    min_prom = float(max(1e-6, min_prom_abs))
    for x in range(1, n - 1):
        v = float(acc_smooth[x])
        if not (v >= float(acc_smooth[x - 1]) and v > float(acc_smooth[x + 1])):
            continue
        li = x - 1
        while li > 0 and float(acc_smooth[li - 1]) <= float(acc_smooth[li]):
            li -= 1
        ri = x + 1
        while ri < n - 1 and float(acc_smooth[ri + 1]) <= float(acc_smooth[ri]):
            ri += 1
        base = max(float(acc_smooth[li]), float(acc_smooth[ri]))
        prom = max(0.0, v - base)
        if prom < min_prom:
            continue
        peaks.append(
            {
                "x": int(x),
                "height": float(v),
                "prominence": float(prom),
                "left_base": int(li),
                "right_base": int(ri),
            }
        )
    peaks.sort(key=lambda p: float(p["prominence"]), reverse=True)
    kept: list[dict] = []
    for p in peaks:
        px = int(p["x"])
        if any(abs(px - int(k["x"])) < min_dist for k in kept):
            continue
        kept.append(p)
    kept.sort(key=lambda p: int(p["x"]))
    return kept


def _filter_peaks_by_contour_kinks(peaks: list[dict], profile: dict, image_width: int) -> list[dict]:
    import numpy as np
    if not peaks or not profile or not profile.get("valid"): return peaks
    y_top, y_bottom = profile.get("y_top"), profile.get("y_bottom")
    if y_top is None or y_bottom is None: return peaks
    
    filtered_peaks = []
    D_far = max(30, int(0.04 * image_width))
    D_near = 10
    T = max(8.0, 0.01 * image_width)
    search_radius = max(15, int(0.03 * image_width))
    
    for p in peaks:
        px = int(p["x"])
        if px - D_far < 0 or px + D_far >= len(y_top):
            filtered_peaks.append(p)
            continue
            
        s_left = max(0, px - search_radius)
        s_right = min(len(y_top), px + search_radius + 1)
        valid_yt = [y for y in y_top[s_left : s_right] if y >= 0]
        
        yt_center = float(np.median(y_top[px - 3 : px + 4]))
        yb_center = float(np.median(y_bottom[px - 3 : px + 4]))
        if yt_center < 0 or yb_center < 0:
            filtered_peaks.append(p)
            continue
            
        valley_y = float(np.max(valid_yt)) if valid_yt else yt_center
        gable_y = float(np.min(valid_yt)) if valid_yt else yt_center
        
        left_yt = [y for y in y_top[px - D_far : px - D_near] if y >= 0]
        right_yt = [y for y in y_top[px + D_near : px + D_far] if y >= 0]
        left_yb = [y for y in y_bottom[px - D_far : px - D_near] if y >= 0]
        right_yb = [y for y in y_bottom[px + D_near : px + D_far] if y >= 0]
        
        if len(left_yt) < D_far // 2 or len(right_yt) < D_far // 2:
            filtered_peaks.append(p)
            continue
            
        yt_L = float(np.median(left_yt))
        yt_R = float(np.median(right_yt))
        yb_L = float(np.median(left_yb))
        yb_R = float(np.median(right_yb))
        
        is_gable = (gable_y < yt_L - T) and (gable_y < yt_R - T)
        is_valley = (valley_y > yt_L + T) and (valley_y > yt_R + T)
        is_foundation_kink = abs(yb_center - yb_L) > T and abs(yb_center - yb_R) > T and ((yb_center - yb_L) * (yb_center - yb_R) > 0)
        
        if is_gable and not is_foundation_kink:
            continue
            
        if is_valley or is_foundation_kink:
            p["prominence"] = float(p["prominence"]) * 5.0
            p["height"] = float(p["height"]) * 5.0
            
        filtered_peaks.append(p)
    return filtered_peaks

def _peaks_to_split_candidates(
    peaks: list[dict],
    acc_smooth: np.ndarray,
    image_width: int,
    wall_mask_bin: np.ndarray,
    total_support: float,
) -> tuple[list[float], list[dict]]:
    """Convert x-peaks to split candidates and keep per-peak diagnostics."""
    if not peaks:
        return [], []
    w = int(max(1, image_width))
    max_val = float(np.max(acc_smooth)) if acc_smooth.size > 0 else 0.0
    min_height_abs = float(max(3.0, 0.35 * max_val))
    min_prom_abs = float(max(2.5, 0.30 * max_val))
    xs: list[float] = []
    diag: list[dict] = []
    for p in peaks:
        x = int(np.clip(int(p["x"]), 0, w - 1))
        h = float(p["height"])
        prom = float(p["prominence"])
        support_ratio = float(h / max(max_val, 1e-6))
        if h < min_height_abs:
            continue
        if prom < min_prom_abs:
            continue
        ys = np.where(wall_mask_bin[:, x] > 0)[0]
        if ys.size == 0:
            continue
        xs.append(float(x))
        diag.append(
            {
                "x": int(x),
                "height": round(h, 3),
                "prominence": round(prom, 3),
                "support_ratio": round(support_ratio, 4),
                "left_base": int(p["left_base"]),
                "right_base": int(p["right_base"]),
            }
        )
    xs = _nms_unique_x_candidates(xs, image_width)
    if len(xs) > EXTERIOR_LSD_XACC_MAX_SPLITS:
        xs = xs[:EXTERIOR_LSD_XACC_MAX_SPLITS]
    return xs, diag


def _group_segments_by_formula(segments: list[dict], axis: str, image_width: int) -> list[list[dict]]:
    """
    Group segments by similar line representation.
    - axis='h': group by c in ax+by+c=0 (horizontal levels)
    - axis='v': group by x-mid (vertical families)
    """
    items: list[dict] = []
    for s in segments:
        abc = _segment_line_abc(s)
        if abc is None:
            continue
        a, b, c = abc
        x_mid = 0.5 * (float(s["x1"]) + float(s["x2"]))
        y_mid = 0.5 * (float(s["y1"]) + float(s["y2"]))
        items.append({"seg": s, "a": a, "b": b, "c": c, "x_mid": x_mid, "y_mid": y_mid})
    if not items:
        return []
    if axis == "h":
        items.sort(key=lambda r: float(r["c"]))
        gap_thr = max(8.0, 0.02 * float(image_width))
    else:
        items.sort(key=lambda r: float(r["x_mid"]))
        # Vertical clusters were over-merged by chain-neighbor logic.
        # Keep tighter threshold and cluster around group centroid.
        gap_thr = max(8.0, 0.012 * float(image_width))

    groups: list[list[dict]] = []
    if axis == "h":
        cur: list[dict] = [items[0]]
        for rec in items[1:]:
            prev = cur[-1]
            gap = abs(float(rec["c"]) - float(prev["c"]))
            if gap <= gap_thr:
                cur.append(rec)
            else:
                groups.append(cur)
                cur = [rec]
        groups.append(cur)
    else:
        # Centroid-based 1D clustering to avoid transitive over-merge.
        for rec in items:
            x = float(rec["x_mid"])
            best_idx = -1
            best_d = 1e18
            for gi, g in enumerate(groups):
                gx = float(np.mean([float(r["x_mid"]) for r in g]))
                d = abs(x - gx)
                if d < best_d:
                    best_d = d
                    best_idx = gi
            if best_idx >= 0 and best_d <= gap_thr:
                groups[best_idx].append(rec)
            else:
                groups.append([rec])

        # Merge only truly close centroids (single-pass), no chaining.
        groups.sort(key=lambda g: float(np.mean([float(r["x_mid"]) for r in g])))
        merged: list[list[dict]] = []
        for g in groups:
            if not merged:
                merged.append(g)
                continue
            gxc = float(np.mean([float(r["x_mid"]) for r in g]))
            mxc = float(np.mean([float(r["x_mid"]) for r in merged[-1]]))
            if abs(gxc - mxc) <= 0.6 * gap_thr:
                merged[-1].extend(g)
            else:
                merged.append(g)
        groups = merged

    out: list[list[dict]] = []
    for g in groups:
        min_group = 1 if axis == "v" else 2
        if len(g) < min_group:
            continue
        out.append([r["seg"] for r in g])
    return out


def _line_intersection(line1: dict, line2: dict) -> tuple[float, float] | None:
    a1, b1, c1 = float(line1["a"]), float(line1["b"]), float(line1["c"])
    a2, b2, c2 = float(line2["a"]), float(line2["b"]), float(line2["c"])
    denom = a1 * b2 - a2 * b1
    if abs(denom) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / denom
    y = (a2 * c1 - a1 * c2) / denom
    return float(x), float(y)


def _build_planes_from_direction_families(
    families: list[list[dict]],
    image_width: int,
) -> tuple[list[dict], dict]:
    """
    Build planes from V-H pairs in lower contour zone.
    Vertical groups are paired with nearest horizontal groups.
    """
    build_dbg: dict = {
        "input_families": int(len(families)),
        "input_segments_total": int(sum(len(f) for f in families)),
        "horiz_segments_total": 0,
        "vert_segments_total": 0,
        "horiz_groups_total": 0,
        "vert_groups_total": 0,
        "horiz_groups_kept": 0,
        "vert_groups_kept": 0,
        "h_lines_total": 0,
        "v_lines_total": 0,
        "pairing_attempts": 0,
        "pairing_no_horizontal": 0,
        "pairing_no_anchor": 0,
        "pairing_built": 0,
    }

    all_segments = [s for fam in families for s in fam]
    horiz_segments, vert_segments = _split_segments_horizontal_vertical(all_segments)
    build_dbg["horiz_segments_total"] = int(len(horiz_segments))
    build_dbg["vert_segments_total"] = int(len(vert_segments))
    horiz_groups = _group_segments_by_formula(horiz_segments, axis="h", image_width=image_width)
    vert_groups = _group_segments_by_formula(vert_segments, axis="v", image_width=image_width)
    build_dbg["horiz_groups_kept"] = int(len(horiz_groups))
    build_dbg["vert_groups_kept"] = int(len(vert_groups))
    build_dbg["horiz_groups_total"] = int(len(horiz_groups))
    build_dbg["vert_groups_total"] = int(len(vert_groups))

    h_lines: list[dict] = []
    for grp in horiz_groups:
        ln = _fit_line_from_cluster_endpoints(grp)
        if ln is None:
            continue
        y_vals = [0.5 * (float(s["y1"]) + float(s["y2"])) for s in grp]
        h_lines.append(
            {
                "line": ln,
                "segments": grp,
                "n_segments": len(grp),
                "support": float(sum(float(s["length"]) for s in grp)),
                "y_center": float(np.mean(y_vals)) if y_vals else 0.0,
            }
        )
    build_dbg["h_lines_total"] = int(len(h_lines))

    v_lines: list[dict] = []
    for grp in vert_groups:
        ln = _fit_line_from_cluster_endpoints(grp)
        if ln is None:
            continue
        x_vals = [0.5 * (float(s["x1"]) + float(s["x2"])) for s in grp]
        v_lines.append(
            {
                "line": ln,
                "segments": grp,
                "n_segments": len(grp),
                "support": float(sum(float(s["length"]) for s in grp)),
                "x_center": float(np.mean(x_vals)) if x_vals else 0.0,
            }
        )
    build_dbg["v_lines_total"] = int(len(v_lines))

    planes: list[dict] = []
    plane_id = 0
    for v in v_lines:
        build_dbg["pairing_attempts"] = int(build_dbg["pairing_attempts"]) + 1
        x_ref = float(v.get("x_center", 0.0))
        best_h: dict | None = None
        best_d = 1e18
        for h in h_lines:
            inter = _line_intersection(v["line"], h["line"])
            if inter is None:
                continue
            d = abs(float(inter[0]) - x_ref)
            if d < best_d:
                best_d = d
                best_h = h
        if best_h is None:
            build_dbg["pairing_no_horizontal"] = int(build_dbg["pairing_no_horizontal"]) + 1
            continue
        anchor = _line_intersection(v["line"], best_h["line"])
        if anchor is None:
            build_dbg["pairing_no_anchor"] = int(build_dbg["pairing_no_anchor"]) + 1
            continue
        plane_id += 1
        planes.append(
            {
                "id": int(plane_id),
                "family_id": int(plane_id),  # unique id for compatibility
                "vertical_line": v["line"],
                "horizontal_line": best_h["line"],
                "line": v["line"],  # overlay compatibility
                "n_segments": int(v["n_segments"] + best_h["n_segments"]),
                "support": float(v["support"] + best_h["support"]),
                "x_center": float(v.get("x_center", 0.0)),
                "y_center": float(best_h.get("y_center", 0.0)),
                "anchor": {"x": float(anchor[0]), "y": float(anchor[1])},
            }
        )
        build_dbg["pairing_built"] = int(build_dbg["pairing_built"]) + 1
    planes.sort(key=lambda p: float(p["support"]), reverse=True)
    build_dbg["planes_total"] = int(len(planes))
    return planes, build_dbg


def _line_direction_from_abc(a: float, b: float) -> np.ndarray:
    """Direction vector of line ax + by + c = 0."""
    v = np.array([b, -a], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _canonical_line_abc(line: dict) -> dict:
    """Normalize ax+by+c=0 and fix sign for stable comparisons."""
    a = float(line["a"])
    b = float(line["b"])
    c = float(line["c"])
    n = float(np.sqrt(a * a + b * b))
    if n < 1e-8:
        return {"a": 1.0, "b": 0.0, "c": 0.0}
    a /= n
    b /= n
    c /= n
    if a < 0.0 or (abs(a) <= 1e-9 and b < 0.0):
        a, b, c = -a, -b, -c
    return {"a": float(a), "b": float(b), "c": float(c)}


def _line_angle_delta_deg(line1: dict, line2: dict) -> float:
    d1 = _line_direction_from_abc(float(line1["a"]), float(line1["b"]))
    d2 = _line_direction_from_abc(float(line2["a"]), float(line2["b"]))
    cos_a = float(abs(np.clip(np.dot(d1, d2), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cos_a)))


def _line_parallel_dist(line1: dict, line2: dict) -> float:
    """For near-parallel normalized lines, |c1-c2| approximates spacing."""
    c1 = float(_canonical_line_abc(line1)["c"])
    c2 = float(_canonical_line_abc(line2)["c"])
    return abs(c1 - c2)


def _anchor_distance(p1: dict, p2: dict) -> float:
    return float(np.hypot(float(p1["x"]) - float(p2["x"]), float(p1["y"]) - float(p2["y"])))


def _plane_duplicate(
    p1: dict,
    p2: dict,
    image_width: int,
    image_height: int,
) -> tuple[bool, dict]:
    """Rule-based duplicate check for planes built from (V,H) pairs."""
    v1 = p1.get("vertical_line")
    v2 = p2.get("vertical_line")
    h1 = p1.get("horizontal_line")
    h2 = p2.get("horizontal_line")
    a1 = p1.get("anchor")
    a2 = p2.get("anchor")
    if v1 is None or v2 is None or h1 is None or h2 is None or a1 is None or a2 is None:
        return False, {"reason": "missing_components"}

    v_angle = _line_angle_delta_deg(v1, v2)
    h_angle = _line_angle_delta_deg(h1, h2)
    v_dist = _line_parallel_dist(v1, v2)
    h_dist = _line_parallel_dist(h1, h2)
    anc_dist = _anchor_distance(a1, a2)

    tau_v_angle = 8.0
    tau_h_angle = 12.0
    tau_v_dist = max(6.0, 0.014 * float(image_width))
    tau_h_dist = max(8.0, 0.016 * float(image_height))
    tau_anchor = max(16.0, 0.03 * float(max(image_width, image_height)))
    tau_x_center = max(14.0, 0.028 * float(image_width))
    x_center_dist = abs(float(p1.get("x_center", 0.0)) - float(p2.get("x_center", 0.0)))

    # Anchor is unstable (paired with different horizontal lines), so it is soft.
    core_similar = (
        v_angle <= tau_v_angle
        and v_dist <= tau_v_dist
        and x_center_dist <= tau_x_center
    )
    horiz_compatible = (h_angle <= tau_h_angle and h_dist <= tau_h_dist)
    anchor_compatible = anc_dist <= tau_anchor
    is_dup = core_similar and (horiz_compatible or anchor_compatible)
    return is_dup, {
        "v_angle": round(v_angle, 3),
        "h_angle": round(h_angle, 3),
        "v_dist": round(v_dist, 3),
        "h_dist": round(h_dist, 3),
        "anchor_dist": round(anc_dist, 3),
        "tau_v_angle": tau_v_angle,
        "tau_h_angle": tau_h_angle,
        "tau_v_dist": round(tau_v_dist, 3),
        "tau_h_dist": round(tau_h_dist, 3),
        "tau_anchor": round(tau_anchor, 3),
        "x_center_dist": round(x_center_dist, 3),
        "tau_x_center": round(tau_x_center, 3),
        "core_similar": bool(core_similar),
        "horiz_compatible": bool(horiz_compatible),
        "anchor_compatible": bool(anchor_compatible),
    }


def _merge_similar_planes(
    raw_planes: list[dict],
    image_width: int,
    image_height: int,
) -> tuple[list[dict], dict]:
    """Cluster duplicate planes and merge each cluster into one canonical plane."""
    dbg: dict = {
        "raw_total": int(len(raw_planes)),
        "clusters_total": 0,
        "removed_duplicates": 0,
        "merge_clusters": [],
        "pair_checks": 0,
    }
    if not raw_planes:
        return [], dbg

    # 1) Pre-cluster by vertical plane signature to avoid fragmented duplicates.
    pre_items: list[dict] = []
    for idx, p in enumerate(raw_planes):
        v = _canonical_line_abc(p["vertical_line"])
        pre_items.append(
            {
                "idx": idx,
                "x": float(p.get("x_center", 0.0)),
                "c": float(v["c"]),
            }
        )
    pre_items.sort(key=lambda r: r["x"])
    pre_clusters: list[list[int]] = []
    x_thr = max(16.0, 0.03 * float(image_width))
    c_thr = max(8.0, 0.014 * float(image_width))
    cur: list[int] = []
    cur_x = 0.0
    cur_c = 0.0
    for rec in pre_items:
        if not cur:
            cur = [int(rec["idx"])]
            cur_x = float(rec["x"])
            cur_c = float(rec["c"])
            continue
        if abs(float(rec["x"]) - cur_x) <= x_thr and abs(float(rec["c"]) - cur_c) <= c_thr:
            cur.append(int(rec["idx"]))
            cur_x = float(np.mean([float(raw_planes[i].get("x_center", 0.0)) for i in cur]))
            cur_c = float(np.mean([float(_canonical_line_abc(raw_planes[i]["vertical_line"])["c"]) for i in cur]))
        else:
            pre_clusters.append(cur)
            cur = [int(rec["idx"])]
            cur_x = float(rec["x"])
            cur_c = float(rec["c"])
    if cur:
        pre_clusters.append(cur)

    # 2) Refine each pre-cluster by duplicate predicate (union-find style BFS).
    clusters: list[list[int]] = []
    for pre in pre_clusters:
        visited_local: set[int] = set()
        for i in pre:
            if i in visited_local:
                continue
            queue = [i]
            visited_local.add(i)
            comp = [i]
            while queue:
                u = queue.pop()
                for v in pre:
                    if v in visited_local:
                        continue
                    dbg["pair_checks"] = int(dbg["pair_checks"]) + 1
                    dup, _ = _plane_duplicate(raw_planes[u], raw_planes[v], image_width, image_height)
                    if dup:
                        visited_local.add(v)
                        queue.append(v)
                        comp.append(v)
            clusters.append(comp)

    merged: list[dict] = []
    for cid, ids in enumerate(clusters, start=1):
        pls = [raw_planes[i] for i in ids]
        weights = np.array([max(1e-6, float(p.get("support", 1.0))) for p in pls], dtype=np.float64)
        wsum = float(np.sum(weights))
        if wsum <= 1e-9:
            weights = np.ones(len(pls), dtype=np.float64)
            wsum = float(len(pls))

        def _weighted_line(key: str) -> dict:
            canon = [_canonical_line_abc(p[key]) for p in pls]
            a = float(np.sum([canon[k]["a"] * weights[k] for k in range(len(pls))]) / wsum)
            b = float(np.sum([canon[k]["b"] * weights[k] for k in range(len(pls))]) / wsum)
            c = float(np.sum([canon[k]["c"] * weights[k] for k in range(len(pls))]) / wsum)
            return _canonical_line_abc({"a": a, "b": b, "c": c})

        anchors_x = np.array([float(p["anchor"]["x"]) for p in pls], dtype=np.float64)
        anchors_y = np.array([float(p["anchor"]["y"]) for p in pls], dtype=np.float64)
        anchor = {
            "x": float(np.sum(anchors_x * weights) / wsum),
            "y": float(np.sum(anchors_y * weights) / wsum),
        }

        v_line = _weighted_line("vertical_line")
        h_line = _weighted_line("horizontal_line")
        merged.append(
            {
                "id": int(cid),
                "family_id": int(cid),
                "vertical_line": v_line,
                "horizontal_line": h_line,
                "line": v_line,
                "anchor": anchor,
                "support": float(np.sum([float(p.get("support", 0.0)) for p in pls])),
                "n_segments": int(np.sum([int(p.get("n_segments", 0)) for p in pls])),
                "x_center": float(np.sum([float(p.get("x_center", 0.0)) * weights[k] for k, p in enumerate(pls)]) / wsum),
                "y_center": float(np.sum([float(p.get("y_center", 0.0)) * weights[k] for k, p in enumerate(pls)]) / wsum),
                "raw_plane_ids": [int(raw_planes[i].get("id", i + 1)) for i in ids],
                "raw_count": int(len(ids)),
            }
        )
        dbg["merge_clusters"].append(
            {
                "cluster_id": int(cid),
                "raw_ids": [int(raw_planes[i].get("id", i + 1)) for i in ids],
                "raw_count": int(len(ids)),
            }
        )

    # Secondary pass: merge planes still close in vertical signature.
    if len(merged) > 1:
        merged.sort(key=lambda p: float(p.get("x_center", 0.0)))
        second_clusters: list[list[dict]] = []
        x_thr2 = max(18.0, 0.035 * float(image_width))
        c_thr2 = max(10.0, 0.016 * float(image_width))
        for p in merged:
            if not second_clusters:
                second_clusters.append([p])
                continue
            last = second_clusters[-1][-1]
            dx = abs(float(p.get("x_center", 0.0)) - float(last.get("x_center", 0.0)))
            cv = abs(
                float(_canonical_line_abc(p["vertical_line"])["c"])
                - float(_canonical_line_abc(last["vertical_line"])["c"])
            )
            if dx <= x_thr2 and cv <= c_thr2:
                dup, _ = _plane_duplicate(last, p, image_width, image_height)
                if dup:
                    second_clusters[-1].append(p)
                    continue
            second_clusters.append([p])

        if len(second_clusters) != len(merged):
            rem2: list[dict] = []
            for cid, grp in enumerate(second_clusters, start=1):
                if len(grp) == 1:
                    item = grp[0].copy()
                    item["id"] = int(cid)
                    item["family_id"] = int(cid)
                    rem2.append(item)
                    continue
                weights = np.array([max(1e-6, float(g.get("support", 1.0))) for g in grp], dtype=np.float64)
                wsum = float(np.sum(weights))
                if wsum <= 1e-9:
                    weights = np.ones(len(grp), dtype=np.float64)
                    wsum = float(len(grp))

                def _merge_line(key: str) -> dict:
                    cls = [_canonical_line_abc(g[key]) for g in grp]
                    a = float(np.sum([cls[k]["a"] * weights[k] for k in range(len(grp))]) / wsum)
                    b = float(np.sum([cls[k]["b"] * weights[k] for k in range(len(grp))]) / wsum)
                    c = float(np.sum([cls[k]["c"] * weights[k] for k in range(len(grp))]) / wsum)
                    return _canonical_line_abc({"a": a, "b": b, "c": c})

                anchor_x = float(np.sum([float(g["anchor"]["x"]) * weights[k] for k, g in enumerate(grp)]) / wsum)
                anchor_y = float(np.sum([float(g["anchor"]["y"]) * weights[k] for k, g in enumerate(grp)]) / wsum)
                rem2.append(
                    {
                        "id": int(cid),
                        "family_id": int(cid),
                        "vertical_line": _merge_line("vertical_line"),
                        "horizontal_line": _merge_line("horizontal_line"),
                        "line": _merge_line("vertical_line"),
                        "anchor": {"x": anchor_x, "y": anchor_y},
                        "support": float(np.sum([float(g.get("support", 0.0)) for g in grp])),
                        "n_segments": int(np.sum([int(g.get("n_segments", 0)) for g in grp])),
                        "x_center": float(np.sum([float(g.get("x_center", 0.0)) * weights[k] for k, g in enumerate(grp)]) / wsum),
                        "y_center": float(np.sum([float(g.get("y_center", 0.0)) * weights[k] for k, g in enumerate(grp)]) / wsum),
                        "raw_plane_ids": [rid for g in grp for rid in (g.get("raw_plane_ids") or [int(g.get("id", 0))])],
                        "raw_count": int(np.sum([int(g.get("raw_count", 1)) for g in grp])),
                    }
                )
            dbg["secondary_merge_clusters_total"] = int(len(second_clusters))
            dbg["secondary_removed"] = int(len(merged) - len(rem2))
            merged = rem2
        else:
            dbg["secondary_merge_clusters_total"] = int(len(second_clusters))
            dbg["secondary_removed"] = 0

    dbg["clusters_total"] = int(len(clusters))
    dbg["removed_duplicates"] = int(len(raw_planes) - len(merged))
    dbg["merged_total"] = int(len(merged))
    return merged, dbg


def _dedupe_intersections(points: list[dict], image_width: int, image_height: int) -> tuple[list[dict], dict]:
    """Spatial NMS for intersection points."""
    dbg = {"raw_total": int(len(points)), "dedup_total": 0, "removed": 0}
    if not points:
        return [], dbg
    gap = max(8.0, 0.015 * float(max(image_width, image_height)))
    out: list[dict] = []
    for p in sorted(points, key=lambda r: float(r.get("roi_ratio", 0.0)), reverse=True):
        x = float(p["x"])
        y = float(p["y"])
        if any(abs(x - float(q["x"])) < gap and abs(y - float(q["y"])) < gap for q in out):
            continue
        out.append(p)
    dbg["dedup_total"] = int(len(out))
    dbg["removed"] = int(len(points) - len(out))
    return out, dbg


def _intersections_from_planes(
    planes: list[dict],
    band_mask_bin: np.ndarray,
    image_width: int,
    image_height: int,
    neighbor_radius: int,
) -> tuple[list[dict], int]:
    """Find intersections between planes built from V-H pairs."""
    h, w = band_mask_bin.shape[:2]
    results: list[dict] = []
    pairs_checked = 0
    ordered = sorted(planes, key=lambda p: float(p.get("x_center", 0.0)))
    # Intersect only neighboring planes by x-order to avoid combinatorial duplicates.
    for i in range(len(ordered) - 1):
        for j in (i + 1,):
            p1 = ordered[i]
            p2 = ordered[j]
            v1 = p1.get("vertical_line")
            h1_line = p1.get("horizontal_line")
            v2 = p2.get("vertical_line")
            h2_line = p2.get("horizontal_line")
            if v1 is None or h1_line is None or v2 is None or h2_line is None:
                continue
            pairs_checked += 1
            cross_points = [
                _line_intersection(v1, h2_line),
                _line_intersection(v2, h1_line),
                _line_intersection(v1, v2),
            ]
            for pt in cross_points:
                if pt is None:
                    continue
                x, y = pt
                if not (0 <= x < image_width and 0 <= y < image_height):
                    continue
                r = max(1, int(neighbor_radius))
                x0 = max(0, int(round(x)) - r)
                y0 = max(0, int(round(y)) - r)
                x1c = min(w, int(round(x)) + r + 1)
                y1c = min(h, int(round(y)) + r + 1)
                patch = band_mask_bin[y0:y1c, x0:x1c]
                roi_ratio = float(np.count_nonzero(patch)) / max(1, patch.size)
                if roi_ratio < 0.06:
                    continue
                results.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "plane_i": int(p1["id"]),
                        "plane_j": int(p2["id"]),
                        "family_i": int(p1["family_id"]),
                        "family_j": int(p2["family_id"]),
                        "angle": 90.0,
                        "roi_ratio": round(float(roi_ratio), 3),
                    }
                )
    return results, pairs_checked


def _fit_line_from_cluster_endpoints(cluster_segments: list[dict]) -> dict | None:
    """Fit one support line to a merged lower-band segment cluster."""
    if len(cluster_segments) == 1:
        s = cluster_segments[0]
        abc = _segment_line_abc(s)
        if abc is None:
            return None
        a, b, c = abc
        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "support": float(s.get("length", 0.0)),
            "n_segments": 1,
            "y_mean": round(0.5 * (float(s["y1"]) + float(s["y2"])), 2),
        }

    pts = np.array(
        [[s["x1"], s["y1"]] for s in cluster_segments] + [[s["x2"], s["y2"]] for s in cluster_segments],
        dtype=np.float32,
    )
    if len(pts) < 4:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    a = float(vy[0])
    b = float(-vx[0])
    c = float(vx[0] * y0[0] - vy[0] * x0[0])
    norm = float(np.sqrt(a * a + b * b))
    if norm < 1e-8:
        return None
    a /= norm
    b /= norm
    c /= norm
    support = sum(float(s["length"]) for s in cluster_segments)
    return {
        "a": a,
        "b": b,
        "c": c,
        "support": float(support),
        "n_segments": len(cluster_segments),
        "y_mean": round(float(np.mean([float(s.get("y_mean", 0.0)) for s in cluster_segments])), 2),
    }


def _intersections_from_lines(
    wall_lines: list[dict],
    band_mask_bin: np.ndarray,
    image_width: int,
    image_height: int,
    neighbor_radius: int,
) -> list[dict]:
    """Find intersections between lower-band support lines."""
    h, w = band_mask_bin.shape[:2]
    results: list[dict] = []
    for i in range(len(wall_lines)):
        for j in range(i + 1, len(wall_lines)):
            l1 = wall_lines[i]
            l2 = wall_lines[j]
            denom = l1["a"] * l2["b"] - l2["a"] * l1["b"]
            if abs(denom) < 1e-6:
                continue
            x = (l1["b"] * l2["c"] - l2["b"] * l1["c"]) / denom
            y = (l2["a"] * l1["c"] - l1["a"] * l2["c"]) / denom
            if not (0 <= x < image_width and 0 <= y < image_height):
                continue
            r = max(1, neighbor_radius)
            x0 = max(0, int(round(x)) - r)
            y0 = max(0, int(round(y)) - r)
            x1c = min(w, int(round(x)) + r + 1)
            y1c = min(h, int(round(y)) + r + 1)
            patch = band_mask_bin[y0:y1c, x0:x1c]
            roi_ratio = float(np.count_nonzero(patch)) / max(1, patch.size)
            if roi_ratio < 0.12:
                continue
            results.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "wall_i": i,
                    "wall_j": j,
                    "roi_ratio": round(roi_ratio, 3),
                }
            )
    return results


def _corner_intersections_to_split_x(intersections: list[dict], image_width: int, wall_mask_bin: np.ndarray) -> list[float]:
    """Convert intersections to candidate split-X values without dropping edge candidates."""
    xs: list[float] = []
    for inter in intersections:
        x = float(inter["x"])
        if 0.0 <= x <= float(max(0, image_width - 1)):
            xs.append(x)
    return sorted(xs)


def _nms_unique_x_candidates(xs: list[float], image_width: int) -> list[float]:
    """Deduplicate candidate X values by minimum gap."""
    if not xs:
        return []
    min_gap = max(6.0, EXTERIOR_LSD_NMS_GAP_RATIO * float(image_width))
    xs_sorted = sorted(xs)
    result: list[float] = [xs_sorted[0]]
    for x in xs_sorted[1:]:
        if x - result[-1] >= min_gap:
            result.append(x)
    return result


def _apply_multiple_vertical_splits(
    wall_mask_bin: np.ndarray,
    split_xs: list[float],
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[np.ndarray]:
    """Apply multiple vertical split lines sequentially, returning final component masks."""
    components: list[np.ndarray] = [wall_mask_bin.copy()]
    for x in sorted(split_xs):
        pt_a = np.array([x, 0.0], dtype=np.float64)
        pt_b = np.array([x, float(image_height - 1)], dtype=np.float64)
        new_components: list[np.ndarray] = []
        for comp in components:
            split_result = _component_masks_from_split_line(comp, pt_a, pt_b, min_area)
            if split_result is not None and len(split_result) >= 2:
                new_components.extend(split_result)
            else:
                new_components.append(comp)
            if len(new_components) >= EXTERIOR_MAX_WALLS:
                break
        components = new_components
        if len(components) >= EXTERIOR_MAX_WALLS:
            break
    return components


def _split_by_rgb_lsd_corners(
    wall_mask_bin: np.ndarray,
    image_bgr: np.ndarray | None,
    image_width: int,
    image_height: int,
    min_area: int,
    wall_minus_holes: np.ndarray = None,
) -> tuple[list[dict] | None, dict]:
    """Bottom-band RGB+LSD cascade: use only stable lower facade lines and their intersections."""
    dbg: dict = {"enabled": True, "used": False, "method": "rgb_lsd_bottom_band_xacc_peaks"}
    if image_bgr is None:
        dbg["error"] = "image_bgr is None"
        return None, dbg

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask_bin = (wall_mask_bin > 0).astype(np.uint8) * 255
    profile = _extract_bottom_profile(mask_bin)
    if not bool(profile.get("valid")):
        dbg["error"] = "bottom profile is empty"
        return None, dbg
    band_mask = _build_bottom_band_mask(mask_bin, profile, EXTERIOR_LSD_BOTTOM_BAND_RATIO)
    dbg["bottom_band_nonzero"] = int(np.count_nonzero(band_mask))
    dbg["bottom_band_ratio"] = round(float(EXTERIOR_LSD_BOTTOM_BAND_RATIO), 3)

    # Stage 1: LSD only inside the adaptive lower band.
    segments = _lsd_segments_in_bottom_band(gray, band_mask, profile, EXTERIOR_LSD_BOTTOM_NEAR_RATIO, wall_minus_holes)
    dbg["lsd_segments_total"] = len(segments)
    if len(segments) < 4:
        dbg["error"] = f"too few LSD segments: {len(segments)}"
        return None, dbg

    # Stage 2: split lower-band LSD segments by orientation.
    horiz_segments, vert_segments, vps = _split_segments_horizontal_vertical(segments)
    dbg["direction_families_total"] = 2 if (horiz_segments or vert_segments) else 0
    dbg["clusters_total"] = int((1 if horiz_segments else 0) + (1 if vert_segments else 0))
    dbg["direction_families_detail"] = [
        {
            "name": "horizontal",
            "n_segments": len(horiz_segments),
            "total_length": round(sum(float(s.get("length", 0.0)) for s in horiz_segments), 1),
            "y_mean": round(float(np.mean([float(s.get("y_mean", 0.0)) for s in horiz_segments])) if horiz_segments else 0.0, 2),
        },
        {
            "name": "vertical",
            "n_segments": len(vert_segments),
            "total_length": round(sum(float(s.get("length", 0.0)) for s in vert_segments), 1),
            "y_mean": round(float(np.mean([float(s.get("y_mean", 0.0)) for s in vert_segments])) if vert_segments else 0.0, 2),
        },
    ]
    if len(vert_segments) < 2:
        dbg["error"] = f"too few vertical segments: {len(vert_segments)}"
        return None, dbg

    # Stage 3: dominant vertical axes through weighted 1D x-accumulator.
    xacc = _build_vertical_x_accumulator(vert_segments, image_width, vps)
    raw_acc = xacc["raw_acc"]
    total_support = float(xacc["total_support"])
    sigma_px = max(2.0, EXTERIOR_LSD_XACC_SIGMA_RATIO * float(image_width))
    min_dist_px = max(8, int(round(EXTERIOR_LSD_XACC_MIN_PEAK_DIST_RATIO * float(image_width))))
    smooth_acc = _smooth_x_accumulator(raw_acc, sigma_px)
    max_acc = float(np.max(smooth_acc)) if smooth_acc.size > 0 else 0.0
    min_prom_abs = max(1.0, EXTERIOR_LSD_XACC_MIN_PROM_FRAC * max_acc)
    peaks_raw = _detect_x_peaks(smooth_acc, min_dist_px=min_dist_px, min_prom_abs=min_prom_abs)
    peaks_raw = _filter_peaks_by_contour_kinks(peaks_raw, profile, image_width)
    split_x_raw, kept_peaks_diag = _peaks_to_split_candidates(
        peaks_raw,
        smooth_acc,
        image_width,
        mask_bin,
        total_support=total_support,
    )
    dbg["x_acc_debug"] = {
        "vert_segments_total": int(len(vert_segments)),
        "xacc_sigma_px": round(float(sigma_px), 3),
        "xacc_sum_raw": round(float(np.sum(raw_acc)), 3),
        "xacc_sum_smooth": round(float(np.sum(smooth_acc)), 3),
        "peak_count_raw": int(len(peaks_raw)),
        "peak_count_kept": int(len(split_x_raw)),
        "min_dist_px": int(min_dist_px),
        "min_prom_abs": round(float(min_prom_abs), 3),
        "kept_peaks": kept_peaks_diag,
    }
    dbg["plane_pairs_checked"] = 0
    dbg["intersections_raw_total"] = int(len(peaks_raw))
    dbg["intersection_dedupe_debug"] = {
        "raw_total": int(len(peaks_raw)),
        "dedup_total": int(len(split_x_raw)),
        "removed": int(max(0, len(peaks_raw) - len(split_x_raw))),
    }
    dbg["intersections_total"] = int(len(split_x_raw))
    dbg["intersections"] = [{"x": round(float(x), 2), "y": 0.0} for x in split_x_raw]

    # Preserve overlays even when we fail later, so debug PNGs are still saved.
    dbg["_segments_for_overlay"] = segments
    dbg["_band_mask_for_overlay"] = band_mask
    dbg["_planes_for_overlay"] = []
    dbg["_x_peaks_raw_for_overlay"] = peaks_raw
    dbg["_x_peaks_kept_for_overlay"] = split_x_raw

    if not split_x_raw:
        dbg["error"] = "no dominant vertical peaks"
        return None, dbg

    # Stage 4: convert peaks to split-x candidates, including edge candidates.
    raw_xs = split_x_raw
    dbg["split_x_candidates_raw"] = [round(x, 2) for x in raw_xs]
    xs_nms = _nms_unique_x_candidates(raw_xs, image_width)
    dbg["split_x_candidates_after_nms"] = [round(x, 2) for x in xs_nms]
    if not xs_nms:
        dbg["error"] = "no split-x candidates after NMS"
        return None, dbg

    # Stage 6: apply multiple splits
    components = _apply_multiple_vertical_splits(mask_bin, xs_nms, image_width, image_height, min_area)
    dbg["components_after_split"] = len(components)
    if len(components) < 2:
        dbg["error"] = f"split produced {len(components)} components (need >=2)"
        return None, dbg

    walls = _build_walls_from_component_masks(components)
    dbg["walls_built_count"] = len(walls)
    dbg["multiple_splits_used"] = True
    dbg["used"] = True

    # Preserve debug data for image overlays.
    dbg["_intersections_for_overlay"] = [{"x": float(x), "y": 0.0} for x in split_x_raw]
    dbg["_split_xs_for_overlay"] = xs_nms

    return walls, dbg


def _split_by_multiscale_candidates(
    wall_mask: np.ndarray,
    image_width: int,
    image_height: int,
    min_area: int,
    top_k: int = 6,
) -> tuple[list[dict] | None, dict]:
    """
    Relative top-k candidate split (no fixed confidence tiers).
    Tries strongest vertical cut hypotheses and selects the best by resulting wall count + balance.
    """
    dbg: dict = {"enabled": True, "candidates": [], "selected": None, "used": False}
    cands = _multi_scale_corner_candidates(wall_mask)
    dbg["candidates_total"] = len(cands)
    if not cands:
        return None, dbg
    cands = cands[: max(1, int(top_k))]
    dbg["candidates"] = cands
    best_walls: list[dict] | None = None
    best_key = (-1, -1.0)
    best_sel: dict | None = None
    for c in cands:
        x = float(c["x"])
        pt_a = np.array([x, 0.0], dtype=np.float64)
        pt_b = np.array([x, float(image_height - 1)], dtype=np.float64)
        comps = _component_masks_from_split_line(wall_mask, pt_a, pt_b, min_area)
        if comps is None:
            continue
        walls = _build_walls_from_component_masks(comps)
        if len(walls) < 2:
            continue
        areas = [int(cv2.countNonZero(cm)) for cm in comps]
        balance = float(min(areas) / max(areas)) if areas and max(areas) > 0 else 0.0
        key = (len(walls), balance + 1e-3 * float(c["score"]))
        if key > best_key:
            best_key = key
            best_walls = walls
            best_sel = {"x": x, "walls_count": len(walls), "balance": round(balance, 4), "score": c["score"]}
    dbg["selected"] = best_sel
    dbg["used"] = best_walls is not None
    return best_walls, dbg


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


EXTERIOR_MIN_WALL_AREA_RATIO = 0.004
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
# TEMP: forced off to validate non-seam geometry cascade behavior.
EXTERIOR_SEAM_ENABLE = False
EXTERIOR_SEAM_HOUGH_ENABLE = os.getenv("EXTERIOR_SEAM_HOUGH_ENABLE", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_SEAM_ANGLE_THR_DEG = float(os.getenv("EXTERIOR_SEAM_ANGLE_THR_DEG", "12"))
EXTERIOR_SEAM_MIN_LINE_RATIO = float(os.getenv("EXTERIOR_SEAM_MIN_LINE_RATIO", "0.40"))
EXTERIOR_SEAM_CENTER_MARGIN_RATIO = float(os.getenv("EXTERIOR_SEAM_CENTER_MARGIN_RATIO", "0.10"))
EXTERIOR_SEAM_CONTOUR_CENTER_MARGIN_RATIO = float(
    os.getenv("EXTERIOR_SEAM_CONTOUR_CENTER_MARGIN_RATIO", "0.02")
)
EXTERIOR_SEAM_CONTOUR_EDGE_EXCLUDE_RATIO = float(
    os.getenv("EXTERIOR_SEAM_CONTOUR_EDGE_EXCLUDE_RATIO", "0.05")
)
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
EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG = float(os.getenv("EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG", "40.0"))
EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO = float(os.getenv("EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO", "0.08"))
EXTERIOR_SEAM_CONTOUR_MAX_K = int(os.getenv("EXTERIOR_SEAM_CONTOUR_MAX_K", "6"))
EXTERIOR_SEAM_CONTOUR_BONUS_BOTTOM = float(os.getenv("EXTERIOR_SEAM_CONTOUR_BONUS_BOTTOM", "0.28"))
EXTERIOR_SEAM_CONTOUR_BONUS_TOP = float(os.getenv("EXTERIOR_SEAM_CONTOUR_BONUS_TOP", "0.10"))
EXTERIOR_SEAM_CONTOUR_X_TOL_RATIO = float(os.getenv("EXTERIOR_SEAM_CONTOUR_X_TOL_RATIO", "0.03"))
EXTERIOR_SEAM_PROFILE_ENABLE = os.getenv("EXTERIOR_SEAM_PROFILE_ENABLE", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
EXTERIOR_SEAM_PROFILE_MEDIAN_K = int(os.getenv("EXTERIOR_SEAM_PROFILE_MEDIAN_K", "9"))
EXTERIOR_SEAM_PROFILE_WINDOW = int(os.getenv("EXTERIOR_SEAM_PROFILE_WINDOW", "12"))
EXTERIOR_SEAM_PROFILE_PROMINENCE_PX = float(os.getenv("EXTERIOR_SEAM_PROFILE_PROMINENCE_PX", "6.0"))
EXTERIOR_SEAM_PROFILE_MIN_SEG_LEN_PX = int(os.getenv("EXTERIOR_SEAM_PROFILE_MIN_SEG_LEN_PX", "10"))
EXTERIOR_SEAM_PROFILE_NMS_X_GAP_RATIO = float(os.getenv("EXTERIOR_SEAM_PROFILE_NMS_X_GAP_RATIO", "0.02"))
EXTERIOR_SEAM_PROFILE_MIN_SCALE_HITS = int(os.getenv("EXTERIOR_SEAM_PROFILE_MIN_SCALE_HITS", "2"))
EXTERIOR_SEAM_PROFILE_RDP_EPS_PX = float(os.getenv("EXTERIOR_SEAM_PROFILE_RDP_EPS_PX", "7.0"))
EXTERIOR_SEAM_PROFILE_QUALITY_MIN = float(os.getenv("EXTERIOR_SEAM_PROFILE_QUALITY_MIN", "0.45"))
EXTERIOR_SEAM_PROFILE_QUALITY_HIGH = float(os.getenv("EXTERIOR_SEAM_PROFILE_QUALITY_HIGH", "0.65"))
EXTERIOR_SEAM_PROFILE_STABILITY_WIN = int(os.getenv("EXTERIOR_SEAM_PROFILE_STABILITY_WIN", "18"))
EXTERIOR_SEAM_PROFILE_SLOPE_EPS_PX = float(os.getenv("EXTERIOR_SEAM_PROFILE_SLOPE_EPS_PX", "0.8"))
EXTERIOR_SEAM_PROFILE_SIGN_FLIPS_REF = float(os.getenv("EXTERIOR_SEAM_PROFILE_SIGN_FLIPS_REF", "4.0"))
EXTERIOR_SEAM_PROFILE_SEG_REF_PX = float(os.getenv("EXTERIOR_SEAM_PROFILE_SEG_REF_PX", "36.0"))
EXTERIOR_SEAM_PROFILE_PROM_REF_PX = float(os.getenv("EXTERIOR_SEAM_PROFILE_PROM_REF_PX", "14.0"))
EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_RATIO = float(os.getenv("EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_RATIO", "0.035"))
EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_MIN_PX = float(os.getenv("EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_MIN_PX", "16.0"))
EXTERIOR_SEAM_PROFILE_TANDEM_MIN_NEIGHBOR_GAP_RATIO = float(
    os.getenv("EXTERIOR_SEAM_PROFILE_TANDEM_MIN_NEIGHBOR_GAP_RATIO", "0.045")
)
EXTERIOR_DEBUG_KINK_LABEL_TOP_K = int(os.getenv("EXTERIOR_DEBUG_KINK_LABEL_TOP_K", "30"))
EXTERIOR_DEBUG_KINK_LABEL_MIN_GAP_PX = int(os.getenv("EXTERIOR_DEBUG_KINK_LABEL_MIN_GAP_PX", "14"))
EXTERIOR_SEAM_TANDEM_X_TOL_RATIO = float(os.getenv("EXTERIOR_SEAM_TANDEM_X_TOL_RATIO", "0.04"))
EXTERIOR_SEAM_TANDEM_BOOST = float(os.getenv("EXTERIOR_SEAM_TANDEM_BOOST", "0.12"))
EXTERIOR_SEAM_BOTTOM_CONF_MIN = float(os.getenv("EXTERIOR_SEAM_BOTTOM_CONF_MIN", "0.42"))
EXTERIOR_SEAM_TOP_FALLBACK_CONF_MIN = float(os.getenv("EXTERIOR_SEAM_TOP_FALLBACK_CONF_MIN", "0.46"))
EXTERIOR_SEAM_CORNER_SNAP_TOL_RATIO = float(os.getenv("EXTERIOR_SEAM_CORNER_SNAP_TOL_RATIO", "0.02"))
EXTERIOR_SEAM_SECOND_PASS = os.getenv("EXTERIOR_SEAM_SECOND_PASS", "1") in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}
def _line_verticality(pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """1.0 — почти вертикальная линия, 0.0 — почти горизонтальная."""
    dx = float(pt_b[0] - pt_a[0])
    dy = float(pt_b[1] - pt_a[1])
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return 0.0
    return abs(dy) / n


def _interior_x_bounds(mask_bin: np.ndarray, width: int) -> tuple[float, float]:
    """
    Возвращает допустимый внутренний диапазон X:
    от левой/правой границы маски отступаем margin, чтобы не брать края дома как изломы.
    """
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return 0.0, float(max(0, width - 1))
    x_min = float(xs.min())
    x_max = float(xs.max())
    span = max(1.0, x_max - x_min)
    margin = max(6.0, EXTERIOR_SEAM_CONTOUR_EDGE_EXCLUDE_RATIO * span)
    left = x_min + margin
    right = x_max - margin
    if right <= left:
        # На очень узких масках ослабляем фильтр, чтобы не выкинуть все кандидаты.
        left = x_min + 2.0
        right = x_max - 2.0
    return left, right


def _median_filter_1d(vals: np.ndarray, k: int) -> np.ndarray:
    if vals.size == 0:
        return vals
    kk = max(1, int(k))
    if kk % 2 == 0:
        kk += 1
    if kk <= 1:
        return vals.astype(np.float64, copy=True)
    pad = kk // 2
    x = vals.astype(np.float64, copy=False)
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = float(np.median(xp[i : i + kk]))
    return out


def _profile_kink_candidates(
    mask_bin: np.ndarray,
    bw: int,
    bh: int,
    zone: str,
    x_left_interior: float,
    x_right_interior: float,
) -> list[dict]:
    cands, _ = _profile_kink_candidates_with_debug(
        mask_bin, bw, bh, zone, x_left_interior, x_right_interior
    )
    return cands


def _profile_kink_candidates_with_debug(
    mask_bin: np.ndarray,
    bw: int,
    bh: int,
    zone: str,
    x_left_interior: float,
    x_right_interior: float,
) -> tuple[list[dict], dict]:
    if zone not in {"bottom", "top"}:
        return [], {"zone": zone}
    ys = np.full((bw,), np.nan, dtype=np.float64)
    for x in range(bw):
        idx = np.where(mask_bin[:, x] > 0)[0]
        if idx.size == 0:
            continue
        ys[x] = float(idx[-1] if zone == "bottom" else idx[0])

    valid = np.where(np.isfinite(ys))[0]
    if valid.size < 24:
        return [], {"zone": zone, "valid_count": int(valid.size)}

    # Интерполируем пропуски по X, затем сглаживаем профиль.
    x_all = np.arange(bw, dtype=np.float64)
    ys_i = ys.copy()
    ys_i[~np.isfinite(ys_i)] = np.interp(x_all[~np.isfinite(ys_i)], x_all[valid], ys[valid])
    ys_s = _median_filter_1d(ys_i, EXTERIOR_SEAM_PROFILE_MEDIAN_K)

    # Кусочно-линейное выравнивание профиля перед поиском изломов.
    pts = np.stack([x_all, ys_s], axis=1).astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, EXTERIOR_SEAM_PROFILE_RDP_EPS_PX, False)
    piece = approx.reshape(-1, 2).astype(np.float64)
    if piece.shape[0] < 3:
        return [], {
            "zone": zone,
            "raw_profile": np.stack([x_all, ys], axis=1).tolist(),
            "smoothed_profile": np.stack([x_all, ys_s], axis=1).tolist(),
            "piecewise_profile": piece.tolist(),
            "kinks": [],
        }

    base_bonus = EXTERIOR_SEAM_CONTOUR_BONUS_BOTTOM if zone == "bottom" else EXTERIOR_SEAM_CONTOUR_BONUS_TOP
    cands_raw: list[dict] = []

    def _sign_flips_near(xf: float) -> int:
        xi = int(round(xf))
        r = max(6, int(EXTERIOR_SEAM_PROFILE_STABILITY_WIN))
        lo = max(0, xi - r)
        hi = min(bw - 1, xi + r)
        if hi - lo < 4:
            return 0
        seg = ys_s[lo : hi + 1]
        d = np.diff(seg)
        if d.size < 3:
            return 0
        eps = float(max(0.1, EXTERIOR_SEAM_PROFILE_SLOPE_EPS_PX))
        d = d[np.abs(d) >= eps]
        if d.size < 3:
            return 0
        s = np.sign(d)
        return int(np.count_nonzero(s[1:] * s[:-1] < 0))

    for i in range(1, piece.shape[0] - 1):
        p_prev = piece[i - 1]
        p = piece[i]
        p_next = piece[i + 1]
        x = float(p[0])
        y = float(p[1])
        if x <= x_left_interior or x >= x_right_interior:
            continue
        v1 = p - p_prev
        v2 = p_next - p
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < float(EXTERIOR_SEAM_PROFILE_MIN_SEG_LEN_PX) or n2 < float(EXTERIOR_SEAM_PROFILE_MIN_SEG_LEN_PX):
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_a)))
        delta = 180.0 - angle
        if delta < EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG:
            continue
        y_norm = y / max(1.0, float(bh - 1))
        if zone == "bottom":
            if y_norm < EXTERIOR_SEAM_CONTOUR_BOTTOM_Y_MIN:
                continue
        else:
            if y_norm > EXTERIOR_SEAM_CONTOUR_TOP_Y_MAX:
                continue

        # Prominence: расстояние до хорды между соседними вершинами.
        chord = p_next - p_prev
        chord_n = float(np.linalg.norm(chord))
        if chord_n < 1e-6:
            continue
        prom = float(abs(np.cross(chord, p - p_prev)) / chord_n)
        if prom < EXTERIOR_SEAM_PROFILE_PROMINENCE_PX:
            continue

        delta_norm = float(np.clip(delta / 90.0, 0.0, 1.0))
        score_hint = base_bonus * (0.65 + 0.35 * delta_norm)
        cands_raw.append(
            {
                "mid_x_local": x,
                "y_local": y,
                "pt_a_local": np.array([x, 0.0], dtype=np.float64),
                "pt_b_local": np.array([x, float(bh - 1)], dtype=np.float64),
                "source_bonus": float(score_hint),
                "delta_norm": float(delta_norm),
                "source": f"profile_{zone}",
                "delta_deg": float(delta),
                "prominence_px": float(prom),
                "left_seg_len": float(n1),
                "right_seg_len": float(n2),
                "sign_flips": _sign_flips_near(x),
            }
        )

    cands_precluster: list[dict] = []
    if cands_raw:
        prom_min = max(1e-6, EXTERIOR_SEAM_PROFILE_PROMINENCE_PX)
        prom_ref = max(prom_min + 1e-6, EXTERIOR_SEAM_PROFILE_PROM_REF_PX)
        seg_ref = max(float(EXTERIOR_SEAM_PROFILE_MIN_SEG_LEN_PX), EXTERIOR_SEAM_PROFILE_SEG_REF_PX)
        ang_min = float(EXTERIOR_SEAM_CONTOUR_MIN_DELTA_DEG)
        ang_ref = max(ang_min + 1.0, 82.0)
        flips_ref = max(1.0, EXTERIOR_SEAM_PROFILE_SIGN_FLIPS_REF)

        for rec in cands_raw:
            prom = float(rec.get("prominence_px", 0.0))
            delta = float(rec.get("delta_deg", 0.0))
            seg_min = min(float(rec.get("left_seg_len", 0.0)), float(rec.get("right_seg_len", 0.0)))
            flips = float(rec.get("sign_flips", 0.0))

            p_score = float(np.clip((prom - prom_min) / max(1e-6, (prom_ref - prom_min)), 0.0, 1.0))
            a_score = float(np.clip((delta - ang_min) / max(1e-6, (ang_ref - ang_min)), 0.0, 1.0))
            l_score = float(np.clip(seg_min / max(1e-6, seg_ref), 0.0, 1.0))
            s_score = float(np.clip(1.0 - (flips / flips_ref), 0.0, 1.0))
            q = 0.35 * p_score + 0.25 * a_score + 0.20 * l_score + 0.20 * s_score
            if q < EXTERIOR_SEAM_PROFILE_QUALITY_MIN:
                continue

            tier = "high" if q >= EXTERIOR_SEAM_PROFILE_QUALITY_HIGH else "medium"
            rec["quality_score"] = float(q)
            rec["quality_tier"] = tier
            rec["source_bonus"] = float(rec.get("source_bonus", 0.0)) * (0.85 + 0.30 * q)
            cands_precluster.append(rec)

    # Сливаем близкие профильные точки в один структурный излом (не лимит, а кластеризация по X).
    cands: list[dict] = []
    if cands_precluster:
        gap_px = max(EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_MIN_PX, EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_RATIO * float(bw))
        by_x = sorted(cands_precluster, key=lambda r: float(r.get("mid_x_local", 0.0)))
        clusters: list[list[dict]] = []
        cur: list[dict] = [by_x[0]]
        for rec in by_x[1:]:
            x_cur = float(rec.get("mid_x_local", 0.0))
            x_prev = float(cur[-1].get("mid_x_local", 0.0))
            if abs(x_cur - x_prev) <= gap_px:
                cur.append(rec)
            else:
                clusters.append(cur)
                cur = [rec]
        clusters.append(cur)

        for grp in clusters:
            grp_sorted = sorted(
                grp,
                key=lambda r: (
                    float(r.get("quality_score", 0.0)),
                    float(r.get("prominence_px", 0.0)),
                    min(float(r.get("left_seg_len", 0.0)), float(r.get("right_seg_len", 0.0))),
                ),
                reverse=True,
            )
            best = dict(grp_sorted[0])
            xs = [float(r.get("mid_x_local", 0.0)) for r in grp]
            best["cluster_size"] = int(len(grp))
            best["cluster_span_px"] = float(max(xs) - min(xs)) if xs else 0.0
            cands.append(best)

    dbg = {
        "zone": zone,
        "raw_profile": np.stack([x_all, ys], axis=1).tolist(),
        "smoothed_profile": np.stack([x_all, ys_s], axis=1).tolist(),
        "piecewise_profile": piece.tolist(),
        "kinks": [
            {
                "x": round(float(c["mid_x_local"]), 2),
                "y": round(float(c["y_local"]), 2),
                "delta_deg": round(float(c.get("delta_deg", 0.0)), 2),
                "prominence_px": round(float(c.get("prominence_px", 0.0)), 2),
                "left_seg_len": round(float(c.get("left_seg_len", 0.0)), 2),
                "right_seg_len": round(float(c.get("right_seg_len", 0.0)), 2),
                "sign_flips": int(c.get("sign_flips", 0)),
                "quality_score": round(float(c.get("quality_score", 0.0)), 3),
                "quality_tier": str(c.get("quality_tier", "")),
                "cluster_size": int(c.get("cluster_size", 1)),
                "cluster_span_px": round(float(c.get("cluster_span_px", 0.0)), 2),
            }
            for c in cands
        ],
    }
    return cands, dbg


def _collect_corner_anchor_xs(mask_bin: np.ndarray, bw: int, bh: int) -> list[float]:
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(contour, True)
    if per < 20.0:
        return []
    x_left_interior, x_right_interior = _interior_x_bounds(mask_bin, bw)
    min_edge = max(6.0, EXTERIOR_SEAM_CONTOUR_MIN_EDGE_RATIO * float(min(bw, bh)))
    xs: list[float] = []
    for frac in EXTERIOR_CONTOUR_EPS_FRACS:
        f = float(frac)
        if not (0.005 <= f <= 0.04):
            continue
        approx = cv2.approxPolyDP(contour, f * per, True)
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
            x = float(p[0])
            if x <= x_left_interior or x >= x_right_interior:
                continue
            xs.append(x)
    if not xs:
        return []
    xs.sort()
    uniq: list[float] = [xs[0]]
    gap = max(4.0, EXTERIOR_SEAM_CORNER_SNAP_TOL_RATIO * float(bw))
    for x in xs[1:]:
        if abs(x - uniq[-1]) >= gap:
            uniq.append(x)
    return uniq


def _snap_x_to_anchor(x: float, anchors: list[float], tol_px: float) -> float:
    if not anchors:
        return float(x)
    best = min(anchors, key=lambda a: abs(a - x))
    if abs(best - x) <= tol_px:
        return float(best)
    return float(x)


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

    # Use Distance Transform to find the point deepest inside the mask (Pole of Inaccessibility).
    # This guarantees the center is visually pleasing and never falls inside a hole.
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = float(max_loc[0]), float(max_loc[1])

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


def _component_masks_from_connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Connected components only (without contour/watershed extra splitting)."""
    mask_bin = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    component_masks: list[np.ndarray] = []
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            comp = np.zeros_like(mask_clean)
            comp[labels == i] = 255
            component_masks.append(comp)

    if not component_masks and cv2.countNonZero(mask_clean) >= int(min_area):
        component_masks = [mask_clean]

    return component_masks


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

    x_left_interior, x_right_interior = _interior_x_bounds(mask_crop, bw)
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
            if px <= x_left_interior or px >= x_right_interior:
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
                "y_local": py,
                "pt_a_local": np.array([px, 0.0], dtype=np.float64),
                "pt_b_local": np.array([px, float(bh - 1)], dtype=np.float64),
                "source_bonus": float(score_hint),
                "delta_norm": float(delta_norm),
                "source": f"contour_{zone}",
            }
            if zone == "bottom":
                raw_bottom.append(rec)
            else:
                raw_top.append(rec)

    if EXTERIOR_SEAM_PROFILE_ENABLE:
        raw_bottom.extend(
            _profile_kink_candidates(
                mask_crop, bw, bh, "bottom", x_left_interior, x_right_interior
            )
        )
        raw_top.extend(
            _profile_kink_candidates(
                mask_crop, bw, bh, "top", x_left_interior, x_right_interior
            )
        )

    def _unique_by_x(records: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[int] = set()
        step = max(1.0, float(bw) * 0.03)
        for rec in sorted(
            records,
            key=lambda c: (float(c.get("source_bonus", 0.0)), float(c.get("delta_norm", 0.0))),
            reverse=True,
        ):
            x = float(rec["mid_x_local"])
            x_bin = int(round(x / step))
            if x_bin in seen:
                continue
            seen.add(x_bin)
            out.append(rec)
        return out

    bottom_cands = _unique_by_x(raw_bottom)
    top_cands = _unique_by_x(raw_top)

    # Локальная изоляция нижних профильных точек: плотные группы чаще являются бугорками шума.
    bottom_profile_idx = [i for i, b in enumerate(bottom_cands) if str(b.get("source", "")) == "profile_bottom"]
    for i in bottom_profile_idx:
        xi = float(bottom_cands[i].get("mid_x_local", 0.0))
        other = [
            abs(xi - float(bottom_cands[j].get("mid_x_local", 0.0)))
            for j in bottom_profile_idx
            if j != i
        ]
        nearest = min(other) if other else 1e9
        bottom_cands[i]["neighbor_dx"] = float(nearest)

    def _conf(rec: dict) -> float:
        base = float(rec.get("source_bonus", 0.0)) + 0.35 * float(rec.get("delta_norm", 0.0))
        q = float(rec.get("quality_score", 0.0))
        if str(rec.get("source", "")).startswith("profile_"):
            base += 0.10 * max(0.0, min(1.0, q))
        return base

    x_tol = max(
        6.0,
        max(EXTERIOR_SEAM_CONTOUR_X_TOL_RATIO, EXTERIOR_SEAM_TANDEM_X_TOL_RATIO) * float(bw),
    )
    used_top: set[int] = set()
    for b in bottom_cands:
        bx = float(b["mid_x_local"])
        best_j = -1
        best_dx = 1e9
        for j, t in enumerate(top_cands):
            if j in used_top:
                continue
            dx = abs(float(t["mid_x_local"]) - bx)
            if dx <= x_tol and dx < best_dx:
                best_dx = dx
                best_j = j
        if best_j >= 0:
            # Тандем разрешаем только с качественным нижним профильным изломом.
            src_b = str(b.get("source", ""))
            q_b = float(b.get("quality_score", 0.0))
            cluster_n = int(b.get("cluster_size", 1))
            neighbor_dx = float(b.get("neighbor_dx", 1e9))
            min_neighbor_gap = max(10.0, EXTERIOR_SEAM_PROFILE_TANDEM_MIN_NEIGHBOR_GAP_RATIO * float(bw))
            allow_tandem = True
            if src_b == "profile_bottom":
                allow_tandem = (
                    q_b >= EXTERIOR_SEAM_PROFILE_QUALITY_HIGH
                    and cluster_n <= 2
                    and neighbor_dx >= min_neighbor_gap
                )
            if allow_tandem:
                used_top.add(best_j)
                b["tandem_match"] = True
                b["source_bonus"] = float(b.get("source_bonus", 0.0)) + EXTERIOR_SEAM_TANDEM_BOOST
                top_cands[best_j]["tandem_match"] = True
                top_cands[best_j]["source_bonus"] = float(top_cands[best_j].get("source_bonus", 0.0)) + (
                    EXTERIOR_SEAM_TANDEM_BOOST * 0.7
                )

    out: list[dict] = list(bottom_cands)
    matched_tops = [t for t in top_cands if bool(t.get("tandem_match", False))]
    out.extend(matched_tops)

    bottom_best = max((_conf(b) for b in bottom_cands), default=0.0)
    top_best = max((_conf(t) for t in top_cands), default=0.0)

    # Если низ слабый/неполный, разрешаем fallback на сильные верхние изломы.
    if (not bottom_cands and top_best >= EXTERIOR_SEAM_TOP_FALLBACK_CONF_MIN) or (
        bottom_best < EXTERIOR_SEAM_BOTTOM_CONF_MIN and top_best >= EXTERIOR_SEAM_TOP_FALLBACK_CONF_MIN
    ):
        for t in top_cands:
            if t not in out and _conf(t) >= EXTERIOR_SEAM_TOP_FALLBACK_CONF_MIN:
                out.append(t)

    out = _unique_by_x(out)
    out.sort(
        key=lambda c: (
            float(c.get("source_bonus", 0.0)),
            float(c.get("delta_norm", 0.0)),
            1.0 if bool(c.get("tandem_match", False)) else 0.0,
        ),
        reverse=True,
    )
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

    x_left_interior, x_right_interior = _interior_x_bounds(mask, w)
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
            if px <= x_left_interior or px >= x_right_interior:
                continue
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

    profile_bottom_raw = (
        _profile_kink_candidates(mask, w, h, "bottom", x_left_interior, x_right_interior)
        if EXTERIOR_SEAM_PROFILE_ENABLE
        else []
    )
    profile_top_raw = (
        _profile_kink_candidates(mask, w, h, "top", x_left_interior, x_right_interior)
        if EXTERIOR_SEAM_PROFILE_ENABLE
        else []
    )
    profile_bottom = [
        {
            "x": round(float(r.get("mid_x_local", 0.0)), 2),
            "y": round(float(r.get("y_local", 0.0)), 2),
            "delta_norm": round(float(r.get("delta_norm", 0.0)), 3),
            "score_hint": round(float(r.get("source_bonus", 0.0)), 3),
            "quality_score": round(float(r.get("quality_score", 0.0)), 3),
            "quality_tier": str(r.get("quality_tier", "")),
            "sign_flips": int(r.get("sign_flips", 0)),
            "cluster_size": int(r.get("cluster_size", 1)),
            "cluster_span_px": round(float(r.get("cluster_span_px", 0.0)), 2),
            "neighbor_dx": round(float(r.get("neighbor_dx", 0.0)), 2) if "neighbor_dx" in r else None,
        }
        for r in profile_bottom_raw
    ]
    profile_top = [
        {
            "x": round(float(r.get("mid_x_local", 0.0)), 2),
            "y": round(float(r.get("y_local", 0.0)), 2),
            "delta_norm": round(float(r.get("delta_norm", 0.0)), 3),
            "score_hint": round(float(r.get("source_bonus", 0.0)), 3),
            "quality_score": round(float(r.get("quality_score", 0.0)), 3),
            "quality_tier": str(r.get("quality_tier", "")),
            "sign_flips": int(r.get("sign_flips", 0)),
            "cluster_size": int(r.get("cluster_size", 1)),
            "cluster_span_px": round(float(r.get("cluster_span_px", 0.0)), 2),
        }
        for r in profile_top_raw
    ]

    out_bottom.sort(key=lambda d: (float(d["x"])))
    out_top.sort(key=lambda d: (float(d["x"])))
    return {
        "bottom": out_bottom,
        "top": out_top,
        "profile_bottom": profile_bottom,
        "profile_top": profile_top,
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
        "all_scored_lines": [],
        "seam_candidates_raw": [],
        "seam_candidates_total": 0,
        "seam_candidates_sources": {},
        "contour_candidates_count": 0,
        "hough_enabled": bool(EXTERIOR_SEAM_HOUGH_ENABLE),
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
    lines = None
    if EXTERIOR_SEAM_HOUGH_ENABLE:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(40, int(0.15 * bh)),
            minLineLength=max(30, int(EXTERIOR_SEAM_MIN_LINE_RATIO * bh)),
            maxLineGap=max(10, int(0.06 * bh)),
        )

    center_margin_hough = EXTERIOR_SEAM_CENTER_MARGIN_RATIO
    center_margin_contour = min(center_margin_hough, EXTERIOR_SEAM_CONTOUR_CENTER_MARGIN_RATIO)
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
    dbg["seam_candidates_total"] = len(seam_candidates)
    src_stats: dict[str, int] = {}
    raw_recs: list[dict] = []
    for c in seam_candidates:
        src = str(c.get("source", ""))
        src_stats[src] = src_stats.get(src, 0) + 1
        raw_recs.append(
            {
                "source": src,
                "x_local": round(float(c.get("mid_x_local", 0.0)), 2),
                "delta_norm": round(float(c.get("delta_norm", 0.0)), 3),
                "source_bonus": round(float(c.get("source_bonus", 0.0)), 4),
                "tandem_match": bool(c.get("tandem_match", False)),
                "quality_score": round(float(c.get("quality_score", 0.0)), 3),
                "quality_tier": str(c.get("quality_tier", "")),
                "cluster_size": int(c.get("cluster_size", 1)),
                "cluster_span_px": round(float(c.get("cluster_span_px", 0.0)), 2),
                "neighbor_dx": round(float(c.get("neighbor_dx", 0.0)), 2) if "neighbor_dx" in c else None,
            }
        )
    dbg["seam_candidates_sources"] = src_stats
    dbg["seam_candidates_raw"] = raw_recs
    anchors_x = _collect_corner_anchor_xs(mask_crop, bw, bh)
    snap_tol_px = max(4.0, EXTERIOR_SEAM_CORNER_SNAP_TOL_RATIO * float(bw))

    for cand in seam_candidates:
        original_mid_x = float(cand["mid_x_local"])
        mid_x = original_mid_x
        mid_x_norm = mid_x / max(1.0, float(bw - 1))
        src = str(cand.get("source", ""))
        margin = center_margin_contour if (src.startswith("contour_") or src.startswith("profile_")) else center_margin_hough
        if mid_x_norm < margin or mid_x_norm > (1.0 - margin):
            continue
        if src.startswith("contour_") or src.startswith("profile_"):
            mid_x = _snap_x_to_anchor(mid_x, anchors_x, snap_tol_px)
            mid_x_norm = mid_x / max(1.0, float(bw - 1))
        mid_y = max(0, min(bh - 1, int(round(0.5 * (bh - 1)))))
        mid_xi = int(round(mid_x))
        mid_xi = max(0, min(bw - 1, mid_xi))
        if mask_crop[mid_y, mid_xi] == 0:
            ys_nz = np.where(mask_crop[:, mid_xi] > 0)[0]
            if ys_nz.size == 0:
                continue
            mid_y = int(ys_nz[ys_nz.size // 2])

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
        score += float(cand.get("source_bonus", 0.0))
        delta_norm = float(cand.get("delta_norm", 0.0))
        if src in {"contour_bottom", "profile_bottom"}:
            score += 0.10 + 0.20 * max(0.0, min(1.0, delta_norm))
        elif src in {"contour_top", "profile_top"}:
            score += 0.04 + 0.08 * max(0.0, min(1.0, delta_norm))
        if bool(cand.get("tandem_match", False)):
            score += EXTERIOR_SEAM_TANDEM_BOOST * 0.6
        candidates_count += 1
        candidate_lines.append(
            {
                "score": float(score),
                "mid_x_local": float(mid_x),
                "line_local": (pt_a.copy(), pt_b.copy()),
                "source": str(cand.get("source", "")),
                "delta_norm": float(delta_norm),
                "balance": float(balance),
                "edge_score": float(edge_score),
                "center_score": float(max(0.0, center_score)),
                "x_before_snap": float(original_mid_x),
                "x_after_snap": float(mid_x),
                "tandem_match": bool(cand.get("tandem_match", False)),
                "quality_score": float(cand.get("quality_score", 0.0)),
                "quality_tier": str(cand.get("quality_tier", "")),
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
    all_scored = sorted(candidate_lines, key=lambda c: float(c["score"]), reverse=True)
    all_scored_dbg: list[dict] = []
    for cand in all_scored:
        pt_a_l, pt_b_l = cand["line_local"]
        pt_a_g = np.array([pt_a_l[0] + x1i, pt_a_l[1] + y1i], dtype=np.float64)
        pt_b_g = np.array([pt_b_l[0] + x1i, pt_b_l[1] + y1i], dtype=np.float64)
        all_scored_dbg.append(
            {
                "score": round(float(cand["score"]), 4),
                "source": str(cand.get("source", "")),
                "line": [
                    round(float(pt_a_g[0]), 2),
                    round(float(pt_a_g[1]), 2),
                    round(float(pt_b_g[0]), 2),
                    round(float(pt_b_g[1]), 2),
                ],
                "delta_norm": round(float(cand.get("delta_norm", 0.0)), 3),
                "balance": round(float(cand.get("balance", 0.0)), 4),
                "edge_score": round(float(cand.get("edge_score", 0.0)), 4),
                "center_score": round(float(cand.get("center_score", 0.0)), 4),
                "x_before_snap": round(float(cand.get("x_before_snap", 0.0)), 2),
                "x_after_snap": round(float(cand.get("x_after_snap", 0.0)), 2),
                "tandem_match": bool(cand.get("tandem_match", False)),
                "quality_score": round(float(cand.get("quality_score", 0.0)), 3),
                "quality_tier": str(cand.get("quality_tier", "")),
                "cluster_size": int(cand.get("cluster_size", 1)),
                "cluster_span_px": round(float(cand.get("cluster_span_px", 0.0)), 2),
                "neighbor_dx": round(float(cand.get("neighbor_dx", 0.0)), 2) if "neighbor_dx" in cand else None,
            }
        )
    dbg["all_scored_lines"] = all_scored_dbg

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
    elif component_masks is not None:
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
    else:
        return None, seam_debug

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
    t_pipe0 = time.perf_counter()
    geom_trace: dict = {
        "trace_version": 1,
        "debug_level": EXTERIOR_GEOM_DEBUG_LEVEL,
        "feature_flags": {
            "geom_cascade_enable": EXTERIOR_GEOM_CASCADE_ENABLE,
            "sam_batch_enable": EXTERIOR_SAM_BATCH_ENABLE,
        },
        "input": {
            "image_size": {"width": int(image_width), "height": int(image_height)},
            "image_digest": call_digest,
        },
        "stages": [],
    }

    def _stage(name: str, started_at: float, **kwargs: object) -> None:
        rec = {"name": name, "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2)}
        rec.update(kwargs)
        geom_trace["stages"].append(rec)

    t0 = time.perf_counter()
    building_bbox = await detect_building_bbox(image_bytes)
    _stage("detect_building_bbox", t0, found=bool(building_bbox), bbox=building_bbox)
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

    t0 = time.perf_counter()
    wall_mask = await sam_mask_from_bbox(image_bytes, building_bbox)
    _stage("sam_wall_mask", t0, nonzero=int(np.count_nonzero(wall_mask)))

    if wall_mask.shape != (image_height, image_width):
        wall_mask = cv2.resize(wall_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    # ВАЖНО: сначала делим фасад на стены по цельной маске (без выреза окон/дверей),
    # затем отдельно считаем holes и отдаём wall_minus_holes для визуализации/клиппинга.
    no_holes_mask = np.zeros_like(wall_mask)
    t0 = time.perf_counter()
    wall_for_split = postprocess_masks(wall_mask, no_holes_mask)
    wall_for_split = _clean_mask_for_geometry(wall_for_split)
    _stage(
        "prepare_wall_for_split",
        t0,
        wall_nonzero=int(np.count_nonzero(wall_mask)),
        wall_for_split_nonzero=int(np.count_nonzero(wall_for_split)),
    )

    t0 = time.perf_counter()
    image_bgr = _decode_image_bgr(image_bytes, image_width, image_height)
    _stage("decode_image", t0, ok=image_bgr is not None)
    t0 = time.perf_counter()
    openings = await detect_openings_bboxes(image_bytes, building_bbox)
    _stage(
        "detect_openings_bboxes",
        t0,
        windows_count=len(openings.get("windows") or []),
        doors_count=len(openings.get("doors") or []),
    )
    holes_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    all_opening_bboxes_raw = openings["windows"] + openings["doors"]
    all_opening_bboxes = _filter_opening_bboxes(all_opening_bboxes_raw)
    sam_batch_chunks = 0
    sam_batch_total_ms = 0.0
    sam_fallback_used = False
    if EXTERIOR_SAM_BATCH_ENABLE and all_opening_bboxes:
        try:
            t_batch0 = time.perf_counter()
            for i in range(0, len(all_opening_bboxes), EXTERIOR_SAM_BATCH_SIZE):
                chunk = all_opening_bboxes[i : i + EXTERIOR_SAM_BATCH_SIZE]
                t0 = time.perf_counter()
                chunk_masks = await sam_masks_from_bboxes_batch(image_bytes, chunk)
                sam_batch_total_ms += (time.perf_counter() - t0) * 1000.0
                sam_batch_chunks += 1
                for opening_mask in chunk_masks:
                    if opening_mask.shape == holes_mask.shape:
                        holes_mask = cv2.bitwise_or(holes_mask, opening_mask)
            _stage("sam_batch_openings", t_batch0, chunks=sam_batch_chunks, total_ms=round(float(sam_batch_total_ms), 2))
        except Exception:
            sam_fallback_used = True
    if (not EXTERIOR_SAM_BATCH_ENABLE or sam_fallback_used) and all_opening_bboxes:
        t_fallback0 = time.perf_counter()
        for ob in all_opening_bboxes:
            try:
                opening_mask = await sam_mask_from_bbox(image_bytes, ob)
                if opening_mask.shape == holes_mask.shape:
                    holes_mask = cv2.bitwise_or(holes_mask, opening_mask)
            except Exception:
                continue
        _stage("sam_single_openings_fallback", t_fallback0, used=True, boxes=len(all_opening_bboxes))
    if holes_mask.shape != (image_height, image_width):
        holes_mask = cv2.resize(holes_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    t0 = time.perf_counter()
    wall_minus_holes = postprocess_masks(wall_mask, holes_mask)
    _stage("postprocess_wall_minus_holes", t0, nonzero=int(np.count_nonzero(wall_minus_holes)))
    min_area = int(image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO)
    comp_count = _count_large_components(wall_for_split, min_area)
    split_method_geo = "geometry_fallback"
    walls_geo: list[dict]
    seam_debug: dict = {}
    geom_cascade_debug: dict = {"enabled": EXTERIOR_GEOM_CASCADE_ENABLE, "used": False}
    if comp_count <= 1:
        t0 = time.perf_counter()
        walls_seam, seam_debug = _auto_split_by_corner_seam(
            wall_for_split, image_bgr, building_bbox, image_width, image_height
        )
        _stage(
            "split_seam",
            t0,
            used=bool(walls_seam),
            walls_count=len(walls_seam or []),
            candidates=int(seam_debug.get("candidates_count", 0)),
            best_score=round(float(seam_debug.get("best_score", 0.0)), 4),
        )
        if walls_seam and len(walls_seam) >= 2:
            walls_geo = walls_seam
            split_method_geo = "seam"
        else:
            walls_geom_rel: list[dict] | None = None
            if EXTERIOR_GEOM_CASCADE_ENABLE:
                t0 = time.perf_counter()
                walls_geom_rel, geom_cascade_debug = _split_by_rgb_lsd_corners(
                    wall_for_split, image_bgr, image_width, image_height, min_area, wall_minus_holes
                )
                _stage(
                    "split_geom_cascade",
                    t0,
                    used=bool(walls_geom_rel),
                    walls_count=len(walls_geom_rel or []),
                    lsd_segments=int(geom_cascade_debug.get("lsd_segments_total", 0)),
                    clusters=int(geom_cascade_debug.get("clusters_total", 0)),
                    intersections=int(geom_cascade_debug.get("intersections_total", 0)),
                    splits_applied=len(geom_cascade_debug.get("split_x_candidates_after_nms") or []),
                )
            if walls_geom_rel and len(walls_geom_rel) >= 2:
                walls_geo = walls_geom_rel
                split_method_geo = "geom_rgb_lsd_bottom_band_multiple_splits"
            else:
                t0 = time.perf_counter()
                walls_kink = _split_by_bottom_contour_kink_once(
                    wall_for_split, image_width, image_height, min_area
                )
                _stage("split_bottom_kink", t0, used=bool(walls_kink), walls_count=len(walls_kink or []))
                if walls_kink and len(walls_kink) >= 2:
                    walls_geo = walls_kink
                    split_method_geo = "bottom_kink"
                else:
                    component_masks = _component_masks_from_connected_components(wall_for_split, min_area)
                    walls_geo = _build_walls_from_component_masks(component_masks)
                    split_method_geo = "connected_components_fallback"
                    _stage(
                        "split_connected_components_fallback",
                        time.perf_counter(),
                        walls_count=len(walls_geo),
                    )
    else:
        t0 = time.perf_counter()
        component_masks = _component_masks_from_connected_components(wall_for_split, min_area)
        walls_geo = _build_walls_from_component_masks(component_masks)
        split_method_geo = "connected_components"
        _stage("split_connected_components", t0, walls_count=len(walls_geo), components=int(comp_count))



    # ── Normals-planes арбитр ─────────────────────────────────────────────────────────────
    # Если EXTERIOR_DEPTH_SPLIT_ENABLE=1, запрашиваем нормали поверхности (DSINE) и
    # кластеризуем пиксели стены по азимуту нормалей. Разные грани = разные нормали.
    # VP-снаппинг: граница кластеров сдвигается к ближайшему LSD-ребру (geom_cascade_debug).
    # При сбое или однородных нормалях (фронталь) — откат на walls_geo (геометрический каскад).
    walls = walls_geo
    split_method = split_method_geo
    split_arbiter: dict = {
        "chosen": "geometry_only",
        "plane_compare_ran": False,
        "reason": "normals_arbiter_disabled",
    }

    if EXTERIOR_DEPTH_SPLIT_ENABLE:
        try:
            from plane_split import request_normals, cluster_planes_by_normals

            # LSD-рёбра для VP-снаппинга (если геом-каскад их нашёл)
            lsd_vertical_xs: list[int] = [
                int(x) for x in (geom_cascade_debug.get("_split_xs_for_overlay") or [])
            ]

            t0 = time.perf_counter()
            normals_map = request_normals(image_bgr)
            _stage("normals_request", t0, success=normals_map is not None)

            if normals_map is not None:
                t0 = time.perf_counter()
                plane_masks, np_debug = cluster_planes_by_normals(
                    wall_mask=wall_for_split,
                    normals=normals_map,
                    image_width=image_width,
                    image_height=image_height,
                    lsd_vertical_xs=lsd_vertical_xs or None,
                    holes_mask=holes_mask,
                    debug_dir=call_dir if EXTERIOR_DEBUG_SAVE else "",
                )
                _stage(
                    "normal_cluster",
                    t0,
                    planes=np_debug.get("planes_found"),
                    angle_deg=np_debug.get("plane_angle_deg"),
                    reason=np_debug.get("reason"),
                    masks=len(plane_masks),
                    vp_agreement=np_debug.get("vp_snap", {}).get("vp_agreement", False),
                )

                split_arbiter = {
                    "chosen": "geometry_only",
                    "plane_compare_ran": True,
                    "normals_debug": np_debug,
                    "reason": np_debug.get("reason", "unknown"),
                }

                if plane_masks and len(plane_masks) >= 2:
                    walls_normals = _build_walls_from_component_masks(plane_masks)
                    if walls_normals:
                        walls = walls_normals
                        split_method = "normal_planes"
                        split_arbiter["chosen"] = "normal_planes"
                        split_arbiter["walls_count"] = len(walls_normals)
                    else:
                        split_arbiter["reason"] = "normals_walls_empty_fallback_geo"
                else:
                    split_arbiter["reason"] = np_debug.get("reason", "no_planes_fallback_geo")
            else:
                split_arbiter = {
                    "chosen": "geometry_only",
                    "plane_compare_ran": True,
                    "reason": "normals_request_failed",
                }
        except Exception as _norm_exc:
            logger.warning("[exterior] plane_split error (fallback to geometry): %s", _norm_exc)
            split_arbiter = {
                "chosen": "geometry_only",
                "plane_compare_ran": True,
                "reason": f"normals_exception: {_norm_exc}",
            }

    if EXTERIOR_DEBUG_SAVE:
        try:
            corner_dbg = _debug_contour_corner_points(wall_for_split)
            # Save masks as PNGs for manual inspection.
            cv2.imwrite(os.path.join(call_dir, "wall.png"), wall_mask)
            cv2.imwrite(os.path.join(call_dir, "wall_for_split.png"), wall_for_split)
            cv2.imwrite(os.path.join(call_dir, "holes.png"), holes_mask)
            cv2.imwrite(os.path.join(call_dir, "wall_minus_holes.png"), wall_minus_holes)

            # Save an overlay visualization (polygons drawn on top of mask).
            overlay = cv2.cvtColor(wall_for_split, cv2.COLOR_GRAY2BGR)
            for w in walls:
                poly = w.get("polygon") or []
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(call_dir, "polygons_overlay.png"), overlay)

            # Save explicit kink points with angle labels.
            kinks_overlay = cv2.cvtColor(wall_for_split, cv2.COLOR_GRAY2BGR)

            def _draw_kinks(items: list[dict], color: tuple[int, int, int], prefix: str) -> None:
                # Сортируем по силе и прореживаем близкие подписи для читаемости.
                def _strength(rec: dict) -> float:
                    if "delta_deg" in rec:
                        return float(rec.get("delta_deg", 0.0))
                    return float(rec.get("delta_norm", 0.0)) * 90.0

                ranked = sorted(items, key=_strength, reverse=True)
                placed: list[tuple[int, int]] = []
                shown = 0
                for rec in ranked:
                    if shown >= max(1, EXTERIOR_DEBUG_KINK_LABEL_TOP_K):
                        break
                    x = int(round(float(rec.get("x", rec.get("mid_x_local", 0.0)))))
                    y = int(round(float(rec.get("y", rec.get("y_local", 0.0)))))
                    if x <= 0 or y <= 0 or x >= image_width - 1 or y >= image_height - 1:
                        continue
                    if any(abs(x - px) < EXTERIOR_DEBUG_KINK_LABEL_MIN_GAP_PX and abs(y - py) < EXTERIOR_DEBUG_KINK_LABEL_MIN_GAP_PX for px, py in placed):
                        continue
                    placed.append((x, y))
                    shown += 1
                    cv2.circle(kinks_overlay, (x, y), 6, color, -1)
                    cv2.circle(kinks_overlay, (x, y), 8, (0, 0, 0), 1)
                    lbl = (
                        f"{prefix}:{float(rec.get('delta_deg', rec.get('delta_norm', 0.0) * 90.0)):.1f}"
                    )
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    tx = x + 8
                    ty = max(16, y - 8)
                    cv2.rectangle(
                        kinks_overlay,
                        (tx - 2, ty - th - 3),
                        (tx + tw + 2, ty + 3),
                        (255, 255, 255),
                        thickness=-1,
                    )
                    cv2.putText(
                        kinks_overlay,
                        lbl,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            _draw_kinks(corner_dbg.get("bottom") or [], (0, 255, 255), "B")
            _draw_kinks(corner_dbg.get("top") or [], (255, 255, 0), "T")
            _draw_kinks(corner_dbg.get("profile_bottom") or [], (0, 165, 255), "PB")
            _draw_kinks(corner_dbg.get("profile_top") or [], (255, 0, 255), "PT")
            cv2.imwrite(os.path.join(call_dir, "kinks_overlay.png"), kinks_overlay)

            if geom_cascade_debug.get("_band_mask_for_overlay") is not None:
                cv2.imwrite(
                    os.path.join(call_dir, "bottom_band.png"),
                    geom_cascade_debug.get("_band_mask_for_overlay"),
                )

            # Save LSD overlay: detected segments, fitted lines, intersections, split lines.
            if geom_cascade_debug.get("_segments_for_overlay"):
                lsd_overlay = image_bgr.copy() if image_bgr is not None else cv2.cvtColor(wall_for_split, cv2.COLOR_GRAY2BGR)
                mask_alpha = cv2.cvtColor(wall_for_split, cv2.COLOR_GRAY2BGR)
                lsd_overlay = cv2.addWeighted(lsd_overlay, 0.7, mask_alpha, 0.3, 0)
                band_mask_overlay = geom_cascade_debug.get("_band_mask_for_overlay")
                if band_mask_overlay is not None:
                    contours_band, _ = cv2.findContours(
                        (band_mask_overlay > 0).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(lsd_overlay, contours_band, -1, (255, 255, 0), 1)
                for seg in geom_cascade_debug["_segments_for_overlay"]:
                    p1 = (int(round(seg["x1"])), int(round(seg["y1"])))
                    p2 = (int(round(seg["x2"])), int(round(seg["y2"])))
                    cv2.line(lsd_overlay, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
                for plane in geom_cascade_debug.get("_planes_for_overlay") or []:
                    ln = plane.get("line") or {}
                    a = float(ln.get("a", 0.0))
                    b = float(ln.get("b", 0.0))
                    c = float(ln.get("c", 0.0))
                    if abs(b) > 1e-6:
                        y0 = int(round((-c - a * 0.0) / b))
                        y1 = int(round((-c - a * float(image_width - 1)) / b))
                        cv2.line(lsd_overlay, (0, y0), (image_width - 1, y1), (0, 200, 255), 1, cv2.LINE_AA)
                for p in geom_cascade_debug.get("_x_peaks_raw_for_overlay") or []:
                    xi = int(round(float(p.get("x", 0.0))))
                    cv2.line(lsd_overlay, (xi, 0), (xi, image_height - 1), (0, 180, 255), 1, cv2.LINE_AA)
                for sx in geom_cascade_debug.get("_x_peaks_kept_for_overlay") or []:
                    xi = int(round(float(sx)))
                    cv2.line(lsd_overlay, (xi, 0), (xi, image_height - 1), (255, 255, 0), 2, cv2.LINE_AA)
                for inter in geom_cascade_debug.get("_intersections_for_overlay") or []:
                    cx = int(round(inter["x"]))
                    cy = int(round(inter["y"]))
                    cv2.circle(lsd_overlay, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.circle(lsd_overlay, (cx, cy), 10, (255, 255, 255), 2)
                    lbl = f"({cx},{cy})"
                    cv2.putText(lsd_overlay, lbl, (cx + 12, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                for sx in geom_cascade_debug.get("_split_xs_for_overlay") or []:
                    xi = int(round(sx))
                    cv2.line(lsd_overlay, (xi, 0), (xi, image_height - 1), (255, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(call_dir, "lsd_overlay.png"), lsd_overlay)

            # Save profile denoise stages: raw -> median-smoothed -> piecewise.
            bw_dbg = int(wall_for_split.shape[1])
            bh_dbg = int(wall_for_split.shape[0])
            x_margin_dbg = max(2.0, EXTERIOR_SEAM_CONTOUR_EDGE_EXCLUDE_RATIO * float(bw_dbg))
            x_left_dbg = x_margin_dbg
            x_right_dbg = float(bw_dbg - 1) - x_margin_dbg
            _, pb_dbg = _profile_kink_candidates_with_debug(
                wall_for_split, bw_dbg, bh_dbg, "bottom", x_left_dbg, x_right_dbg
            )
            _, pt_dbg = _profile_kink_candidates_with_debug(
                wall_for_split, bw_dbg, bh_dbg, "top", x_left_dbg, x_right_dbg
            )
            denoise_overlay = cv2.cvtColor(wall_for_split, cv2.COLOR_GRAY2BGR)

            def _draw_profile_stage(stage: dict, color_smooth: tuple[int, int, int], color_piece: tuple[int, int, int]) -> None:
                smooth = stage.get("smoothed_profile") or []
                piece = stage.get("piecewise_profile") or []
                kinks = stage.get("kinks") or []
                if len(smooth) >= 2:
                    spts = np.array([[int(round(p[0])), int(round(p[1]))] for p in smooth], dtype=np.int32)
                    cv2.polylines(denoise_overlay, [spts], isClosed=False, color=color_smooth, thickness=1)
                if len(piece) >= 2:
                    ppts = np.array([[int(round(p[0])), int(round(p[1]))] for p in piece], dtype=np.int32)
                    cv2.polylines(denoise_overlay, [ppts], isClosed=False, color=color_piece, thickness=2)
                for rec in kinks:
                    x = int(round(float(rec.get("x", 0.0))))
                    y = int(round(float(rec.get("y", 0.0))))
                    if x <= 0 or y <= 0 or x >= bw_dbg - 1 or y >= bh_dbg - 1:
                        continue
                    cv2.circle(denoise_overlay, (x, y), 4, color_piece, -1)
                    lbl = f"d={float(rec.get('delta_deg', 0.0)):.0f}"
                    cv2.putText(
                        denoise_overlay,
                        lbl,
                        (x + 5, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            _draw_profile_stage(pb_dbg, (0, 255, 0), (0, 165, 255))   # bottom
            _draw_profile_stage(pt_dbg, (255, 0, 0), (255, 0, 255))   # top
            cv2.imwrite(os.path.join(call_dir, "profile_denoise_overlay.png"), denoise_overlay)

            # Strip non-serializable overlay data before JSON dump.
            for _overlay_key in (
                "_segments_for_overlay",
                "_intersections_for_overlay",
                "_split_xs_for_overlay",
                "_band_mask_for_overlay",
                "_planes_for_overlay",
                "_x_peaks_raw_for_overlay",
                "_x_peaks_kept_for_overlay",
            ):
                geom_cascade_debug.pop(_overlay_key, None)

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
                    "split_mask_used": "wall_for_split",
                    "contour_corner_debug": corner_dbg,
                    "geom_cascade_debug": geom_cascade_debug,
                    "openings_bboxes_raw_count": len(all_opening_bboxes_raw),
                    "openings_bboxes_filtered_count": len(all_opening_bboxes),
                    "sam_batch_enable": EXTERIOR_SAM_BATCH_ENABLE,
                    "sam_batch_size": EXTERIOR_SAM_BATCH_SIZE,
                    "sam_batch_chunks": sam_batch_chunks,
                    "sam_batch_total_ms": round(float(sam_batch_total_ms), 2),
                    "sam_fallback_used": sam_fallback_used,
                    "trace_digest": call_digest,
                    "trace_stage_count": len(geom_trace.get("stages", [])),
                    "profile_denoise_debug": {
                        "bottom_vertices": len((pb_dbg or {}).get("piecewise_profile") or []),
                        "top_vertices": len((pt_dbg or {}).get("piecewise_profile") or []),
                        "bottom_kinks": len((pb_dbg or {}).get("kinks") or []),
                        "top_kinks": len((pt_dbg or {}).get("kinks") or []),
                        "rdp_eps_px": EXTERIOR_SEAM_PROFILE_RDP_EPS_PX,
                        "quality_min": EXTERIOR_SEAM_PROFILE_QUALITY_MIN,
                        "quality_high": EXTERIOR_SEAM_PROFILE_QUALITY_HIGH,
                        "prom_ref_px": EXTERIOR_SEAM_PROFILE_PROM_REF_PX,
                        "seg_ref_px": EXTERIOR_SEAM_PROFILE_SEG_REF_PX,
                        "cluster_gap_ratio": EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_RATIO,
                        "cluster_gap_min_px": EXTERIOR_SEAM_PROFILE_CLUSTER_GAP_MIN_PX,
                        "tandem_min_neighbor_gap_ratio": EXTERIOR_SEAM_PROFILE_TANDEM_MIN_NEIGHBOR_GAP_RATIO,
                    },
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
            if EXTERIOR_GEOM_DEBUG_SAVE_JSON:
                geom_trace["pipeline_total_ms"] = round((time.perf_counter() - t_pipe0) * 1000.0, 2)
                geom_trace["result"] = {
                    "split_method": split_method,
                    "walls_count": len(walls),
                    "sam_batch_chunks": sam_batch_chunks,
                    "sam_fallback_used": sam_fallback_used,
                }
                with open(os.path.join(call_dir, "debug_trace.json"), "w", encoding="utf-8") as f:
                    f.write(_json.dumps(geom_trace, ensure_ascii=False, indent=2))
        except Exception as dbg_err:
            print(f"[exterior-debug] save failed: {dbg_err}")

    # Always log a minimal summary to console (useful even if debug save off).
    try:
        polygon_lens = [len(w.get("polygon") or []) for w in walls]
        geom_trace["pipeline_total_ms"] = round((time.perf_counter() - t_pipe0) * 1000.0, 2)
        print(
            f"[detect-exterior] walls={len(walls)} polygon_lens={polygon_lens[:10]} "
            f"wall_minus_holes_nonzero={int(np.count_nonzero(wall_minus_holes))} "
            f"split={split_method} arb={split_arbiter.get('chosen', '')} "
            f"geom_ms={geom_trace.get('pipeline_total_ms', 0)} call={call_ts}_{call_digest}"
        )
    except Exception:
        pass

    for _overlay_key in (
        "_segments_for_overlay",
        "_intersections_for_overlay",
        "_split_xs_for_overlay",
        "_band_mask_for_overlay",
        "_planes_for_overlay",
        "_x_peaks_raw_for_overlay",
        "_x_peaks_kept_for_overlay",
    ):
        geom_cascade_debug.pop(_overlay_key, None)

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
            "geom_cascade_debug": geom_cascade_debug,
            "split_arbiter": split_arbiter,
            "trace": {
                "digest": call_digest,
                "stages": geom_trace.get("stages", [])[:EXTERIOR_GEOM_DEBUG_TOPN],
                "pipeline_total_ms": geom_trace.get("pipeline_total_ms", 0.0),
            },
        },
    }
