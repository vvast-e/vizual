"""
Exterior detection pipeline.
Calls local model_service (GroundingDINO + SAM) via HTTP,
postprocesses masks, returns wall/holes/wall_minus_holes.
"""
import base64
import io
import os

import cv2
import httpx
import numpy as np

GDINO_URL = os.getenv("EXTERIOR_GDINO_URL", "http://127.0.0.1:8001/gdino")
SAM_URL = os.getenv("EXTERIOR_SAM_URL", "http://127.0.0.1:8001/sam")
TIMEOUT_S = int(os.getenv("EXTERIOR_TIMEOUT_MS", "30000")) / 1000.0
SCORE_THRESH_BUILDING = float(os.getenv("EXTERIOR_SCORE_THRESH_BUILDING", "0.3"))
SCORE_THRESH_OPENINGS = float(os.getenv("EXTERIOR_SCORE_THRESH_OPENINGS", "0.35"))


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

    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect)
    corners = _order_rect_corners_tl_bl_br_tr(box_pts)

    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = float(np.mean([c[0] for c in corners]))
        cy = float(np.mean([c[1] for c in corners]))

    return {
        "corners": corners,
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
            "center": w["center"],
        })

    return walls


async def run_exterior_pipeline(image_bytes: bytes, image_width: int, image_height: int) -> dict:
    """
    Full exterior detection pipeline:
    1) GroundingDINO → building bbox
    2) SAM → wall mask
    3) GroundingDINO → window/door bboxes
    4) SAM → individual opening masks → combined holes mask
    5) Postprocess → wall_minus_holes
    """
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

    walls = split_walls_from_mask(wall_minus_holes, image_width, image_height)

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
        },
    }
