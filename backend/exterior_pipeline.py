"""
Exterior detection pipeline using SegFormer (Semantic Segmentation).
Calls local model_service via HTTP, extracts Wall and Hole masks,
and returns wall polygons.
"""
import base64
import hashlib
import os
import time

import cv2
import httpx
import numpy as np

SEGFORMER_URL = os.getenv("EXTERIOR_SEGFORMER_URL", "http://127.0.0.1:8001/segformer")
TIMEOUT_S = int(os.getenv("EXTERIOR_TIMEOUT_MS", "30000")) / 1000.0
EXTERIOR_MIN_WALL_AREA_RATIO = float(os.getenv("EXTERIOR_MIN_WALL_AREA_RATIO", "0.01"))
EXTERIOR_POLYGON_EPS_RATIO = float(os.getenv("EXTERIOR_POLYGON_EPS_RATIO", "0.004"))

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
EXTERIOR_DEBUG_DIR = os.path.join(PROJECT_ROOT, "exterior-debug")

def encode_mask_png_base64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Failed to encode mask PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _order_rect_corners_tl_bl_br_tr(box_pts: np.ndarray) -> list[list[int]]:
    """Sort 4 points of minAreaRect into [TL, BL, BR, TR] order."""
    pts = box_pts.reshape(4, 2).astype(float)
    cx = pts[:, 0].mean()
    
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
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    # Fallback quad
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect)
    corners = _order_rect_corners_tl_bl_br_tr(box_pts)

    perimeter = cv2.arcLength(contour, True)
    eps = EXTERIOR_POLYGON_EPS_RATIO * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    polygon = approx.reshape(-1, 2).astype(int).tolist()
    
    if len(polygon) < 3:
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

async def run_exterior_pipeline(image_bytes: bytes, image_width: int, image_height: int) -> dict:
    """
    1) Call SegFormer -> returns 3-channel PNG (R=Wall, G=Holes, B=Balconies)
    2) wall_minus_holes = Wall AND NOT Holes AND NOT Balconies
    3) Find connected components -> Walls
    """
    call_ts = int(time.time() * 1000)
    call_digest = hashlib.md5(image_bytes).hexdigest()[:10]
    call_dir = os.path.join(EXTERIOR_DEBUG_DIR, f"call_{call_ts}_{call_digest}")
    os.makedirs(call_dir, exist_ok=True)

    empty = np.zeros((image_height, image_width), dtype=np.uint8)
    empty_b64 = encode_mask_png_base64(empty)
    default_resp = {
        "walls": [],
        "image_size": {"width": image_width, "height": image_height},
        "masks": {
            "wall": empty_b64,
            "holes": empty_b64,
            "wall_minus_holes": empty_b64,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            resp = await client.post(
                SEGFORMER_URL,
                files={"image": ("photo.jpg", image_bytes, "image/jpeg")}
            )
        resp.raise_for_status()
    except Exception as e:
        print(f"[detect-exterior] SegFormer request failed: {e}")
        return default_resp

    # Decode 3-channel mask
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    masks_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if masks_bgr is None:
        print("[detect-exterior] Failed to decode SegFormer response")
        return default_resp
        
    if masks_bgr.shape[:2] != (image_height, image_width):
        masks_bgr = cv2.resize(masks_bgr, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    wall_mask = masks_bgr[:, :, 2]      # Red channel
    holes_mask = masks_bgr[:, :, 1]     # Green channel
    balcony_mask = masks_bgr[:, :, 0]   # Blue channel
    
    # We combine holes and balconies to subtract from the wall
    exclude_mask = cv2.bitwise_or(holes_mask, balcony_mask)

    # Smooth the exclusions slightly so we don't have 1-pixel jagged edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    exclude_dilated = cv2.dilate(exclude_mask, kernel, iterations=1)
    
    wall_minus_holes = cv2.bitwise_and(wall_mask, cv2.bitwise_not(exclude_dilated))

    # Clean up small noise in the final mask
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    wall_minus_holes = cv2.morphologyEx(wall_minus_holes, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)
    wall_minus_holes = cv2.morphologyEx(wall_minus_holes, cv2.MORPH_OPEN, smooth_kernel, iterations=1)

    # Find components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_minus_holes, connectivity=8)
    min_area = image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO

    component_masks = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            comp = np.zeros_like(wall_minus_holes)
            comp[labels == i] = 255
            component_masks.append(comp)

    raw_walls = []
    for comp_mask in component_masks:
        wall_data = _extract_wall_from_component(comp_mask)
        if wall_data:
            raw_walls.append(wall_data)

    raw_walls.sort(key=lambda w: w["center"][0])
    walls = []
    for idx, w in enumerate(raw_walls):
        w["id"] = idx + 1
        walls.append(w)

    print(f"[detect-exterior] found {len(walls)} walls using SegFormer")
    
    try:
        cv2.imwrite(os.path.join(call_dir, "wall.png"), wall_mask)
        cv2.imwrite(os.path.join(call_dir, "exclude.png"), exclude_mask)
        cv2.imwrite(os.path.join(call_dir, "wall_minus_holes.png"), wall_minus_holes)
    except Exception as dbg_err:
        pass

    return {
        "walls": walls,
        "image_size": {"width": image_width, "height": image_height},
        "masks": {
            "wall": encode_mask_png_base64(wall_mask),
            "holes": encode_mask_png_base64(exclude_mask),
            "wall_minus_holes": encode_mask_png_base64(wall_minus_holes),
        },
    }

def split_exterior_by_line(
    mask_base64: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Fallback if front-end still sends split lines."""
    return []
