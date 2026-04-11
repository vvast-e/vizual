"""
Exterior detection pipeline using SegFormer (Semantic Segmentation).
Calls local model_service via HTTP, extracts Wall and Hole masks,
splits into flat walls (to preserve perspective), and returns wall polygons.
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

EXTERIOR_MIN_WALL_AREA_RATIO = float(os.getenv("EXTERIOR_MIN_WALL_AREA_RATIO", "0.015"))
EXTERIOR_POLYGON_EPS_RATIO = float(os.getenv("EXTERIOR_POLYGON_EPS_RATIO", "0.004"))
EXTERIOR_SEAM_ENABLE = os.getenv("EXTERIOR_SEAM_ENABLE", "1") in {"1", "true", "TRUE", "yes", "YES"}

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
EXTERIOR_DEBUG_DIR = os.path.join(PROJECT_ROOT, "exterior-debug")

def encode_mask_png_base64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Failed to encode mask PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _align_perspective_corners(component_mask: np.ndarray) -> list[list[int]]:
    """
    Finds perspective corners [TL, BL, BR, TR] bounded strictly to the top/bottom edges of the wall.
    Prevents minAreaRect from creating severely rotated quads that break frontend texturing.
    """
    y_coords, x_coords = np.where(component_mask > 0)
    if len(y_coords) == 0:
        return [[0,0], [0,1], [1,1], [1,0]]

    # 1. Find bounding box of the mask
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # 2. Extract top and bottom bands (e.g. top 15% and bottom 15%)
    h = max_y - min_y
    band = max(10, int(h * 0.15))
    
    top_band = np.where((component_mask > 0) & (np.mgrid[0:component_mask.shape[0], 0:component_mask.shape[1]][0] <= min_y + band))
    bot_band = np.where((component_mask > 0) & (np.mgrid[0:component_mask.shape[0], 0:component_mask.shape[1]][0] >= max_y - band))

    # Fallback to simple bbox if bands are too thin
    if len(top_band[0]) == 0 or len(bot_band[0]) == 0:
        return [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]

    # 3. Find left/right extremes in the top band
    tl_x = np.min(top_band[1])
    tl_y = top_band[0][np.argmin(top_band[1])]
    tr_x = np.max(top_band[1])
    tr_y = top_band[0][np.argmax(top_band[1])]

    # 4. Find left/right extremes in the bottom band
    bl_x = np.min(bot_band[1])
    bl_y = bot_band[0][np.argmin(bot_band[1])]
    br_x = np.max(bot_band[1])
    br_y = bot_band[0][np.argmax(bot_band[1])]

    return [
        [int(tl_x), int(tl_y)],
        [int(bl_x), int(bl_y)],
        [int(br_x), int(br_y)],
        [int(tr_x), int(tr_y)],
    ]

def _extract_wall_from_component(component_mask: np.ndarray) -> dict | None:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    # New: aligned perspective corners (prevents diagonal rotations)
    corners = _align_perspective_corners(component_mask)

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

def _auto_split_by_corner_seam(
    wall_mask: np.ndarray,
    image_bgr: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[np.ndarray]:
    """
    Finds a vertical edge (corner of a house) in the grayscale image
    within the mask area, and splits the house into 2 separate walls
    to preserve 3D perspective mapping.
    """
    if not EXTERIOR_SEAM_ENABLE:
        return [wall_mask]
        
    x_coords, y_coords = np.where(wall_mask > 0)
    if len(x_coords) == 0:
        return [wall_mask]
        
    y1i, y2i = np.min(x_coords), np.max(x_coords)
    x1i, x2i = np.min(y_coords), np.max(y_coords)
    
    bw = x2i - x1i + 1
    bh = y2i - y1i + 1
    if bw < 50 or bh < 50:
        return [wall_mask]

    crop = image_bgr[y1i:y2i+1, x1i:x2i+1]
    mask_crop = wall_mask[y1i:y2i+1, x1i:x2i+1]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=(mask_crop > 0).astype(np.uint8) * 255)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(40, int(0.15 * bh)),
        minLineLength=max(30, int(0.40 * bh)),
        maxLineGap=max(10, int(0.06 * bh)),
    )
    
    if lines is None:
        return [wall_mask]

    min_area = cv2.countNonZero(mask_crop) * 0.25 # Each half must be at least 25% of total area
    best_score = -1.0
    best_line_local = None

    for seg in lines.reshape(-1, 4):
        lx1, ly1, lx2, ly2 = [float(v) for v in seg.tolist()]
        dx = lx2 - lx1
        dy = ly2 - ly1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < 0.40 * bh:
            continue
            
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if abs(angle - 90.0) > 12: # Only nearly vertical lines
            continue

        mid_x = (lx1 + lx2) * 0.5
        mid_x_norm = mid_x / max(1.0, float(bw - 1))
        
        # Line must be somewhat central (not at the very edge)
        if mid_x_norm < 0.2 or mid_x_norm > 0.8:
            continue

        pt_a = np.array([mid_x, 0.0], dtype=np.float64)
        pt_b = np.array([mid_x, float(bh - 1)], dtype=np.float64)
        split_res = _split_mask_by_line(mask_crop, float(pt_a[0]), float(pt_a[1]), float(pt_b[0]), float(pt_b[1]))
        if split_res is None:
            continue
            
        left, right = split_res
        area_l = cv2.countNonZero(left)
        area_r = cv2.countNonZero(right)
        
        # strict limit to prevent cutting tiny balconies
        if area_l < min_area or area_r < min_area:
            continue
            
        balance = min(area_l, area_r) / max(area_l, area_r)
        
        # evaluate gradient strength along this line
        g_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        stripe = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(stripe, (int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1])), 255, 7)
        vals = np.abs(g_x[stripe > 0])
        edge_score = float(np.clip(vals.mean() / 255.0, 0.0, 1.0)) if vals.size > 0 else 0.0
        
        score = (0.5 * balance) + (0.5 * edge_score)
        
        if score > best_score:
            best_score = score
            best_line_local = (pt_a, pt_b)

    if best_line_local is None:
        return [wall_mask]

    pt_a_local, pt_b_local = best_line_local
    split_global = _split_mask_by_line(
        wall_mask,
        float(pt_a_local[0] + x1i),
        float(pt_a_local[1] + y1i),
        float(pt_b_local[0] + x1i),
        float(pt_b_local[1] + y1i),
    )
    
    if split_global is None:
        return [wall_mask]

    left_g, right_g = split_global
    return [left_g, right_g]

async def run_exterior_pipeline(image_bytes: bytes, image_width: int, image_height: int) -> dict:
    """
    1) Call SegFormer -> returns 3-channel PNG (R=Wall, G=Holes, B=Balconies)
    2) Find vertical corner seam to split house into 2 walls (for perspective)
    3) wall_minus_holes = Wall AND NOT Holes AND NOT Balconies
    4) Find connected components -> Walls
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
        return default_resp
        
    if masks_bgr.shape[:2] != (image_height, image_width):
        masks_bgr = cv2.resize(masks_bgr, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    wall_mask = masks_bgr[:, :, 2]      # Red channel
    holes_mask = masks_bgr[:, :, 1]     # Green channel
    balcony_mask = masks_bgr[:, :, 0]   # Blue channel
    
    # 1. Clean up wall mask FIRST (fill tiny holes/errors before we cut real windows)
    # Using a much smaller kernel (3x3) so we don't accidentally erase large architectural features
    smooth_kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, smooth_kernel_small, iterations=1)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, smooth_kernel_small, iterations=1)

    # 2. Split into Left/Right walls if there is a strong corner seam
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if image_bgr.shape[:2] != (image_height, image_width):
        image_bgr = cv2.resize(image_bgr, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    split_walls = _auto_split_by_corner_seam(wall_mask, image_bgr, image_width, image_height)

    # 3. Process holes & exclusions (windows, sky, trees)
    # Dilate holes slightly so window frames are carved out
    exclude_mask = cv2.bitwise_or(holes_mask, balcony_mask)
    kernel_holes = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    exclude_dilated = cv2.dilate(exclude_mask, kernel_holes, iterations=1)

    component_masks = []
    min_area = image_width * image_height * EXTERIOR_MIN_WALL_AREA_RATIO

    # 4. Subtract holes from each wall piece, clean it gently, and extract components
    for part_mask in split_walls:
        part_minus_holes = cv2.bitwise_and(part_mask, cv2.bitwise_not(exclude_dilated))
        
        # Very gentle smoothing for the final mask
        smooth_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        part_minus_holes = cv2.morphologyEx(part_minus_holes, cv2.MORPH_OPEN, smooth_final, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(part_minus_holes, connectivity=8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                comp = np.zeros_like(part_minus_holes)
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
    
    # Combined mask for frontend
    wall_minus_holes_combined = cv2.bitwise_and(wall_mask, cv2.bitwise_not(exclude_dilated))

    return {
        "walls": walls,
        "image_size": {"width": image_width, "height": image_height},
        "masks": {
            "wall": encode_mask_png_base64(wall_mask),
            "holes": encode_mask_png_base64(exclude_mask),
            "wall_minus_holes": encode_mask_png_base64(wall_minus_holes_combined),
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
