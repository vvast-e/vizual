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
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

# Ensure we run in backend directory so relative paths in submodules work
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
WARP_DEBUG_DIR = os.path.join(PROJECT_ROOT, "warp-debug")
WARP_DEBUG_SAVE = True

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
    """Order 4 points as [TL, TR, BR, BL]. pts shape: (4,2)."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


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


def _warp_texture_overlay_sync(
    texture_bytes: bytes,
    corners: list[list[float]],
    image_width: int,
    image_height: int,
    texture_scale: float,
    opacity: float,
    polygon: list[list[float]] | None = None,
) -> bytes:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be > 0")
    if len(corners) != 4:
        raise ValueError("corners must have 4 points")

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
    use_polygon = bool(polygon) and len(polygon) >= 3
    poly_pts: np.ndarray | None = None
    if use_polygon:
        poly_pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        poly_pts[:, 0] = np.clip(poly_pts[:, 0], 0, W - 1)
        poly_pts[:, 1] = np.clip(poly_pts[:, 1], 0, H - 1)
        poly_pts = _sanitize_polygon_points(poly_pts)

    dst_pts_raw = np.array(corners, dtype=np.float32).reshape(4, 2)
    dst_pts = _order_points_tl_tr_br_bl(dst_pts_raw)

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

    warped: np.ndarray
    if use_polygon and poly_pts is not None and len(poly_pts) >= 3:
        # Inverse UV mapping: perspective from corners, full fill inside polygon.
        warped = np.zeros((H, W, 4), dtype=np.uint8)
        try:
            M_inv = np.linalg.inv(M)
            ys, xs = np.where(mask > 0)
            if xs.size > 0:
                ones = np.ones_like(xs, dtype=np.float64)
                pts = np.stack([xs.astype(np.float64), ys.astype(np.float64), ones], axis=0)
                src = M_inv @ pts
                wv = src[2]
                valid = np.abs(wv) > 1e-8
                sx = np.zeros_like(wv)
                sy = np.zeros_like(wv)
                sx[valid] = src[0][valid] / wv[valid]
                sy[valid] = src[1][valid] / wv[valid]

                # Repeat texture coordinates to cover polygon outside projected quad.
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
            print(f"[warp-debug] mode=inv_uv_polygon pixels={int(xs.size)}")
        except Exception as inv_err:
            print(f"[warp-debug] inv_uv failed, fallback quad warp: {inv_err}")
            warped = cv2.warpPerspective(
                tiled,
                M,
                dsize=(W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
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
    h, w = img.shape[:2]

    from exterior_pipeline import run_exterior_pipeline

    try:
        result = await run_exterior_pipeline(contents, w, h)
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
):
    """Split exterior facade mask by user-drawn line. Returns new walls[]."""
    from exterior_pipeline import split_exterior_by_line

    try:
        walls = split_exterior_by_line(
            mask_base64, float(x1), float(y1), float(x2), float(y2),
            int(image_width), int(image_height),
        )
        return {"walls": walls}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/warp-wall-texture")
async def warp_wall_texture(
    texture: UploadFile = File(...),
    corners: str = Form(...),
    polygon: str | None = Form(None),
    image_width: int = Form(...),
    image_height: int = Form(...),
    texture_scale: float = Form(0.25),
    opacity: float = Form(0.85),
):
    try:
        texture_bytes = await texture.read()
        corners_parsed = json.loads(corners)
        polygon_parsed = None
        if polygon:
            polygon_parsed = json.loads(polygon)
        print(
            f"[warp-wall-texture] corners_len={len(corners_parsed) if isinstance(corners_parsed, list) else '??'} "
            f"polygon_len={len(polygon_parsed) if isinstance(polygon_parsed, list) else 0} "
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
        )
        if WARP_DEBUG_SAVE:
            try:
                os.makedirs(WARP_DEBUG_DIR, exist_ok=True)
                payload = {
                    "corners": corners_parsed,
                    "polygon": polygon_parsed,
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
