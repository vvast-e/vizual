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


def _warp_texture_overlay_sync(
    texture_bytes: bytes,
    corners: list[list[float]],
    image_width: int,
    image_height: int,
    texture_scale: float,
    opacity: float,
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

    dst_pts_raw = np.array(corners, dtype=np.float32).reshape(4, 2)
    dst_pts = _order_points_tl_tr_br_bl(dst_pts_raw)

    W = int(image_width)
    H = int(image_height)
    tiled = _tile_texture_rgba(tex, W, H, texture_scale)

    src_pts = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        tiled,
        M,
        dsize=(W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [dst_pts.astype(np.int32)], 255)

    safe_opacity = float(max(0.0, min(1.0, opacity)))
    alpha = (mask.astype(np.float32) / 255.0) * safe_opacity
    warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0
    out_alpha = (warped_alpha * alpha * 255.0).clip(0, 255).astype(np.uint8)
    warped[:, :, 3] = out_alpha

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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/warp-wall-texture")
async def warp_wall_texture(
    texture: UploadFile = File(...),
    corners: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),
    texture_scale: float = Form(0.25),
    opacity: float = Form(0.85),
):
    try:
        texture_bytes = await texture.read()
        corners_parsed = json.loads(corners)
        png_bytes = await asyncio.to_thread(
            _warp_texture_overlay_sync,
            texture_bytes,
            corners_parsed,
            int(image_width),
            int(image_height),
            float(texture_scale),
            float(opacity),
        )
        if WARP_DEBUG_SAVE:
            try:
                os.makedirs(WARP_DEBUG_DIR, exist_ok=True)
                payload = {
                    "corners": corners_parsed,
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
            except Exception as save_err:
                print(f"[warp-debug] save failed: {save_err}")
        return Response(content=png_bytes, media_type="image/png")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid corners JSON")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
