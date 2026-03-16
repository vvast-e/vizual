"""
FastAPI backend for wall detection (room-wall-visualizer).
Run from backend directory: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio
import os
import tempfile
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Ensure we run in backend directory so relative paths in submodules work
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)

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


@app.get("/api/health")
def health():
    return {"status": "ok"}
