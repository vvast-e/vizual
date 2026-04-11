"""
Local model service: SegFormer semantic segmentation — PyTorch.
Run: uvicorn model_service:app --port 8001
"""
import io
import os
from contextlib import asynccontextmanager

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
_HF_HOME = os.path.join(_PROJECT_ROOT, "models", "huggingface")
os.makedirs(_HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_HOME)

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

DEVICE = "cpu"

segformer_model = None
segformer_processor = None

# ADE20K labels to map for our exterior pipeline
WALL_TARGET_TERMS = ["wall", "building", "house", "facade", "edifice", "hut", "hovel", "shed", "cabin"]
HOLE_TARGET_TERMS = ["window", "windowpane", "door", "double door", "glass", "blind", "screen door"]
BALCONY_TERMS = ["balcony"] # Can be treated as holes or excluded entirely

wall_class_ids = []
hole_class_ids = []
balcony_class_ids = []

def _load_models():
    global segformer_model, segformer_processor, wall_class_ids, hole_class_ids, balcony_class_ids

    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    # SegFormer B1 is extremely fast on CPU and gives excellent architectural segmentation
    model_id = "nvidia/segformer-b1-finetuned-ade-512-512"
    print(f"[model_service] Loading SegFormer from {model_id}...")
    segformer_processor = SegformerImageProcessor.from_pretrained(model_id)
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(DEVICE)
    segformer_model.eval()

    # Pre-compute class mappings
    id2label = segformer_model.config.id2label
    for cls_id, label in id2label.items():
        label_lower = label.lower()
        if any(term in label_lower for term in WALL_TARGET_TERMS):
            wall_class_ids.append(cls_id)
        elif any(term in label_lower for term in HOLE_TARGET_TERMS):
            hole_class_ids.append(cls_id)
        elif any(term in label_lower for term in BALCONY_TERMS):
            balcony_class_ids.append(cls_id)
            
    print(f"[model_service] Mapped {len(wall_class_ids)} Wall classes and {len(hole_class_ids)} Hole classes.")
    print("[model_service] SegFormer loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="Model Service (SegFormer)", lifespan=lifespan)

@app.post("/segformer")
async def segformer_predict(image: UploadFile = File(...)):
    """
    Returns a PNG image containing 3 channels (BGR):
    - R (Red):   Pixels belonging to Walls / Buildings
    - G (Green): Pixels belonging to Windows / Doors
    - B (Blue):  Pixels belonging to Balconies
    """
    if segformer_model is None or segformer_processor is None:
        raise HTTPException(503, "SegFormer not loaded yet")

    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    inputs = segformer_processor(images=pil_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = segformer_model(**inputs)
        
    logits = outputs.logits
    # Rescale logits back to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False
    )
    
    # Get the class index with highest probability for each pixel
    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # Map classes to binary masks
    wall_mask = np.isin(pred, wall_class_ids).astype(np.uint8) * 255
    holes_mask = np.isin(pred, hole_class_ids).astype(np.uint8) * 255
    balcony_mask = np.isin(pred, balcony_class_ids).astype(np.uint8) * 255

    # Encode all masks into a single 3-channel image (BGR format for OpenCV)
    # Red = Wall, Green = Holes, Blue = Balconies
    combined = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    combined[:, :, 2] = wall_mask      # Red
    combined[:, :, 1] = holes_mask     # Green
    combined[:, :, 0] = balcony_mask   # Blue

    ok, buf = cv2.imencode(".png", combined)
    if not ok:
        raise HTTPException(500, "Failed to encode prediction masks")

    return Response(content=buf.tobytes(), media_type="image/png")
