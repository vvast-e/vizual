"""
Local model service: GroundingDINO (bbox) + SAM (mask refinement) — PyTorch.
Run: uvicorn model_service:app --port 8001

Models loaded via transformers directly (no ONNX export needed).
"""
import io
import json
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

DEVICE = "cpu"

gdino_model = None
gdino_processor = None
sam_model = None
sam_processor = None


def _load_all_models():
    global gdino_model, gdino_processor, sam_model, sam_processor

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers import SamModel, SamProcessor

    # GroundingDINO-tiny
    gdino_id = "IDEA-Research/grounding-dino-tiny"
    print(f"[model_service] Loading GroundingDINO from {gdino_id}...")
    gdino_processor = AutoProcessor.from_pretrained(gdino_id, use_fast=False)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(DEVICE)
    gdino_model.eval()
    print("[model_service] GroundingDINO loaded.")

    # MobileSAM
    sam_id = "ChaoningZhang/MobileSAM"
    print(f"[model_service] Loading MobileSAM from {sam_id}...")
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)
    sam_model.eval()
    print("[model_service] MobileSAM loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    yield


app = FastAPI(title="Model Service (GroundingDINO + SAM) ONNX", lifespan=lifespan)


def _read_image_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.post("/gdino")
async def grounding_dino_detect(
    image: UploadFile = File(...),
    prompt: str = Form("house . building . facade"),
    score_threshold: float = Form(0.3),
):
    """Return bounding boxes for objects matching the text prompt."""
    if gdino_model is None or gdino_processor is None:
        raise HTTPException(503, "GroundingDINO not loaded yet")

    image_bytes = await image.read()
    pil_image = _read_image_pil(image_bytes)

    inputs = gdino_processor(images=pil_image, text=prompt, return_tensors="pt")

    outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=score_threshold,
        text_threshold=score_threshold,
        target_sizes=[pil_image.size[::-1]],
    )[0]

    bboxes = []
    for box, score, label in zip(
        results["boxes"].cpu().numpy(),
        results["scores"].cpu().numpy(),
        results["labels"],
    ):
        x1, y1, x2, y2 = box.tolist()
        bboxes.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score": round(float(score), 4),
            "label": label,
        })

    return {"bboxes": bboxes}


@app.post("/sam")
async def sam_segment(
    image: UploadFile = File(...),
    bbox: str = Form(...),
):
    """Return a binary mask (PNG, 1-channel, 0/255) for the given bbox prompt."""
    if sam_model is None or sam_processor is None:
        raise HTTPException(503, "SAM not loaded yet")

    image_bytes = await image.read()
    pil_image = _read_image_pil(image_bytes)

    bbox_parsed = json.loads(bbox)
    if len(bbox_parsed) != 4:
        raise HTTPException(400, "bbox must be [x1, y1, x2, y2]")

    input_boxes = [[[
        float(bbox_parsed[0]),
        float(bbox_parsed[1]),
        float(bbox_parsed[2]),
        float(bbox_parsed[3]),
    ]]]

    inputs = sam_processor(
        pil_image,
        input_boxes=input_boxes,
        return_tensors="pt",
    )

    outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    mask_np = masks[0][0, 0].numpy().astype(np.uint8) * 255

    ok, buf = cv2.imencode(".png", mask_np)
    if not ok:
        raise HTTPException(500, "Failed to encode mask PNG")

    return Response(content=buf.tobytes(), media_type="image/png")
