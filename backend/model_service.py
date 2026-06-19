"""
Local model service: GroundingDINO (bbox) + SAM (mask refinement) — PyTorch.
Run: uvicorn model_service:app --port 8001

Models loaded via transformers directly (no ONNX export needed).
"""
import io
import json
import os
import base64
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

# Устройство: можно переключить на "cuda" через env MODEL_DEVICE для прода с GPU.
DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# DSINE — монокулярные нормали поверхности для деления фасада на планарные грани.
# Модель: baegwangbin/DSINE (CVPR 2024 Oral). Загружается из torch.hub cache.
# Weights: camenduru/DSINE/resolve/main/dsine.pt (скачивается при первом запуске).
DSINE_HUB_DIR: str = ""   # заполняется при загрузке

gdino_model = None
gdino_processor = None
sam_model = None
sam_processor = None
dsine_model = None   # DSINE_v02 instance (на CPU/DEVICE)


def _load_dsine():
    """Загрузить DSINE_v02 на DEVICE. Обходит хардкод .to(0) через патч torch.Tensor.to."""
    import types, sys
    import torch
    global dsine_model, DSINE_HUB_DIR

    hub_dir = torch.hub.get_dir()
    dsine_src = os.path.join(hub_dir, "baegwangbin_DSINE_main")

    # Скачать репо, если ещё нет.
    if not os.path.isdir(dsine_src):
        print("[model_service] Downloading DSINE repo...")
        torch.hub.load("baegwangbin/DSINE", "DSINE", trust_repo=True)  # упадёт на cuda, но репо скачает
    DSINE_HUB_DIR = dsine_src

    if dsine_src not in sys.path:
        sys.path.insert(0, dsine_src)

    # Патч: перехватить .to(int) → .to(DEVICE) во время __init__ (иначе → cuda:0)
    _orig_tensor_to = torch.Tensor.to
    def _safe_to(self, *args, **kwargs):
        if args and isinstance(args[0], int):
            return _orig_tensor_to(self, DEVICE)
        return _orig_tensor_to(self, *args, **kwargs)
    torch.Tensor.to = _safe_to

    try:
        args = types.SimpleNamespace(
            NNET_architecture="v02", NNET_output_dim=3, NNET_output_type="R",
            NNET_feature_dim=64, NNET_hidden_dim=64, NNET_encoder_B=5,
            NNET_decoder_NF=2048, NNET_decoder_BN=False, NNET_decoder_down=8,
            NNET_learned_upsampling=False,
            NRN_prop_ps=5, NRN_num_iter_train=5, NRN_num_iter_test=5, NRN_ray_relu=False,
        )
        from models.dsine.v02 import DSINE_v02
        model = DSINE_v02(args)
    finally:
        torch.Tensor.to = _orig_tensor_to  # восстановить патч в любом случае

    # Веса: скачать из HuggingFace если нет в кэше.
    weights_path = os.path.join(hub_dir, "checkpoints", "dsine.pt")
    if not os.path.exists(weights_path):
        print("[model_service] Downloading DSINE weights (~278 MB)...")
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt",
            file_name="dsine.pt",
            map_location=DEVICE,
        )
    else:
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)

    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    model.pixel_coords = model.pixel_coords.to(DEVICE)
    model = model.to(DEVICE)
    print(f"[model_service] DSINE loaded ({sum(p.numel() for p in model.parameters())//1_000_000}M params).")
    return model


def _load_all_models():
    global gdino_model, gdino_processor, sam_model, sam_processor, dsine_model

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers import SamModel, SamProcessor

    # GroundingDINO-tiny
    gdino_id = "IDEA-Research/grounding-dino-tiny"
    print(f"[model_service] Loading GroundingDINO from {gdino_id}...")
    gdino_processor = AutoProcessor.from_pretrained(gdino_id, backend="torchvision")
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(DEVICE)
    gdino_model.eval()
    print("[model_service] GroundingDINO loaded.")

    # SAM ViT-Base (stable, tested)
    sam_id = "facebook/sam-vit-base"
    print(f"[model_service] Loading SAM from {sam_id}...")
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)
    sam_model.eval()
    print("[model_service] SAM loaded.")

    # DSINE — нормали поверхности для деления фасада на планарные грани.
    print("[model_service] Loading DSINE surface normals model...")
    dsine_model = _load_dsine()


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
        results["boxes"].detach().cpu().numpy(),
        results["scores"].detach().cpu().numpy(),
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

    # SAM returns 3 masks per prompt. We pick the one with the highest IoU score
    # to avoid fragmented or weirdly shaped boundaries.
    iou_scores = outputs.iou_scores.detach().cpu().numpy()
    best_mask_idx = int(np.argmax(iou_scores[0, 0]))

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
    )

    mask_np = masks[0][0, best_mask_idx].numpy().astype(np.uint8) * 255

    ok, buf = cv2.imencode(".png", mask_np)
    if not ok:
        raise HTTPException(500, "Failed to encode mask PNG")

    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/sam_batch")
async def sam_segment_batch(
    image: UploadFile = File(...),
    bboxes: str = Form(...),
    multimask_output: bool = Form(False),
):
    """Return masks for a batch of bbox prompts."""
    if sam_model is None or sam_processor is None:
        raise HTTPException(503, "SAM not loaded yet")

    image_bytes = await image.read()
    pil_image = _read_image_pil(image_bytes)

    try:
        parsed = json.loads(bboxes)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid bboxes json: {e}") from e
    if not isinstance(parsed, list) or len(parsed) == 0:
        raise HTTPException(400, "bboxes must be a non-empty JSON array")
    if not all(isinstance(b, list) and len(b) == 4 for b in parsed):
        raise HTTPException(400, "each bbox must be [x1, y1, x2, y2]")

    input_boxes = [[[
        float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ] for b in parsed]]

    inputs = sam_processor(
        pil_image,
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    outputs = sam_model(**inputs, multimask_output=bool(multimask_output))

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
    )

    iou_scores = outputs.iou_scores.detach().cpu().numpy()[0]
    all_masks = masks[0]
    masks_b64: list[str] = []
    scores: list[float] = []

    for i in range(all_masks.shape[0]):
        best_idx = int(np.argmax(iou_scores[i]))
        mask_np = all_masks[i, best_idx].numpy().astype(np.uint8) * 255
        ok, buf = cv2.imencode(".png", mask_np)
        if not ok:
            raise HTTPException(500, f"failed to encode mask png for bbox index {i}")
        masks_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        scores.append(float(iou_scores[i, best_idx]))

    return {"masks": masks_b64, "scores": scores, "count": len(masks_b64)}


@app.post("/normals")
async def estimate_normals(
    image: UploadFile = File(...),
):
    """
    Вернуть карту нормалей поверхности (DSINE) для изображения.
    Ответ: raw float32 bytes (numpy array, shape H×W×3, C-order), единичные векторы.
    Размеры — в заголовках X-Normals-Height / X-Normals-Width / X-Normals-Channels.
    Система координат камеры: X вправо, Y вниз, Z от камеры.
    """
    if dsine_model is None:
        raise HTTPException(503, "DSINE model not loaded yet")

    image_bytes = await image.read()
    img_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Cannot decode image")

    import torch.nn.functional as F
    from torchvision import transforms as T

    orig_H, orig_W = img_bgr.shape[:2]

    # RGB float32 [0,1] → tensor 1×3×H×W
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Pad до кратного 32 (требование DSINE)
    pad_h = (32 - orig_H % 32) % 32
    pad_w = (32 - orig_W % 32) % 32
    t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    # ImageNet нормализация
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = norm(t)

    # Intrinsics: FOV=60° (разумный дефолт для фасадных фото)
    fov = float(os.getenv("DSINE_FOV_DEG", "60"))
    f = orig_W / (2 * np.tan(np.radians(fov / 2)))
    intrins = torch.tensor(
        [[f, 0, orig_W / 2], [0, f, orig_H / 2], [0, 0, 1]],
        dtype=torch.float32,
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_norm = dsine_model(t, intrins=intrins)[-1]  # (1, 3, H+pad, W+pad)
        pred_norm = pred_norm[:, :, :orig_H, :orig_W]    # обрезать паддинг

    # (1, 3, H, W) → (H, W, 3) float32, уже единичные векторы
    normals_np = pred_norm[0].permute(1, 2, 0).cpu().float().numpy()

    h, w, c = normals_np.shape
    raw_bytes = normals_np.astype(np.float32).tobytes()

    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Normals-Height": str(h),
            "X-Normals-Width": str(w),
            "X-Normals-Channels": str(c),
        },
    )
