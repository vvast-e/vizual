"""
Local model service: GroundingDINO (bbox) + SAM (mask refinement) + DSINE (normals).
Run: uvicorn model_service:app --port 8001

Acceleration:
  - GDino and DSINE support ONNX int8 inference (set USE_ONNX_GDINO=1 / USE_ONNX_DSINE=1).
  - If ONNX artefacts absent or disabled, falls back to torch dynamic int8 (GDino)
    or plain torch (DSINE).
  - INFER_MAX_SIDE=1024: input images are capped before ML inference; results are
    scaled back — no change to API contracts.
"""
import io
import json
import os
import base64
import sys
import types
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

DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# ── ONNX / inference settings ──────────────────────────────────────────────────
MODELS_ONNX_DIR = os.getenv("MODELS_ONNX_DIR", os.path.join(_BACKEND_DIR, "models", "onnx"))
_use_onnx_env = lambda key: os.getenv(key, "auto").lower()  # "1"/"0"/"auto"

# Maximum long-side for ML inference (results scaled back to original size)
INFER_MAX_SIDE = int(os.getenv("INFER_MAX_SIDE", "1024"))

DSINE_HUB_DIR: str = ""

# Model holders
gdino_model = None        # torch or None
gdino_processor = None
gdino_ort_session = None  # onnxruntime.InferenceSession or None
gdino_use_onnx: bool = False

sam_model = None
sam_processor = None

dsine_model = None        # torch or None
dsine_ort_session = None  # onnxruntime.InferenceSession or None
dsine_use_onnx: bool = False


# ── helpers ────────────────────────────────────────────────────────────────────

def _ort_session(path: str):
    """Create ORT session with CPU optimizations."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "4"))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess


def _resize_cap(img: np.ndarray, max_side: int = INFER_MAX_SIDE):
    """Downscale img so max(H,W) <= max_side. Returns (resized_bgr, scale)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _pil_resize_cap(pil_img: Image.Image, max_side: int = INFER_MAX_SIDE):
    """Downscale PIL image so max(H,W) <= max_side. Returns (resized_pil, scale)."""
    w, h = pil_img.size
    if max(h, w) <= max_side:
        return pil_img, 1.0
    scale = max_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((new_w, new_h), Image.LANCZOS), scale


# ── DSINE loading ──────────────────────────────────────────────────────────────

def _load_dsine_torch():
    global dsine_model, DSINE_HUB_DIR

    hub_dir = torch.hub.get_dir()
    dsine_src = os.path.join(hub_dir, "baegwangbin_DSINE_main")

    if not os.path.isdir(dsine_src):
        print("[model_service] Downloading DSINE repo...")
        try:
            torch.hub.load("baegwangbin/DSINE", "DSINE", trust_repo=True)
        except Exception:
            pass
    DSINE_HUB_DIR = dsine_src

    if dsine_src not in sys.path:
        sys.path.insert(0, dsine_src)

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
        torch.Tensor.to = _orig_tensor_to

    weights_path = os.path.join(hub_dir, "checkpoints", "dsine.pt")
    if not os.path.exists(weights_path):
        print("[model_service] Downloading DSINE weights (~278 MB)...")
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt",
            file_name="dsine.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    else:
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)

    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    model.pixel_coords = model.pixel_coords.to(DEVICE)
    model = model.to(DEVICE)
    print(f"[model_service] DSINE torch loaded ({sum(p.numel() for p in model.parameters())//1_000_000}M params).")
    return model


# ── GDino loading ──────────────────────────────────────────────────────────────

def _apply_gdino_dynamic_int8(model):
    """Quantize Linear layers to int8 in-place (no ONNX needed). ~1.5-2x CPU speedup."""
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("[model_service] GDino: torch dynamic int8 quantization applied.")
    except Exception as exc:
        print(f"[model_service] GDino: dynamic int8 failed ({exc}), using fp32.")
    return model


# ── main loader ────────────────────────────────────────────────────────────────

def _load_all_models():
    global gdino_model, gdino_processor, gdino_ort_session, gdino_use_onnx
    global sam_model, sam_processor
    global dsine_model, dsine_ort_session, dsine_use_onnx

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers import SamModel, SamProcessor

    # ── GDino ──────────────────────────────────────────────────────────────────
    gdino_id = "IDEA-Research/grounding-dino-tiny"
    print(f"[model_service] Loading GroundingDINO from {gdino_id}...")
    gdino_processor = AutoProcessor.from_pretrained(gdino_id, backend="torchvision")

    onnx_gdino = os.path.join(MODELS_ONNX_DIR, "gdino_int8.onnx")
    use_onnx_gdino = _use_onnx_env("USE_ONNX_GDINO")
    if (use_onnx_gdino in ("1", "auto")) and os.path.exists(onnx_gdino):
        try:
            gdino_ort_session = _ort_session(onnx_gdino)
            gdino_use_onnx = True
            print(f"[model_service] GDino: ONNX int8 session loaded ({onnx_gdino}).")
            # Torch model not needed for inference, but processor is required for post-process
            gdino_model = None
        except Exception as exc:
            print(f"[model_service] GDino: ONNX load failed ({exc}), falling back to torch.")
            gdino_use_onnx = False

    if not gdino_use_onnx:
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(DEVICE)
        gdino_model.eval()
        if use_onnx_gdino != "0":  # don't quantize if explicitly disabled
            gdino_model = _apply_gdino_dynamic_int8(gdino_model)
        print("[model_service] GDino: torch model ready.")

    # ── SAM ────────────────────────────────────────────────────────────────────
    sam_id = "facebook/sam-vit-base"
    print(f"[model_service] Loading SAM from {sam_id}...")
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)
    sam_model.eval()
    print("[model_service] SAM loaded.")

    # ── DSINE ──────────────────────────────────────────────────────────────────
    print("[model_service] Loading DSINE surface normals model...")
    use_onnx_dsine = _use_onnx_env("USE_ONNX_DSINE")
    # Try int8 first, then fp32 ONNX — both give ORT graph-fusion speedup.
    _dsine_candidates = [
        os.path.join(MODELS_ONNX_DIR, "dsine_int8.onnx"),
        os.path.join(MODELS_ONNX_DIR, "dsine_fp32.onnx"),
    ]
    if use_onnx_dsine in ("1", "auto"):
        for _candidate in _dsine_candidates:
            if os.path.exists(_candidate):
                try:
                    dsine_ort_session = _ort_session(_candidate)
                    dsine_use_onnx = True
                    print(f"[model_service] DSINE: ONNX session loaded ({_candidate}).")
                    break
                except Exception as exc:
                    print(f"[model_service] DSINE: ONNX load failed for {_candidate} ({exc}), trying next.")

    if not dsine_use_onnx:
        dsine_model = _load_dsine_torch()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    yield


app = FastAPI(title="Model Service (GroundingDINO + SAM + DSINE)", lifespan=lifespan)


def _read_image_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ── /gdino ─────────────────────────────────────────────────────────────────────

@app.post("/gdino")
async def grounding_dino_detect(
    image: UploadFile = File(...),
    prompt: str = Form("house . building . facade"),
    score_threshold: float = Form(0.3),
):
    """Return bounding boxes for objects matching the text prompt."""
    if not gdino_use_onnx and gdino_model is None:
        raise HTTPException(503, "GroundingDINO not loaded yet")
    if gdino_processor is None:
        raise HTTPException(503, "GroundingDINO processor not loaded yet")

    image_bytes = await image.read()
    pil_image = _read_image_pil(image_bytes)
    pil_small, scale = _pil_resize_cap(pil_image)

    inputs = gdino_processor(images=pil_small, text=prompt, return_tensors="pt")

    if gdino_use_onnx:
        import onnxruntime as ort  # noqa: F401 (already imported via session)
        ort_logits, ort_boxes = gdino_ort_session.run(
            ["logits", "pred_boxes"],
            {
                "pixel_values":    inputs["pixel_values"].numpy(),
                "input_ids":       inputs["input_ids"].numpy(),
                "attention_mask":  inputs["attention_mask"].numpy(),
                "token_type_ids":  inputs["token_type_ids"].numpy(),
                "text_token_mask": inputs["text_token_mask"].numpy(),
            },
        )

        class _FakeOut:
            logits = torch.tensor(ort_logits)
            pred_boxes = torch.tensor(ort_boxes)

        results = gdino_processor.post_process_grounded_object_detection(
            _FakeOut(),
            inputs["input_ids"],
            box_threshold=score_threshold,
            text_threshold=score_threshold,
            target_sizes=[pil_small.size[::-1]],
        )[0]
    else:
        outputs = gdino_model(**inputs)
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=score_threshold,
            text_threshold=score_threshold,
            target_sizes=[pil_small.size[::-1]],
        )[0]

    inv = 1.0 / scale if scale != 1.0 else 1.0
    bboxes = []
    for box, score, label in zip(
        results["boxes"].detach().cpu().numpy(),
        results["scores"].detach().cpu().numpy(),
        results["labels"],
    ):
        x1, y1, x2, y2 = [v * inv for v in box.tolist()]
        bboxes.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score": round(float(score), 4),
            "label": label,
        })

    return {"bboxes": bboxes}


# ── /sam ───────────────────────────────────────────────────────────────────────

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


# ── /sam_batch ─────────────────────────────────────────────────────────────────

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


# ── /normals ───────────────────────────────────────────────────────────────────

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
    if not dsine_use_onnx and dsine_model is None:
        raise HTTPException(503, "DSINE model not loaded yet")

    image_bytes = await image.read()
    img_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Cannot decode image")

    orig_H, orig_W = img_bgr.shape[:2]

    if dsine_use_onnx:
        # ── ONNX path ──────────────────────────────────────────────────────────
        # Подготовка ОБЯЗАНА совпадать с export_dsine_onnx.py (общий модуль dsine_canvas).
        from dsine_canvas import make_dsine_canvas
        fov = float(os.getenv("DSINE_FOV_DEG", "60"))
        t, intrins, _, content_hw = make_dsine_canvas(img_bgr, fov_deg=fov)
        content_H, content_W = content_hw

        pred = dsine_ort_session.run(
            ["normals"],
            {"image": t.numpy(), "intrins": intrins.numpy()},
        )[0]  # (1, 3, CANVAS, CANVAS)

        # Обрезать паддинг → (content_H, content_W, 3)
        normals_np = pred[0].transpose(1, 2, 0)[:content_H, :content_W]

        # Ресайз обратно к исходному размеру + перенормировка
        if (content_H, content_W) != (orig_H, orig_W):
            normals_np = cv2.resize(normals_np, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
            norms = np.linalg.norm(normals_np, axis=2, keepdims=True)
            normals_np = normals_np / (norms + 1e-8)

    else:
        # ── torch path (оригинальная логика) ──────────────────────────────────
        import torch.nn.functional as F
        from torchvision import transforms as T

        img_small, scale = _resize_cap(img_bgr)
        infer_H, infer_W = img_small.shape[:2]

        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        pad_h = (32 - infer_H % 32) % 32
        pad_w = (32 - infer_W % 32) % 32
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t = norm(t)

        fov = float(os.getenv("DSINE_FOV_DEG", "60"))
        f = infer_W / (2 * np.tan(np.radians(fov / 2)))
        intrins = torch.tensor(
            [[f, 0, infer_W / 2], [0, f, infer_H / 2], [0, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_norm = dsine_model(t, intrins=intrins)[-1]
            pred_norm = pred_norm[:, :, :infer_H, :infer_W]

        normals_np = pred_norm[0].permute(1, 2, 0).cpu().float().numpy()

        if scale != 1.0:
            normals_np = cv2.resize(normals_np, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
            norms = np.linalg.norm(normals_np, axis=2, keepdims=True)
            normals_np = normals_np / (norms + 1e-8)

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
