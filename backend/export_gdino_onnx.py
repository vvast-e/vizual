"""
Export GroundingDINO-tiny → ONNX + int8 quantization.
Если ONNX-экспорт не заводится на torch 2.0.1 (deformable attention, кастом-опы)
— скрипт печатает причину и завершается с кодом 2. Рантайм подхватит
torch dynamic int8 фолбэк автоматически.

Usage (from backend/ with venv active):
    python export_gdino_onnx.py [--test-image PATH]

Output (при успехе):
    models/onnx/gdino_fp32.onnx
    models/onnx/gdino_int8.onnx   ← used at runtime
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)

_HF_HOME = os.path.join(os.path.dirname(BACKEND_DIR), "models", "huggingface")
os.makedirs(_HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_HOME)

ONNX_DIR = os.path.join(BACKEND_DIR, "models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

CANVAS = 1024
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "house . building . facade"   # тот же промпт, что в рантайме


# ── 1. Загрузка ────────────────────────────────────────────────────────────────

def load_gdino():
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    print(f"[export] Loading GroundingDINO from {GDINO_ID}...")
    processor = AutoProcessor.from_pretrained(GDINO_ID, backend="torchvision")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID)
    model.eval()
    print("[export] GroundingDINO loaded.")
    return model, processor


# ── 2. Подготовка фиксированных входов ────────────────────────────────────────

def make_inputs(pil_image: Image.Image, processor):
    inputs = processor(images=pil_image, text=PROMPT, return_tensors="pt")
    return inputs


# ── 3. Экспорт ────────────────────────────────────────────────────────────────

class GDinoExportWrapper(object):
    """Обёртка — не nn.Module, используем torch.onnx.export через forward."""
    pass


import torch
import torch.nn as nn


class _GDinoForward(nn.Module):
    """Берёт pixel_values + текстовые тензоры, отдаёт logits + pred_boxes."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids, text_token_mask):
        out = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            text_token_mask=text_token_mask,
        )
        return out.logits, out.pred_boxes


def export_onnx(model, inputs) -> str:
    fp32_path = os.path.join(ONNX_DIR, "gdino_fp32.onnx")
    wrapper = _GDinoForward(model)
    wrapper.eval()

    pv = inputs["pixel_values"]
    iids = inputs["input_ids"]
    am = inputs["attention_mask"]
    ttids = inputs["token_type_ids"]
    ttm = inputs["text_token_mask"]

    print(f"[export] Exporting fp32 ONNX to {fp32_path} ...")
    print(f"  pixel_values: {tuple(pv.shape)}")
    print(f"  input_ids:    {tuple(iids.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (pv, iids, am, ttids, ttm),
            fp32_path,
            opset_version=17,
            input_names=["pixel_values", "input_ids", "attention_mask",
                         "token_type_ids", "text_token_mask"],
            output_names=["logits", "pred_boxes"],
            do_constant_folding=True,
        )
    print(f"[export] fp32 saved: {fp32_path}")
    return fp32_path


def quantize_int8(fp32_path: str) -> str:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = os.path.join(ONNX_DIR, "gdino_int8.onnx")
    print(f"[export] Quantizing int8 → {int8_path} ...")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"[export] int8 saved: {int8_path}")
    return int8_path


# ── 4. Валидация ──────────────────────────────────────────────────────────────

def validate(model_torch, processor, int8_path: str, pil_image: Image.Image,
             inputs, threshold_iou: float = 0.8):
    import onnxruntime as ort

    with torch.no_grad():
        out = model_torch(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            text_token_mask=inputs["text_token_mask"],
        )
        torch_boxes_raw = out.pred_boxes[0].numpy()  # (N_queries, 4) cx/cy/w/h
        torch_logits = out.logits[0].numpy()

    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    ort_logits, ort_boxes = sess.run(
        ["logits", "pred_boxes"],
        {
            "pixel_values":    inputs["pixel_values"].numpy(),
            "input_ids":       inputs["input_ids"].numpy(),
            "attention_mask":  inputs["attention_mask"].numpy(),
            "token_type_ids":  inputs["token_type_ids"].numpy(),
            "text_token_mask": inputs["text_token_mask"].numpy(),
        },
    )

    # Сравниваем финальные bbox через post_process (кол-во + перекрытие)
    results_torch = processor.post_process_grounded_object_detection(
        type("O", (), {"logits": torch.tensor(torch_logits).unsqueeze(0),
                       "pred_boxes": torch.tensor(torch_boxes_raw).unsqueeze(0)})(),
        inputs["input_ids"],
        box_threshold=0.3, text_threshold=0.3,
        target_sizes=[pil_image.size[::-1]],
    )[0]
    results_ort = processor.post_process_grounded_object_detection(
        type("O", (), {"logits": torch.tensor(ort_logits),
                       "pred_boxes": torch.tensor(ort_boxes)})(),
        inputs["input_ids"],
        box_threshold=0.3, text_threshold=0.3,
        target_sizes=[pil_image.size[::-1]],
    )[0]

    n_torch = len(results_torch["boxes"])
    n_ort   = len(results_ort["boxes"])
    print(f"[validate] torch detections: {n_torch}, ORT detections: {n_ort}")

    # Если есть хоть какой-то результат — считаем IoU попарно лучших боксов
    if n_torch > 0 and n_ort > 0:
        b_torch = results_torch["boxes"][0].numpy()
        b_ort   = results_ort["boxes"][0].numpy()
        ix1 = max(b_torch[0], b_ort[0]); iy1 = max(b_torch[1], b_ort[1])
        ix2 = min(b_torch[2], b_ort[2]); iy2 = min(b_torch[3], b_ort[3])
        inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
        a_t = (b_torch[2]-b_torch[0]) * (b_torch[3]-b_torch[1])
        a_o = (b_ort[2]-b_ort[0]) * (b_ort[3]-b_ort[1])
        iou = inter / (a_t + a_o - inter + 1e-6)
        ok = iou >= threshold_iou
        print(f"[validate] top-1 IoU torch vs ORT: {iou:.3f} [{'OK' if ok else 'FAIL'}]")
        return ok
    elif n_torch == 0 and n_ort == 0:
        print("[validate] No detections in both — counts match, OK")
        return True
    else:
        print("[validate] Detection count mismatch — check model")
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image", default=None)
    args = parser.parse_args()

    model, processor = load_gdino()

    # Подготовить тестовое фото
    test_path = args.test_image
    if test_path and os.path.exists(test_path):
        img_bgr = cv2.imread(test_path)
        img_bgr = cv2.resize(img_bgr, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    else:
        print("[export] No test image provided, using synthetic 1024×1024 image")
        pil_image = Image.fromarray(np.random.randint(0, 255, (CANVAS, CANVAS, 3), dtype=np.uint8))

    inputs = make_inputs(pil_image, processor)

    try:
        fp32_path = export_onnx(model, inputs)
    except Exception as exc:
        print(f"\n[export] ONNX export FAILED: {exc}")
        print("[export] → Falling back to torch dynamic int8 at runtime (USE_ONNX_GDINO=0)")
        sys.exit(2)

    int8_path = quantize_int8(fp32_path)
    validate(model, processor, int8_path, pil_image, inputs)

    print("\n[export] Done.")
    print(f"  fp32: {fp32_path}")
    print(f"  int8: {int8_path}")
    print("  → Set USE_ONNX_GDINO=1 in model_service env to use these.")


if __name__ == "__main__":
    main()
