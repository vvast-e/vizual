"""
Экспорт GroundingDINO и SAM в ONNX через torch.onnx.export.

Запуск:
    cd backend
    python export_onnx.py

Результат:
    models/onnx/grounding-dino-tiny/model.onnx  + processor файлы
    models/onnx/mobile-sam/model.onnx           + processor файлы
"""
import io
import json
import os
import shutil
from pathlib import Path

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
_ONNX_DIR = os.path.join(_PROJECT_ROOT, "models", "onnx")
_HF_HOME = os.path.join(_PROJECT_ROOT, "models", "huggingface")
os.makedirs(_ONNX_DIR, exist_ok=True)
os.makedirs(_HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_HOME)

import numpy as np
import torch
from PIL import Image


def export_grounding_dino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-tiny"
    out_dir = os.path.join(_ONNX_DIR, "grounding-dino-tiny")
    onnx_path = os.path.join(out_dir, "model.onnx")

    if os.path.exists(onnx_path):
        print(f"[export] GroundingDINO ONNX already exists: {onnx_path}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"[export] Loading GroundingDINO from HuggingFace...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()

    # Сохраняем processor (для загрузки в ONNX-сервисе)
    processor.save_pretrained(out_dir)

    # Создаём dummy inputs для экспорта
    dummy_image = torch.randn(1, 3, 800, 800)
    dummy_input_ids = torch.ones(1, 256, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 256, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, 256, dtype=torch.long)

    input_names = ["pixel_values", "input_ids", "attention_mask", "token_type_ids"]
    output_names = ["logits", "pred_boxes"]

    dynamic_axes = {
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "token_type_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    }

    print(f"[export] Exporting to {onnx_path} (opset=14, may take a few minutes)...")
    torch.onnx.export(
        model,
        (
            dummy_image,
            dummy_input_ids,
            dummy_attention_mask,
            dummy_token_type_ids,
        ),
        onnx_path,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Удаляем тяжёлые PyTorch-файлы
    for f in Path(out_dir).glob("*.bin"):
        f.unlink()
    for f in Path(out_dir).glob("*.safetensors"):
        f.unlink()

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[export] GroundingDINO exported: {onnx_path} ({size_mb:.0f} MB)")
    return out_dir


def export_sam():
    from transformers import SamModel, SamProcessor

    model_id = "facebook/sam-vit-base"
    out_dir = os.path.join(_ONNX_DIR, "sam-vit-base")
    onnx_path = os.path.join(out_dir, "model.onnx")

    if os.path.exists(onnx_path):
        print(f"[export] SAM ONNX already exists: {onnx_path}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"[export] Loading SAM from HuggingFace...")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    model.eval()

    # Сохраняем processor
    processor.save_pretrained(out_dir)

    # Dummy inputs для SAM
    dummy_pixel_values = torch.randn(1, 3, 1024, 1024)
    dummy_input_boxes = torch.tensor([[[100.0, 100.0, 500.0, 500.0]]])

    input_names = ["pixel_values", "input_boxes"]
    output_names = ["iou_scores", "pred_masks"]

    dynamic_axes = {
        "pixel_values": {0: "batch"},
        "input_boxes": {0: "batch", 1: "num_boxes"},
        "iou_scores": {0: "batch"},
        "pred_masks": {0: "batch"},
    }

    print(f"[export] Exporting to {onnx_path} (opset=14, may take a few minutes)...")

    # SAM требует image_embeddings, поэтому оборачиваем
    class SAMWrapper(torch.nn.Module):
        def __init__(self, sam_model):
            super().__init__()
            self.model = sam_model

        def forward(self, pixel_values, input_boxes):
            image_embeddings = self.model.get_image_embeddings(pixel_values)
            outputs = self.model(
                image_embeddings=image_embeddings,
                input_boxes=input_boxes,
            )
            return outputs.iou_scores, outputs.pred_masks

    wrapper = SAMWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_pixel_values, dummy_input_boxes),
        onnx_path,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Удаляем тяжёлые PyTorch-файлы
    for f in Path(out_dir).glob("*.bin"):
        f.unlink()
    for f in Path(out_dir).glob("*.safetensors"):
        f.unlink()

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[export] SAM exported: {onnx_path} ({size_mb:.0f} MB)")
    return out_dir


if __name__ == "__main__":
    export_grounding_dino()
    export_sam()
    print("\n[export] All models exported successfully!")
