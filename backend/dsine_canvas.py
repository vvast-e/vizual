"""
dsine_canvas.py — общая подготовка входного тензора для DSINE ONNX-модели.

DSINE ONNX экспортирован с фикс-входом CANVAS×CANVAS (1024×1024).
Эта функция должна использоваться как при экспорте (export_dsine_onnx.py),
так и в рантайме (model_service.py /normals ONNX-ветка) — чтобы формы входа совпадали.
"""
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T

CANVAS = 1024   # фикс-размер стороны квадрата — должен совпадать с export


def make_dsine_canvas(img_bgr: np.ndarray, fov_deg: float = 60.0):
    """
    Подготовить тензор 1×3×CANVAS×CANVAS и intrinsics для DSINE ONNX.

    Алгоритм:
      1. Ресайз по длинной стороне до CANVAS (сохраняем пропорции).
      2. Zero-pad до квадрата CANVAS×CANVAS (снизу и справа).
      3. ImageNet-нормализация.
      4. Intrinsics строятся для ресайзнутого изображения (new_W, new_H),
         а не для канваса — это корректно, т.к. паддинг не меняет геометрию.

    Returns:
        t          : torch.Tensor (1, 3, CANVAS, CANVAS) float32
        intrins    : torch.Tensor (1, 3, 3) float32
        orig_hw    : (orig_H, orig_W) — исходный размер
        content_hw : (new_H, new_W)  — реальная часть в канвасе (без паддинга)
    """
    orig_H, orig_W = img_bgr.shape[:2]

    # 1. Ресайз по длинной стороне
    scale = CANVAS / max(orig_H, orig_W)
    new_H = int(round(orig_H * scale))
    new_W = int(round(orig_W * scale))
    resized = cv2.resize(img_bgr, (new_W, new_H), interpolation=cv2.INTER_AREA)

    # 2. RGB float [0,1] → тензор 1×3×new_H×new_W
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    # 3. Pad до CANVAS×CANVAS
    pad_h = CANVAS - new_H
    pad_w = CANVAS - new_W
    t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    # 4. ImageNet-нормализация
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = norm(t)

    # 5. Intrinsics: FOV=fov_deg для new_W (реальной ширины без паддинга)
    f = new_W / (2 * np.tan(np.radians(fov_deg / 2)))
    intrins = torch.tensor(
        [[f, 0, new_W / 2], [0, f, new_H / 2], [0, 0, 1]],
        dtype=torch.float32,
    ).unsqueeze(0)

    return t, intrins, (orig_H, orig_W), (new_H, new_W)
