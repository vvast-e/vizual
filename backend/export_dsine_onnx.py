"""
Export DSINE_v02 → ONNX + int8 quantization.

Usage (from backend/ with venv active):
    python export_dsine_onnx.py [--test-image PATH]

Output:
    models/onnx/dsine_fp32.onnx
    models/onnx/dsine_int8.onnx   ← used at runtime
"""
import argparse
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)

ONNX_DIR = os.path.join(BACKEND_DIR, "models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

DEVICE = "cpu"

from dsine_canvas import make_dsine_canvas, CANVAS


# ── 1. Загрузка модели (копирует логику _load_dsine из model_service.py) ───────

def load_dsine():
    hub_dir = torch.hub.get_dir()
    dsine_src = os.path.join(hub_dir, "baegwangbin_DSINE_main")

    if not os.path.isdir(dsine_src):
        print("[export] Downloading DSINE repo...")
        try:
            torch.hub.load("baegwangbin/DSINE", "DSINE", trust_repo=True)
        except Exception:
            pass

    if dsine_src not in sys.path:
        sys.path.insert(0, dsine_src)

    _orig = torch.Tensor.to
    def _safe_to(self, *args, **kwargs):
        if args and isinstance(args[0], int):
            return _orig(self, DEVICE)
        return _orig(self, *args, **kwargs)
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
        torch.Tensor.to = _orig

    weights_path = os.path.join(hub_dir, "checkpoints", "dsine.pt")
    if not os.path.exists(weights_path):
        print("[export] Downloading DSINE weights (~278 MB)...")
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
    print(f"[export] DSINE loaded ({sum(p.numel() for p in model.parameters())//1_000_000}M params)")
    return model


# make_canvas_inputs → переехало в dsine_canvas.make_dsine_canvas


# ── 3. Обёртка модели для фиксированных входов ────────────────────────────────

class DSINEExportWrapper(torch.nn.Module):
    """Обёртка, принимающая (image_1x3xHxW, intrins_1x3x3) и отдающая нормали."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, intrins: torch.Tensor) -> torch.Tensor:
        preds = self.model(image, intrins=intrins)
        return preds[-1]  # (1, 3, H, W)


# ── 4. Экспорт ────────────────────────────────────────────────────────────────

def export_onnx(model, t, intrins):
    fp32_path = os.path.join(ONNX_DIR, "dsine_fp32.onnx")
    wrapper = DSINEExportWrapper(model)
    wrapper.eval()

    print(f"[export] Exporting fp32 ONNX to {fp32_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (t, intrins),
            fp32_path,
            opset_version=17,
            input_names=["image", "intrins"],
            output_names=["normals"],
            dynamic_axes={
                "image":   {0: "batch"},
                "intrins": {0: "batch"},
                "normals": {0: "batch"},
            },
            do_constant_folding=True,
        )
    print(f"[export] fp32 saved: {fp32_path}")
    return fp32_path


def quantize_int8(fp32_path: str) -> str:
    """
    Quantize MatMul/Gemm ops to int8. Conv ops are skipped — DSINE decoder has
    dynamic (non-initializer) Conv weights that ORT's Conv quantizer can't handle.
    If even that fails, copies fp32 as the "int8" artefact so the runtime still
    gets the ORT graph-fusion speedup.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = os.path.join(ONNX_DIR, "dsine_int8.onnx")
    print(f"[export] Quantizing int8 (MatMul/Gemm only) → {int8_path} ...")
    try:
        quantize_dynamic(
            fp32_path, int8_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        print(f"[export] int8 saved: {int8_path}")
    except Exception as exc:
        import shutil
        print(f"[export] int8 quantization failed ({exc})")
        print(f"[export] Falling back: copying fp32 as ORT artefact → {int8_path}")
        shutil.copy2(fp32_path, int8_path)
    return int8_path


# ── 5. Валидация ──────────────────────────────────────────────────────────────

def validate(model_torch, int8_path: str, t: torch.Tensor, intrins: torch.Tensor,
             crop_hw: tuple, threshold: float = 0.97):
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = DSINEExportWrapper(model_torch)(t, intrins)
    torch_np = torch_out[0].permute(1, 2, 0).numpy()[:crop_hw[0], :crop_hw[1]]

    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["normals"], {"image": t.numpy(), "intrins": intrins.numpy()})[0]
    ort_np = ort_out[0].transpose(1, 2, 0)[:crop_hw[0], :crop_hw[1]]

    dot = np.sum(torch_np * ort_np, axis=-1)
    cos_sim = float(np.mean(np.clip(dot, -1, 1)))
    ok = cos_sim >= threshold
    status = "OK" if ok else "FAIL"
    print(f"[validate] cosine-similarity torch vs int8 ONNX: {cos_sim:.4f} [{status}] (threshold {threshold})")
    if not ok:
        print("[validate] WARNING: quality degradation detected, check int8 model!")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image", default=None,
                        help="Path to test image for validation (optional)")
    args = parser.parse_args()

    model = load_dsine()

    # Построить тестовый тензор
    test_img_path = args.test_image
    if test_img_path and os.path.exists(test_img_path):
        img_bgr = cv2.imread(test_img_path)
    else:
        # Синтетическое фото-заменитель: серый шум
        print("[export] No test image provided, using synthetic 800×600 noise")
        img_bgr = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)

    fov = float(os.getenv("DSINE_FOV_DEG", "60"))
    t, intrins, orig_hw, canvas_hw = make_dsine_canvas(img_bgr, fov_deg=fov)
    print(f"[export] Canvas: {canvas_hw[1]}x{canvas_hw[0]} (orig {orig_hw[1]}x{orig_hw[0]})")

    fp32_path = export_onnx(model, t, intrins)
    int8_path = quantize_int8(fp32_path)
    validate(model, int8_path, t, intrins, crop_hw=canvas_hw)

    print("\n[export] Done.")
    print(f"  fp32: {fp32_path}")
    print(f"  int8: {int8_path}")
    print("  → Set USE_ONNX_DSINE=1 in model_service env to use these.")


if __name__ == "__main__":
    main()
