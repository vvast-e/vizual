"""
Тест локального инференса YOLO на фото фасада.

Примеры:
  cd backend && python test_yolo.py --mode world
  cd backend && python test_yolo.py --mode hf --hf-token <TOKEN>
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PHOTO_PATH = _PROJECT_ROOT / "photo_2026-03-26_12-17-57.jpg"
_OUTPUT_PATH_WORLD = _PROJECT_ROOT / "yolo_test_result.jpg"
_OUTPUT_PATH_HF = _PROJECT_ROOT / "yolo_archivision_result.jpg"
_OUTPUT_JSON_HF = _PROJECT_ROOT / "yolo_archivision_result.json"

# Промпты для zero-shot детекции
PROMPTS = ["building facade", "window", "door", "balcony"]
CONF_THRESHOLD = 0.25
HF_MODEL_ID = "TanTan2025/archivision-yolo"
HF_MODEL_URL = f"https://huggingface.co/{HF_MODEL_ID}/resolve/main/best.pt"
HF_MODEL_LOCAL = _PROJECT_ROOT / "backend" / "archivision_best.pt"
DEFAULT_TARGET_CLASSES = ("facade", "wall", "door", "roof", "balcony", "column", "arch", "stairs", "chimney")
CLASS_THRESHOLDS_DEFAULT = {
    "facade": 0.20,
    "wall": 0.20,
    "door": 0.25,
    "roof": 0.25,
    "balcony": 0.25,
    "column": 0.25,
    "arch": 0.25,
    "stairs": 0.25,
    "chimney": 0.25,
    "window": 0.50,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["world", "hf"], default="hf")
    p.add_argument("--image", default=str(_PHOTO_PATH))
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "")
    p.add_argument(
        "--target-classes",
        default=",".join(DEFAULT_TARGET_CLASSES),
        help="Список классов через запятую для вывода в финальный результат (по умолчанию без window).",
    )
    p.add_argument(
        "--include-window",
        action="store_true",
        help="Добавить class=window в финальный результат (с отдельным порогом).",
    )
    p.add_argument(
        "--class-thresholds",
        default="",
        help='Переопределить пороги: "facade=0.18,door=0.3,window=0.55"',
    )
    return p.parse_args()


def _parse_class_thresholds(raw: str) -> dict[str, float]:
    out = dict(CLASS_THRESHOLDS_DEFAULT)
    if not raw.strip():
        return out
    for chunk in raw.split(","):
        part = chunk.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        k = key.strip().lower()
        try:
            v = float(val.strip())
        except ValueError:
            continue
        out[k] = max(0.0, min(1.0, v))
    return out


def _download_hf_model(token: str) -> Path:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    HF_MODEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if HF_MODEL_LOCAL.exists() and HF_MODEL_LOCAL.stat().st_size > 1_000_000:
        print(f"[hf] Использую кеш модели: {HF_MODEL_LOCAL}")
        return HF_MODEL_LOCAL
    print(f"[hf] Скачиваю модель {HF_MODEL_ID} ...")
    with requests.get(HF_MODEL_URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(HF_MODEL_LOCAL, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[hf] Модель сохранена: {HF_MODEL_LOCAL}")
    return HF_MODEL_LOCAL


def _run_world(image_path: Path, conf: float) -> None:
    if not image_path.exists():
        print(f"[ERROR] Фото не найдено: {image_path}")
        return

    print(f"[test] Загружаю YOLO-World v2 small...")
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(PROMPTS)

    print(f"[test] Прогоняю фото: {image_path}")
    results = model(str(image_path), conf=conf)
    result = results[0]

    # Вывод результатов
    bboxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box, score, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box.tolist()
            label = result.names[int(cls_id)]
            bboxes.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "score": round(float(score), 4),
                "label": label,
            })
            print(f"  {label}: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] score={score:.4f}")

    print(f"\n[test] Найдено объектов: {len(bboxes)}")

    # Визуализация
    img = cv2.imread(str(image_path))
    colors = {
        "building facade": (0, 255, 0),    # зелёный
        "window": (255, 0, 0),              # синий
        "door": (0, 165, 255),              # оранжевый
        "balcony": (255, 255, 0),           # голубой
    }
    for det in bboxes:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        score = det["score"]
        color = colors.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(_OUTPUT_PATH_WORLD), img)
    print(f"[test] Результат сохранён: {_OUTPUT_PATH_WORLD}")


def _run_hf_local(
    image_path: Path,
    conf: float,
    token: str,
    target_classes_raw: str,
    include_window: bool,
    class_thresholds_raw: str,
) -> None:
    if not image_path.exists():
        print(f"[ERROR] Фото не найдено: {image_path}")
        return
    model_path = _download_hf_model(token)
    print(f"[hf] Загружаю локальную модель: {model_path}")
    model = YOLO(str(model_path))
    print(f"[hf] Прогоняю фото: {image_path}")
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    result = results[0]

    classes = result.names
    target_classes = {c.strip().lower() for c in target_classes_raw.split(",") if c.strip()}
    if include_window:
        target_classes.add("window")
    class_thresholds = _parse_class_thresholds(class_thresholds_raw)

    print(f"[hf] Целевые классы: {sorted(target_classes)}")
    print(f"[hf] Пороги классов: {class_thresholds}")

    detections: list[dict] = []
    detections_raw: list[dict] = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()
        for box, score, cls_id in zip(boxes, confs, clses):
            x1, y1, x2, y2 = box.tolist()
            label = str(classes.get(int(cls_id), str(int(cls_id)))).lower()
            rec = (
                {
                    "label": label,
                    "score": round(float(score), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )
            detections_raw.append(rec)
            if label not in target_classes:
                continue
            min_score = class_thresholds.get(label, conf)
            if float(score) < min_score:
                continue
            detections.append(rec)
    with open(_OUTPUT_JSON_HF, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": HF_MODEL_ID,
                "image": str(image_path),
                "target_classes": sorted(target_classes),
                "class_thresholds": class_thresholds,
                "detections_raw_count": len(detections_raw),
                "detections": detections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot = result.plot()
    cv2.imwrite(str(_OUTPUT_PATH_HF), plot)
    print(f"[hf] Найдено объектов всего: {len(detections_raw)}")
    print(f"[hf] После фильтрации: {len(detections)}")
    print(f"[hf] Визуализация: {_OUTPUT_PATH_HF}")
    print(f"[hf] JSON: {_OUTPUT_JSON_HF}")
    for d in detections[:20]:
        print(f"  {d['label']}: {d['bbox']} score={d['score']:.4f}")


def main():
    args = _parse_args()
    image_path = Path(args.image)
    if args.mode == "world":
        _run_world(image_path, args.conf)
        return
    _run_hf_local(
        image_path,
        args.conf,
        args.hf_token,
        args.target_classes,
        args.include_window,
        args.class_thresholds,
    )

if __name__ == "__main__":
    main()
