"""
Optional wall-plane hints from a single RGB image (no extra model download).

Uses normalized Sobel features + position inside wall_minus_holes, k-means (scipy),
to propose 2..K planar regions when lighting/texture differs between facade planes.

Controlled by EXTERIOR_PLANE_DEPTH_ENABLE in exterior_pipeline.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

try:
    from scipy.cluster.vq import kmeans, vq
except ImportError:  # pragma: no cover
    kmeans = None  # type: ignore[misc, assignment]
    vq = None  # type: ignore[misc, assignment]

EXTERIOR_DEPTH_MAX_K = int(os.getenv("EXTERIOR_DEPTH_MAX_K", "4"))
EXTERIOR_DEPTH_MIN_PIXELS = int(os.getenv("EXTERIOR_DEPTH_MIN_PIXELS", "800"))
# Ниже этого aspect bbox маски считаем «высокими» — оставляем в фичах и y (крыша/цоколь).
EXTERIOR_DEPTH_WIDE_MASK_ASPECT = float(os.getenv("EXTERIOR_DEPTH_WIDE_MASK_ASPECT", "1.08"))
# Усиление горизонтальной позиции внутри bbox (после нормализации фичей).
EXTERIOR_DEPTH_NX_WEIGHT = float(os.getenv("EXTERIOR_DEPTH_NX_WEIGHT", "2.2"))
# Для k>=3: отбрасываем кластеры, у которых центры размазаны в основном по Y (полосы).
EXTERIOR_DEPTH_MIN_H_SPREAD_RATIO = float(os.getenv("EXTERIOR_DEPTH_MIN_H_SPREAD_RATIO", "0.42"))


def _centroid_spread_ratio_xy(masks: list[np.ndarray], rw: float, rh: float) -> float:
    """Насколько сильнее кластеры разнесены по x, чем по y (отн. размера bbox маски). ~1 = круг, >>1 = вертикальные грани."""
    xs_m: list[float] = []
    ys_m: list[float] = []
    for m in masks:
        yy, xx = np.where(m > 0)
        if xx.size < 20:
            continue
        xs_m.append(float(np.mean(xx)))
        ys_m.append(float(np.mean(yy)))
    if len(xs_m) < 2:
        return 1.0
    sx = (max(xs_m) - min(xs_m)) / max(float(rw), 1.0)
    sy = (max(ys_m) - min(ys_m)) / max(float(rh), 1.0)
    return float(sx / max(sy, 0.06))


def plane_cluster_masks(
    image_bgr: np.ndarray,
    wall_mask: np.ndarray,
    min_area: int,
    max_planes: int,
) -> tuple[list[np.ndarray], float, dict]:
    """
    Returns (list of binary uint8 masks covering disjoint subsets of wall_mask),
    confidence in [0,1], debug dict. Empty list if skipped or failed.
    """
    dbg: dict = {"skipped": False, "reason": ""}
    if kmeans is None or vq is None:
        dbg["reason"] = "scipy.cluster.vq unavailable"
        return [], 0.0, dbg

    if image_bgr.shape[:2] != wall_mask.shape[:2]:
        dbg["reason"] = "shape_mismatch"
        return [], 0.0, dbg

    h, w = wall_mask.shape[:2]
    rx, ry, rw, rh = cv2.boundingRect((wall_mask > 0).astype(np.uint8) * 255)
    mask_aspect = rw / max(1.0, float(rh))
    ys, xs = np.where(wall_mask > 0)
    if ys.size < EXTERIOR_DEPTH_MIN_PIXELS:
        dbg["skipped"] = True
        dbg["reason"] = "too_few_pixels"
        return [], 0.0, dbg

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.bilateralFilter(gray, 5, 0.1, 5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Позиция внутри bbox здания: для широкого фасада k-means иначе цепляется к глобальному y
    # (небо/земля, горизонтальные градиенты) и даёт полосы вместо левой/правой стены.
    nx = (xs.astype(np.float32) - float(rx)) / max(float(rw), 1.0)
    ny = (ys.astype(np.float32) - float(ry)) / max(float(rh), 1.0)

    wide = mask_aspect >= EXTERIOR_DEPTH_WIDE_MASK_ASPECT
    if wide:
        feats = np.stack([gx[ys, xs], gy[ys, xs], nx], axis=1)
    else:
        feats = np.stack([gx[ys, xs], gy[ys, xs], nx, ny], axis=1)

    fmean = feats.mean(axis=0)
    fstd = feats.std(axis=0) + 1e-6
    feats_n = (feats - fmean) / fstd
    if wide:
        feats_n[:, 2] *= EXTERIOR_DEPTH_NX_WEIGHT

    max_k = min(max_planes, EXTERIOR_DEPTH_MAX_K, ys.size // max(min_area, 1))
    if max_k < 2:
        dbg["reason"] = "max_k_lt_2"
        return [], 0.0, dbg

    best_masks: list[np.ndarray] = []
    best_score = -1.0
    best_k = 0
    best_spread_ratio = 0.0

    for k in range(2, max_k + 1):
        try:
            codebook, _distortion = kmeans(feats_n, k, iter=30, thresh=1e-05)
        except Exception as e:
            dbg["kmeans_error"] = str(e)
            continue
        if codebook.shape[0] < k:
            continue
        labels, _ = vq(feats_n, codebook)

        masks_k: list[np.ndarray] = []
        areas: list[int] = []
        for lab in range(k):
            m = np.zeros((h, w), dtype=np.uint8)
            sel = labels == lab
            if not np.any(sel):
                continue
            m[ys[sel], xs[sel]] = 255
            area = int(cv2.countNonZero(m))
            if area < min_area:
                masks_k = []
                break
            masks_k.append(m)
            areas.append(area)
        if len(masks_k) < 2:
            continue

        spread_ratio = _centroid_spread_ratio_xy(masks_k, float(rw), float(rh))
        if k >= 3 and spread_ratio < EXTERIOR_DEPTH_MIN_H_SPREAD_RATIO:
            continue

        # Separation: pairwise distance of codebook rows vs mean intra spread
        sep = 0.0
        pairs = 0
        for i in range(k):
            for j in range(i + 1, k):
                sep += float(np.linalg.norm(codebook[i] - codebook[j]))
                pairs += 1
        sep /= max(1, pairs)
        intra = float(np.mean(np.std(feats_n, axis=0)))
        score_k = sep / (intra + 0.15)
        score_k = float(np.clip(score_k / 3.0, 0.0, 1.0))

        bal = min(areas) / max(areas)
        score_k = 0.7 * score_k + 0.3 * bal
        # Штраф за «полосы по высоте»: даже при хорошем sep кластеры должны разъезжаться по x
        q = float(np.clip((spread_ratio - 0.25) / 1.15, 0.15, 1.0))
        score_k *= 0.35 + 0.65 * q
        # Широкий фасад в кадре чаще = несколько плоскостей в ряд — слегка поощняем k=3..4
        if mask_aspect > 1.15 and k >= 3:
            score_k += min(0.05 * (k - 2), 0.1)

        if score_k > best_score:
            best_score = score_k
            best_masks = masks_k
            best_k = k
            best_spread_ratio = spread_ratio

    dbg["best_k"] = best_k
    dbg["wide_feature_mode"] = bool(wide)
    dbg["spread_ratio_xy"] = round(float(best_spread_ratio), 4)
    dbg["mask_aspect"] = round(float(mask_aspect), 3)
    dbg["score_depth_raw"] = float(best_score)
    if not best_masks:
        dbg["reason"] = "no_valid_k"
        return [], 0.0, dbg

    return best_masks, best_score, dbg
