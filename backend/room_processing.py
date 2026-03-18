"""Room image loading/saving and brightness transfer. From IvanDumych/room-wall-visualizer."""

import cv2
import numpy as np
from PIL import Image


def load_img(img_path: str) -> np.ndarray:
    image = Image.open(img_path)
    return np.asarray(image)


def save_image(img: np.ndarray, img_path: str) -> None:
    im = Image.fromarray(img)
    im.save(img_path)


def brightness_transfer(image: np.ndarray, wall_decorated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Transfer brightness from original image to decorated walls (HSV V channel)."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h1, s1, v1 = cv2.split(hsv_image)
    hsv_image = cv2.cvtColor(wall_decorated, cv2.COLOR_BGR2HSV)
    h2, s2, v2 = cv2.split(hsv_image)
    v1_wall = v1[np.where(mask != 0)]
    hist, bin_edges = np.histogram(v1_wall.ravel(), bins=np.arange(0, 260, 5))
    result = np.argmax(hist)
    d = bin_edges[result]
    delta = np.array(v1[np.where(mask != 0)], dtype="float32") - d
    v2_wall = np.array(v2[np.where(mask != 0)], dtype="float32")
    v2_wall += delta * 1.5
    v2_wall[v2_wall > 250.0] = 250
    v2_wall[v2_wall < 0.0] = 0
    v2[np.where(mask != 0)] = np.array(v2_wall, dtype="uint8")
    hsv_image = cv2.merge([h2, s2, v2])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
