"""Wall corners and texture mapping. From IvanDumych/room-wall-visualizer."""

import math
import numpy as np
import cv2
from PIL import Image


def load_img(img_path: str) -> np.ndarray:
    image = Image.open(img_path)
    return np.asarray(image)


def image_resize(image: np.ndarray, width: int | None = None, height: int | None = None, inter: int = cv2.INTER_AREA) -> np.ndarray:
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def get_wall_corners(image: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return list of wall polygons (each 4 corners). Colors: left=(255,0,0), center=(0,255,0), right=(0,0,255)."""
    image = image[..., ::-1]
    rgb_unique = set(tuple(rgb) for rgb in image.reshape(image.shape[0] * image.shape[1], 3))
    result = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for color in colors:
        if color not in rgb_unique:
            continue
        mask = np.all(image == color, axis=-1)
        mask = mask.astype(np.uint8)
        img = np.copy(image)
        img[np.where(mask != 1)] = [0, 0, 0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)
        area = cv2.contourArea(approx)
        if area < (image.shape[0] * image.shape[1]) / 20:
            continue
        approx = [tuple(point[0]) for point in approx]
        points = _countour_rect_corners(approx)
        points.sort()
        if points[0][1] > points[1][1]:
            points[0], points[1] = points[1], points[0]
        if points[2][1] > points[3][1]:
            points[2], points[3] = points[3], points[2]
        result.append(points)
    for i in range(1, len(result)):
        right_top_point = result[i - 1][2]
        right_bottom_points = result[i - 1][3]
        result[i][0] = (right_top_point[0], result[i][0][1])
        result[i][1] = right_bottom_points
    for points in result:
        points[3], points[2] = points[2], points[3]
    return result


def _countour_rect_corners(approx: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = []
    max_x = max(approx, key=lambda p: p[0])[0]
    min_x = min(approx, key=lambda p: p[0])[0]
    max_y = max(approx, key=lambda p: p[1])[1]
    min_y = min(approx, key=lambda p: p[1])[1]
    width = max_x - min_x
    height = max_y - min_y
    tresh_left_x = width / 6 + min_x
    tresh_y = (height * 3) / 5 + min_y
    filtered = [p for p in approx if p[0] < tresh_left_x and p[1] < tresh_y]
    top_left_point = min(filtered, key=lambda p: (p[1], p[0])) if filtered else (min_x, min_y)
    tresh_right_x = (width * 4) / 5 + min_x
    filtered = [p for p in approx if p[0] > tresh_right_x and p[1] < tresh_y]
    top_right_point = min(filtered, key=lambda p: (p[1], -p[0])) if filtered else (max_x, min_y)
    if top_left_point[1] > 25 and top_right_point[1] < 10:
        top_right_point = _find_approx_top(approx, top_left_point, top_right_point, 1)
    elif top_left_point[1] < 10 and top_right_point[1] > 25:
        top_left_point = _find_approx_top(approx, top_right_point, top_left_point, -1)
    filtered = [p for p in approx if p[0] < tresh_left_x and p[1] > tresh_y]
    bottom_left_point = min(filtered, key=lambda p: p[0]) if filtered else (min_x, max_y)
    filtered = [p for p in approx if p[0] > tresh_right_x and p[1] > tresh_y]
    bottom_right_point = min(filtered, key=lambda p: -p[0]) if filtered else (max_x, max_y)
    points.append(top_left_point)
    points.append(top_right_point)
    points.append((points[0][0], bottom_left_point[1]))
    points.append((points[1][0], bottom_right_point[1]))
    return points


def _find_approx_top(approx: list, top_left_point: tuple, top_right_point: tuple, sign: int = 1) -> tuple:
    points_zero_y = [p for p in approx if p[1] < 10]
    second_top_right_point = min(points_zero_y, key=lambda p: (sign * p[0])) if points_zero_y else top_right_point
    if second_top_right_point != top_right_point:
        a = (-top_left_point[0] - sign, top_left_point[1])
        b = (-top_left_point[0], top_left_point[1])
        c = (-second_top_right_point[0], second_top_right_point[1])
        angle = _get_angle(a, b, c)
        if angle > 180:
            angle = 180 - angle
        d = abs(top_right_point[0] - top_left_point[0])
        stroke_c = math.tan(math.radians(angle)) * d
        top_right_point = (top_right_point[0], int(top_right_point[1] + (top_left_point[1] - stroke_c)))
    return top_right_point


def _get_angle(a: tuple, b: tuple, c: tuple) -> float:
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def get_ceiling_corners(image: np.ndarray) -> list[tuple[int, int]] | None:
    """Extract ceiling quad [TL, BL, BR, TR] from estimation map (white pixels = ceiling).

    Returns None if ceiling region is absent or too small.
    """
    # estimation map arrives as BGR from cv2; ceiling = white (255,255,255) in any channel order
    rgb = image[..., ::-1]
    mask = np.all(rgb == (255, 255, 255), axis=-1).astype(np.uint8)
    if not mask.any():
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    image_area = image.shape[0] * image.shape[1]
    if area < image_area / 20:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    pts = np.array([p[0] for p in approx], dtype=np.float32)

    # Need exactly 4 corners; use bounding-box corners if approximation gives more/fewer
    if len(pts) != 4:
        x, y, w, h = cv2.boundingRect(contour)
        pts = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]], dtype=np.float32)

    # Order as [TL, BL, BR, TR] — matches renderPerspectiveWallTexture expectations
    cx, cy = pts.mean(axis=0)
    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmax(-pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmax(pts[:, 0] - pts[:, 1])]
    ordered = [tuple(map(int, p)) for p in (tl, bl, br, tr)]
    return ordered


def wall_polygon_centroid(points: list[tuple[int, int]]) -> tuple[float, float]:
    """Centroid of a polygon (for button placement)."""
    n = len(points)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return (cx, cy)
