import numpy as np
import cv2

def _segment_line(seg):
    x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = np.hypot(a, b)
    if norm > 1e-8:
        a, b, c = a / norm, b / norm, c / norm
    return a, b, c

def detect_vanishing_point(segments, max_iters=1000, inlier_deg=4.0):
    """
    RANSAC for finding a single Vanishing Point given a list of segments.
    """
    if len(segments) < 2:
        return None, []

    lines = []
    midpoints = []
    for s in segments:
        a, b, c = _segment_line(s)
        lines.append((a, b, c))
        midpoints.append(((s["x1"] + s["x2"]) / 2.0, (s["y1"] + s["y2"]) / 2.0))

    best_vp = None
    best_inliers = []
    n = len(segments)
    
    # We precompute random pairs to speed up
    # If n is small, test all pairs
    if n * (n - 1) / 2 <= max_iters:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [tuple(np.random.choice(n, 2, replace=False)) for _ in range(max_iters)]

    cos_thresh = np.cos(np.radians(inlier_deg))

    for i, j in pairs:
        a1, b1, c1 = lines[i]
        a2, b2, c2 = lines[j]
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6: # parallel lines in 2D = VP at infinity
            continue
            
        vx = (b1 * c2 - b2 * c1) / denom
        vy = (a2 * c1 - a1 * c2) / denom

        inliers = []
        for k in range(n):
            a, b, c = lines[k]
            mx, my = midpoints[k]
            dx = vx - mx
            dy = vy - my
            norm = np.hypot(dx, dy)
            
            if norm < 1e-6:
                inliers.append(k)
                continue
            
            # The direction of the line k is (-b, a)
            # The vector to the VP is (dx/norm, dy/norm)
            cos_theta = abs(-b * dx + a * dy) / norm
            if cos_theta > cos_thresh:
                inliers.append(k)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_vp = (vx, vy)

    return best_vp, best_inliers

def extract_manhattan_vps(segments):
    """
    Extracts up to 3 vanishing points (1 vertical, 2 horizontal)
    Returns: 
        vp_vert, vert_segments, 
        vp_horiz1, horiz1_segments, 
        vp_horiz2, horiz2_segments
    """
    if len(segments) == 0:
        return None, [], None, [], None, []

    # 1. Extract Vertical VP
    # Heuristic: vertical lines usually have angles between 45 and 135 degrees
    vert_candidates = []
    horiz_candidates = []
    for s in segments:
        deg = np.degrees(s["theta"]) % 180
        if 45 <= deg <= 135:
            vert_candidates.append(s)
        else:
            horiz_candidates.append(s)

    vp_vert, vert_inlier_idx = detect_vanishing_point(vert_candidates, max_iters=1000, inlier_deg=5.0)
    # SOFT FILTERING: Keep all vertical candidates so we don't lose inner corners with distortion.
    # The projection step will align them perfectly anyway.
    vert_segments = vert_candidates

    # Combine the non-inliers back to horizontal candidates
    if vp_vert:
        vert_inlier_set = set(vert_inlier_idx)
        for i, s in enumerate(vert_candidates):
            if i not in vert_inlier_set:
                horiz_candidates.append(s)

    # 2. Extract First Horizontal VP
    vp_h1, h1_inlier_idx = detect_vanishing_point(horiz_candidates, max_iters=1000, inlier_deg=5.0)
    # SOFT FILTERING for horizontal
    h1_segments = [horiz_candidates[i] for i in h1_inlier_idx] if vp_h1 else []
    
    # 3. Extract Second Horizontal VP
    rem_horiz = []
    if vp_h1:
        h1_inlier_set = set(h1_inlier_idx)
        for i, s in enumerate(horiz_candidates):
            if i not in h1_inlier_set:
                rem_horiz.append(s)
    else:
        rem_horiz = horiz_candidates

    vp_h2, h2_inlier_idx = detect_vanishing_point(rem_horiz, max_iters=1000, inlier_deg=5.0)
    h2_segments = [rem_horiz[i] for i in h2_inlier_idx] if vp_h2 else []

    # Filter out garbage (any segment that is not an inlier to VP_v, VP_h1, or VP_h2 is likely roof or garbage)
    return vp_vert, vert_segments, vp_h1, h1_segments, vp_h2, h2_segments

def project_to_horizon(vp_h1, vp_h2, image_width, image_height):
    """
    Calculates the horizon line equation: Ax + By + C = 0
    Returns A, B, C or None if it can't be established.
    """
    if vp_h1 and vp_h2:
        x1, y1 = vp_h1
        x2, y2 = vp_h2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        norm = np.hypot(A, B)
        if norm > 1e-8:
            return A/norm, B/norm, C/norm
    
    # Fallback to a default horizontal line across the middle if VPs missing
    return 0.0, 1.0, -image_height / 2.0

def project_verticals(vert_segments, vp_vert, horizon_line, image_width):
    """
    Instead of a 1D X-accumulator, we intersect the vertical segments (which pass through VP_vert)
    with the Horizon Line. This gives us perspective-corrected 1D positions.
    """
    A_h, B_h, C_h = horizon_line
    
    projected_xs = []
    for s in vert_segments:
        # The line passes through midpoint and VP_vert
        mx = (s["x1"] + s["x2"]) / 2.0
        my = (s["y1"] + s["y2"]) / 2.0
        
        if vp_vert:
            vx, vy = vp_vert
            A_v = vy - my
            B_v = mx - vx
            C_v = vx * my - mx * vy
        else:
            A_v, B_v, C_v = _segment_line(s)
            
        # Intersect with horizon
        denom = A_v * B_h - A_h * B_v
        if abs(denom) > 1e-6:
            px = (B_v * C_h - B_h * C_v) / denom
            projected_xs.append({"x_proj": px, "seg": s})
        else:
            projected_xs.append({"x_proj": mx, "seg": s}) # Fallback to mx
            
    return projected_xs
