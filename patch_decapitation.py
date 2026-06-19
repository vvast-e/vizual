import re

with open('backend/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic_pattern = r'    elif n_hull > 4:\n        from facade_geometry import split_gable_roof_polygon\n[\s\S]*?method = f"visvalingam\({n_hull}->4\)"\n    else:'

new_logic = """    elif n_hull > 4:
        # Smart Roof Decapitation (Gable/Hip multi-point removal):
        xs = hull_pts[:, 0]
        ys = hull_pts[:, 1]
        min_x, max_x = np.min(xs), np.max(xs)
        width = max_x - min_x
        
        # Find Eaves candidates
        left_mask = xs <= min_x + 0.15 * width
        right_mask = xs >= max_x - 0.15 * width
        
        apex_y = float(np.min(ys))
        quad = None
        
        if np.any(left_mask) and np.any(right_mask):
            eave_l_idx = np.where(left_mask)[0][np.argmin(ys[left_mask])]
            eave_r_idx = np.where(right_mask)[0][np.argmin(ys[right_mask])]
            
            eave_l = hull_pts[eave_l_idx]
            eave_r = hull_pts[eave_r_idx]
            
            # A true gable/hip roof apex must be significantly higher than BOTH eaves
            if apex_y < eave_l[1] - 10.0 and apex_y < eave_r[1] - 10.0:
                dx = eave_r[0] - eave_l[0]
                dy = eave_r[1] - eave_l[1]
                
                new_hull = []
                for pt in hull_pts:
                    px, py = float(pt[0]), float(pt[1])
                    if np.array_equal(pt, eave_l) or np.array_equal(pt, eave_r):
                        new_hull.append(pt)
                        continue
                        
                    if min(eave_l[0], eave_r[0]) < px < max(eave_l[0], eave_r[0]):
                        if abs(dx) > 1e-6:
                            line_y = eave_l[1] + (px - eave_l[0]) * dy / dx
                        else:
                            line_y = min(eave_l[1], eave_r[1])
                            
                        if py < line_y - 2.0:
                            continue
                            
                    new_hull.append(pt)
                    
                hull_pts = np.array(new_hull, dtype=np.float32)
                method = f"visvalingam({n_hull}->4, decapitated)"
            else:
                method = f"visvalingam({n_hull}->4)"
        else:
            method = f"visvalingam({n_hull}->4)"
            
        if len(hull_pts) < 4:
            # Fallback if decapitation removed too many points
            hull_pts = hull[:, 0].astype(np.float32)
            
        quad = _visvalingam_reduce(hull_pts, 4)
    else:"""

content = re.sub(old_logic_pattern, new_logic, content)

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied Smart Roof Decapitation fix to main.py!")
