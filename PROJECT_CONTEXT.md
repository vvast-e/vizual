# Vizual - Wall Material Visualizer

Full-stack web app for visualizing textures/colors on walls in room photos.

## Stack
- Frontend: React 19 + TS + Vite + TailwindCSS + Fabric.js + Three.js
- Backend: FastAPI + Python 3.11 + PyTorch + OpenCV
- ML: Interior (ResNet wall seg + layout est), Exterior (GroundingDINO + SAM)

## Structure
- src/ - React frontend (pages/, components/, hooks/, store/, lib/)
- backend/ - FastAPI + ML models
- docker/ - Dockerfiles + nginx

## Pages
- /upload - Photo upload (interior/exterior toggle)
- /editor - Canvas2D + material panel
- /admin - Texture management

## API
- POST /api/detect-walls (interior)
- POST /api/detect-exterior (exterior)
- GET/POST/DELETE /api/materials

## State (Zustand)
- useVisualizerStore: viewMode, sceneMode, photoDataUrl
- useWallStore: walls[], wallTextures, selectedWallId
- useMaterialStore: selectedMaterial, selectedColor
- useUIStore: hideWallMasks, editWallCorners
- useHistoryStore: past[], future[], undo(), redo() - UNDO/REDO

## Recent Changes (2025-04-16)
- Toolbar: icons, dynamic "Show/Hide areas", titles
- Custom mask: polygon with click-to-close (already done)
- Undo/Redo: useHistoryStore + toolbar integration

## Recent Changes (2026-04-17)
- Backend: Rewrote `_corners_from_polygon_auto` v3 — hull-vertex + Visvalingam-Whyatt
  - hull==4 → use vertices directly (preserves real perspective Y-coords)
  - hull>4 → Visvalingam-Whyatt reduce to 4 (removes least-important vertex iteratively)
  - hull==3 → expand triangle: find eaves from original polygon points
  - v2 (scan-lines) was wrong: forced horizontal top/bottom edges, destroyed perspective
  - v1 (minAreaRect/extremes) produced duplicate corners → det2=0
- Backend: CRITICAL FIX — always re-derive corners from polygon in `_warp_texture_overlay_sync`
  - Old code: `if len(corners) != 4` → never triggered (frontend always sends 4, with duplicates)
  - New code: `if use_polygon_early` → always re-derive when polygon available
  - Same fix applied to multi-region (gable) path
- Backend: `_order_points_tl_tr_br_bl` uses centroid-angle algorithm (replaced broken sum/diff)
- Backend: `compute_perspective_from_polygon` now delegates to `_corners_from_polygon_auto`
- Frontend: Removed all 42 `[DEBUG]` console.log statements from Canvas2D.tsx, useCanvas2D.ts, texture-processor.ts
- Frontend: Added Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z) keyboard hotkeys for undo/redo in Canvas2D.tsx
- Frontend: Fixed selectedColor/selectedMaterial mutual exclusion in widgets
- Frontend: Mock materials reduced to only existing textures (planken, vagonka)

## Key Backend Functions (backend/main.py)
- `_order_points_tl_tr_br_bl` (~line 84): Centroid-angle ordering for any convex quad
- `_visvalingam_reduce` (~line 290): Reduce polygon to N vertices preserving shape
- `_expand_triangle_to_quad` (~line 310): Convert 3-vertex hull to wall quad
- `_corners_from_polygon_auto` (~line 350): Hull-vertex + Visvalingam, N-point polygon → 4 corners
- `_warp_texture_overlay_sync` (~line 570): Main texture warping pipeline (always re-derives corners)
- `_warp_texture_overlay_multi_regions_sync` (~line 470): Two-region gable warp (also re-derives)
- `/api/warp-wall-texture` (~line 1050): Texture warp endpoint
