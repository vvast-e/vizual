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
