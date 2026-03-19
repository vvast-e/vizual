import { useRef, useEffect, useCallback, useState } from 'react'
import { useCanvas2D } from '@/hooks/useCanvas2D'
import { useExport } from '@/hooks/useExport'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import { detectWalls, detectExterior, splitExteriorWalls } from '@/hooks/useWallDetection'
import { PhotoUploader } from './PhotoUploader'
import { MaskEditor } from './MaskEditor'
import { defaultExportFilename } from '@/lib/export-utils'

export interface Canvas2DProps {
  width?: number
  height?: number
  className?: string
}

export function Canvas2D({
  width = 800,
  height = 600,
  className = '',
}: Canvas2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)

  const [brushSize, setBrushSizeState] = useState(20)
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const maskTool = useUIStore((s) => s.maskTool)
  const setMaskTool = useUIStore((s) => s.setMaskTool)
  const hideWallMasks = useUIStore((s) => s.hideWallMasks)
  const setHideWallMasks = useUIStore((s) => s.setHideWallMasks)
  const sceneMode = useUIStore((s) => s.sceneMode)
  const setSceneMode = useUIStore((s) => s.setSceneMode)

  const walls = useWallStore((s) => s.walls)
  const wallImageSize = useWallStore((s) => s.wallImageSize)
  const selectedWallId = useWallStore((s) => s.selectedWallId)
  const isDetecting = useWallStore((s) => s.isDetecting)
  const setWalls = useWallStore((s) => s.setWalls)
  const setDetecting = useWallStore((s) => s.setDetecting)
  const setWallTexture = useWallStore((s) => s.setWallTexture)
  const wallTextures = useWallStore((s) => s.wallTextures)
  const exteriorMaskBase64 = useWallStore((s) => s.exteriorMaskBase64)
  const setExteriorMaskBase64 = useWallStore((s) => s.setExteriorMaskBase64)

  const splitFacadeMode = useUIStore((s) => s.splitFacadeMode)
  const setSplitFacadeMode = useUIStore((s) => s.setSplitFacadeMode)

  const handleSplitLineComplete = useCallback(
    (p1: { x: number; y: number }, p2: { x: number; y: number }) => {
      const mask = exteriorMaskBase64
      if (!mask || !wallImageSize) return
      splitExteriorWalls(mask, p1.x, p1.y, p2.x, p2.y, wallImageSize.width, wallImageSize.height)
        .then((res) => {
          setWalls(res.walls, wallImageSize)
        })
        .catch((err) => console.error('[Canvas2D] splitExteriorWalls failed', err))
    },
    [exteriorMaskBase64, wallImageSize, setWalls]
  )

  const {
    isReady,
    loadPhotoFromFile,
    loadPhotoFromDataUrl,
    exportToPng,
    clearCanvas,
    drawingMode,
    setDrawingMode,
    setBrushSize,
    clearMask,
    setWallOverlays,
    setExteriorMaskOverlay,
    applyTexture,
    applyTextureToWall,
    highlightSelectedWall,
    syncCornerHandles,
    finishLasso,
    clearMaskToolState,
    textureScale,
    setTextureScale,
    hasTextureLayer,
  } = useCanvas2D({
    canvasRef,
    containerWidth: width,
    containerHeight: height,
    onSplitLineComplete: handleSplitLineComplete,
  })

  const { exportToPng: downloadPng } = useExport()

  const handleBrushSizeChange = useCallback(
    (size: number) => {
      setBrushSizeState(size)
      setBrushSize(size)
    },
    [setBrushSize]
  )

  useEffect(() => {
    setDrawingMode(maskTool === 'brush')
  }, [maskTool, setDrawingMode])

  useEffect(() => {
    if (photoDataUrl && isReady) {
      loadPhotoFromDataUrl(photoDataUrl)
    }
  }, [photoDataUrl, isReady, loadPhotoFromDataUrl])

  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log('%c[Canvas2D] useEffect(setWallOverlays) FIRED', 'color:cyan;font-weight:bold', {
      isReady,
      wallsCount: walls.length,
      wallImageSize,
      wallTextures: JSON.parse(JSON.stringify(wallTextures)),
      hideWallMasks,
      setWallOverlaysRef: String(setWallOverlays).slice(0, 60),
    })
    if (isReady && walls.length > 0 && wallImageSize) {
      setWallOverlays(walls, wallImageSize)
    }
  }, [isReady, walls, wallImageSize, wallTextures, hideWallMasks, setWallOverlays])

  useEffect(() => {
    if (isReady) {
      highlightSelectedWall(selectedWallId)
    }
  }, [isReady, selectedWallId, highlightSelectedWall])

  const editWallCorners = useUIStore((s) => s.editWallCorners)
  const setEditWallCorners = useUIStore((s) => s.setEditWallCorners)

  useEffect(() => {
    if (!isReady) return
    syncCornerHandles(selectedWallId)
  }, [isReady, selectedWallId, editWallCorners, syncCornerHandles])

  const handleExport = useCallback(() => {
    const dataUrl = exportToPng()
    if (dataUrl) downloadPng(dataUrl, defaultExportFilename('visualizer'))
  }, [exportToPng, downloadPng])

  const handleApplyTexture = useCallback(async () => {
    const url = selectedMaterial?.texture.url
    if (!url) return
    // eslint-disable-next-line no-console
    console.log('%c[Canvas2D] handleApplyTexture START', 'color:lime;font-weight:bold', {
      url,
      selectedWallId,
      wallsCount: walls.length,
      wallImageSize,
      currentWallTextures: JSON.parse(JSON.stringify(useWallStore.getState().wallTextures)),
    })
    try {
      if (selectedWallId != null && walls.length > 0 && wallImageSize) {
        const wall = walls.find((w) => w.id === selectedWallId)
        if (wall && wall.corners.length >= 3) {
          // eslint-disable-next-line no-console
          console.log('[Canvas2D] calling setWallTexture', { selectedWallId, url })
          setWallTexture(selectedWallId, url)
          // eslint-disable-next-line no-console
          console.log('[Canvas2D] wallTextures AFTER setWallTexture:', JSON.parse(JSON.stringify(useWallStore.getState().wallTextures)))
          try {
            // eslint-disable-next-line no-console
            console.log('[Canvas2D] calling applyTextureToWall...')
            await applyTextureToWall(url, wall.corners, wallImageSize, selectedWallId)
            // eslint-disable-next-line no-console
            console.log('%c[Canvas2D] applyTextureToWall DONE', 'color:lime', {
              wallTextures: JSON.parse(JSON.stringify(useWallStore.getState().wallTextures)),
            })
          } catch (e) {
            // eslint-disable-next-line no-console
            console.error('[Canvas2D] applyTextureToWall FAILED, reverting texture', e)
            setWallTexture(selectedWallId, null)
            throw e
          }
          return
        }
      }
      await applyTexture(url, 'repeat')
    } catch (err) {
      console.error('Не удалось наложить текстуру:', url, err)
    }
  }, [selectedMaterial?.texture.url, selectedWallId, walls, wallImageSize, applyTexture, applyTextureToWall, setWallTexture])

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center gap-1 rounded border border-gray-200 bg-white p-1">
        <button
          type="button"
          onClick={() => setSceneMode('interior')}
          className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
            sceneMode === 'interior'
              ? 'bg-gray-800 text-white'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Интерьер
        </button>
        <button
          type="button"
          onClick={() => setSceneMode('exterior')}
          className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
            sceneMode === 'exterior'
              ? 'bg-gray-800 text-white'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Экстерьер
        </button>
      </div>
      <div className="relative overflow-hidden rounded border border-gray-200 bg-gray-100" style={{ width, height, maxWidth: '100%' }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="block touch-none"
          style={{ width, height, display: 'block' }}
          aria-label="Холст для наложения текстуры на фото"
        />
        {!photoDataUrl && (
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <PhotoUploader
              onFileSelect={(file) => {
                loadPhotoFromFile(file)
                setDetecting(true)

                if (sceneMode === 'exterior') {
                  detectExterior(file)
                    .then((res) => {
                      // eslint-disable-next-line no-console
                      console.log('%c[Canvas2D] detectExterior RESULT', 'color:orange;font-weight:bold', {
                        image_size: res.image_size,
                        hasMasks: Boolean(res.masks),
                        buildingBbox: res.debug?.building_bbox,
                        wallsCount: res.walls?.length ?? 0,
                      })
                      setWalls(res.walls ?? [], res.image_size)
                      setExteriorMaskBase64(res.masks?.wall_minus_holes ?? null)
                      setExteriorMaskOverlay(res.masks?.wall_minus_holes ?? null, res.image_size)
                    })
                    .catch((err) => {
                      console.error('[Canvas2D] detectExterior failed', err)
                    })
                    .finally(() => setDetecting(false))
                } else {
                  setExteriorMaskOverlay(null, null)
                  setExteriorMaskBase64(null)
                  detectWalls(file)
                    .then((res) => {
                      // eslint-disable-next-line no-console
                      console.log('%c[Canvas2D] detectWalls RESULT -> setWalls', 'color:cyan;font-weight:bold', {
                        wallsCount: res.walls?.length ?? 0,
                        wallIds: Array.isArray(res.walls) ? res.walls.map((w) => w.id) : [],
                        image_size: res.image_size,
                      })
                      setWalls(res.walls, res.image_size)
                    })
                    .catch(() => {})
                    .finally(() => setDetecting(false))
                }
              }}
              className="h-full min-h-[200px] w-full max-w-md"
            />
          </div>
        )}
        {isDetecting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">
            {sceneMode === 'exterior' ? 'Определение фасада…' : 'Определение стен…'}
          </div>
        )}
      </div>
      {photoDataUrl && (
        <>
          <MaskEditor
            drawingMode={drawingMode}
            onDrawingModeChange={setDrawingMode}
            brushSize={brushSize}
            onBrushSizeChange={handleBrushSizeChange}
            onClearMask={clearMask}
            maskTool={maskTool}
            onMaskToolChange={setMaskTool}
            onClearMaskToolState={clearMaskToolState}
            onFinishLasso={finishLasso}
          />
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handleApplyTexture}
              disabled={!selectedMaterial}
              className="rounded bg-gray-800 px-3 py-1.5 text-sm text-white hover:bg-gray-700 disabled:opacity-50"
            >
              Наложить текстуру
            </button>
            {sceneMode === 'exterior' && (
              <button
                type="button"
                onClick={() => setSplitFacadeMode(!splitFacadeMode)}
                className={`rounded border px-3 py-1.5 text-sm ${
                  splitFacadeMode ? 'border-rose-700 bg-rose-50 text-rose-800' : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                }`}
              >
                Разделить фасад
              </button>
            )}
            <button
              type="button"
              onClick={() => setEditWallCorners(!editWallCorners)}
              className={`rounded border px-3 py-1.5 text-sm ${
                editWallCorners ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-gray-300 text-gray-700 hover:bg-gray-100'
              }`}
            >
              Править углы
            </button>
            <button
              type="button"
              aria-pressed={hideWallMasks}
              onClick={() => setHideWallMasks(!hideWallMasks)}
              className={`rounded border px-3 py-1.5 text-sm ${
                hideWallMasks ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-gray-300 text-gray-700 hover:bg-gray-100'
              }`}
            >
              Скрыть маски
            </button>
            {hasTextureLayer && (
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600" htmlFor="texture-scale">
                  Масштаб текстуры:
                </label>
                <input
                  id="texture-scale"
                  type="range"
                  min={5}
                  max={100}
                  value={Math.round(textureScale * 100)}
                  onChange={(e) => setTextureScale(Number(e.target.value) / 100)}
                  className="h-2 w-28 cursor-pointer accent-gray-800"
                  aria-label="Масштаб текстуры"
                />
                <span className="text-xs text-gray-500 tabular-nums">
                  {Math.round(textureScale * 100)}%
                </span>
              </div>
            )}
            <button
              type="button"
              onClick={handleExport}
              className="rounded border border-gray-800 px-3 py-1.5 text-sm text-gray-800 hover:bg-gray-100"
            >
              Экспорт PNG
            </button>
            <button
              type="button"
              onClick={clearCanvas}
              className="rounded border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100"
            >
              Очистить
            </button>
          </div>
          <p className="text-xs text-gray-500">
            Колёсико — зум. Кисть — рисуйте область. Прямоугольник — выделите рамкой. Лассо — кликайте по точкам, затем «Завершить лассо». Кликните по маркеру стены, чтобы выбрать её для наложения текстуры.
          </p>
        </>
      )}
    </div>
  )
}
