import { useRef, useEffect, useCallback, useState } from 'react'
import { useCanvas2D } from '@/hooks/useCanvas2D'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import { MaskEditor } from './MaskEditor'

export interface Canvas2DProps {
  width?: number
  height?: number
  className?: string
  customMaskMode?: boolean
  onCustomMaskComplete?: (corners: [number, number][]) => void
}

export function Canvas2D({
  width = 800,
  height = 600,
  className = '',
  customMaskMode = false,
  onCustomMaskComplete,
}: Canvas2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)

  const [brushSize, setBrushSizeState] = useState(20)
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const maskTool = useUIStore((s) => s.maskTool)
  const setMaskTool = useUIStore((s) => s.setMaskTool)
  const hideWallMasks = useUIStore((s) => s.hideWallMasks)
  const setHideWallMasks = useUIStore((s) => s.setHideWallMasks)
  const wallVisibility = useUIStore((s) => s.wallVisibility)
  const sceneMode = useUIStore((s) => s.sceneMode)

  const walls = useWallStore((s) => s.walls)
  const wallImageSize = useWallStore((s) => s.wallImageSize)
  const selectedWallId = useWallStore((s) => s.selectedWallId)
  const isDetecting = useWallStore((s) => s.isDetecting)
  const setWallTexture = useWallStore((s) => s.setWallTexture)
  const wallTextures = useWallStore((s) => s.wallTextures)

  const editWallCorners = useUIStore((s) => s.editWallCorners)
  const setEditWallCorners = useUIStore((s) => s.setEditWallCorners)
  const editCornersMode = useUIStore((s) => s.editCornersMode)
  const setEditCornersMode = useUIStore((s) => s.setEditCornersMode)

  const {
    isReady,
    loadPhotoFromDataUrl,
    clearCanvas,
    drawingMode,
    setDrawingMode,
    setBrushSize,
    clearMask,
    setWallOverlays,
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
    customMaskMode,
    onCustomMaskComplete,
  })

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
    if (isReady && walls.length > 0 && wallImageSize) {
      setWallOverlays(walls, wallImageSize)
    }
  }, [isReady, walls, wallImageSize, wallTextures, hideWallMasks, wallVisibility, setWallOverlays])

  useEffect(() => {
    if (isReady) {
      highlightSelectedWall(selectedWallId)
    }
  }, [isReady, selectedWallId, highlightSelectedWall])

  useEffect(() => {
    if (!isReady) return
    syncCornerHandles(selectedWallId)
  }, [isReady, selectedWallId, editWallCorners, editCornersMode, syncCornerHandles])

  const handleApplyTexture = useCallback(async () => {
    const url = selectedMaterial?.texture.url
    if (!url) return
    try {
      if (selectedWallId != null && walls.length > 0 && wallImageSize) {
        const wall = walls.find((w) => w.id === selectedWallId)
        if (wall && wall.corners.length >= 3) {
          setWallTexture(selectedWallId, url)
          try {
            await applyTextureToWall(url, wall.corners, wallImageSize, selectedWallId)
          } catch (e) {
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
              className="rounded-lg bg-gray-800 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-gray-700 disabled:opacity-50"
            >
              Применить текстуру
            </button>
            <button
              type="button"
              onClick={() => setEditWallCorners(!editWallCorners)}
              className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                editWallCorners ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              Править углы
            </button>
            {editWallCorners && sceneMode === 'exterior' && (
              <div className="flex items-center rounded-lg border border-blue-200 bg-blue-50 p-0.5">
                <button
                  type="button"
                  onClick={() => setEditCornersMode('polygon')}
                  className={`rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                    editCornersMode === 'polygon'
                      ? 'bg-blue-600 text-white'
                      : 'text-blue-800 hover:bg-blue-100'
                  }`}
                >
                  Форма
                </button>
                <button
                  type="button"
                  onClick={() => setEditCornersMode('perspective')}
                  className={`rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                    editCornersMode === 'perspective'
                      ? 'bg-blue-600 text-white'
                      : 'text-blue-800 hover:bg-blue-100'
                  }`}
                >
                  Перспектива
                </button>
              </div>
            )}
            <button
              type="button"
              aria-pressed={hideWallMasks}
              onClick={() => setHideWallMasks(!hideWallMasks)}
              className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                hideWallMasks ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              Скрыть маски
            </button>
            {hasTextureLayer && (
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600" htmlFor="texture-scale">
                  Масштаб:
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
              onClick={clearCanvas}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-50"
            >
              Очистить
            </button>
          </div>
        </>
      )}
      <div className="relative overflow-hidden rounded-lg border border-gray-200 bg-gray-100 shadow-sm" style={{ width, height, maxWidth: '100%' }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="block touch-none"
          style={{ width, height, display: 'block' }}
          aria-label="Холст для наложения текстуры на фото"
        />
        {isDetecting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">
            {sceneMode === 'exterior' ? 'Определение фасада…' : 'Определение стен…'}
          </div>
        )}
      </div>
      {photoDataUrl && (
        <p className="text-xs text-gray-400">
          Клик по стене — выбор. Рисуйте маски кистью, прямоугольником или лассо.
        </p>
      )}
    </div>
  )
}
