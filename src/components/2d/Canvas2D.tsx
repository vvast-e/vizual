import { useRef, useEffect, useCallback, useState } from 'react'
import { useCanvas2D } from '@/hooks/useCanvas2D'
import { useExport } from '@/hooks/useExport'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import { detectWalls } from '@/hooks/useWallDetection'
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

  const walls = useWallStore((s) => s.walls)
  const wallImageSize = useWallStore((s) => s.wallImageSize)
  const selectedWallId = useWallStore((s) => s.selectedWallId)
  const isDetecting = useWallStore((s) => s.isDetecting)
  const setWalls = useWallStore((s) => s.setWalls)
  const setDetecting = useWallStore((s) => s.setDetecting)

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
    getBackgroundBounds,
    setWallOverlays,
    applyTexture,
    applyTextureToWall,
    highlightSelectedWall,
    finishLasso,
    clearMaskToolState,
    textureScale,
    setTextureScale,
    hasTextureLayer,
  } = useCanvas2D({
    canvasRef,
    containerWidth: width,
    containerHeight: height,
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
    if (isReady && walls.length > 0 && wallImageSize) {
      setWallOverlays(walls, wallImageSize)
    }
  }, [isReady, walls, wallImageSize, setWallOverlays])

  useEffect(() => {
    if (isReady) {
      highlightSelectedWall(selectedWallId)
    }
  }, [isReady, selectedWallId, highlightSelectedWall])

  const handleExport = useCallback(() => {
    const dataUrl = exportToPng()
    if (dataUrl) downloadPng(dataUrl, defaultExportFilename('visualizer'))
  }, [exportToPng, downloadPng])

  const handleApplyTexture = useCallback(async () => {
    const url = selectedMaterial?.texture.url
    if (!url) return
    try {
      if (selectedWallId != null && walls.length > 0 && wallImageSize) {
        const wall = walls.find((w) => w.id === selectedWallId)
        const bounds = getBackgroundBounds()
        if (wall && bounds && wall.corners.length >= 3) {
          const scaleX = bounds.width / wallImageSize.width
          const scaleY = bounds.height / wallImageSize.height
          const scaledCorners = wall.corners.map(
            (c): [number, number] => [bounds.left + c[0] * scaleX, bounds.top + c[1] * scaleY]
          )
          // Диагностика перспективы
          // eslint-disable-next-line no-console
          console.log('[WallTexture] selectedWallId:', selectedWallId)
          // eslint-disable-next-line no-console
          console.log('[WallTexture] wall.corners (image coords):', wall.corners)
          // eslint-disable-next-line no-console
          console.log('[WallTexture] image_size:', wallImageSize)
          // eslint-disable-next-line no-console
          console.log('[WallTexture] background bounds:', bounds)
          // eslint-disable-next-line no-console
          console.log('[WallTexture] scaledCorners (canvas coords):', scaledCorners)
          await applyTextureToWall(url, scaledCorners)
          return
        }
      }
      await applyTexture(url, 'repeat')
    } catch (err) {
      console.error('Не удалось наложить текстуру:', url, err)
    }
  }, [selectedMaterial?.texture.url, selectedWallId, walls, wallImageSize, getBackgroundBounds, applyTexture, applyTextureToWall])

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
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
                detectWalls(file)
                  .then((res) => setWalls(res.walls, res.image_size))
                  .catch(() => {})
                  .finally(() => setDetecting(false))
              }}
              className="h-full min-h-[200px] w-full max-w-md"
            />
          </div>
        )}
        {isDetecting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">
            Определение стен…
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
