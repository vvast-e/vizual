import { useRef, useEffect, useCallback, useState } from 'react'
import { useCanvas2D } from '@/hooks/useCanvas2D'
import { useExport } from '@/hooks/useExport'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
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
    applyTexture,
    finishLasso,
    clearMaskToolState,
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

  const handleExport = useCallback(() => {
    const dataUrl = exportToPng()
    if (dataUrl) downloadPng(dataUrl, defaultExportFilename('visualizer'))
  }, [exportToPng, downloadPng])

  const handleApplyTexture = useCallback(async () => {
    const url = selectedMaterial?.texture.url
    if (!url) return
    try {
      await applyTexture(url, 'repeat')
    } catch (err) {
      console.error('Не удалось наложить текстуру:', url, err)
    }
  }, [selectedMaterial?.texture.url, applyTexture])

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
            <PhotoUploader onFileSelect={loadPhotoFromFile} className="h-full min-h-[200px] w-full max-w-md" />
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
            Колёсико — зум. Кисть — рисуйте область. Прямоугольник — выделите рамкой. Лассо — кликайте по точкам, затем «Завершить лассо». Без инструмента — перетаскивание панорамы.
          </p>
        </>
      )}
    </div>
  )
}
