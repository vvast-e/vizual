import { useRef, useEffect, useCallback, useState } from 'react'
import { useCanvas2D } from '@/hooks/useCanvas2D'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import { useColorize } from '@/hooks/useColorize'

export interface Canvas2DProps {
  className?: string
  customMaskMode?: boolean
  onCustomMaskComplete?: (corners: [number, number][]) => void
}

export function Canvas2D({
  className = '',
  customMaskMode = false,
  onCustomMaskComplete,
}: Canvas2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 500 })
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)

  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const hideWallMasks = useUIStore((s) => s.hideWallMasks)
  const setHideWallMasks = useUIStore((s) => s.setHideWallMasks)
  const wallVisibility = useUIStore((s) => s.wallVisibility)
  const sceneMode = useUIStore((s) => s.sceneMode)

  const walls = useWallStore((s) => s.walls)
  const wallImageSize = useWallStore((s) => s.wallImageSize)
  const exteriorMaskBase64 = useWallStore((s) => s.exteriorMaskBase64)
  const selectedWallId = useWallStore((s) => s.selectedWallId)
  const isDetecting = useWallStore((s) => s.isDetecting)
  const setWallTexture = useWallStore((s) => s.setWallTexture)
  const wallTextures = useWallStore((s) => s.wallTextures)

  const editWallCorners = useUIStore((s) => s.editWallCorners)
  const setEditWallCorners = useUIStore((s) => s.setEditWallCorners)
  const editCornersMode = useUIStore((s) => s.editCornersMode)
  const setEditCornersMode = useUIStore((s) => s.setEditCornersMode)
  const exteriorSplitLineActive = useUIStore((s) => s.exteriorSplitLineActive)
  const setExteriorSplitLineActive = useUIStore((s) => s.setExteriorSplitLineActive)

  const { selectedColor, colorizeOpacity, getColorizedTextureUrl } = useColorize()

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect
        if (w > 0 && h > 0) {
          setCanvasSize({ width: Math.round(w), height: Math.round(h) })
        }
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const {
    isReady,
    loadPhotoFromDataUrl,
    setWallOverlays,
    setExteriorMaskOverlay,
    applyTexture,
    applyTextureToWall,
    applyTextureToAllWalls,
    highlightSelectedWall,
    syncCornerHandles,
    clearFacadeSplitDraft,
    textureScale,
    setTextureScale,
    hasTextureLayer,
    isPhotoLoaded,
    canvasInstanceRef,
  } = useCanvas2D({
    canvasRef,
    containerWidth: canvasSize.width,
    containerHeight: canvasSize.height,
    customMaskMode,
    onCustomMaskComplete,
  })

  useEffect(() => {
    if (photoDataUrl && isReady) {
      loadPhotoFromDataUrl(photoDataUrl)
    }
  }, [photoDataUrl, isReady, loadPhotoFromDataUrl])

  useEffect(() => {
    if (isReady && isPhotoLoaded && walls.length > 0 && wallImageSize) {
      setWallOverlays(walls, wallImageSize)
    }
  }, [isReady, isPhotoLoaded, walls, wallImageSize, wallTextures, hideWallMasks, wallVisibility, setWallOverlays])

  useEffect(() => {
    if (!isReady || !isPhotoLoaded) return
    if (sceneMode === 'exterior' && exteriorMaskBase64 && wallImageSize) {
      void setExteriorMaskOverlay(exteriorMaskBase64, wallImageSize)
    } else {
      void setExteriorMaskOverlay(null, null)
    }
  }, [
    isReady,
    isPhotoLoaded,
    sceneMode,
    exteriorMaskBase64,
    wallImageSize,
    setExteriorMaskOverlay,
  ])

  useEffect(() => {
    if (!isReady) return
    const c = canvasInstanceRef.current
    if (!c) return
    if (exteriorSplitLineActive && sceneMode === 'exterior') {
      c.defaultCursor = 'crosshair'
    } else {
      c.defaultCursor = 'default'
    }
    c.requestRenderAll()
  }, [isReady, exteriorSplitLineActive, sceneMode, canvasInstanceRef])

  useEffect(() => {
    if (isReady) {
      highlightSelectedWall(selectedWallId)
    }
  }, [isReady, selectedWallId, highlightSelectedWall])

  useEffect(() => {
    if (!isReady) return
    syncCornerHandles(selectedWallId)
  }, [isReady, selectedWallId, editWallCorners, editCornersMode, syncCornerHandles])

  /** Получить URL текстуры с учётом HSV-колоризации */
  const getTextureUrl = useCallback(async (rawUrl: string): Promise<string> => {
    if (selectedColor && selectedColor.hex !== '#ffffff' && colorizeOpacity > 0.001) {
      return getColorizedTextureUrl(rawUrl)
    }
    return rawUrl
  }, [selectedColor, colorizeOpacity, getColorizedTextureUrl])

  const handleApplyTexture = useCallback(async () => {
    const rawUrl = selectedMaterial?.texture.url
    if (!rawUrl) return
    try {
      const url = await getTextureUrl(rawUrl)
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
      console.error('Не удалось наложить текстуру:', rawUrl, err)
    }
  }, [selectedMaterial?.texture.url, selectedWallId, walls, wallImageSize, applyTexture, applyTextureToWall, setWallTexture, getTextureUrl])

  const handleApplyFacadeProfile = useCallback(async () => {
    const rawUrl = selectedMaterial?.texture.url
    if (!rawUrl || walls.length === 0 || sceneMode !== 'exterior') return
    try {
      const url = await getTextureUrl(rawUrl)
      await applyTextureToAllWalls(url)
    } catch (err) {
      console.error('Не удалось применить профиль ко всем областям фасада:', rawUrl, err)
    }
  }, [selectedMaterial?.texture.url, walls.length, sceneMode, getTextureUrl, applyTextureToAllWalls])

  // Авто-применение при смене цвета (debounce 300ms)
  const colorKeyRef = useRef<string>('')
  useEffect(() => {
    const key = `${selectedColor?.hex ?? ''}_${colorizeOpacity}`
    if (colorKeyRef.current === '') {
      colorKeyRef.current = key
      return
    }
    if (colorKeyRef.current === key) return
    colorKeyRef.current = key

    if (!isReady || !selectedMaterial || !hasTextureLayer) return
    const timer = setTimeout(() => {
      handleApplyTexture()
    }, 300)
    return () => clearTimeout(timer)
  }, [selectedColor?.hex, colorizeOpacity, isReady, selectedMaterial, hasTextureLayer, handleApplyTexture])

  return (
    <div className={`flex flex-1 flex-col gap-2 overflow-hidden ${className}`}>
      {photoDataUrl && (
        <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handleApplyTexture}
              disabled={!selectedMaterial}
              className="tour-apply-texture rounded-lg bg-gray-800 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-gray-700 disabled:opacity-50"
            >
              Применить профиль
            </button>
            {sceneMode === 'exterior' && (
              <button
                type="button"
                onClick={handleApplyFacadeProfile}
                disabled={!selectedMaterial || walls.length === 0}
                className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50 disabled:opacity-50"
              >
                Объединить фасад
              </button>
            )}
            <button
              type="button"
              onClick={() => setEditWallCorners(!editWallCorners)}
              className={`tour-edit-mask rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                editWallCorners ? 'border-blue-700 bg-blue-50 text-blue-800' : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              Править область
            </button>
            {sceneMode === 'exterior' && exteriorMaskBase64 && wallImageSize && (
              <button
                type="button"
                onClick={() => {
                  if (exteriorSplitLineActive) {
                    setExteriorSplitLineActive(false)
                    clearFacadeSplitDraft()
                  } else {
                    setExteriorSplitLineActive(true)
                  }
                }}
                className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                  exteriorSplitLineActive
                    ? 'border-orange-600 bg-orange-50 text-orange-900'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                {exteriorSplitLineActive ? 'Отменить разрез' : 'Разрез по линии'}
              </button>
            )}
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
              Скрыть области
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
                  onChange={(e) => setTextureScale(Number(e.target.value) / 100, selectedWallId)}
                  className="h-2 w-28 cursor-pointer accent-gray-800"
                  aria-label="Масштаб текстуры"
                />
                <span className="text-xs text-gray-500 tabular-nums">
                  {Math.round(textureScale * 100)}%
                </span>
              </div>
            )}

          </div>
      )}
      <div
        ref={containerRef}
        className="tour-canvas-container relative flex-1 overflow-hidden rounded-lg border border-gray-200 bg-gray-100 shadow-sm"
        style={{ minHeight: 200 }}
      >
        <canvas
          ref={canvasRef}
          width={canvasSize.width}
          height={canvasSize.height}
          className="block touch-none"
          style={{ width: '100%', height: '100%', display: 'block' }}
          aria-label="Холст для наложения текстуры на фото"
        />
        {isDetecting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">
            {sceneMode === 'exterior' ? 'Определение фасада…' : 'Определение стен…'}
          </div>
        )}
      </div>
    </div>
  )
}
