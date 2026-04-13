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
  const [facadeMergeActive, setFacadeMergeActive] = useState(false)
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
  const actionBtnBaseClass =
    'inline-flex h-8 items-center justify-center rounded-md border px-3 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50'

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
    if (!isReady || !photoDataUrl) return
    if (!isPhotoLoaded) return

    // При переходе /upload -> /editor возможна гонка первого кадра.
    // Повторяем отрисовку оверлеев на следующих frame, чтобы гарантировать показ масок.
    const draw = () => setWallOverlays(walls, wallImageSize)
    draw()
    const raf1 = window.requestAnimationFrame(draw)
    const raf2 = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(draw)
    })
    return () => {
      window.cancelAnimationFrame(raf1)
      window.cancelAnimationFrame(raf2)
    }
  }, [
    isReady,
    isPhotoLoaded,
    photoDataUrl,
    walls,
    wallImageSize,
    hideWallMasks,
    wallVisibility,
    setWallOverlays,
  ])

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
      if (sceneMode === 'exterior' && walls.length > 0 && wallImageSize) {
        if (facadeMergeActive) {
          const profiledIds = walls
            .map((w) => w.id)
            .filter((id) => wallTextures[id] != null)
          if (profiledIds.length === 0) {
            console.warn('Режим объединения фасада активен, но нет областей с уже наложенным профилем')
            return
          }
          for (const id of profiledIds) {
            const wall = walls.find((w) => w.id === id)
            if (!wall || wall.corners.length < 3) continue
            setWallTexture(id, url)
            try {
              await applyTextureToWall(url, wall.corners, wallImageSize, id)
            } catch (e) {
              setWallTexture(id, null)
              throw e
            }
          }
          return
        }
        const fallbackId = selectedWallId ?? walls[0]?.id ?? null
        if (fallbackId != null) {
          const wall = walls.find((w) => w.id === fallbackId)
          if (wall && wall.corners.length >= 3) {
            setWallTexture(fallbackId, url)
            try {
              await applyTextureToWall(url, wall.corners, wallImageSize, fallbackId)
            } catch (e) {
              setWallTexture(fallbackId, null)
              throw e
            }
            return
          }
        }
      }
      await applyTexture(url, 'repeat')
    } catch (err) {
      console.error('Не удалось наложить текстуру:', rawUrl, err)
    }
  }, [
    selectedMaterial?.texture.url,
    selectedWallId,
    walls,
    wallImageSize,
    wallTextures,
    sceneMode,
    facadeMergeActive,
    applyTexture,
    applyTextureToWall,
    setWallTexture,
    getTextureUrl,
  ])

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
    <div className={`flex flex-1 flex-col gap-1 overflow-hidden ${className}`}>
      {photoDataUrl && (
        <div className="rounded-lg border border-gray-200 bg-white p-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={handleApplyTexture}
              disabled={!selectedMaterial}
              className={`tour-apply-texture ${actionBtnBaseClass} min-w-[156px] border-gray-900 bg-gray-900 text-white hover:bg-black`}
              title="Применить выбранный профиль к активной области"
            >
              Применить профиль
            </button>
            {sceneMode === 'exterior' && (
              <button
                type="button"
                onClick={() => setFacadeMergeActive((v) => !v)}
                disabled={walls.length === 0}
                className={`${actionBtnBaseClass} min-w-[156px] ${
                  facadeMergeActive
                    ? 'border-blue-600 bg-blue-50 text-blue-800'
                    : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                }`}
                title="Когда включено, профиль применяется сразу ко всем областям фасада"
              >
                Объединить фасад
              </button>
            )}
            <button
              type="button"
              onClick={() => setEditWallCorners(!editWallCorners)}
              className={`tour-edit-mask ${actionBtnBaseClass} min-w-[140px] ${
                editWallCorners
                  ? 'border-blue-600 bg-blue-50 text-blue-800'
                  : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
              }`}
              title="Включить ручную правку формы и перспективы выбранной области"
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
                className={`${actionBtnBaseClass} min-w-[140px] ${
                  exteriorSplitLineActive
                    ? 'border-blue-600 bg-blue-50 text-blue-800'
                    : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                }`}
                title="Разрезать фасад по линии двумя кликами на фото"
              >
                Разрез по линии
              </button>
            )}
            <button
              type="button"
              aria-pressed={hideWallMasks}
              onClick={() => setHideWallMasks(!hideWallMasks)}
              className={`${actionBtnBaseClass} min-w-[140px] ${
                hideWallMasks
                  ? 'border-blue-600 bg-blue-50 text-blue-800'
                  : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
              }`}
              title="Показать или скрыть контуры областей на фото"
            >
              Скрыть области
            </button>
          </div>

          <div className="mt-1.5 flex min-h-8 flex-wrap items-center gap-1.5 rounded-md bg-gray-50 px-1.5 py-1">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">Режимы</span>
            <div className="flex items-center rounded-md border border-gray-200 bg-white p-0.5">
              <button
                type="button"
                onClick={() => setEditCornersMode('polygon')}
                disabled={!editWallCorners || sceneMode !== 'exterior'}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${
                  editCornersMode === 'polygon'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
                title="Точная правка контура области по точкам"
              >
                Форма
              </button>
              <button
                type="button"
                onClick={() => setEditCornersMode('perspective')}
                disabled={!editWallCorners || sceneMode !== 'exterior'}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${
                  editCornersMode === 'perspective'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
                title="Правка перспективы четырьмя углами"
              >
                Перспектива
              </button>
            </div>
            <div
              className="ml-auto flex items-center gap-2 rounded-md border border-gray-200 bg-white px-2 py-1"
              title="Масштаб отображения текстуры на выбранной области"
            >
              <label className="text-xs font-medium text-gray-600" htmlFor="texture-scale">
                Масштаб
              </label>
              <input
                id="texture-scale"
                type="range"
                min={5}
                max={100}
                value={Math.round(textureScale * 100)}
                onChange={(e) => setTextureScale(Number(e.target.value) / 100, selectedWallId)}
                disabled={!hasTextureLayer}
                className="h-2 w-24 cursor-pointer accent-gray-800 disabled:opacity-40"
                aria-label="Масштаб текстуры"
              />
              <span className="w-10 text-right text-xs text-gray-500 tabular-nums">
                {Math.round(textureScale * 100)}%
              </span>
            </div>
          </div>
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
