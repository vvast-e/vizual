import { useRef, useEffect, useCallback, useState } from 'react'
import { useCanvas2D } from '@/hooks/useCanvas2D'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import { useColorize } from '@/hooks/useColorize'
import { useHistoryStore } from '@/store/useHistoryStore'

function LayersIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
    </svg>
  )
}

function PencilIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
    </svg>
  )
}

function EyeIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  )
}

function EyeOffIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
    </svg>
  )
}

function ScissorsIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12L9.121 12.121m0 5.758a3 3 0 10-4.243 4.243 3 3 0 004.243-4.243zm0-5.758a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z" />
    </svg>
  )
}

function UndoIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
    </svg>
  )
}

function RedoIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 10h-10a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6" />
    </svg>
  )
}

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
  const exteriorSplitLineActive = useUIStore((s) => s.exteriorSplitLineActive)
  const setExteriorSplitLineActive = useUIStore((s) => s.setExteriorSplitLineActive)

  const { selectedColor, colorizeOpacity, getColorizedTextureUrl } = useColorize()
  const { undo, redo, canUndo, canRedo } = useHistoryStore()
  const actionBtnBaseClass =
    'inline-flex h-8 items-center justify-center rounded-md border px-3 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 gap-1'

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

  // Ctrl+Z / Ctrl+Y hotkeys for undo/redo
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return
      const ctrl = e.ctrlKey || e.metaKey
      if (!ctrl) return
      if (e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        if (canUndo()) undo()
      } else if (e.key === 'y' || (e.key === 'z' && e.shiftKey)) {
        e.preventDefault()
        if (canRedo()) redo()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [undo, redo, canUndo, canRedo])

  const {
    isReady,
    loadPhotoFromDataUrl,
    setWallOverlays,
    setExteriorMaskOverlay,
    clearTextureFromWall,
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
    clearTextureFromWall,
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
    const willColorize = !!(selectedColor && selectedColor.hex !== '#ffffff' && colorizeOpacity > 0.001)
    if (willColorize) {
      return await getColorizedTextureUrl(rawUrl)
    }
    return rawUrl
  }, [selectedColor, colorizeOpacity, getColorizedTextureUrl])


  // Sync wall textures removals
  useEffect(() => {
    if (!isReady) return
    walls.forEach(w => {
      if (wallTextures[w.id] == null) {
        clearTextureFromWall(w.id)
      }
    })
  }, [wallTextures, isReady, clearTextureFromWall, walls])

  const handleApplyTexture = useCallback(async () => {
    const rawUrl = selectedMaterial?.texture.url
    if (!rawUrl) {
      return
    }
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
              className={`tour-apply-texture ${actionBtnBaseClass} min-w-[180px] border-gray-900 bg-gray-900 text-white hover:bg-black`}
              title="Наложить выбранный профиль на текущую область фасада"
            >
              <LayersIcon />
              Применить профиль
            </button>
            {sceneMode === 'exterior' && (
              <button
                type="button"
                onClick={() => setFacadeMergeActive((v) => !v)}
                disabled={walls.length === 0}
                className={`${actionBtnBaseClass} min-w-[140px] ${
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
              disabled={selectedWallId == null}
              className={`tour-edit-mask ${actionBtnBaseClass} min-w-[160px] ${
                editWallCorners
                  ? 'border-blue-600 bg-blue-50 text-blue-800'
                  : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
              }`}
              title="Включить режим правки формы выбранной области фасада"
            >
              <PencilIcon />
              Редактировать область
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
                disabled={selectedWallId == null}
                className={`${actionBtnBaseClass} min-w-[140px] ${
                  exteriorSplitLineActive
                    ? 'border-blue-600 bg-blue-50 text-blue-800'
                    : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                }`}
                title="Нарисовать вертикальную линию стыка между стенами"
              >
                <ScissorsIcon />
                Разрезать область
              </button>
            )}
            <button
              type="button"
              aria-pressed={hideWallMasks}
              onClick={() => setHideWallMasks(!hideWallMasks)}
              className={`${actionBtnBaseClass} min-w-[160px] ${
                hideWallMasks
                  ? 'border-blue-600 bg-blue-50 text-blue-800'
                  : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
              }`}
              title="Показать/скрыть синие контуры областей фасада"
            >
              {hideWallMasks ? <EyeOffIcon /> : <EyeIcon />}
              {hideWallMasks ? 'Показать области' : 'Скрыть области'}
            </button>
            <button
              type="button"
              onClick={() => undo()}
              disabled={!canUndo()}
              className={`${actionBtnBaseClass} min-w-[100px] border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-40`}
              title="Отменить последнее изменение области (Ctrl+Z)"
            >
              <UndoIcon />
              Отменить
            </button>
            <button
              type="button"
              onClick={() => redo()}
              disabled={!canRedo()}
              className={`${actionBtnBaseClass} min-w-[100px] border-gray-300 bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-40`}
              title="Повторить отменённое изменение (Ctrl+Y)"
            >
              <RedoIcon />
              Вернуть
            </button>
          </div>

          <div className="mt-1.5 flex min-h-8 flex-wrap items-center gap-1.5 rounded-md bg-gray-50 px-1.5 py-1">
            <div className="ml-auto flex items-center gap-2 rounded-md border border-gray-200 bg-white px-2 py-1"
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
