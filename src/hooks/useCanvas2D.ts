import { useEffect, useRef, useCallback, useState } from 'react'
import { Canvas, FabricImage, Point, PencilBrush, Group, Rect, Path, Circle } from 'fabric'
import type { FabricObject } from 'fabric'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import type { WallData } from '@/store/useWallStore'
import { useHistoryStore } from '@/store/useHistoryStore'
import { MAX_PHOTO_SIZE_BYTES, ALLOWED_IMAGE_TYPES } from '@/lib/constants'
import { splitExteriorWalls } from '@/hooks/useWallDetection'
import { autoLinkPerspectiveToForm } from '@/lib/perspective-helper'

export interface UseCanvas2DOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  containerWidth?: number
  containerHeight?: number
  /** Режим рисования кастомной маски (4 точки перспективы) */
  customMaskMode?: boolean
  /** Вызывается когда 4 точки перспективы нарисованы */
  onCustomMaskComplete?: (corners: [number, number][]) => void
}

export function useCanvas2D({
  canvasRef,
  containerWidth = 800,
  containerHeight = 600,
  customMaskMode = false,
  onCustomMaskComplete,
}: UseCanvas2DOptions) {
  const canvasInstanceRef = useRef<Canvas | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [drawingMode, setDrawingModeState] = useState(false)
  const MASK_DATA_KEY = 'isMask'
  const backgroundImageRef = useRef<FabricImage | null>(null)

  const maskTool = useUIStore((s) => s.maskTool)
  const maskToolRef = useRef(maskTool)
  maskToolRef.current = maskTool
  const setPhotoDataUrl = useVisualizerStore((s) => s.setPhotoDataUrl)

  const rectStartRef = useRef<{ x: number; y: number } | null>(null)
  const rectPreviewRef = useRef<Rect | null>(null)
  const lassoPointsRef = useRef<{ x: number; y: number }[]>([])
  const lassoPreviewRef = useRef<Path | null>(null)

  const [textureScale, setTextureScaleState] = useState(0.25)
  const [hasTextureLayer, setHasTextureLayer] = useState(false)
  const [isPhotoLoaded, setIsPhotoLoaded] = useState(false)
  const [beforeAfter, setBeforeAfter] = useState(false)
  
  const textureLayersRef = useRef<Record<string, FabricObject>>({})
  const wallOverlaysRef = useRef<FabricObject[]>([])
  const wallDebugShapesRef = useRef<FabricObject[]>([])
  const wallIdByObjectRef = useRef<WeakMap<FabricObject, number>>(new WeakMap())
  const WALL_BUTTON_DATA_KEY = 'wallId'
  const cornerHandlesRef = useRef<FabricObject[]>([])
  const CORNER_HANDLE_DATA_KEY = 'cornerHandle'
  const exteriorMaskRef = useRef<FabricObject | null>(null)
  const EXTERIOR_MASK_DATA_KEY = 'isExteriorMask'
  const facadeSplitP1SceneRef = useRef<{ x: number; y: number } | null>(null)
  const facadeSplitImageP1Ref = useRef<{ ix: number; iy: number } | null>(null)
  const facadeSplitTargetWallIdRef = useRef<number | null>(null)
  const facadeSplitPreviewLineRef = useRef<Path | null>(null)

  const interiorMaskImageRef = useRef<{ b64: string; img: HTMLImageElement } | null>(null)

  const loadInteriorMaskImage = useCallback(async (): Promise<HTMLImageElement | null> => {
    const b64 = useWallStore.getState().interiorMaskBase64
    if (!b64) {
      interiorMaskImageRef.current = null
      return null
    }
    const cached = interiorMaskImageRef.current
    if (cached && cached.b64 === b64) return cached.img
    const img = new Image()
    img.src = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve()
      img.onerror = () => reject(new Error('Failed to decode interior mask'))
    })
    interiorMaskImageRef.current = { b64, img }
    return img
  }, [])

  const customMaskModeRef = useRef(customMaskMode)
  customMaskModeRef.current = customMaskMode
  const customMaskPointsRef = useRef<{ x: number; y: number }[]>([])
  const customMaskPreviewRef = useRef<FabricObject[]>([])
  const customMaskHistoryRef = useRef<{ x: number; y: number }[][]>([])
  const customMaskFutureRef = useRef<{ x: number; y: number }[][]>([])
  const onCustomMaskCompleteRef = useRef(onCustomMaskComplete)
  onCustomMaskCompleteRef.current = onCustomMaskComplete

  const rebuildCustomMaskPreview = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    for (const obj of customMaskPreviewRef.current) {
      canvas.remove(obj)
    }
    customMaskPreviewRef.current = []
    const pts = customMaskPointsRef.current
    if (pts.length === 0) {
      canvas.requestRenderAll()
      return
    }
    pts.forEach((p, idx) => {
      const dot = new Circle({
        left: p.x - 5,
        top: p.y - 5,
        radius: 5,
        fill: '#10b981',
        stroke: '#064e3b',
        strokeWidth: 1,
        selectable: false,
        evented: false,
        originX: 'left',
        originY: 'top',
      })
      canvas.add(dot)
      customMaskPreviewRef.current.push(dot)
      if (idx > 0) {
        const prev = pts[idx - 1]
        const line = new Path(`M ${prev.x} ${prev.y} L ${p.x} ${p.y}`, {
          stroke: '#10b981',
          strokeWidth: 2,
          selectable: false,
          evented: false,
        })
        canvas.add(line)
        customMaskPreviewRef.current.push(line)
      }
    })
    canvas.requestRenderAll()
  }, [])

  const undoCustomMask = useCallback(() => {
    if (customMaskHistoryRef.current.length === 0) return
    const prev = customMaskHistoryRef.current.pop()
    if (!prev) return
    customMaskFutureRef.current.push([...customMaskPointsRef.current])
    customMaskPointsRef.current = [...prev]
    rebuildCustomMaskPreview()
  }, [rebuildCustomMaskPreview])

  const redoCustomMask = useCallback(() => {
    if (customMaskFutureRef.current.length === 0) return
    const next = customMaskFutureRef.current.pop()
    if (!next) return
    customMaskHistoryRef.current.push([...customMaskPointsRef.current])
    customMaskPointsRef.current = [...next]
    rebuildCustomMaskPreview()
  }, [rebuildCustomMaskPreview])

  const initCanvas = useCallback(() => {
    const el = canvasRef.current
    if (!el) return null
    const canvas = new Canvas(el, {
      selection: false,
      preserveObjectStacking: true,
      skipTargetFind: false,
    })
    canvas.setDimensions({ width: containerWidth, height: containerHeight })
    canvas.defaultCursor = 'default'
    canvas.hoverCursor = 'default'
    canvas.moveCursor = 'default'
    canvas.setZoom(1)
    canvas.viewportTransform = [1, 0, 0, 1, 0, 0]
    const brush = new PencilBrush(canvas)
    brush.color = 'rgba(0,0,0,0.8)'
    brush.width = 20
    canvas.freeDrawingBrush = brush

    const removeRectPreview = () => {
      const prev = rectPreviewRef.current
      if (prev) {
        canvas.remove(prev)
        rectPreviewRef.current = null
      }
      rectStartRef.current = null
    }

    canvas.on('mouse:down', (opt) => {
      const scenePoint = canvas.getScenePoint(opt.e as MouseEvent)

      if (customMaskModeRef.current) {
        const pts = customMaskPointsRef.current
        // Сохраняем предыдущее состояние для Undo
        customMaskHistoryRef.current.push([...pts])
        customMaskFutureRef.current = []
        const newPoint = { x: scenePoint.x, y: scenePoint.y }
        pts.push(newPoint)

        const dot = new Circle({
          left: newPoint.x - 5,
          top: newPoint.y - 5,
          radius: 5,
          fill: '#10b981',
          stroke: '#064e3b',
          strokeWidth: 1,
          selectable: false,
          evented: false,
          originX: 'left',
          originY: 'top',
        })
        canvas.add(dot)
        customMaskPreviewRef.current.push(dot)

        if (pts.length > 1) {
          const prev = pts[pts.length - 2]
          const line = new Path(`M ${prev.x} ${prev.y} L ${newPoint.x} ${newPoint.y}`, {
            stroke: '#10b981',
            strokeWidth: 2,
            selectable: false,
            evented: false,
          })
          canvas.add(line)
          customMaskPreviewRef.current.push(line)
        }

        // Завершение рисования: клик рядом с первой точкой при >=3 вершинах.
        if (pts.length >= 3) {
          const first = pts[0]
          const dx = newPoint.x - first.x
          const dy = newPoint.y - first.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          const CLOSE_RADIUS = 15
          if (dist <= CLOSE_RADIUS) {
            // Используем первую точку как последнюю для замыкания.
            pts[pts.length - 1] = { x: first.x, y: first.y }
            const prev = pts[pts.length - 2]
            const closeLine = new Path(`M ${prev.x} ${prev.y} L ${first.x} ${first.y}`, {
              stroke: '#10b981',
              strokeWidth: 2,
              selectable: false,
              evented: false,
            })
            canvas.add(closeLine)
            customMaskPreviewRef.current.push(closeLine)

            const bounds = getBackgroundBounds()
            const wallImageSize = useWallStore.getState().wallImageSize
            if (bounds && wallImageSize && bounds.width > 0 && bounds.height > 0) {
              const corners: [number, number][] = pts.map((p) => {
                const ix = Math.round(((p.x - bounds.left) / bounds.width) * wallImageSize.width)
                const iy = Math.round(((p.y - bounds.top) / bounds.height) * wallImageSize.height)
                const ixCl = Math.max(0, Math.min(wallImageSize.width - 1, ix))
                const iyCl = Math.max(0, Math.min(wallImageSize.height - 1, iy))
                return [ixCl, iyCl] as [number, number]
              })
              const cb = onCustomMaskCompleteRef.current
              if (cb) {
                cb(corners)
              }
            }

            for (const obj of customMaskPreviewRef.current) {
              canvas.remove(obj)
            }
            customMaskPreviewRef.current = []
            customMaskPointsRef.current = []
            customMaskHistoryRef.current = []
            customMaskFutureRef.current = []
          }
        }

        canvas.requestRenderAll()
        return
      }

      const uiSplit = useUIStore.getState()
      if (uiSplit.exteriorSplitLineActive && uiSplit.sceneMode === 'exterior') {
        const bg = backgroundImageRef.current
        const wallImageSize = useWallStore.getState().wallImageSize
        const maskB64 = useWallStore.getState().exteriorMaskBase64
        if (!bg || !wallImageSize || !maskB64) {
          return
        }
        const left = (bg.left ?? 0) as number
        const top = (bg.top ?? 0) as number
        const bw = ((bg.width ?? 0) as number) * ((bg.scaleX ?? 1) as number)
        const bh = ((bg.height ?? 0) as number) * ((bg.scaleY ?? 1) as number)
        if (bw <= 0 || bh <= 0) {
          return
        }
        const ix = Math.round(((scenePoint.x - left) / bw) * wallImageSize.width)
        const iy = Math.round(((scenePoint.y - top) / bh) * wallImageSize.height)
        const ixCl = Math.max(0, Math.min(wallImageSize.width - 1, ix))
        const iyCl = Math.max(0, Math.min(wallImageSize.height - 1, iy))

        if (!facadeSplitP1SceneRef.current) {
          const point = new Point(scenePoint.x, scenePoint.y)
          let targetWallId: number | null = null
          for (let i = wallOverlaysRef.current.length - 1; i >= 0; i--) {
            const obj = wallOverlaysRef.current[i]
            if (
              typeof (obj as unknown as { containsPoint?: (p: Point) => boolean }).containsPoint ===
                'function' &&
              (obj as unknown as { containsPoint: (p: Point) => boolean }).containsPoint(point)
            ) {
              const wallId = wallIdByObjectRef.current.get(obj)
              if (wallId != null) {
                targetWallId = wallId
                break
              }
            }
          }
          if (targetWallId == null) {
            const sel = useWallStore.getState().selectedWallId
            if (sel != null) targetWallId = sel
          }
          if (targetWallId == null) {
            console.warn('[facade-split] Кликните по стене, которую нужно разрезать (или выберите стену)')
            return
          }
          facadeSplitTargetWallIdRef.current = targetWallId
          facadeSplitP1SceneRef.current = { x: scenePoint.x, y: scenePoint.y }
          facadeSplitImageP1Ref.current = { ix: ixCl, iy: iyCl }
          canvas.requestRenderAll()
          return
        }
        const p0 = facadeSplitImageP1Ref.current
        if (!p0) {
          facadeSplitP1SceneRef.current = null
          return
        }
        if (facadeSplitPreviewLineRef.current) {
          canvas.remove(facadeSplitPreviewLineRef.current)
          facadeSplitPreviewLineRef.current = null
        }
        facadeSplitP1SceneRef.current = null
        facadeSplitImageP1Ref.current = null
        const splitWallId = facadeSplitTargetWallIdRef.current
        facadeSplitTargetWallIdRef.current = null
        useUIStore.getState().setExteriorSplitLineActive(false)
        canvas.defaultCursor = 'default'

        const wallsSnapshot = useWallStore.getState().walls
        const splitTarget =
          splitWallId != null && wallsSnapshot.length > 0
            ? { targetWallId: splitWallId, walls: wallsSnapshot }
            : undefined

        // Сохраняем состояние ДО разрезания
        useHistoryStore.getState().push()

        void splitExteriorWalls(
          maskB64,
          p0.ix,
          p0.iy,
          ixCl,
          iyCl,
          wallImageSize.width,
          wallImageSize.height,
          splitTarget
        )
          .then((res) => {
            useWallStore.getState().setWalls(res.walls, wallImageSize, false)
          })
          .catch((err) => {
            console.error('[facade-split]', err)
          })
        canvas.requestRenderAll()
        return
      }

      const target = (opt as unknown as { target?: FabricObject | null }).target
      const data = target ? (target as unknown as { data?: Record<string, unknown> }).data : undefined
      if (data && typeof data[WALL_BUTTON_DATA_KEY] === 'number') {
        useWallStore.getState().selectWall(data[WALL_BUTTON_DATA_KEY] as number)
        return
      }

      const tool = maskToolRef.current
      const point = new Point(scenePoint.x, scenePoint.y)

      if (!target && wallOverlaysRef.current.length > 0) {
        for (let i = wallOverlaysRef.current.length - 1; i >= 0; i--) {
          const obj = wallOverlaysRef.current[i]
          if (typeof (obj as unknown as { containsPoint?: (p: Point) => boolean }).containsPoint === 'function' && (obj as unknown as { containsPoint: (p: Point) => boolean }).containsPoint(point)) {
            const wallId = wallIdByObjectRef.current.get(obj)
            if (wallId != null) {
              useWallStore.getState().selectWall(wallId)
              return
            }
          }
        }
      }

      if (tool === 'rect') {
        removeRectPreview()
        rectStartRef.current = { x: scenePoint.x, y: scenePoint.y }
        return
      }
      if (tool === 'lasso') {
        lassoPointsRef.current.push({ x: scenePoint.x, y: scenePoint.y })
        return
      }
      if (tool === 'brush' || canvas.isDrawingMode) return
      if (tool !== null) return

      if (target && target !== backgroundImageRef.current) {
        return
      }
    })

    canvas.on('mouse:move', (opt) => {
      const tool = maskToolRef.current
      const scenePoint = canvas.getScenePoint(opt.e as MouseEvent)

      const uiMove = useUIStore.getState()
      if (
        uiMove.exteriorSplitLineActive &&
        uiMove.sceneMode === 'exterior' &&
        facadeSplitP1SceneRef.current
      ) {
        const p1 = facadeSplitP1SceneRef.current
        if (facadeSplitPreviewLineRef.current) {
          canvas.remove(facadeSplitPreviewLineRef.current)
        }
        const d = `M ${p1.x} ${p1.y} L ${scenePoint.x} ${scenePoint.y}`
        const preview = new Path(d, {
          stroke: '#f97316',
          strokeWidth: 2,
          selectable: false,
          evented: false,
          strokeDashArray: [8, 4],
        })
        canvas.add(preview)
        facadeSplitPreviewLineRef.current = preview
        canvas.requestRenderAll()
        return
      }

      if (tool === 'rect' && rectStartRef.current) {
        const center = rectStartRef.current
        const halfWidth = Math.abs(scenePoint.x - center.x)
        const halfHeight = Math.abs(scenePoint.y - center.y)
        const width = Math.max(halfWidth * 2, 1)
        const height = Math.max(halfHeight * 2, 1)
        const left = center.x - width / 2
        const top = center.y - height / 2
        if (rectPreviewRef.current) canvas.remove(rectPreviewRef.current)
        const preview = new Rect({
          left,
          top,
          width,
          height,
          fill: 'rgba(0,0,0,0.4)',
          stroke: '#333',
          strokeWidth: 1,
          originX: 'left',
          originY: 'top',
          selectable: false,
          evented: false,
        })
        canvas.add(preview)
        rectPreviewRef.current = preview
        canvas.requestRenderAll()
        return
      }

      if (tool === 'lasso' && lassoPointsRef.current.length > 0) {
        const points = lassoPointsRef.current
        if (lassoPreviewRef.current) canvas.remove(lassoPreviewRef.current)
        const segments = points.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`))
        segments.push(`L ${scenePoint.x} ${scenePoint.y}`)
        const d = segments.join(' ')
        const preview = new Path(d, {
          stroke: '#333',
          strokeWidth: 2,
          strokeDashArray: [6, 4],
          fill: 'rgba(0,0,0,0.15)',
          selectable: false,
          evented: false,
        })
        canvas.add(preview)
        lassoPreviewRef.current = preview
        canvas.requestRenderAll()
        return
      }
    })

    canvas.on('mouse:up', (opt) => {
      const tool = maskToolRef.current
      if (tool === 'rect' && rectStartRef.current) {
        const scenePoint = canvas.getScenePoint(opt.e as MouseEvent)
        const center = rectStartRef.current
        const halfWidth = Math.abs(scenePoint.x - center.x)
        const halfHeight = Math.abs(scenePoint.y - center.y)
        const width = halfWidth * 2
        const height = halfHeight * 2
        if (width >= 2 && height >= 2) {
          const left = center.x - width / 2
          const top = center.y - height / 2
          const rect = new Rect({
            left,
            top,
            width,
            height,
            fill: 'black',
            originX: 'left',
            originY: 'top',
            selectable: false,
            evented: false,
          })
          ;(rect as unknown as { set: (o: Record<string, unknown>) => void }).set({
            data: { [MASK_DATA_KEY]: true },
          })
          canvas.add(rect)
        }
        removeRectPreview()
        canvas.requestRenderAll()
        return
      }
    })

    canvas.on('path:created', (opt) => {
      const path = opt.path
      if (path) {
        ;(path as { set: (o: Record<string, unknown>) => void }).set({ data: { [MASK_DATA_KEY]: true } })
      }
    })

    canvasInstanceRef.current = canvas
    setIsReady(true)
    return canvas
  }, [canvasRef, containerWidth, containerHeight])

  const finishLasso = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    const points = lassoPointsRef.current
    if (points.length < 3) return
    if (lassoPreviewRef.current) {
      canvas.remove(lassoPreviewRef.current)
      lassoPreviewRef.current = null
    }
    const d = points.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ') + ' Z'
    const path = new Path(d, {
      fill: 'black',
      selectable: false,
      evented: false,
    })
    ;(path as unknown as { set: (o: Record<string, unknown>) => void }).set({
      data: { [MASK_DATA_KEY]: true },
    })
    canvas.add(path)
    lassoPointsRef.current = []
    canvas.requestRenderAll()
  }, [])

  useEffect(() => {
    const canvas = initCanvas()
    return () => {
      canvas?.dispose()
      canvasInstanceRef.current = null
      setIsReady(false)
    }
  }, [initCanvas])

  useEffect(() => {
    if (!customMaskMode) {
      customMaskPointsRef.current = []
      const canvas = canvasInstanceRef.current
      if (canvas && customMaskPreviewRef.current.length > 0) {
        for (const obj of customMaskPreviewRef.current) {
          canvas.remove(obj)
        }
        customMaskPreviewRef.current = []
        canvas.requestRenderAll()
      }
    }
  }, [customMaskMode])

  const loadPhotoFromDataUrl = useCallback(async (dataUrl: string) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    setIsPhotoLoaded(false)
    try {
      const img = await FabricImage.fromURL(dataUrl)
      const imageW = img.width ?? 0
      const imageH = img.height ?? 0
      if (imageW <= 0 || imageH <= 0) return
      const scale = Math.min(containerWidth / imageW, containerHeight / imageH)
      const drawW = Math.max(1, Math.round(imageW * scale))
      const drawH = Math.max(1, Math.round(imageH * scale))
      img.set({
        scaleX: drawW / imageW,
        scaleY: drawH / imageH,
        left: (containerWidth - drawW) / 2,
        top: (containerHeight - drawH) / 2,
        originX: 'left',
        originY: 'top',
        selectable: false,
        evented: false,
        lockMovementX: true,
        lockMovementY: true,
        lockScalingX: true,
        lockScalingY: true,
        lockRotation: true,
        hoverCursor: 'default',
      })
      ;(img as unknown as { set: (o: Record<string, unknown>) => void }).set({
        data: { isBackground: true },
      })
      canvas.clear()
      canvas.add(img)
      canvas.renderAll()
      backgroundImageRef.current = img
      setPhotoDataUrl(dataUrl)
      setIsPhotoLoaded(true)
    } catch {
      // ignore load error
    }
  }, [setPhotoDataUrl, containerWidth, containerHeight])

  const loadPhotoFromFile = useCallback(
    (file: File) => {
      if (file.size > MAX_PHOTO_SIZE_BYTES) return
      if (!ALLOWED_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_TYPES)[number])) return
      const reader = new FileReader()
      reader.onload = () => {
        const dataUrl = reader.result as string
        loadPhotoFromDataUrl(dataUrl)
      }
      reader.readAsDataURL(file)
    },
    [loadPhotoFromDataUrl]
  )

  const exportToPng = useCallback((): string | null => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return null
    return canvas.toDataURL({ format: 'png', multiplier: 1 })
  }, [])

  const clearCanvas = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    canvas.clear()
    canvas.setZoom(1)
    canvas.viewportTransform = [1, 0, 0, 1, 0, 0]
    canvas.renderAll()
    backgroundImageRef.current = null
    textureLayersRef.current = {}
    exteriorMaskRef.current = null
    facadeSplitP1SceneRef.current = null
    facadeSplitImageP1Ref.current = null
    facadeSplitTargetWallIdRef.current = null
    if (facadeSplitPreviewLineRef.current) {
      canvas.remove(facadeSplitPreviewLineRef.current)
      facadeSplitPreviewLineRef.current = null
    }
    wallOverlaysRef.current = []
    wallDebugShapesRef.current = []
    setHasTextureLayer(false)
    setPhotoDataUrl(null)
    setIsPhotoLoaded(false)
    useWallStore.getState().setWalls([], null)
  }, [setPhotoDataUrl])

  const setDrawingMode = useCallback((enabled: boolean) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    canvas.isDrawingMode = enabled
    if (enabled && canvas.freeDrawingBrush) {
      canvas.freeDrawingBrush.color = 'rgba(0,0,0,0.8)'
      canvas.freeDrawingBrush.width = 20
    }
    setDrawingModeState(enabled)
    canvas.requestRenderAll()
  }, [])

  const setBrushSize = useCallback((size: number) => {
    const canvas = canvasInstanceRef.current
    if (!canvas?.freeDrawingBrush) return
    canvas.freeDrawingBrush.width = Math.max(1, Math.min(100, size))
  }, [])

  const hasMaskData = useCallback((obj: FabricObject): boolean => {
    const data = (obj as unknown as { data?: Record<string, unknown> }).data
    return Boolean(data && typeof data === 'object' && data[MASK_DATA_KEY])
  }, [])

  const clearMask = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    const objects = canvas.getObjects()
    const toRemove = objects.filter(hasMaskData)
    toRemove.forEach((obj) => canvas.remove(obj))
    canvas.requestRenderAll()
  }, [hasMaskData])

  const getMaskObjects = useCallback((): FabricObject[] => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return []
    return canvas.getObjects().filter(hasMaskData)
  }, [hasMaskData])

  const getBackgroundBounds = useCallback((): { left: number; top: number; width: number; height: number } | null => {
    const bg = backgroundImageRef.current
    if (!bg) return null
    const left = (bg.left ?? 0) as number
    const top = (bg.top ?? 0) as number
    const baseW = (bg.width ?? 0) as number
    const baseH = (bg.height ?? 0) as number
    const sx = (bg.scaleX ?? 1) as number
    const sy = (bg.scaleY ?? 1) as number
    const width = baseW * sx
    const height = baseH * sy
    if (width <= 0 || height <= 0) return null
    return { left, top, width, height }
  }, [])

  const getBackgroundTransform = useCallback((): {
    left: number
    top: number
    scaleX: number
    scaleY: number
    originX: 'left' | 'center' | 'right'
    originY: 'top' | 'center' | 'bottom'
    angle: number
    skewX: number
    skewY: number
    flipX: boolean
    flipY: boolean
  } | null => {
    const bg = backgroundImageRef.current
    if (!bg) return null
    return {
      left: (bg.left ?? 0) as number,
      top: (bg.top ?? 0) as number,
      scaleX: (bg.scaleX ?? 1) as number,
      scaleY: (bg.scaleY ?? 1) as number,
      originX: ((bg.originX ?? 'left') as 'left' | 'center' | 'right'),
      originY: ((bg.originY ?? 'top') as 'top' | 'center' | 'bottom'),
      angle: (bg.angle ?? 0) as number,
      skewX: (bg.skewX ?? 0) as number,
      skewY: (bg.skewY ?? 0) as number,
      flipX: Boolean(bg.flipX),
      flipY: Boolean(bg.flipY),
    }
  }, [])

  const setExteriorMaskOverlay = useCallback(
    async (maskBase64Png: string | null, imageSize: { width: number; height: number } | null) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return

      if (exteriorMaskRef.current) {
        canvas.remove(exteriorMaskRef.current)
        exteriorMaskRef.current = null
      }

      if (!maskBase64Png || !imageSize) {
        canvas.requestRenderAll()
        return
      }

      const bgTx = getBackgroundTransform()
      if (!bgTx) return

      const dataUrl = `data:image/png;base64,${maskBase64Png}`
      const img = await FabricImage.fromURL(dataUrl)
      const w = img.width ?? 0
      const h = img.height ?? 0
      if (w <= 0 || h <= 0) return

      img.set({
        left: bgTx.left,
        top: bgTx.top,
        originX: bgTx.originX,
        originY: bgTx.originY,
        scaleX: bgTx.scaleX,
        scaleY: bgTx.scaleY,
        angle: bgTx.angle,
        skewX: bgTx.skewX,
        skewY: bgTx.skewY,
        flipX: bgTx.flipX,
        flipY: bgTx.flipY,
        selectable: false,
        evented: false,
        opacity: 0,
      })
      ;(img as unknown as { set: (o: Record<string, unknown>) => void }).set({
        data: { [EXTERIOR_MASK_DATA_KEY]: true },
      })

      canvas.add(img)
      canvas.moveObjectTo(img, 1)
      exteriorMaskRef.current = img
      canvas.requestRenderAll()
    },
    [getBackgroundTransform]
  )

  const setWallOverlays = useCallback(
    (walls: WallData[], wallImageSize: { width: number; height: number } | null) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      wallDebugShapesRef.current.forEach((obj) => canvas.remove(obj))
      wallDebugShapesRef.current = []
      wallOverlaysRef.current.forEach((obj) => canvas.remove(obj))
      wallOverlaysRef.current = []
      const keepIds = new Set(walls.map((w) => String(w.id)))
      for (const [layerKey, layerObj] of Object.entries(textureLayersRef.current)) {
        if (layerKey === 'background' || keepIds.has(layerKey)) continue
        canvas.remove(layerObj)
        delete textureLayersRef.current[layerKey]
      }
      if (walls.length === 0 || !wallImageSize) {
        for (const [layerKey, layerObj] of Object.entries(textureLayersRef.current)) {
          if (layerKey === 'background') continue
          canvas.remove(layerObj)
          delete textureLayersRef.current[layerKey]
        }
        setHasTextureLayer(Boolean(textureLayersRef.current.background))
        canvas.requestRenderAll()
        return
      }
      const bounds = getBackgroundBounds()
      if (!bounds) return
      const scaleX = bounds.width / wallImageSize.width
      const scaleY = bounds.height / wallImageSize.height
      const hideWallMasks = useUIStore.getState().hideWallMasks
      const wallVisibility = useUIStore.getState().wallVisibility
      
      for (const wall of walls) {
        const isVisible = wallVisibility[wall.id] !== false
        const overlayPoly =
          wall.polygon && wall.polygon.length >= 3 ? wall.polygon : wall.corners
        if (!hideWallMasks && isVisible && overlayPoly.length >= 3) {
          const d =
            overlayPoly
              .map((c, i) => {
                const x = bounds.left + c[0] * scaleX
                const y = bounds.top + c[1] * scaleY
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
              })
              .join(' ') + ' Z'
          const shape = new Path(d, {
            fill: 'transparent',
            stroke: 'rgba(37,99,235,0.8)',
            strokeWidth: 2,
            selectable: false,
            evented: false,
          })
          canvas.add(shape)
          wallDebugShapesRef.current.push(shape)
        }

        if (!isVisible) continue

        const polyForCenter = wall.polygon && wall.polygon.length >= 3 ? wall.polygon : wall.corners
        const centerIx = polyForCenter.reduce((acc, c) => acc + c[0], 0) / polyForCenter.length
        const centerIy = polyForCenter.reduce((acc, c) => acc + c[1], 0) / polyForCenter.length
        const cx = bounds.left + centerIx * scaleX
        const cy = bounds.top + centerIy * scaleY
        const btn = new Rect({
          width: 36,
          height: 22,
          left: cx - 18,
          top: cy - 11,
          fill: 'white',
          stroke: '#333',
          strokeWidth: 1,
          rx: 11,
          ry: 11,
          originX: 'left',
          originY: 'top',
          selectable: false,
          evented: true,
          hoverCursor: 'pointer',
        })
        ;(btn as unknown as { set: (o: Record<string, unknown>) => void }).set({
          data: { [WALL_BUTTON_DATA_KEY]: wall.id },
        })
        wallIdByObjectRef.current.set(btn, wall.id)
        canvas.add(btn)
        wallOverlaysRef.current.push(btn)
      }
      canvas.requestRenderAll()
    },
    [getBackgroundBounds]
  )


  const clearTextureFromWall = useCallback((wallId: number | null) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    const layerKey = wallId != null ? String(wallId) : 'background'
    if (textureLayersRef.current[layerKey]) {
      canvas.remove(textureLayersRef.current[layerKey])
      delete textureLayersRef.current[layerKey]
      canvas.requestRenderAll()
    }
  }, [])

  const applyTexture = useCallback(
    async (
      textureUrl: string,
      repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat',
      options?: { clipPathOverride?: FabricObject }
    ) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) {
        return
      }
      
      const layerKey = 'background'
      if (textureLayersRef.current[layerKey]) {
        canvas.remove(textureLayersRef.current[layerKey])
      }
      
      const { applyPatternToCanvas } = await import('@/lib/texture-processor')
      let clipPath: Group | FabricObject | undefined = options?.clipPathOverride
      if (!clipPath) {
        const maskObjs = getMaskObjects()
        if (maskObjs.length > 0) {
          const clones = await Promise.all(maskObjs.map((o) => o.clone()))
          clipPath = new Group(clones)
        }
      }
      let sceneBounds: { left: number; top: number; width: number; height: number } | undefined
      const bgBounds = getBackgroundBounds()
      if (bgBounds) sceneBounds = bgBounds
      const layer = await applyPatternToCanvas(
        canvas,
        textureUrl,
        repeat,
        clipPath,
        sceneBounds,
        textureScale
      )
      textureLayersRef.current[layerKey] = layer
      setHasTextureLayer(true)
    },
    [getMaskObjects, textureScale, getBackgroundBounds]
  )

  const applyTextureToWall = useCallback(
    async (
      textureUrl: string,
      wallCorners: [number, number][],
      _wallImageSize: { width: number; height: number } | null,
      selectedWallId: number | null
    ) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) {
        return
      }
      
      const layerKey = selectedWallId != null ? String(selectedWallId) : 'background'
      if (textureLayersRef.current[layerKey]) {
        canvas.remove(textureLayersRef.current[layerKey])
      }

      const sceneMode = useUIStore.getState().sceneMode
      const wall = selectedWallId != null ? useWallStore.getState().walls.find((w) => w.id === selectedWallId) : null
      const wallPolygon = wall?.polygon && wall.polygon.length >= 3 ? wall.polygon : undefined

      if (sceneMode === 'exterior' && wallPolygon && _wallImageSize) {
        const bgTx = getBackgroundTransform()
        if (!bgTx) {
          return
        }

        const { warpWallTexture } = await import('@/hooks/warpWallTexture')
        const regions =
          wall?.regions && wall.regions.length === 2 ? wall.regions : undefined
        const maskBase64 = useWallStore.getState().exteriorMaskBase64

        const blob = await warpWallTexture({
          textureUrl,
          corners: wallCorners,
          polygon: wallPolygon,
          regions,
          imageSize: _wallImageSize,
          textureScale,
          opacity: 1,
          maskBase64,
        })
        const img = new Image()
        const objectUrl = URL.createObjectURL(blob)
        img.src = objectUrl
        await new Promise((resolve, reject) => {
          img.onload = () => resolve(true)
          img.onerror = () => reject(new Error('Failed to load warped texture blob'))
        })
        URL.revokeObjectURL(objectUrl)

        const fabricImg = new FabricImage(img, {
          left: bgTx.left,
          top: bgTx.top,
          selectable: false,
          evented: false,
          opacity: 1,
          originX: bgTx.originX,
          originY: bgTx.originY,
          scaleX: bgTx.scaleX,
          scaleY: bgTx.scaleY,
          angle: bgTx.angle,
          skewX: bgTx.skewX,
          skewY: bgTx.skewY,
          flipX: bgTx.flipX,
          flipY: bgTx.flipY,
        })
        canvas.add(fabricImg)
        textureLayersRef.current[layerKey] = fabricImg
        setHasTextureLayer(true)
        canvas.requestRenderAll()
        return
      }

      const { renderPerspectiveWallTexture } = await import('@/lib/texture-processor')

      const width = canvas.getWidth() ?? containerWidth
      const height = canvas.getHeight() ?? containerHeight
      const maskImage =
        sceneMode === 'interior'
          ? await loadInteriorMaskImage().catch(() => null)
          : null
      const { canvas: textureCanvas, offsetX, offsetY, localCorners, localPolygon } = await renderPerspectiveWallTexture(
        textureUrl,
        wallCorners,
        width,
        height,
        textureScale,
        wallPolygon,
        maskImage
      )
      const img = new Image()
      img.src = textureCanvas.toDataURL()
      await new Promise((resolve) => { img.onload = resolve })
      const fabricImg = new FabricImage(img, {
        left: offsetX,
        top: offsetY,
        selectable: false,
        evented: false,
        opacity: 0.85,
      })

      const clipPoints = localPolygon && localPolygon.length >= 3 ? localPolygon : localCorners
      const clipPath = new Path(
        clipPoints
          .map((p, i) => (i === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`))
          .join(' ') + ' Z',
        { selectable: false, evented: false }
      )
      fabricImg.set({ clipPath })
      canvas.add(fabricImg)
      textureLayersRef.current[layerKey] = fabricImg
      setHasTextureLayer(true)
      canvas.requestRenderAll()
    },
    [containerWidth, containerHeight, textureScale, getBackgroundTransform, loadInteriorMaskImage]
  )

  const highlightSelectedWall = useCallback((selectedId: number | null) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    for (const obj of wallOverlaysRef.current) {
      const wallId = wallIdByObjectRef.current.get(obj)
      if (wallId != null && wallId === selectedId) {
        obj.set({ fill: '#3b82f6', stroke: '#1d4ed8' })
      } else {
        obj.set({ fill: 'white', stroke: '#333' })
      }
    }
    canvas.requestRenderAll()
  }, [])

  const syncCornerHandles = useCallback((selectedWallId: number | null) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return

    for (const h of cornerHandlesRef.current) canvas.remove(h)
    cornerHandlesRef.current = []

    const { editWallCorners, editCornersMode } = useUIStore.getState()
    if (!editWallCorners || selectedWallId == null) {
      canvas.requestRenderAll()
      return
    }

    const { walls, wallImageSize, updateWallCorners, updateWallPolygonVertex } = useWallStore.getState()
    const wall = walls.find((w) => w.id === selectedWallId)
    if (!wall || !wallImageSize || wall.corners.length < 3) {
      canvas.requestRenderAll()
      return
    }

    const bounds = getBackgroundBounds()
    if (!bounds) return
    const scaleX = bounds.width / wallImageSize.width
    const scaleY = bounds.height / wallImageSize.height

    const handleRadius = 8

    const hasPolygon = wall.polygon && wall.polygon.length >= 3
    const isPerspectiveMode = editCornersMode === 'perspective' || !hasPolygon

    if (isPerspectiveMode && wall.corners && wall.corners.length === 4) {
      const pts = wall.corners.map(([x, y]) => ({ x: bounds.left + x * scaleX, y: bounds.top + y * scaleY }))
      const d = `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y} L ${pts[2].x} ${pts[2].y} L ${pts[3].x} ${pts[3].y} Z`
      const quadOutline = new Path(d, {
        stroke: 'rgba(59,130,246,0.8)',
        strokeWidth: 2,
        strokeDashArray: [5, 5],
        fill: 'transparent',
        selectable: false,
        evented: false,
      })
      canvas.add(quadOutline)
      cornerHandlesRef.current.push(quadOutline)
      const updatePerspectiveOutline = (corners: [number, number][]) => {
        if (corners.length !== 4) return
        const nextPts = corners.map(([cx, cy]) => ({ x: bounds.left + cx * scaleX, y: bounds.top + cy * scaleY }))
        const nextD = `M ${nextPts[0].x} ${nextPts[0].y} L ${nextPts[1].x} ${nextPts[1].y} L ${nextPts[2].x} ${nextPts[2].y} L ${nextPts[3].x} ${nextPts[3].y} Z`
        const nextPath = new Path(nextD, { selectable: false, evented: false }).path
        quadOutline.set({ path: nextPath })
      }

      wall.corners.forEach(([ix, iy], idx) => {
        const x = bounds.left + ix * scaleX
        const y = bounds.top + iy * scaleY

        const handle = new Rect({
          left: x - handleRadius,
          top: y - handleRadius,
          width: handleRadius * 2,
          height: handleRadius * 2,
          fill: 'rgba(255,255,255,0.95)',
          stroke: 'rgba(59,130,246,1)',
          strokeWidth: 2,
          originX: 'left',
          originY: 'top',
          selectable: true,
          evented: true,
          hoverCursor: 'move',
          hasControls: false,
          hasBorders: false,
          lockScalingX: true,
          lockScalingY: true,
          lockRotation: true,
        })
        ;(handle as unknown as { set: (o: Record<string, unknown>) => void }).set({
          data: { [CORNER_HANDLE_DATA_KEY]: true, wallId: selectedWallId, cornerIndex: idx, isPerspective: true },
        })

        handle.on('moving', () => {
          // Читаем актуальные corners из store, а не из замыкания
          const currentWall = useWallStore.getState().walls.find((w) => w.id === selectedWallId)
          if (!currentWall) return
          const hx = ((handle.left ?? (x - handleRadius)) as number) + handleRadius
          const hy = ((handle.top ?? (y - handleRadius)) as number) + handleRadius
          const newIx = Math.round((hx - bounds.left) / scaleX)
          const newIy = Math.round((hy - bounds.top) / scaleY)
          const next = currentWall.corners.map((c, i) => (i === idx ? ([newIx, newIy] as [number, number]) : c))
          updatePerspectiveOutline(next)
          updateWallCorners(selectedWallId, next)
          canvas.requestRenderAll()
        })

        handle.on('modified', () => {
          // Автоматически привязываем перспективу к форме
          const wall = useWallStore.getState().walls.find((w) => w.id === selectedWallId)
          if (wall?.polygon) {
            const newCorners = autoLinkPerspectiveToForm(wall)
            updateWallCorners(selectedWallId, newCorners)
          }
          useHistoryStore.getState().push()
        })

        canvas.add(handle)
        cornerHandlesRef.current.push(handle)
      })
    }

    if (!isPerspectiveMode && hasPolygon && wall.polygon) {
      wall.polygon.forEach(([ix, iy], idx) => {
        const x = bounds.left + ix * scaleX
        const y = bounds.top + iy * scaleY

        const handle = new Circle({
          left: x,
          top: y,
          radius: handleRadius,
          fill: 'rgba(255,255,255,0.95)',
          stroke: 'rgba(16,185,129,1)',
          strokeWidth: 2,
          originX: 'center',
          originY: 'center',
          selectable: true,
          evented: true,
          hoverCursor: 'move',
          hasControls: false,
          hasBorders: false,
          lockScalingX: true,
          lockScalingY: true,
          lockRotation: true,
        })
        ;(handle as unknown as { set: (o: Record<string, unknown>) => void }).set({
          data: { [CORNER_HANDLE_DATA_KEY]: true, wallId: selectedWallId, cornerIndex: idx, isPerspective: false },
        })

        handle.on('moving', () => {
          // Читаем актуальный polygon из store, а не из замыкания
          const currentWall = useWallStore.getState().walls.find((w) => w.id === selectedWallId)
          if (!currentWall?.polygon) return
          const hx = (handle.left ?? x) as number
          const hy = (handle.top ?? y) as number
          const newIx = Math.round((hx - bounds.left) / scaleX)
          const newIy = Math.round((hy - bounds.top) / scaleY)
          updateWallPolygonVertex(selectedWallId, idx, [newIx, newIy])
        })

        handle.on('modified', () => {
          // Автоматически привязываем перспективу к форме
          const wall = useWallStore.getState().walls.find((w) => w.id === selectedWallId)
          if (wall?.polygon) {
            const newCorners = autoLinkPerspectiveToForm(wall)
            updateWallCorners(selectedWallId, newCorners)
          }
          useHistoryStore.getState().push()
        })

        canvas.add(handle)
        cornerHandlesRef.current.push(handle)
      })
    }

    canvas.requestRenderAll()
  }, [getBackgroundBounds])

  const setTextureScale = useCallback((scale: number, selectedWallId: number | null) => {
    const s = Math.max(0.05, Math.min(1, scale))
    setTextureScaleState(s)
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    const layerKey = selectedWallId != null ? String(selectedWallId) : 'background'
    const layer = textureLayersRef.current[layerKey]
    if (!layer) return
    const fill = (layer as unknown as { fill?: { patternTransform?: number[] } }).fill
    if (fill && Array.isArray(fill.patternTransform)) {
      fill.patternTransform = [s, 0, 0, s, 0, 0]
      canvas.requestRenderAll()
    }
  }, [])

  const clearMaskToolState = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (rectPreviewRef.current && canvas) {
      canvas.remove(rectPreviewRef.current)
      rectPreviewRef.current = null
    }
    rectStartRef.current = null
    if (lassoPreviewRef.current && canvas) {
      canvas.remove(lassoPreviewRef.current)
      lassoPreviewRef.current = null
    }
    lassoPointsRef.current = []
    canvas?.requestRenderAll()
  }, [])

  // Переключение "до/после" — скрывает/показывает все слои текстур
  useEffect(() => {
    for (const layer of Object.values(textureLayersRef.current)) {
      ;(layer as unknown as { set: (o: Record<string, unknown>) => void }).set({ visible: !beforeAfter })
    }
    canvasInstanceRef.current?.requestRenderAll()
  }, [beforeAfter])

  const clearFacadeSplitDraft = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (facadeSplitPreviewLineRef.current && canvas) {
      canvas.remove(facadeSplitPreviewLineRef.current)
      facadeSplitPreviewLineRef.current = null
    }
    facadeSplitP1SceneRef.current = null
    facadeSplitImageP1Ref.current = null
    facadeSplitTargetWallIdRef.current = null
    canvas?.requestRenderAll()
  }, [])

  return {
    canvasInstanceRef,
    isReady,
    drawingMode,
    loadPhotoFromDataUrl,
    loadPhotoFromFile,
    exportToPng,
    clearCanvas,
    setDrawingMode,
    setBrushSize,
    clearMask,
    getMaskObjects,
    getBackgroundBounds,
    setWallOverlays,
    setExteriorMaskOverlay,
    clearTextureFromWall,
    applyTexture,
    applyTextureToWall,
    highlightSelectedWall,
    syncCornerHandles,
    finishLasso,
    clearMaskToolState,
    clearFacadeSplitDraft,
    textureScale,
    setTextureScale,
    hasTextureLayer,
    isPhotoLoaded,
    beforeAfter,
    setBeforeAfter,
    undoCustomMask,
    redoCustomMask,
    get canUndoCustomMask() {
      return customMaskHistoryRef.current.length > 0
    },
    get canRedoCustomMask() {
      return customMaskFutureRef.current.length > 0
    },
  }
}
