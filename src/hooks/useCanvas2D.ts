import { useEffect, useRef, useCallback, useState } from 'react'
import { Canvas, FabricImage, Point, PencilBrush, Group, Rect, Path, Circle } from 'fabric'
import type { FabricObject } from 'fabric'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'
import type { WallData } from '@/store/useWallStore'
import { MAX_PHOTO_SIZE_BYTES, ALLOWED_IMAGE_TYPES } from '@/lib/constants'

const MIN_ZOOM = 0.1
const MAX_ZOOM = 5
const ZOOM_STEP = 0.1

export interface UseCanvas2DOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  containerWidth?: number
  containerHeight?: number
  /** Вызывается при двух кликах в режиме "Разделить фасад". Координаты уже в image_size. */
  onSplitLineComplete?: (p1: { x: number; y: number }, p2: { x: number; y: number }) => void
}

export function useCanvas2D({
  canvasRef,
  containerWidth = 800,
  containerHeight = 600,
  onSplitLineComplete,
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
  const textureLayerRef = useRef<FabricObject | null>(null)
  const wallOverlaysRef = useRef<FabricObject[]>([])
  const wallDebugShapesRef = useRef<FabricObject[]>([])
  const wallIdByObjectRef = useRef<WeakMap<FabricObject, number>>(new WeakMap())
  const WALL_BUTTON_DATA_KEY = 'wallId'
  const cornerHandlesRef = useRef<FabricObject[]>([])
  const CORNER_HANDLE_DATA_KEY = 'cornerHandle'
  const exteriorMaskRef = useRef<FabricObject | null>(null)
  const EXTERIOR_MASK_DATA_KEY = 'isExteriorMask'

  const splitFacadeMode = useUIStore((s) => s.splitFacadeMode)
  const splitFacadeModeRef = useRef(splitFacadeMode)
  splitFacadeModeRef.current = splitFacadeMode
  const splitPendingPointRef = useRef<{ x: number; y: number } | null>(null)
  const splitLinePreviewRef = useRef<Path | null>(null)
  const onSplitLineCompleteRef = useRef(onSplitLineComplete)
  onSplitLineCompleteRef.current = onSplitLineComplete

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

    canvas.on('mouse:wheel', (opt) => {
      const ev = opt.e as WheelEvent
      ev.preventDefault()
      const delta = ev.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
      const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, canvas.getZoom() + delta))
      const point = new Point(ev.offsetX, ev.offsetY)
      canvas.zoomToPoint(point, zoom)
    })

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

      if (splitFacadeModeRef.current && useUIStore.getState().sceneMode === 'exterior') {
        const cb = onSplitLineCompleteRef.current
        if (!cb) return
        const pending = splitPendingPointRef.current
        if (!pending) {
          splitPendingPointRef.current = { x: scenePoint.x, y: scenePoint.y }
          return
        }
        const bg = backgroundImageRef.current
        const bounds =
          bg && typeof (bg as unknown as { getBoundingRect?: () => { left: number; top: number; width: number; height: number } }).getBoundingRect === 'function'
            ? (bg as unknown as { getBoundingRect: () => { left: number; top: number; width: number; height: number } }).getBoundingRect()
            : null
        const { wallImageSize } = useWallStore.getState()
        if (bounds && wallImageSize) {
          const scaleX = bounds.width / wallImageSize.width
          const scaleY = bounds.height / wallImageSize.height
          const imgP1 = {
            x: (pending.x - bounds.left) / scaleX,
            y: (pending.y - bounds.top) / scaleY,
          }
          const imgP2 = {
            x: (scenePoint.x - bounds.left) / scaleX,
            y: (scenePoint.y - bounds.top) / scaleY,
          }
          cb(imgP1, imgP2)
        }
        splitPendingPointRef.current = null
        if (splitLinePreviewRef.current) {
          canvas.remove(splitLinePreviewRef.current)
          splitLinePreviewRef.current = null
        }
        useUIStore.getState().setSplitFacadeMode(false)
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

      if (splitFacadeModeRef.current && splitPendingPointRef.current) {
        const p1 = splitPendingPointRef.current
        if (splitLinePreviewRef.current) canvas.remove(splitLinePreviewRef.current)
        const d = `M ${p1.x} ${p1.y} L ${scenePoint.x} ${scenePoint.y}`
        const preview = new Path(d, {
          stroke: '#e11d48',
          strokeWidth: 2,
          strokeDashArray: [8, 4],
          selectable: false,
          evented: false,
        })
        canvas.add(preview)
        splitLinePreviewRef.current = preview
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
    if (!splitFacadeMode) {
      splitPendingPointRef.current = null
      const canvas = canvasInstanceRef.current
      if (splitLinePreviewRef.current && canvas) {
        canvas.remove(splitLinePreviewRef.current)
        splitLinePreviewRef.current = null
        canvas.requestRenderAll()
      }
    }
  }, [splitFacadeMode])

  const loadPhotoFromDataUrl = useCallback(async (dataUrl: string) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
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
    textureLayerRef.current = null
    exteriorMaskRef.current = null
    wallOverlaysRef.current = []
    wallDebugShapesRef.current = []
    setHasTextureLayer(false)
    setPhotoDataUrl(null)
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
        opacity: 0.35,
      })
      ;(img as unknown as { set: (o: Record<string, unknown>) => void }).set({
        data: { [EXTERIOR_MASK_DATA_KEY]: true },
      })

      // добавляем поверх background, но под оверлеями/текстурой
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
      // убрать старые фигуры стен
      wallDebugShapesRef.current.forEach((obj) => canvas.remove(obj))
      wallDebugShapesRef.current = []
      wallOverlaysRef.current.forEach((obj) => canvas.remove(obj))
      wallOverlaysRef.current = []
      if (walls.length === 0 || !wallImageSize) {
        canvas.requestRenderAll()
        return
      }
      const bounds = getBackgroundBounds()
      if (!bounds) return
      const scaleX = bounds.width / wallImageSize.width
      const scaleY = bounds.height / wallImageSize.height
      const hideWallMasks = useUIStore.getState().hideWallMasks
      for (const wall of walls) {
        // многоугольник стены поверх фото, чтобы было видно реальную форму
        const overlayPoly =
          wall.polygon && wall.polygon.length >= 3 ? wall.polygon : wall.corners
        // Показываем синие маски и в интерьере, и в экстерьере (если не скрыты).
        if (!hideWallMasks && overlayPoly.length >= 3) {
          const d =
            overlayPoly
              .map((c, i) => {
                const x = bounds.left + c[0] * scaleX
                const y = bounds.top + c[1] * scaleY
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
              })
              .join(' ') + ' Z'
          const shape = new Path(d, {
            fill: 'rgba(59,130,246,0.12)', // полупрозрачный синий
            stroke: 'rgba(37,99,235,0.8)',
            strokeWidth: 2,
            selectable: false,
            evented: false,
          })
          canvas.add(shape)
          wallDebugShapesRef.current.push(shape)
        }

        const cx = bounds.left + wall.center[0] * scaleX
        const cy = bounds.top + wall.center[1] * scaleY
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

  const applyTexture = useCallback(
    async (
      textureUrl: string,
      repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat',
      options?: { clipPathOverride?: FabricObject }
    ) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      if (textureLayerRef.current) {
        canvas.remove(textureLayerRef.current)
        textureLayerRef.current = null
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
      textureLayerRef.current = layer
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
      const canvas = canvasInstanceRef.current;
      if (!canvas) return;
      if (textureLayerRef.current) {
        canvas.remove(textureLayerRef.current);
        textureLayerRef.current = null;
      }
      const sceneMode = useUIStore.getState().sceneMode
      const wall = selectedWallId != null ? useWallStore.getState().walls.find((w) => w.id === selectedWallId) : null
      const wallPolygon = wall?.polygon && wall.polygon.length >= 3 ? wall.polygon : undefined

      // В экстерьере делаем варп+клип на бэкенде по polygon.
      if (sceneMode === 'exterior' && wallPolygon && _wallImageSize) {
        const bgTx = getBackgroundTransform()
        if (!bgTx) return

        const { warpWallTexture } = await import('@/hooks/warpWallTexture')
        const blob = await warpWallTexture({
          textureUrl,
          corners: wallCorners,
          polygon: wallPolygon,
          imageSize: _wallImageSize,
          textureScale,
          opacity: 0.85,
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
        textureLayerRef.current = fabricImg
        setHasTextureLayer(true)
        canvas.requestRenderAll()
        return
      }

      const { renderPerspectiveWallTexture } = await import('@/lib/texture-processor');

      const width = canvas.getWidth() ?? containerWidth;
      const height = canvas.getHeight() ?? containerHeight;
      const { canvas: textureCanvas, offsetX, offsetY, localCorners, localPolygon } = await renderPerspectiveWallTexture(
        textureUrl,
        wallCorners,
        width,
        height,
        textureScale,
        wallPolygon
      );
      const img = new Image();
      img.src = textureCanvas.toDataURL();
      await new Promise((resolve) => { img.onload = resolve; });
      const fabricImg = new FabricImage(img, {
        left: offsetX,
        top: offsetY,
        selectable: false,
        evented: false,
        opacity: 0.85,
      });

      // ClipPath по локальным координатам (origin = offsetX/offsetY offscreen canvas).
      const clipPoints = localPolygon && localPolygon.length >= 3 ? localPolygon : localCorners
      const clipPath = new Path(
        clipPoints
          .map((p, i) => (i === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`))
          .join(' ') + ' Z',
        { selectable: false, evented: false }
      );
      fabricImg.set({ clipPath });
      canvas.add(fabricImg);
      textureLayerRef.current = fabricImg;
      setHasTextureLayer(true);
      canvas.requestRenderAll();
    },
    [containerWidth, containerHeight, textureScale, getBackgroundTransform]
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

    // remove existing handles
    for (const h of cornerHandlesRef.current) canvas.remove(h)
    cornerHandlesRef.current = []

    const { editWallCorners } = useUIStore.getState()
    if (!editWallCorners || selectedWallId == null) {
      canvas.requestRenderAll()
      return
    }

    const { walls, wallImageSize, updateWallCorners } = useWallStore.getState()
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
    wall.corners.forEach(([ix, iy], idx) => {
      const x = bounds.left + ix * scaleX
      const y = bounds.top + iy * scaleY

      const handle = new Circle({
        left: x,
        top: y,
        radius: handleRadius,
        fill: 'rgba(255,255,255,0.95)',
        stroke: 'rgba(59,130,246,1)',
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
        data: { [CORNER_HANDLE_DATA_KEY]: true, wallId: selectedWallId, cornerIndex: idx },
      })

      handle.on('moving', () => {
        const hx = (handle.left ?? x) as number
        const hy = (handle.top ?? y) as number
        const newIx = Math.round((hx - bounds.left) / scaleX)
        const newIy = Math.round((hy - bounds.top) / scaleY)
        const next = wall.corners.map((c, i) => (i === idx ? ([newIx, newIy] as [number, number]) : c))
        updateWallCorners(selectedWallId, next)
      })

      canvas.add(handle)
      cornerHandlesRef.current.push(handle)
    })

    canvas.requestRenderAll()
  }, [getBackgroundBounds])

  const setTextureScale = useCallback((scale: number) => {
    const s = Math.max(0.05, Math.min(1, scale))
    setTextureScaleState(s)
    const canvas = canvasInstanceRef.current
    const layer = textureLayerRef.current
    if (!canvas || !layer) return
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
    applyTexture,
    applyTextureToWall,
    highlightSelectedWall,
    syncCornerHandles,
    finishLasso,
    clearMaskToolState,
    textureScale,
    setTextureScale,
    hasTextureLayer,
  }
}
