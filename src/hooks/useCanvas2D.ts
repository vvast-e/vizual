import { useEffect, useRef, useCallback, useState } from 'react'
import { Canvas, FabricImage, Point, PencilBrush, Group, Rect, Path, Circle, Text } from 'fabric'
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
}

export function useCanvas2D({
  canvasRef,
  containerWidth = 800,
  containerHeight = 600,
}: UseCanvas2DOptions) {
  const DEBUG_WALL_OVERLAY = true
  const canvasInstanceRef = useRef<Canvas | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [drawingMode, setDrawingModeState] = useState(false)
  const MASK_DATA_KEY = 'isMask'
  const backgroundImageRef = useRef<FabricImage | null>(null)

  const maskTool = useUIStore((s) => s.maskTool)
  const maskToolRef = useRef(maskTool)
  maskToolRef.current = maskTool
  const setPhotoDataUrl = useVisualizerStore((s) => s.setPhotoDataUrl)
  const hideWallMasks = useUIStore((s) => s.hideWallMasks)

  const rectStartRef = useRef<{ x: number; y: number } | null>(null)
  const rectPreviewRef = useRef<Rect | null>(null)
  const lassoPointsRef = useRef<{ x: number; y: number }[]>([])
  const lassoPreviewRef = useRef<Path | null>(null)

  const [textureScale, setTextureScaleState] = useState(0.25)
  const [hasTextureLayer, setHasTextureLayer] = useState(false)
  const textureLayerRef = useRef<FabricObject | null>(null)
  const textureLayerByWallIdRef = useRef<Map<number, FabricObject>>(new Map())
  const wallOverlaysRef = useRef<FabricObject[]>([])
  const wallDebugShapesRef = useRef<FabricObject[]>([])
  const wallDebugLabelsRef = useRef<FabricObject[]>([])
  const wallDebugShapeByIdRef = useRef<Map<number, FabricObject>>(new Map())
  const wallIdByObjectRef = useRef<WeakMap<FabricObject, number>>(new WeakMap())
  const WALL_BUTTON_DATA_KEY = 'wallId'
  const overlayUidCounterRef = useRef(0)

  const editWallCorners = useUIStore((s) => s.editWallCorners)
  const cornerHandlesRef = useRef<FabricObject[]>([])
  const cornerHandleMetaRef = useRef<WeakMap<FabricObject, { wallId: number; cornerIndex: number }>>(new WeakMap())
  const dragDebounceRef = useRef<number | null>(null)

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
      const target = (opt as unknown as { target?: FabricObject | null }).target
      const data = target ? (target as unknown as { data?: Record<string, unknown> }).data : undefined
      if (data && typeof data[WALL_BUTTON_DATA_KEY] === 'number') {
        useWallStore.getState().selectWall(data[WALL_BUTTON_DATA_KEY] as number)
        return
      }

      const tool = maskToolRef.current
      const scenePoint = canvas.getScenePoint(opt.e as MouseEvent)
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
    textureLayerByWallIdRef.current.clear()
    wallOverlaysRef.current = []
    wallDebugShapesRef.current = []
    wallDebugShapeByIdRef.current.clear()
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
    if (!bg || typeof (bg as unknown as { getBoundingRect?: () => { left: number; top: number; width: number; height: number } }).getBoundingRect !== 'function') return null
    const rect = (bg as unknown as { getBoundingRect: () => { left: number; top: number; width: number; height: number } }).getBoundingRect()
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
  }, [])

  const setWallOverlays = useCallback(
    (walls: WallData[], wallImageSize: { width: number; height: number } | null) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      const callId = `swo-${Date.now()}`
      // eslint-disable-next-line no-console
      console.log(`%c[WallOverlay] setWallOverlays() CALLED`, 'color:magenta;font-weight:bold', {
        callId,
        wallsCount: walls.length,
        wallImageSize,
        textureLayerByWallIdKeys: Array.from(textureLayerByWallIdRef.current.keys()),
        wallDebugShapesCount: wallDebugShapesRef.current.length,
        wallOverlaysCount: wallOverlaysRef.current.length,
        canvasObjectCount: canvas.getObjects().length,
        stackTrace: new Error().stack,
      })
      const textureObjs = Array.from(textureLayerByWallIdRef.current.values())
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} removing ${textureObjs.length} texture objs temporarily`)
      textureObjs.forEach((obj) => canvas.remove(obj))
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} removing ${wallDebugShapesRef.current.length} old debug shapes`)
      wallDebugShapesRef.current.forEach((obj) => canvas.remove(obj))
      wallDebugShapesRef.current = []
      wallDebugShapeByIdRef.current.clear()
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} removing ${wallDebugLabelsRef.current.length} old debug labels`)
      wallDebugLabelsRef.current.forEach((obj) => canvas.remove(obj))
      wallDebugLabelsRef.current = []
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} removing ${wallOverlaysRef.current.length} old overlay buttons`)
      wallOverlaysRef.current.forEach((obj) => canvas.remove(obj))
      wallOverlaysRef.current = []
      if (walls.length === 0 || !wallImageSize) {
        textureObjs.forEach((obj) => canvas.add(obj))
        canvas.requestRenderAll()
        return
      }
      const bounds = getBackgroundBounds()
      if (!bounds) {
        // eslint-disable-next-line no-console
        console.warn(`[WallOverlay] ${callId} ABORT: no background bounds`)
        return
      }
      const scaleX = bounds.width / wallImageSize.width
      const scaleY = bounds.height / wallImageSize.height
      const textured = useWallStore.getState().wallTextures
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} wallTextures from store:`, JSON.parse(JSON.stringify(textured)))
      for (const wall of walls) {
        const hasTexture = Boolean(textured[wall.id])
        const skip = wall.corners.length >= 3 && hasTexture
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${callId} wall ${wall.id}: corners=${wall.corners.length}, textured=${hasTexture}, textureValue=${JSON.stringify(textured[wall.id])}, skipBlue=${skip}`)
        const cx = bounds.left + wall.center[0] * scaleX
        const cy = bounds.top + wall.center[1] * scaleY
        if (!hideWallMasks && wall.corners.length >= 3 && !textured[wall.id]) {
          const overlayUid = `wo-${wall.id}-${++overlayUidCounterRef.current}`
          const d =
            wall.corners
              .map((c, i) => {
                const x = bounds.left + c[0] * scaleX
                const y = bounds.top + c[1] * scaleY
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
              })
              .join(' ') + ' Z'
          const shape = new Path(d, {
            fill: 'rgba(59,130,246,0.12)',
            stroke: 'rgba(37,99,235,0.8)',
            strokeWidth: 2,
            selectable: false,
            evented: false,
          })
          ;(shape as unknown as { set: (o: Record<string, unknown>) => void }).set({
            data: { isWallOverlay: true, wallId: wall.id, overlayUid },
          })
          canvas.add(shape)
          wallDebugShapesRef.current.push(shape)
          wallDebugShapeByIdRef.current.set(wall.id, shape)
          // eslint-disable-next-line no-console
          console.log(`%c[WallOverlay] ${callId} >>> CREATED blue shape for wall ${wall.id}`, 'color:red;font-weight:bold')
          // eslint-disable-next-line no-console
          console.log(`[WallOverlay] ${callId} overlay bbox`, {
            wallId: wall.id,
            overlayUid,
            bbox: (shape as unknown as { getBoundingRect: () => unknown }).getBoundingRect(),
          })

          const label = new Text(`W${wall.id}`, {
            left: cx,
            top: cy,
            originX: 'center',
            originY: 'center',
            fontSize: 18,
            fontWeight: 'bold',
            fill: '#ef4444',
            stroke: '#ffffff',
            strokeWidth: 4,
            paintFirst: 'stroke',
            selectable: false,
            evented: false,
            opacity: 0.95,
          })
          ;(label as unknown as { set: (o: Record<string, unknown>) => void }).set({
            data: { isWallOverlayLabel: true, wallId: wall.id, overlayUid },
          })
          canvas.add(label)
          wallDebugLabelsRef.current.push(label)
          // eslint-disable-next-line no-console
          console.log(`[WallOverlay] ${callId} label bbox`, {
            wallId: wall.id,
            overlayUid,
            bbox: (label as unknown as { getBoundingRect: () => unknown }).getBoundingRect(),
          })
        }
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
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} re-adding ${textureObjs.length} texture objs on top`)
      textureObjs.forEach((obj) => canvas.add(obj))
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${callId} DONE. canvas objects: ${canvas.getObjects().length}, debugShapes: ${wallDebugShapesRef.current.length}`)
      canvas.requestRenderAll()
    },
    [getBackgroundBounds, hideWallMasks]
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
      const bg = backgroundImageRef.current
      if (bg && typeof (bg as unknown as { getBoundingRect?: () => { left: number; top: number; width: number; height: number } }).getBoundingRect === 'function') {
        const rect = (bg as unknown as { getBoundingRect: () => { left: number; top: number; width: number; height: number } }).getBoundingRect()
        sceneBounds = { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
      }
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
    [getMaskObjects, textureScale]
  )

  const applyTextureToWall = useCallback(
    async (
      textureUrl: string,
      wallCornersImageCoords: [number, number][],
      wallImageSize: { width: number; height: number },
      wallId: number
    ) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      const requestId = `atw-${wallId}-${Date.now()}`
      // eslint-disable-next-line no-console
      console.log(`%c[WallOverlay] applyTextureToWall START`, 'color:blue;font-weight:bold', {
        requestId,
        wallId,
        textureUrl,
        canvasObjectCount: canvas.getObjects().length,
        existingTextureForWall: textureLayerByWallIdRef.current.has(wallId),
      })
      const existing = textureLayerByWallIdRef.current.get(wallId)
      if (existing) {
        canvas.remove(existing)
        textureLayerByWallIdRef.current.delete(wallId)
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} removed old texture for wall ${wallId}`)
      }
      const bounds = getBackgroundBounds()
      if (!bounds) {
        // eslint-disable-next-line no-console
        console.warn(`[WallOverlay] ${requestId} ABORT: no bounds`)
        return
      }
      const scaleXBounds = bounds.width / wallImageSize.width
      const scaleYBounds = bounds.height / wallImageSize.height
      const targetSceneCorners = wallCornersImageCoords.map(([ix, iy]) => [bounds.left + ix * scaleXBounds, bounds.top + iy * scaleYBounds] as [number, number])
      const xs = targetSceneCorners.map((p) => p[0])
      const ys = targetSceneCorners.map((p) => p[1])
      const targetBbox = {
        left: Math.min(...xs),
        top: Math.min(...ys),
        width: Math.max(...xs) - Math.min(...xs),
        height: Math.max(...ys) - Math.min(...ys),
      }
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${requestId} target wall corners (image)`, wallCornersImageCoords)
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${requestId} target wall corners (scene)`, targetSceneCorners)
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${requestId} target wall bbox (scene)`, targetBbox)
      const { warpWallTexture } = await import('@/hooks/warpWallTexture')
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${requestId} warpWallTexture starting...`)
      const blob = await warpWallTexture({
        textureUrl,
        corners: wallCornersImageCoords,
        imageSize: wallImageSize,
        textureScale,
        opacity: 0.85,
      })
      // eslint-disable-next-line no-console
      console.log(`[WallOverlay] ${requestId} warpWallTexture done, blob size: ${blob.size}`)
      const objectUrl = URL.createObjectURL(blob)
      try {
        const imgEl = await new Promise<HTMLImageElement>((resolve, reject) => {
          const img = new Image()
          img.onload = () => resolve(img)
          img.onerror = reject
          img.src = objectUrl
        })
        const scaleX = scaleXBounds
        const scaleY = scaleYBounds
        const fabricImg = new FabricImage(imgEl, {
          left: bounds.left,
          top: bounds.top,
          originX: 'left',
          originY: 'top',
          scaleX,
          scaleY,
          selectable: false,
          evented: false,
          opacity: 1,
        })
        canvas.add(fabricImg)
        canvas.requestRenderAll()
        textureLayerByWallIdRef.current.set(wallId, fabricImg)
        setHasTextureLayer(true)
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} texture image added to canvas for wall ${wallId}`)
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} texture bbox`, (fabricImg as unknown as { getBoundingRect: () => unknown }).getBoundingRect())

        const allObjs = canvas.getObjects()
        const allDataDump = allObjs.map((obj, idx) => {
          const d = (obj as unknown as { data?: Record<string, unknown> }).data
          return { idx, type: (obj as unknown as { type?: string }).type, data: d ?? null }
        })
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} FULL CANVAS DUMP after texture add:`, allDataDump)

        const storeTextures = useWallStore.getState().wallTextures
        const texturedIds = new Set(
          Object.entries(storeTextures ?? {})
            .filter(([, url]) => url)
            .map(([id]) => Number(id))
        )
        texturedIds.add(wallId)
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} texturedIds for removal:`, Array.from(texturedIds), 'storeTextures:', JSON.parse(JSON.stringify(storeTextures)))

        const aabbIntersects = (a: { left: number; top: number; width: number; height: number }, b: { left: number; top: number; width: number; height: number }) => {
          return a.left < b.left + b.width && a.left + a.width > b.left && a.top < b.top + b.height && a.top + a.height > b.top
        }

        const overlayObjs = canvas.getObjects().filter((obj) => {
          const data = (obj as unknown as { data?: Record<string, unknown> }).data
          return Boolean(data && data.isWallOverlay)
        })
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} overlays on canvas right now: ${overlayObjs.length}`)
        overlayObjs.forEach((obj) => {
          const data = ((obj as unknown as { data?: Record<string, unknown> }).data ?? {}) as any
          const bbox = (obj as unknown as { getBoundingRect: () => { left: number; top: number; width: number; height: number } }).getBoundingRect()
          // eslint-disable-next-line no-console
          console.log(`[WallOverlay] ${requestId} overlay-vs-target`, {
            overlayWallId: data.wallId,
            overlayUid: data.overlayUid,
            overlayBbox: bbox,
            targetWallId: wallId,
            targetBbox,
            intersects: aabbIntersects(bbox, targetBbox),
          })
        })

        const toRemove = canvas.getObjects().filter((obj) => {
          const data = (obj as unknown as { data?: Record<string, unknown> }).data
          const wid = data?.wallId
          const isOverlay = Boolean(data && data.isWallOverlay)
          const widIsNumber = typeof wid === 'number'
          const inSet = widIsNumber && texturedIds.has(wid as number)
          if (isOverlay) {
            // eslint-disable-next-line no-console
            console.log(`[WallOverlay] ${requestId} overlay candidate:`, { wallId: wid, isWallOverlay: isOverlay, widType: typeof wid, inTexturedSet: inSet, willRemove: isOverlay && widIsNumber && inSet })
          }
          return Boolean(data && data.isWallOverlay && widIsNumber && inSet)
        })
        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} toRemove count: ${toRemove.length}`)
        toRemove.forEach((obj) => canvas.remove(obj))
        if (toRemove.length > 0) {
          wallDebugShapesRef.current = wallDebugShapesRef.current.filter((s) => !toRemove.includes(s))
          toRemove.forEach((obj) => {
            const d = (obj as unknown as { data?: Record<string, unknown> }).data
            if (d && typeof d.wallId === 'number') wallDebugShapeByIdRef.current.delete(d.wallId)
          })
          canvas.requestRenderAll()
          // eslint-disable-next-line no-console
          console.log(`%c[WallOverlay] ${requestId} REMOVED ${toRemove.length} blue shapes`, 'color:green;font-weight:bold')
        } else {
          // eslint-disable-next-line no-console
          console.warn(`%c[WallOverlay] ${requestId} WARNING: NO blue shapes found to remove!`, 'color:orange;font-weight:bold')
        }

        // eslint-disable-next-line no-console
        console.log(`[WallOverlay] ${requestId} FINAL canvas objects: ${canvas.getObjects().length}, debugShapes: ${wallDebugShapesRef.current.length}`)
      } finally {
        URL.revokeObjectURL(objectUrl)
      }
    },
    [getBackgroundBounds, textureScale]
  )

  const highlightSelectedWall = useCallback((selectedId: number | null) => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    // eslint-disable-next-line no-console
    console.log('[WallOverlay] highlightSelectedWall', { selectedId, overlayButtonsCount: wallOverlaysRef.current.length })
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

  const clearCornerHandles = useCallback(() => {
    const canvas = canvasInstanceRef.current
    if (!canvas) return
    cornerHandlesRef.current.forEach((h) => canvas.remove(h))
    cornerHandlesRef.current = []
    cornerHandleMetaRef.current = new WeakMap()
    canvas.requestRenderAll()
  }, [])

  const updateWallShapePath = useCallback(
    (wallId: number, corners: [number, number][], wallImageSize: { width: number; height: number }) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      const bounds = getBackgroundBounds()
      if (!bounds) return
      const scaleX = bounds.width / wallImageSize.width
      const scaleY = bounds.height / wallImageSize.height
      const d =
        corners
          .map((c, i) => {
            const x = bounds.left + c[0] * scaleX
            const y = bounds.top + c[1] * scaleY
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
          })
          .join(' ') + ' Z'
      const shape = wallDebugShapeByIdRef.current.get(wallId)
      if (shape && typeof (shape as unknown as { set: (o: Record<string, unknown>) => void }).set === 'function') {
        ;(shape as unknown as { set: (o: Record<string, unknown>) => void }).set({ path: d })
      }
      canvas.requestRenderAll()
    },
    [getBackgroundBounds]
  )

  const syncCornerHandles = useCallback(
    (wallId: number | null) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      clearCornerHandles()
      if (!editWallCorners || wallId == null) return

      const state = useWallStore.getState()
      const wall = state.walls.find((w) => w.id === wallId)
      const wallImageSize = state.wallImageSize
      if (!wall || !wallImageSize || wall.corners.length < 4) return

      const bounds = getBackgroundBounds()
      if (!bounds) return
      const scaleX = bounds.width / wallImageSize.width
      const scaleY = bounds.height / wallImageSize.height

      for (let i = 0; i < 4; i++) {
        const c = wall.corners[i]
        const x = bounds.left + c[0] * scaleX
        const y = bounds.top + c[1] * scaleY
        const h = new Circle({
          left: x,
          top: y,
          radius: 6,
          fill: '#ffffff',
          stroke: '#2563eb',
          strokeWidth: 2,
          originX: 'center',
          originY: 'center',
          selectable: true,
          evented: true,
          hoverCursor: 'move',
        })
        cornerHandleMetaRef.current.set(h, { wallId, cornerIndex: i })
        canvas.add(h)
        cornerHandlesRef.current.push(h)
      }

      canvas.on('object:moving', (e) => {
        const target = (e as unknown as { target?: FabricObject | null }).target
        if (!target) return
        const meta = cornerHandleMetaRef.current.get(target)
        if (!meta) return

        const state2 = useWallStore.getState()
        const wall2 = state2.walls.find((w) => w.id === meta.wallId)
        const wallImageSize2 = state2.wallImageSize
        if (!wall2 || !wallImageSize2) return
        const bounds2 = getBackgroundBounds()
        if (!bounds2) return

        const sx = wallImageSize2.width / bounds2.width
        const sy = wallImageSize2.height / bounds2.height
        const cx = (target.left ?? 0)
        const cy = (target.top ?? 0)
        const ix = (cx - bounds2.left) * sx
        const iy = (cy - bounds2.top) * sy
        const nextCorners = wall2.corners.map((p) => [...p] as [number, number])
        nextCorners[meta.cornerIndex] = [Math.max(0, Math.min(wallImageSize2.width, ix)), Math.max(0, Math.min(wallImageSize2.height, iy))]
        state2.updateWallCorners(meta.wallId, nextCorners)
        updateWallShapePath(meta.wallId, nextCorners, wallImageSize2)

        const texUrl = state2.wallTextures[meta.wallId]
        if (texUrl) {
          if (dragDebounceRef.current) window.clearTimeout(dragDebounceRef.current)
          dragDebounceRef.current = window.setTimeout(() => {
            applyTextureToWall(texUrl, nextCorners, wallImageSize2, meta.wallId).catch(() => {})
          }, 200)
        }
      })
      canvas.requestRenderAll()
    },
    [applyTextureToWall, clearCornerHandles, editWallCorners, getBackgroundBounds, updateWallShapePath]
  )

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
