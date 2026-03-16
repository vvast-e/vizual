import { useEffect, useRef, useCallback, useState } from 'react'
import { Canvas, FabricImage, Point, PencilBrush, Group, Rect, Path } from 'fabric'
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
    if (!bg || typeof (bg as unknown as { getBoundingRect?: () => { left: number; top: number; width: number; height: number } }).getBoundingRect !== 'function') return null
    const rect = (bg as unknown as { getBoundingRect: () => { left: number; top: number; width: number; height: number } }).getBoundingRect()
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
  }, [])

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
      for (const wall of walls) {
        // многоугольник стены поверх фото, чтобы было видно реальную форму
        if (wall.corners.length >= 3) {
          const d =
            wall.corners
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
    async (textureUrl: string, wallCorners: [number, number][]) => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      if (textureLayerRef.current) {
        canvas.remove(textureLayerRef.current)
        textureLayerRef.current = null
      }
      const { renderPerspectiveWallTexture } = await import('@/lib/texture-processor')
      const perspective = await renderPerspectiveWallTexture(
        textureUrl,
        wallCorners,
        containerWidth,
        containerHeight,
        textureScale
      )
      const fabricImg = new FabricImage(perspective.canvas, {
        left: perspective.offsetX,
        top: perspective.offsetY,
        selectable: false,
        evented: false,
        opacity: 0.85,
      })
      if (perspective.localCorners.length >= 3) {
        const d = perspective.localCorners
          .map((p, i) => (i === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`))
          .join(' ') + ' Z'
        fabricImg.set({ clipPath: new Path(d, { selectable: false, evented: false }) })
      }
      canvas.add(fabricImg)
      canvas.requestRenderAll()
      textureLayerRef.current = fabricImg
      setHasTextureLayer(true)
    },
    [containerWidth, containerHeight, textureScale]
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
    applyTexture,
    applyTextureToWall,
    highlightSelectedWall,
    finishLasso,
    clearMaskToolState,
    textureScale,
    setTextureScale,
    hasTextureLayer,
  }
}
