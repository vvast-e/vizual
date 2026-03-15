import { useEffect, useRef, useCallback, useState } from 'react'
import { Canvas, FabricImage, Point, PencilBrush, Group, Rect, Path } from 'fabric'
import type { FabricObject } from 'fabric'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useUIStore } from '@/store/useUIStore'
import { constrainDimensions } from '@/lib/canvas-utils'
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
  const panStartRef = useRef<{ x: number; y: number; vpt: [number, number, number, number, number, number] } | null>(null)
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

  const initCanvas = useCallback(() => {
    const el = canvasRef.current
    if (!el) return null
    const canvas = new Canvas(el, {
      selection: false,
      preserveObjectStacking: true,
    })
    canvas.setDimensions({ width: containerWidth, height: containerHeight })
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
      const tool = maskToolRef.current
      const scenePoint = canvas.getScenePoint(opt.e as MouseEvent)
      const viewportPoint = canvas.getViewportPoint(opt.e as MouseEvent)

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

      const target = (opt as unknown as { target?: FabricObject | null }).target ?? null
      if (target && target !== backgroundImageRef.current) {
        return
      }

      panStartRef.current = {
        x: viewportPoint.x,
        y: viewportPoint.y,
        vpt: [...canvas.viewportTransform] as [number, number, number, number, number, number],
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

      if (tool !== null || !panStartRef.current) return
      const point = canvas.getViewportPoint(opt.e as MouseEvent)
      const vpt: [number, number, number, number, number, number] = [
        panStartRef.current.vpt[0],
        panStartRef.current.vpt[1],
        panStartRef.current.vpt[2],
        panStartRef.current.vpt[3],
        panStartRef.current.vpt[4] + point.x - panStartRef.current.x,
        panStartRef.current.vpt[5] + point.y - panStartRef.current.y,
      ]
      canvas.viewportTransform = vpt
      canvas.requestRenderAll()
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
        panStartRef.current = null
        return
      }
      panStartRef.current = null
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
      const dims = constrainDimensions(img.width ?? 0, img.height ?? 0)
      img.set({
        scaleX: dims.width / (img.width ?? 1),
        scaleY: dims.height / (img.height ?? 1),
        selectable: false,
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
  }, [setPhotoDataUrl])

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
    setPhotoDataUrl(null)
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

  const applyTexture = useCallback(
    async (textureUrl: string, repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat') => {
      const canvas = canvasInstanceRef.current
      if (!canvas) return
      const { applyPatternToCanvas } = await import('@/lib/texture-processor')
      const maskObjs = getMaskObjects()
      let clipPath: Group | undefined
      if (maskObjs.length > 0) {
        const clones = await Promise.all(maskObjs.map((o) => o.clone()))
        clipPath = new Group(clones)
      }
      let sceneBounds: { left: number; top: number; width: number; height: number } | undefined
      const bg = backgroundImageRef.current
      if (bg && typeof (bg as unknown as { getBoundingRect?: () => { left: number; top: number; width: number; height: number } }).getBoundingRect === 'function') {
        const rect = (bg as unknown as { getBoundingRect: () => { left: number; top: number; width: number; height: number } }).getBoundingRect()
        sceneBounds = {
          left: rect.left,
          top: rect.top,
          width: rect.width,
          height: rect.height,
        }
      }
      await applyPatternToCanvas(canvas, textureUrl, repeat, clipPath, sceneBounds)
    },
    [getMaskObjects]
  )

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
    applyTexture,
    finishLasso,
    clearMaskToolState,
  }
}
