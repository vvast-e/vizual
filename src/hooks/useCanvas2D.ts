import { useEffect, useRef, useCallback, useState } from 'react'
import { Canvas, FabricImage, Point, PencilBrush, Group } from 'fabric'
import type { FabricObject } from 'fabric'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { constrainDimensions } from '@/lib/canvas-utils'
import { MAX_PHOTO_SIZE_BYTES, ALLOWED_IMAGE_TYPES } from '@/lib/constants'

const MIN_ZOOM = 0.1
const MAX_ZOOM = 5
const ZOOM_STEP = 0.1

export interface UseCanvas2DOptions {
  /** Ref на DOM canvas элемент */
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  /** Ширина контейнера (для setDimensions) */
  containerWidth?: number
  /** Высота контейнера */
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

  const setPhotoDataUrl = useVisualizerStore((s) => s.setPhotoDataUrl)

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

    canvas.on('mouse:down', (opt) => {
      if (canvas.isDrawingMode) return
      const point = canvas.getViewportPoint(opt.e as MouseEvent)
      panStartRef.current = {
        x: point.x,
        y: point.y,
        vpt: [...canvas.viewportTransform] as [number, number, number, number, number, number],
      }
    })

    canvas.on('mouse:move', (opt) => {
      if (canvas.isDrawingMode || !panStartRef.current) return
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
    })

    canvas.on('mouse:up', () => {
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
      img.set({ scaleX: dims.width / (img.width ?? 1), scaleY: dims.height / (img.height ?? 1) })
      canvas.clear()
      canvas.add(img)
      canvas.renderAll()
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
      await applyPatternToCanvas(canvas, textureUrl, repeat, clipPath)
    },
    [getMaskObjects]
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
    applyTexture,
  }
}
