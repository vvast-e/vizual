/**
 * Обработка текстур: загрузка, паттерны repeat, применение на канвас.
 * Колоризация (blend) — в этапе 4.
 */

import type { Canvas } from 'fabric'
import type { FabricObject } from 'fabric'
import { Rect, Pattern } from 'fabric'

const textureCache = new Map<string, HTMLImageElement>()

function loadImage(url: string): Promise<HTMLImageElement> {
  const cached = textureCache.get(url)
  if (cached) return Promise.resolve(cached)
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      textureCache.set(url, img)
      resolve(img)
    }
    img.onerror = reject
    img.src = url
  })
}

/**
 * Загружает изображение текстуры (с кэшем по URL).
 */
export function loadTextureImage(url: string): Promise<HTMLImageElement> {
  return loadImage(url)
}

/**
 * Создаёт canvas с паттерном (repeat) из изображения.
 * Возвращает data URL или canvas для использования как Pattern source.
 */
export function createPatternCanvas(
  image: HTMLImageElement,
  repeat: 'repeat' | 'repeat-x' | 'repeat-y' | 'no-repeat' = 'repeat',
  width: number = 512,
  height: number = 512
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return canvas
  const pattern = ctx.createPattern(image, repeat)
  if (!pattern) return canvas
  ctx.fillStyle = pattern
  ctx.fillRect(0, 0, width, height)
  return canvas
}

/**
 * Загружает текстуру по URL и возвращает canvas с паттерном.
 */
export async function getPatternCanvas(
  textureUrl: string,
  repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat',
  width: number = 512,
  height: number = 512
): Promise<HTMLCanvasElement> {
  const img = await loadTextureImage(textureUrl)
  return createPatternCanvas(img, repeat, width, height)
}

/**
 * Накладывает паттерн текстуры на Fabric Canvas как слой (Rect с Pattern fill).
 * repeat: 'repeat' | 'repeat-x' | 'repeat-y'.
 */
/**
 * Накладывает цвет на текстуру с сохранением фактуры (blend multiply).
 * @param source — изображение или canvas с текстурой
 * @param hexColor — цвет в формате #rrggbb
 * @param opacity — интенсивность 0..1
 * @returns canvas с колоризованной текстурой
 */
export function applyColorToTexture(
  source: HTMLImageElement | HTMLCanvasElement,
  hexColor: string,
  opacity: number
): HTMLCanvasElement {
  const w = source instanceof HTMLImageElement ? source.naturalWidth : source.width
  const h = source instanceof HTMLImageElement ? source.naturalHeight : source.height
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return canvas
  ctx.drawImage(source, 0, 0)
  ctx.globalCompositeOperation = 'multiply'
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity))
  ctx.fillStyle = hexColor
  ctx.fillRect(0, 0, w, h)
  ctx.globalAlpha = 1
  ctx.globalCompositeOperation = 'source-over'
  return canvas
}

const TEXTURE_LAYER_DATA_KEY = 'isTextureLayer'

/**
 * Накладывает паттерн текстуры на Fabric Canvas как слой (Rect с Pattern fill).
 * Если передан clipPath (например Group масок), текстура применяется только в области маски.
 * @param patternScale масштаб повтора паттерна (0.05–1: чем меньше, тем мельче текстура).
 * @returns добавленный Rect (слой текстуры) для последующего изменения масштаба.
 */
export async function applyPatternToCanvas(
  canvas: Canvas,
  textureUrl: string,
  repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat',
  clipPath?: FabricObject,
  sceneBounds?: { left: number; top: number; width: number; height: number },
  patternScale: number = 0.25
): Promise<Rect> {
  const patternCanvas = await getPatternCanvas(textureUrl, repeat)
  const w = sceneBounds?.width ?? (canvas.width ?? 0)
  const h = sceneBounds?.height ?? (canvas.height ?? 0)
  const scale = Math.max(0.05, Math.min(1, patternScale))
  const pattern = new Pattern({
    source: patternCanvas,
    repeat,
    patternTransform: [scale, 0, 0, scale, 0, 0],
  })
  const rect = new Rect({
    width: w,
    height: h,
    left: sceneBounds?.left ?? 0,
    top: sceneBounds?.top ?? 0,
    fill: pattern,
    opacity: 0.85,
    originX: 'left',
    originY: 'top',
    selectable: false,
    evented: false,
    clipPath: clipPath ?? undefined,
  })
  ;(rect as unknown as { set: (o: Record<string, unknown>) => void }).set({
    data: { [TEXTURE_LAYER_DATA_KEY]: true },
  })
  canvas.add(rect)
  canvas.requestRenderAll()
  return rect
}

function drawTexturedTriangle(
  ctx: CanvasRenderingContext2D,
  src: CanvasImageSource,
  sx0: number, sy0: number,
  sx1: number, sy1: number,
  sx2: number, sy2: number,
  dx0: number, dy0: number,
  dx1: number, dy1: number,
  dx2: number, dy2: number
): void {
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(dx0, dy0)
  ctx.lineTo(dx1, dy1)
  ctx.lineTo(dx2, dy2)
  ctx.closePath()
  ctx.clip()

  const denom = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0)
  if (Math.abs(denom) < 1e-6) { ctx.restore(); return }

  const a = ((dx1 - dx0) * (sy2 - sy0) - (dx2 - dx0) * (sy1 - sy0)) / denom
  const c = ((dx2 - dx0) * (sx1 - sx0) - (dx1 - dx0) * (sx2 - sx0)) / denom
  const e = dx0 - a * sx0 - c * sy0
  const b = ((dy1 - dy0) * (sy2 - sy0) - (dy2 - dy0) * (sy1 - sy0)) / denom
  const d = ((dy2 - dy0) * (sx1 - sx0) - (dy1 - dy0) * (sx2 - sx0)) / denom
  const f = dy0 - b * sx0 - d * sy0

  ctx.setTransform(a, b, c, d, e, f)
  ctx.drawImage(src, 0, 0)
  ctx.restore()
}

function createTiledCanvas(img: HTMLImageElement, width: number, height: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const ctx = c.getContext('2d')!
  const pat = ctx.createPattern(img, 'repeat')
  if (pat) {
    ctx.fillStyle = pat
    ctx.fillRect(0, 0, width, height)
  }
  return c
}

/**
 * Рендерит текстуру на четырёхугольник стены с учётом перспективы (через perspectivejs).
 *
 * @param corners — 4 угла стены в координатах канваса, порядок: [TL, BL, BR, TR]
 */
export async function renderPerspectiveWallTexture(
  textureUrl: string,
  corners: [number, number][],
  canvasWidth: number,
  canvasHeight: number,
  textureScale: number = 0.25
): Promise<{ canvas: HTMLCanvasElement; offsetX: number; offsetY: number; localCorners: [number, number][] }> {
  const img = await loadTextureImage(textureUrl)
  if (corners.length < 4) {
    const empty = document.createElement('canvas')
    empty.width = Math.max(1, canvasWidth)
    empty.height = Math.max(1, canvasHeight)
    return { canvas: empty, offsetX: 0, offsetY: 0, localCorners: [] }
  }

  // --- FIX: srcCanvas размера bbox стены ---
  const xs = corners.map((c) => c[0])
  const ys = corners.map((c) => c[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const bboxW = Math.max(1, Math.ceil(maxX - minX))
  const bboxH = Math.max(1, Math.ceil(maxY - minY))
  const srcCanvas = createPatternCanvas(img, 'repeat', bboxW, bboxH)
  const localCorners = corners.map(([x, y]) => [x - minX, y - minY] as [number, number])

  // offscreen canvas размера bbox
  const offscreen = document.createElement('canvas')
  offscreen.width = bboxW
  offscreen.height = bboxH
  const ctx = offscreen.getContext('2d')
  if (!ctx) {
    return { canvas: offscreen, offsetX: minX, offsetY: minY, localCorners }
  }

  // Рисуем перспективу без perspectivejs: разбиваем quad на 2 треугольника.
  // Ожидаемый порядок углов: [TL, BL, BR, TR]
  const [tl, bl, br, tr] = localCorners

  ctx.clearRect(0, 0, bboxW, bboxH)

  // Triangle 1: TL-BL-BR
  drawTexturedTriangle(
    ctx,
    srcCanvas,
    0, 0,
    0, bboxH,
    bboxW, bboxH,
    tl[0], tl[1],
    bl[0], bl[1],
    br[0], br[1]
  )

  // Triangle 2: TL-BR-TR
  drawTexturedTriangle(
    ctx,
    srcCanvas,
    0, 0,
    bboxW, bboxH,
    bboxW, 0,
    tl[0], tl[1],
    br[0], br[1],
    tr[0], tr[1]
  )

  return { canvas: offscreen, offsetX: minX, offsetY: minY, localCorners }
}

export { textureCache }
