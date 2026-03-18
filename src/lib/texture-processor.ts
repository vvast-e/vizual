/**
 * Обработка текстур: загрузка, паттерны repeat, применение на канвас.
 * Колоризация (blend) — в этапе 4.
 */

import type { Canvas } from 'fabric'
import type { FabricObject } from 'fabric'
import { Rect, Pattern } from 'fabric'
// @ts-expect-error — vendored ESM without type declarations
import { Homography } from './homography-vendor.js'

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

function createScaledTiledCanvas(
  img: HTMLImageElement,
  width: number,
  height: number,
  scale: number
): HTMLCanvasElement {
  const safeScale = Math.max(0.05, Math.min(1, scale))
  const tileW = Math.max(1, Math.round(img.naturalWidth * safeScale))
  const tileH = Math.max(1, Math.round(img.naturalHeight * safeScale))

  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const ctx = c.getContext('2d')
  if (!ctx) return c

  for (let y = 0; y < height; y += tileH) {
    for (let x = 0; x < width; x += tileW) {
      ctx.drawImage(img, x, y, tileW, tileH)
    }
  }

  return c
}

/**
 * Рендерит текстуру на четырёхугольник стены с учётом перспективы (Homography.js).
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

  const xs = corners.map((c) => c[0])
  const ys = corners.map((c) => c[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const bboxW = Math.max(1, Math.ceil(maxX - minX))
  const bboxH = Math.max(1, Math.ceil(maxY - minY))
  const localCorners = corners.map(([x, y]) => [x - minX, y - minY] as [number, number])

  // Источник: тайлим текстуру в canvas такого же размера, что bbox стены,
  // а масштаб паттерна регулируем через drawImage(tileW/tileH).
  const srcCanvas = createScaledTiledCanvas(img, bboxW, bboxH, textureScale)
  const srcImageData = srcCanvas.getContext('2d')!.getImageData(0, 0, bboxW, bboxH)

  // Homography: проектное преобразование прямоугольника srcCanvas → четырёхугольник стены.
  // Входные точки приходят как [TL, BL, BR, TR], а для матчинга с прямоугольником удобнее
  // подать dst-углы в порядке [TL, TR, BR, BL].
  const [tl, bl, br, tr] = localCorners
  const dstQuad: [number, number][] = [tl, tr, br, bl]

  // Homography.js подбирает выходной кадр по dst-точкам и может добавлять “паддинг”, если min(dst) > 0.
  // Нормализуем dstQuad к (0,0), а смещение компенсируем в offsetX/offsetY и clipPath.
  const dstMinX = Math.min(dstQuad[0][0], dstQuad[1][0], dstQuad[2][0], dstQuad[3][0])
  const dstMinY = Math.min(dstQuad[0][1], dstQuad[1][1], dstQuad[2][1], dstQuad[3][1])
  const shiftedDstQuad: [number, number][] = dstQuad.map(
    ([x, y]) => [x - dstMinX, y - dstMinY] as [number, number]
  )

  const hom = new Homography('projective', bboxW, bboxH)
  hom.setSourcePoints([
    [0, 0],
    [bboxW, 0],
    [bboxW, bboxH],
    [0, bboxH],
  ])
  hom.setDestinyPoints(shiftedDstQuad)
  hom.setImage(srcImageData)
  const warped: ImageData = hom.warp()

  const offscreen = document.createElement('canvas')
  offscreen.width = warped.width
  offscreen.height = warped.height
  const ctx = offscreen.getContext('2d')
  if (!ctx) {
    return { canvas: offscreen, offsetX: minX + dstMinX, offsetY: minY + dstMinY, localCorners: shiftedDstQuad }
  }
  ctx.putImageData(warped, 0, 0)

  return { canvas: offscreen, offsetX: minX + dstMinX, offsetY: minY + dstMinY, localCorners: shiftedDstQuad }
}

export { textureCache }
