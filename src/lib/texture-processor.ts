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


/**
 * Базовый размер тайла текстуры в пикселях при scale=1.0.
 * При дефолтном scale=0.25 тайл ~150px — визуально мелкий и корректный.
 */
const BASE_TILE_PX = 600

/**
 * Строит тайлованный паттерн-canvas нужного размера с учётом масштаба.
 * Один тайл = BASE_TILE_PX * textureScale пикселей (по большей стороне),
 * затем повторяется на destW×destH.
 */
function buildTiledSource(
  img: HTMLImageElement,
  destW: number,
  destH: number,
  textureScale: number,
): HTMLCanvasElement {
  const scale = Math.max(0.02, Math.min(2, textureScale))
  const tileW = Math.max(1, Math.round(BASE_TILE_PX * scale))
  const tileH = Math.max(1, Math.round(tileW * (img.naturalHeight / Math.max(1, img.naturalWidth))))

  // Промежуточный холст одного тайла
  const tileCanvas = document.createElement('canvas')
  tileCanvas.width = tileW
  tileCanvas.height = tileH
  const tc = tileCanvas.getContext('2d')
  if (tc) tc.drawImage(img, 0, 0, tileW, tileH)

  // Финальный холст с повтором тайла
  const dst = document.createElement('canvas')
  dst.width = Math.max(1, destW)
  dst.height = Math.max(1, destH)
  const dc = dst.getContext('2d')
  if (dc) {
    const pat = dc.createPattern(tileCanvas, 'repeat')
    if (pat) {
      dc.fillStyle = pat
      dc.fillRect(0, 0, dst.width, dst.height)
    }
  }
  return dst
}

/**
 * Строит источник текстуры для балки: одна доска, растянутая поперёк,
 * повторяется вдоль длины как продольные сегменты (волокно по длине балки).
 * Устраняет эффект «реек/жалюзи», который возникает при квадратном тайлинге.
 *
 * Ориентация: если destW >= destH — балка горизонтальная (along = W),
 * иначе — вертикальная (along = H). Картинка поворачивается 90° для вертикальных балок.
 */
function buildBeamSource(
  img: HTMLImageElement,
  destW: number,
  destH: number,
  textureScale: number,
): HTMLCanvasElement {
  const dst = document.createElement('canvas')
  dst.width = Math.max(1, destW)
  dst.height = Math.max(1, destH)
  const dc = dst.getContext('2d')
  if (!dc) return dst

  const isHorizontal = destW >= destH
  const along = isHorizontal ? destW : destH   // длина балки
  const across = isHorizontal ? destH : destW  // ширина балки

  const scale = Math.max(0.05, Math.min(4, textureScale))
  // Длина одного сегмента вдоль балки (≈ квадратные сегменты, масштабируемые ползунком)
  const segLen = Math.max(1, Math.round(across * 4 * scale))
  const numSegs = Math.max(1, Math.round(along / segLen))
  const actualSegLen = along / numSegs

  for (let i = 0; i < numSegs; i++) {
    const segStart = i * actualSegLen

    if (isHorizontal) {
      // Горизонтальная балка: каждый сегмент — вертикальная полоска,
      // картинка рисуется «в ширину» поперёк балки (растянута на across = destH)
      // и повторяется по длине
      dc.drawImage(img, segStart, 0, actualSegLen, across)
    } else {
      // Вертикальная балка: каждый сегмент — горизонтальная полоска,
      // картинка рисуется повёрнуто (растянута на across = destW)
      dc.save()
      dc.translate(across, segStart)
      dc.rotate(Math.PI / 2)
      dc.drawImage(img, 0, 0, actualSegLen, across)
      dc.restore()
    }
  }

  return dst
}

/**
 * Рендерит текстуру на четырёхугольник стены с учётом перспективы
 * через разбиение на 2 треугольника (drawTexturedTriangle).
 *
 * @param corners — 4 угла стены в координатах канваса, порядок: [TL, BL, BR, TR]
 * @param polygon — опциональный многоугольник стены (для обрезки)
 * @param fullCanvasSize — если задан, рисует в полноразмерный холст (wallImageSize) вместо bbox.
 *   Позволяет класть слой через bgTx как экстерьерный PNG — без clipPath и ручной арифметики координат.
 */
export async function renderPerspectiveWallTexture(
  textureUrl: string,
  corners: [number, number][],
  canvasWidth: number,
  canvasHeight: number,
  textureScale: number = 0.25,
  polygon?: [number, number][],
  maskImage?: HTMLImageElement | null,
  fullCanvasSize?: { width: number; height: number } | null,
  sourceMode: 'tile' | 'beam' = 'tile'
): Promise<{
  canvas: HTMLCanvasElement
  offsetX: number
  offsetY: number
  localCorners: [number, number][]
  localPolygon?: [number, number][]
}> {
  const img = await loadTextureImage(textureUrl)
  if (corners.length < 4) {
    const empty = document.createElement('canvas')
    empty.width = Math.max(1, canvasWidth)
    empty.height = Math.max(1, canvasHeight)
    return { canvas: empty, offsetX: 0, offsetY: 0, localCorners: [], localPolygon: [] }
  }

  const pointsForBBox = polygon && polygon.length >= 3 ? polygon : corners
  const xs = pointsForBBox.map((c) => c[0])
  const ys = pointsForBBox.map((c) => c[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const bboxW = Math.max(1, Math.ceil(maxX - minX))
  const bboxH = Math.max(1, Math.ceil(maxY - minY))

  // Флаг отладочного логирования (только в dev-режиме)
  const DEBUG_INTERIOR = (import.meta.env?.DEV ?? false) || (typeof localStorage !== 'undefined' && localStorage.getItem('vizual-debug') === '1')

  // Режим full-canvas: рисуем в полноразмерный холст (wallImageSize) по абсолютным координатам.
  // Слой затем кладётся через bgTx (как экстерьерный PNG) — без clipPath.
  if (fullCanvasSize && fullCanvasSize.width > 0 && fullCanvasSize.height > 0) {
    const fullW = fullCanvasSize.width
    const fullH = fullCanvasSize.height

    // Паттерн с учётом textureScale — тайл масштабирован, затем повторён на bbox
    const srcCanvas = buildTiledSource(img, bboxW, bboxH, textureScale)
    const offscreen = document.createElement('canvas')
    offscreen.width = fullW
    offscreen.height = fullH
    const ctx = offscreen.getContext('2d')
    if (!ctx) {
      return { canvas: offscreen, offsetX: 0, offsetY: 0, localCorners: corners, localPolygon: polygon }
    }
    ctx.clearRect(0, 0, fullW, fullH)

    // Если есть polygon — ограничиваем клипом для точного контура (внутри самого canvas)
    const clipPts = polygon && polygon.length >= 3 ? polygon : corners
    if (DEBUG_INTERIOR) {
      console.log('[interior-wall] render full-canvas', {
        bboxW, bboxH, fullW, fullH,
        clipPtsCount: clipPts.length,
        sourceMode,
        hasMask: !!(maskImage && maskImage.width > 0),
        textureScale,
      })
    }
    ctx.save()
    ctx.beginPath()
    clipPts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)))
    ctx.closePath()
    ctx.clip()

    // Рисуем quad: [TL, BL, BR, TR] — абсолютные координаты corners
    const [tl, bl, br, tr] = corners
    // Triangle 1: TL-BL-BR
    drawTexturedTriangle(
      ctx, srcCanvas,
      0, 0,   0, bboxH,   bboxW, bboxH,
      tl[0], tl[1],  bl[0], bl[1],  br[0], br[1]
    )
    // Triangle 2: TL-BR-TR
    drawTexturedTriangle(
      ctx, srcCanvas,
      0, 0,   bboxW, bboxH,   bboxW, 0,
      tl[0], tl[1],  br[0], br[1],  tr[0], tr[1]
    )
    ctx.restore()

    // Маска окон/дверей: вырезаем прозрачные области
    if (maskImage && maskImage.width > 0 && maskImage.height > 0) {
      const prevOp = ctx.globalCompositeOperation
      ctx.globalCompositeOperation = 'destination-in'
      ctx.drawImage(maskImage, 0, 0, fullW, fullH)
      ctx.globalCompositeOperation = prevOp
    }

    return {
      canvas: offscreen,
      offsetX: 0,
      offsetY: 0,
      localCorners: corners,
      localPolygon: polygon,
    }
  }

  // Режим bbox (для балок и обратной совместимости)
  const srcCanvas = sourceMode === 'beam'
    ? buildBeamSource(img, bboxW, bboxH, textureScale)
    : buildTiledSource(img, bboxW, bboxH, textureScale)
  const localCorners = corners.map(([x, y]) => [x - minX, y - minY] as [number, number])
  const localPolygon = (polygon && polygon.length >= 3
    ? polygon
    : corners
  ).map(([x, y]) => [x - minX, y - minY] as [number, number])

  const offscreen = document.createElement('canvas')
  offscreen.width = bboxW
  offscreen.height = bboxH
  const ctx = offscreen.getContext('2d')
  if (!ctx) {
    return { canvas: offscreen, offsetX: minX, offsetY: minY, localCorners, localPolygon }
  }

  // Рисуем перспективу: порядок углов [TL, BL, BR, TR]
  const [tl, bl, br, tr] = localCorners

  ctx.clearRect(0, 0, bboxW, bboxH)

  if (sourceMode === 'beam') {
    // Субдивизия балки на K срезов вдоль длинной оси для уменьшения аффинного перекоса.
    // Кусочно-аффинная аппроксимация резко снижает диагональный артефакт.
    const K = 12
    if (bboxW >= bboxH) {
      // Горизонтальная балка: разрезаем вдоль X (TL→TR и BL→BR)
      for (let i = 0; i < K; i++) {
        const t0 = i / K
        const t1 = (i + 1) / K
        const sTL: [number, number] = [tl[0] + (tr[0] - tl[0]) * t0, tl[1] + (tr[1] - tl[1]) * t0]
        const sTR: [number, number] = [tl[0] + (tr[0] - tl[0]) * t1, tl[1] + (tr[1] - tl[1]) * t1]
        const sBL: [number, number] = [bl[0] + (br[0] - bl[0]) * t0, bl[1] + (br[1] - bl[1]) * t0]
        const sBR: [number, number] = [bl[0] + (br[0] - bl[0]) * t1, bl[1] + (br[1] - bl[1]) * t1]
        // Источник: горизонтальный срез [t0·W … t1·W] × [0 … H]
        drawTexturedTriangle(ctx, srcCanvas,
          t0 * bboxW, 0,    t0 * bboxW, bboxH,    t1 * bboxW, bboxH,
          sTL[0], sTL[1],   sBL[0], sBL[1],        sBR[0], sBR[1])
        drawTexturedTriangle(ctx, srcCanvas,
          t0 * bboxW, 0,    t1 * bboxW, bboxH,    t1 * bboxW, 0,
          sTL[0], sTL[1],   sBR[0], sBR[1],        sTR[0], sTR[1])
      }
    } else {
      // Вертикальная балка: разрезаем вдоль Y (TL→BL и TR→BR)
      for (let i = 0; i < K; i++) {
        const t0 = i / K
        const t1 = (i + 1) / K
        const sTL: [number, number] = [tl[0] + (bl[0] - tl[0]) * t0, tl[1] + (bl[1] - tl[1]) * t0]
        const sBL: [number, number] = [tl[0] + (bl[0] - tl[0]) * t1, tl[1] + (bl[1] - tl[1]) * t1]
        const sTR: [number, number] = [tr[0] + (br[0] - tr[0]) * t0, tr[1] + (br[1] - tr[1]) * t0]
        const sBR: [number, number] = [tr[0] + (br[0] - tr[0]) * t1, tr[1] + (br[1] - tr[1]) * t1]
        // Источник: вертикальный срез [0 … W] × [t0·H … t1·H]
        drawTexturedTriangle(ctx, srcCanvas,
          0, t0 * bboxH,    0, t1 * bboxH,    bboxW, t1 * bboxH,
          sTL[0], sTL[1],   sBL[0], sBL[1],   sBR[0], sBR[1])
        drawTexturedTriangle(ctx, srcCanvas,
          0, t0 * bboxH,    bboxW, t1 * bboxH,    bboxW, t0 * bboxH,
          sTL[0], sTL[1],   sBR[0], sBR[1],       sTR[0], sTR[1])
      }
    }
  } else {
    // Стены/тайл: стандартные 2 треугольника (полный quad)
    // Triangle 1: TL-BL-BR
    drawTexturedTriangle(ctx, srcCanvas,
      0, 0,     0, bboxH,    bboxW, bboxH,
      tl[0], tl[1],  bl[0], bl[1],  br[0], br[1])
    // Triangle 2: TL-BR-TR
    drawTexturedTriangle(ctx, srcCanvas,
      0, 0,     bboxW, bboxH,    bboxW, 0,
      tl[0], tl[1],  br[0], br[1],  tr[0], tr[1])
  }

  // Mask: carve openings (windows/doors) out of the rendered texture.
  // The mask is expected in the same coordinate system as `corners` (wallImageSize),
  // so we sample the same bbox region without rescaling.
  if (maskImage && maskImage.width > 0 && maskImage.height > 0) {
    const prevOp = ctx.globalCompositeOperation
    ctx.globalCompositeOperation = 'destination-in'
    ctx.drawImage(maskImage, minX, minY, bboxW, bboxH, 0, 0, bboxW, bboxH)
    ctx.globalCompositeOperation = prevOp
  }

  return { canvas: offscreen, offsetX: minX, offsetY: minY, localCorners, localPolygon }
}

export { textureCache }
