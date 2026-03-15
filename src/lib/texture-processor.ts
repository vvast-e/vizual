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

/**
 * Накладывает паттерн текстуры на Fabric Canvas как слой (Rect с Pattern fill).
 * Если передан clipPath (например Group масок), текстура применяется только в области маски.
 */
export async function applyPatternToCanvas(
  canvas: Canvas,
  textureUrl: string,
  repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat',
  clipPath?: FabricObject,
  sceneBounds?: { left: number; top: number; width: number; height: number }
): Promise<void> {
  const patternCanvas = await getPatternCanvas(textureUrl, repeat)
  const w = sceneBounds?.width ?? (canvas.width ?? 0)
  const h = sceneBounds?.height ?? (canvas.height ?? 0)
  const pattern = new Pattern({
    source: patternCanvas,
    repeat,
    patternTransform: [0.25, 0, 0, 0.25, 0, 0],
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
  canvas.add(rect)
  canvas.requestRenderAll()
}

export { textureCache }
