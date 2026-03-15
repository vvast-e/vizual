/**
 * Утилиты для 2D канваса (размеры, ограничения, координаты).
 */

import type { CanvasDimensions } from '@/types/canvas'

const DEFAULT_MAX_WIDTH = 1920
const DEFAULT_MAX_HEIGHT = 1080

/**
 * Ограничивает размеры по максимальной стороне, сохраняя пропорции.
 */
export function constrainDimensions(
  width: number,
  height: number,
  maxW: number = DEFAULT_MAX_WIDTH,
  maxH: number = DEFAULT_MAX_HEIGHT
): CanvasDimensions {
  if (width <= maxW && height <= maxH) return { width, height }
  const r = Math.min(maxW / width, maxH / height)
  return {
    width: Math.round(width * r),
    height: Math.round(height * r),
  }
}

/**
 * Вычисляет размеры контейнера под канвас (fit inside).
 */
export function fitInContainer(
  canvasW: number,
  canvasH: number,
  containerW: number,
  containerH: number
): { width: number; height: number; scale: number; offsetX: number; offsetY: number } {
  const scale = Math.min(containerW / canvasW, containerH / canvasH, 1)
  const width = Math.round(canvasW * scale)
  const height = Math.round(canvasH * scale)
  const offsetX = (containerW - width) / 2
  const offsetY = (containerH - height) / 2
  return { width, height, scale, offsetX, offsetY }
}
