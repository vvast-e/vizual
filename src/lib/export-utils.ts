/**
 * Экспорт канваса/сцены в PNG (download).
 */

import type { Color, Material } from '@/types/material'

/**
 * Скачивает data URL как файл с заданным именем.
 */
export function downloadDataUrl(dataUrl: string, filename: string): void {
  const link = document.createElement('a')
  link.href = dataUrl
  link.download = filename
  link.rel = 'noopener'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/**
 * Генерирует имя файла с датой.
 */
export function defaultExportFilename(prefix: string = 'visualizer'): string {
  const date = new Date()
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  const h = String(date.getHours()).padStart(2, '0')
  const min = String(date.getMinutes()).padStart(2, '0')
  return `${prefix}-${y}${m}${d}-${h}${min}.png`
}

interface ExportLegendMeta {
  material: Material | null
  color: Color | null
}

/**
 * Генерирует PNG с watermark и легендой.
 */
export function buildExportImageDataUrl(
  canvas: HTMLCanvasElement,
  meta: ExportLegendMeta
): string {
  const w = Math.max(1, canvas.width)
  const h = Math.max(1, canvas.height)
  const out = document.createElement('canvas')
  out.width = w
  out.height = h
  const ctx = out.getContext('2d')
  if (!ctx) return canvas.toDataURL('image/png')

  ctx.drawImage(canvas, 0, 0, w, h)

  const pad = Math.max(12, Math.round(Math.min(w, h) * 0.018))
  const radius = Math.max(8, Math.round(Math.min(w, h) * 0.012))
  const titleSize = Math.max(13, Math.min(20, Math.round(Math.min(w, h) * 0.022)))
  const bodySize = Math.max(12, Math.min(18, Math.round(Math.min(w, h) * 0.018)))
  const lineGap = Math.max(6, Math.round(bodySize * 0.45))

  const materialText = `Материал: ${meta.material?.name ?? '—'}`
  const colorText = `Цвет: ${meta.color?.name ?? meta.color?.hex ?? 'Без цвета'}`
  const watermark = 'leslab.ru'

  // Классический watermark: повтор по диагонали по всей картинке.
  const wmSize = Math.max(28, Math.min(64, Math.round(Math.min(w, h) * 0.055)))
  const stepX = Math.max(260, Math.round(wmSize * 6.0))
  const stepY = Math.max(200, Math.round(wmSize * 4.4))
  ctx.save()
  ctx.translate(w / 2, h / 2)
  ctx.rotate((-28 * Math.PI) / 180)
  ctx.font = `700 ${wmSize}px Inter, Arial, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = 'rgba(255, 255, 255, 0.12)'
  for (let yOff = -h * 1.4; yOff <= h * 1.4; yOff += stepY) {
    const rowShift = Math.round(yOff / stepY) % 2 === 0 ? 0 : stepX / 2
    for (let xOff = -w * 1.6; xOff <= w * 1.6; xOff += stepX) {
      ctx.fillText(watermark, xOff + rowShift, yOff)
    }
  }
  ctx.restore()

  ctx.font = `600 ${titleSize}px Inter, Arial, sans-serif`
  const w1 = ctx.measureText(materialText).width
  ctx.font = `500 ${bodySize}px Inter, Arial, sans-serif`
  const w2 = ctx.measureText(colorText).width
  const swatch = Math.max(12, Math.round(bodySize * 0.9))
  const legendW = Math.ceil(Math.max(w1, w2 + swatch + 8) + pad * 2)
  const legendH = Math.ceil(pad * 2 + titleSize + lineGap + bodySize)
  const legendX = w - legendW - pad
  const legendY = h - legendH - pad * 3

  ctx.fillStyle = 'rgba(17, 24, 39, 0.58)'
  ctx.beginPath()
  ctx.moveTo(legendX + radius, legendY)
  ctx.lineTo(legendX + legendW - radius, legendY)
  ctx.quadraticCurveTo(legendX + legendW, legendY, legendX + legendW, legendY + radius)
  ctx.lineTo(legendX + legendW, legendY + legendH - radius)
  ctx.quadraticCurveTo(legendX + legendW, legendY + legendH, legendX + legendW - radius, legendY + legendH)
  ctx.lineTo(legendX + radius, legendY + legendH)
  ctx.quadraticCurveTo(legendX, legendY + legendH, legendX, legendY + legendH - radius)
  ctx.lineTo(legendX, legendY + radius)
  ctx.quadraticCurveTo(legendX, legendY, legendX + radius, legendY)
  ctx.closePath()
  ctx.fill()

  ctx.fillStyle = 'rgba(255, 255, 255, 0.96)'
  ctx.font = `600 ${titleSize}px Inter, Arial, sans-serif`
  ctx.fillText(materialText, legendX + pad, legendY + pad + titleSize)

  const colorLineY = legendY + pad + titleSize + lineGap + bodySize
  ctx.fillStyle = meta.color?.hex ?? '#ffffff'
  ctx.fillRect(legendX + pad, colorLineY - swatch + 2, swatch, swatch)
  ctx.strokeStyle = 'rgba(255,255,255,0.85)'
  ctx.lineWidth = 1
  ctx.strokeRect(legendX + pad, colorLineY - swatch + 2, swatch, swatch)

  ctx.fillStyle = 'rgba(255, 255, 255, 0.96)'
  ctx.font = `500 ${bodySize}px Inter, Arial, sans-serif`
  ctx.fillText(colorText, legendX + pad + swatch + 8, colorLineY)

  return out.toDataURL('image/png')
}
