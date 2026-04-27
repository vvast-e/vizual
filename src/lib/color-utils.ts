/**
 * HSV-утилиты для перекраски текстур с сохранением фактуры.
 *
 * Алгоритм: заменяем H на целевой, сдвигаем S к целевому (blend),
 * V оставляем оригинальным — так сохраняются тени и рисунок текстуры.
 */

/** RGB → HSV. r,g,b ∈ [0,255], возвращает h ∈ [0,360), s,v ∈ [0,1] */
export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255
  const gf = g / 255
  const bf = b / 255
  const max = Math.max(rf, gf, bf)
  const min = Math.min(rf, gf, bf)
  const d = max - min
  let h = 0
  if (d > 0) {
    if (max === rf) h = ((gf - bf) / d + 6) % 6
    else if (max === gf) h = (bf - rf) / d + 2
    else h = (rf - gf) / d + 4
    h *= 60
  }
  const s = max === 0 ? 0 : d / max
  return [h, s, max]
}

/** HSV → RGB. h ∈ [0,360), s,v ∈ [0,1], возвращает [r,g,b] ∈ [0,255] */
export function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = v - c
  let r1 = 0, g1 = 0, b1 = 0
  if (h < 60) { r1 = c; g1 = x }
  else if (h < 120) { r1 = x; g1 = c }
  else if (h < 180) { g1 = c; b1 = x }
  else if (h < 240) { g1 = x; b1 = c }
  else if (h < 300) { r1 = x; b1 = c }
  else { r1 = c; b1 = x }
  return [
    Math.round((r1 + m) * 255),
    Math.round((g1 + m) * 255),
    Math.round((b1 + m) * 255),
  ]
}

/** Парсинг hex-цвета (#rrggbb) → [r, g, b] */
export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ]
}

/**
 * Применяет HSV-сдвиг цвета к ImageData текстуры.
 * - H заменяется на целевой
 * - S смешивается (lerp) с целевой насыщенностью по intensity
 * - V остаётся оригинальным (сохраняет фактуру, тени, рисунок)
 *
 * @param source — изображение или canvas с текстурой
 * @param targetHex — целевой цвет (#rrggbb)
 * @param intensity — интенсивность перекраски 0..1
 * @returns canvas с перекрашенной текстурой
 */
export function applyHsvColorShift(
  source: HTMLImageElement | HTMLCanvasElement,
  targetHex: string,
  intensity: number
): HTMLCanvasElement {
  const w = source instanceof HTMLImageElement ? source.naturalWidth : source.width
  const h = source instanceof HTMLImageElement ? source.naturalHeight : source.height
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return canvas

  ctx.drawImage(source, 0, 0)
  const imageData = ctx.getImageData(0, 0, w, h)
  const data = imageData.data

  const [tr, tg, tb] = hexToRgb(targetHex)
  const [tH, tS, tV] = rgbToHsv(tr, tg, tb)
  const t = Math.max(0, Math.min(1, intensity))

  // Если intensity=0, ничего не делаем.
  if (t < 0.001) return canvas

  // Безопасная граница для режима Hard Light, чтобы чистый белый или чёрный
  // цвет не делал текстуру полностью плоской (не убивал весь контраст).
  const safeTv = Math.max(0.1, Math.min(0.9, tV))

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    // alpha не трогаем

    const [, oS, oV] = rgbToHsv(r, g, b)

    // Новый H = целевой H
    const newH = tH
    // Новый S = lerp(originalS, targetS, intensity)
    const newS = oS + (tS - oS) * t
    
    // Сдвиг яркости (V) через режим Hard Light
    let blendV = oV
    if (safeTv < 0.5) {
      blendV = 2.0 * oV * safeTv
    } else {
      blendV = 1.0 - 2.0 * (1.0 - oV) * (1.0 - safeTv)
    }
    const newV = oV + (blendV - oV) * t

    const [nr, ng, nb] = hsvToRgb(newH, newS, newV)
    data[i] = nr
    data[i + 1] = ng
    data[i + 2] = nb
  }

  ctx.putImageData(imageData, 0, 0)
  return canvas
}
