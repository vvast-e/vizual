/**
 * Утилиты для перекраски текстур с сохранением фактуры.
 *
 * Алгоритм: HSL Color Transfer — переносим тон и насыщенность краски,
 * а яркость текстуры центрируем вокруг яркости краски, сохраняя рисунок
 * (сучки, волокна, трещины) как отклонения от средней яркости.
 *
 * Имитирует реальную физику морилки/масла:
 * - тёмные краски (больше пигмента) → слабее видна текстура
 * - светлые краски (тонкий слой) → текстура просвечивает сильнее
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

/** RGB → HSL. r,g,b ∈ [0,255], возвращает h ∈ [0,360), s,l ∈ [0,1] */
export function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255
  const gf = g / 255
  const bf = b / 255
  const max = Math.max(rf, gf, bf)
  const min = Math.min(rf, gf, bf)
  const l = (max + min) / 2

  if (max === min) return [0, 0, l]

  const d = max - min
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min)

  let h = 0
  if (max === rf) h = (gf - bf) / d + (gf < bf ? 6 : 0)
  else if (max === gf) h = (bf - rf) / d + 2
  else h = (rf - gf) / d + 4
  h *= 60

  return [h, s, l]
}

/** HSL → RGB. h ∈ [0,360), s,l ∈ [0,1], возвращает [r,g,b] ∈ [0,255] */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) {
    const v = Math.round(l * 255)
    return [v, v, v]
  }

  const hueToChannel = (p: number, q: number, t: number): number => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const hNorm = h / 360

  return [
    Math.round(hueToChannel(p, q, hNorm + 1 / 3) * 255),
    Math.round(hueToChannel(p, q, hNorm) * 255),
    Math.round(hueToChannel(p, q, hNorm - 1 / 3) * 255),
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
 * Перекраска текстуры методом HSL Color Transfer.
 *
 * Имитирует нанесение морилки/масла на дерево:
 * 1. Вычисляем среднюю яркость текстуры (один раз)
 * 2. Для каждого пикселя находим «деталь» = отклонение от средней яркости
 *    (сучки, волокна, трещины — это всё отклонения)
 * 3. Новая яркость = яркость краски + деталь × preserveFactor
 * 4. Тон и насыщенность берём от краски
 *
 * preserveFactor адаптивный:
 * - тёмные краски (много пигмента) → preserveFactor ≈ 0.30 (текстура слабо видна)
 * - светлые краски (тонкий слой) → preserveFactor ≈ 0.61 (текстура хорошо видна)
 *
 * @param source — изображение или canvas с текстурой
 * @param targetHex — целевой цвет покраски (#rrggbb)
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

  const [pr, pg, pb] = hexToRgb(targetHex)
  const [paintH, paintS, paintL] = rgbToHsl(pr, pg, pb)
  const t = Math.max(0, Math.min(1, intensity))

  if (t < 0.001) return canvas

  // ─── Проход 1: средняя яркость текстуры ───
  // Используем быструю формулу HSL-lightness: (max + min) / 2 / 255
  let sumL = 0
  let count = 0
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 10) continue // пропускаем прозрачные
    const pixMax = Math.max(data[i], data[i + 1], data[i + 2])
    const pixMin = Math.min(data[i], data[i + 1], data[i + 2])
    sumL += (pixMax + pixMin) / 510
    count++
  }
  const meanTexL = count > 0 ? sumL / count : 0.5

  // ─── Адаптивный коэффициент сохранения текстуры ───
  // Физика: тёмная морилка = много пигмента → текстура менее заметна
  //         светлая побелка = тонкий слой → текстура просвечивает
  const basePF = 0.65
  const preserveFactor = basePF * (0.4 + 0.6 * paintL)

  // ─── Проход 2: перекраска ───
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 10) continue

    const r = data[i], g = data[i + 1], b = data[i + 2]

    // Яркость текущего пикселя (HSL lightness, быстрая формула)
    const pixMax = Math.max(r, g, b)
    const pixMin = Math.min(r, g, b)
    const texL = (pixMax + pixMin) / 510

    // Деталь текстуры = отклонение от средней яркости
    // сучок (тёмный) → detail < 0, светлая полоса → detail > 0
    const detail = texL - meanTexL

    // Новая яркость: центрируем рисунок вокруг яркости краски
    const newL = Math.max(0, Math.min(1, paintL + detail * preserveFactor))

    // Тон + насыщенность от краски, яркость с деталями текстуры
    const [nr, ng, nb] = hslToRgb(paintH, paintS, newL)

    // Смешиваем с оригиналом по intensity
    data[i]     = Math.round(r + (nr - r) * t)
    data[i + 1] = Math.round(g + (ng - g) * t)
    data[i + 2] = Math.round(b + (nb - b) * t)
  }

  ctx.putImageData(imageData, 0, 0)
  return canvas
}
