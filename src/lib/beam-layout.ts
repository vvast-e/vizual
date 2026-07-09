import type { WallData } from '@/store/useWallStore'

export type BeamDirection = 'depth' | 'width' | 'grid'

/** Одна балка — quad из 4 точек в координатах wallImageSize */
export interface BeamQuad {
  id: number
  quad: [number, number][]
}

function lerp2(
  a: [number, number],
  b: [number, number],
  t: number,
): [number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

function edgeLen(a: [number, number], b: [number, number]): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

function depthBeams(
  TL: [number, number], BL: [number, number], BR: [number, number], TR: [number, number],
  n: number, hw: number, idOffset = 0,
): BeamQuad[] {
  const result: BeamQuad[] = []
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n
    const topLeft  = lerp2(TL, TR, t - hw)
    const topRight = lerp2(TL, TR, t + hw)
    const botLeft  = lerp2(BL, BR, t - hw)
    const botRight = lerp2(BL, BR, t + hw)
    result.push({ id: idOffset + i + 1, quad: [topLeft, botLeft, botRight, topRight] })
  }
  return result
}

function widthBeams(
  TL: [number, number], BL: [number, number], BR: [number, number], TR: [number, number],
  n: number, hw: number, idOffset = 0,
): BeamQuad[] {
  const result: BeamQuad[] = []
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n
    const leftTop  = lerp2(TL, BL, t - hw)
    const leftBot  = lerp2(TL, BL, t + hw)
    const rightTop = lerp2(TR, BR, t - hw)
    const rightBot = lerp2(TR, BR, t + hw)
    result.push({ id: idOffset + i + 1, quad: [leftTop, leftBot, rightBot, rightTop] })
  }
  return result
}

/**
 * Генерирует N балок поверх quad потолка [TL, BL, BR, TR].
 * direction='depth'  → балки идут вдоль (от дальней стены к зрителю)
 * direction='width'  → балки идут поперёк
 * direction='grid'   → сетка: N вдоль + N поперёк (кессонный потолок)
 * halfWidth — половина ширины балки в долях [0..0.5]
 *
 * Для direction='depth' полуширина пересчитывается так, чтобы абсолютная
 * ширина балки совпадала с direction='width' при том же halfWidth.
 * Это нужно потому, что «вдоль» использует горизонтальный размах (TL→TR),
 * а «поперёк» — вертикальный (TL→BL); на перспективно сжатом потолке
 * они существенно различаются, и без коррекции балки «вдоль» получаются
 * широкими (почти квадратными), а не узкими длинными.
 */
export function autoLayoutBeams(
  ceiling: WallData,
  count: number,
  halfWidth: number,
  direction: BeamDirection,
): BeamQuad[] {
  if (ceiling.corners.length < 4) return []

  const [TL, BL, BR, TR] = ceiling.corners as [
    [number, number],
    [number, number],
    [number, number],
    [number, number],
  ]

  const n = Math.max(1, Math.round(count))
  const hw = Math.max(0.01, Math.min(0.45, halfWidth))

  if (direction === 'width') return widthBeams(TL, BL, BR, TR, n, hw)

  // Средние размахи горизонтального и вертикального рёбер потолка
  const horizSpan = (edgeLen(TL, TR) + edgeLen(BL, BR)) / 2
  const vertSpan  = (edgeLen(TL, BL) + edgeLen(TR, BR)) / 2

  // Скорректированная полуширина для «вдоль»: hw * vertSpan — целевая абсолютная ширина
  // (такая же, как у «поперёк»), делённая на horizSpan даёт долю для горизонтального ребра.
  // Нижний clamp 0.002 (не 0.01) — иначе ползунок застывает на мелких значениях.
  const hwDepth = Math.max(0.002, Math.min(0.45, hw * (vertSpan / Math.max(1, horizSpan))))

  if (direction === 'depth') return depthBeams(TL, BL, BR, TR, n, hwDepth)

  // grid: обе направления, id уникальны; «поперёк» остаётся с исходным hw
  return [
    ...depthBeams(TL, BL, BR, TR, n, hwDepth, 0),
    ...widthBeams(TL, BL, BR, TR, n, hw, n),
  ]
}
