/**
 * Алгоритм вычисления перспективных углов из формы (полигона)
 * 
 * Идея: прямоугольная стена в 3D пространстве проецируется на 2D как произвольный четырехугольник.
 * Зная форму четырехугольника, можно вычислить "правильные" перспективные углы.
 * 
 * Алгоритм для многоточечного полигона (5+ точек):
 * 1. Разделение точек на левую и правую границы по медиане X
 * 2. Фильтрация точек крыши (фронтона) по углу наклона
 * 3. Определение вертикалей (крайние точки каждой группы)
 * 4. Определение боковых сторон (нижняя и верхняя горизонтали)
 * 5. Вычисление 4 вершин трапеции через пересечение линий
 */

import type { WallData } from '../store/useWallStore'

const ROOF_HEIGHT_RATIO = 0.3

export function inferPerspectiveCorners(
  polygon: [number, number][]
): [number, number][] {
  if (polygon.length < 3) {
    return polygon as [number, number][]
  }

  if (polygon.length === 3) {
    return inferFromTriangle(polygon)
  }

  if (polygon.length === 4) {
    return inferFromQuadrilateral(polygon)
  }

  return inferFromMultiPointPolygon(polygon)
}

function inferFromMultiPointPolygon(polygon: [number, number][]): [number, number][] {
  const points = polygon as [number, number][]

  const sortedByX = [...points].sort((a, b) => a[0] - b[0])
  const minX = sortedByX[0][0]
  const maxX = sortedByX[sortedByX.length - 1][0]
  const medianX = (minX + maxX) / 2

  const leftPoints: [number, number][] = []
  const rightPoints: [number, number][] = []

  for (const p of points) {
    if (p[0] < medianX) {
      leftPoints.push(p)
    } else {
      rightPoints.push(p)
    }
  }

  leftPoints.sort((a, b) => a[1] - b[1])
  rightPoints.sort((a, b) => a[1] - b[1])

  const leftFiltered = filterRoofPoints(leftPoints)
  const rightFiltered = filterRoofPoints(rightPoints)

  const leftBottom = leftFiltered[0]
  const leftTop = leftFiltered[leftFiltered.length - 1]
  const rightBottom = rightFiltered[0]
  const rightTop = rightFiltered[rightFiltered.length - 1]

  return sortCornersClockwise([
    [leftTop[0], leftTop[1]],
    [rightTop[0], rightTop[1]],
    [rightBottom[0], rightBottom[1]],
    [leftBottom[0], leftBottom[1]],
  ])
}

function filterRoofPoints(points: [number, number][]): [number, number][] {
  if (points.length < 3) {
    return points
  }

  const sortedByY = [...points].sort((a, b) => a[1] - b[1])
  const rangeY = sortedByY[sortedByY.length - 1][1] - sortedByY[0][1]
  const roofThreshold = sortedByY[0][1] + rangeY * ROOF_HEIGHT_RATIO

  return sortedByY.filter(p => p[1] < roofThreshold)
}

function inferFromTriangle(polygon: [number, number][]): [number, number][] {
  const edges = [
    { len: dist(polygon[0], polygon[1]), idx: [0, 1] },
    { len: dist(polygon[1], polygon[2]), idx: [1, 2] },
    { len: dist(polygon[2], polygon[0]), idx: [2, 0] },
  ].sort((a, b) => b.len - a.len)

  const bottomEdge = edges[0].idx
  const [p1, p2] = bottomEdge.map(i => polygon[i]) as [[number, number], [number, number]]
  const midBottom: [number, number] = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
  const height = edges[0].len * 0.8
  const topPoint: [number, number] = [midBottom[0], midBottom[1] - height]

  return sortCornersClockwise([
    topPoint,
    [p1[0] + (p2[0] - p1[0]) * 0.3, topPoint[1]] as [number, number],
    p2,
    p1,
  ])
}

function inferFromQuadrilateral(polygon: [number, number][]): [number, number][] {
  const quad = polygon.slice(0, 4) as [number, number][]
  const metrics = computeQuadMetrics(quad)

  if (metrics.isConvex && metrics.aspectRatio > 0.3 && metrics.aspectRatio < 3) {
    return sortCornersClockwise(quad)
  }

  return correctQuadToPerspective(quad, metrics)
}

function computeQuadMetrics(quad: [number, number][]) {
  const edges = [
    dist(quad[0], quad[1]),
    dist(quad[1], quad[2]),
    dist(quad[2], quad[3]),
    dist(quad[3], quad[0]),
  ]

  const convex = isConvexQuad(quad)
  const avgHorizontal = (edges[0] + edges[2]) / 2
  const avgVertical = (edges[1] + edges[3]) / 2
  const aspectRatio = avgHorizontal / (avgVertical || 1)

  return { isConvex: convex, aspectRatio, edges }
}

function isConvexQuad(quad: [number, number][]): boolean {
  const crossProducts = []
  for (let i = 0; i < 4; i++) {
    const p1 = quad[i]
    const p2 = quad[(i + 1) % 4]
    const p3 = quad[(i + 2) % 4]
    const v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    const v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    crossProducts.push(v1[0] * v2[1] - v1[1] * v2[0])
  }
  const hasPos = crossProducts.some(c => c > 0)
  const hasNeg = crossProducts.some(c => c < 0)
  return !(hasPos && hasNeg)
}

function correctQuadToPerspective(
  quad: [number, number][],
  metrics: ReturnType<typeof computeQuadMetrics>
): [number, number][] {
  const isHorizontalLonger = metrics.edges[0] + metrics.edges[2] > metrics.edges[1] + metrics.edges[3]
  
  if (!isHorizontalLonger) {
    return sortCornersClockwise(quad)
  }

  const bottomWidth = dist(quad[0], quad[3])
  const topWidth = dist(quad[1], quad[2])
  const ratio = topWidth / (bottomWidth || 1)

  if (ratio > 0.7) {
    return sortCornersClockwise(quad)
  }

  const center = quad.reduce((acc, p) => [acc[0] + p[0] / 4, acc[1] + p[1] / 4], [0, 0] as [number, number])
  const avgWidth = (bottomWidth + topWidth) / 2
  const avgHeight = (metrics.edges[1] + metrics.edges[3]) / 2
  const perspectiveScale = 1.2

  return sortCornersClockwise([
    [center[0] - avgWidth / 2 / perspectiveScale, center[1] - avgHeight / 2],
    [center[0] + avgWidth / 2 / perspectiveScale, center[1] - avgHeight / 2],
    [center[0] + avgWidth / 2, center[1] + avgHeight / 2],
    [center[0] - avgWidth / 2, center[1] + avgHeight / 2],
  ])
}

function sortCornersClockwise(corners: [number, number][]): [number, number][] {
  const center = corners.reduce((acc, c) => [acc[0] + c[0] / 4, acc[1] + c[1] / 4], [0, 0] as [number, number])
  
  const sorted = [...corners].sort((a, b) => {
    const angleA = Math.atan2(a[1] - center[1], a[0] - center[0])
    const angleB = Math.atan2(b[1] - center[1], b[0] - center[0])
    return angleA - angleB
  })

  const sums = sorted.map(c => c[0] + c[1])
  const minSumIdx = sums.indexOf(Math.min(...sums))

  const result: [number, number][] = []
  for (let i = 0; i < 4; i++) {
    result.push(sorted[(minSumIdx + i) % 4])
  }
  return result
}

function dist(p1: number[], p2: number[]): number {
  return Math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
}

export function isValidPerspectiveShape(
  polygon: [number, number][]
): { valid: boolean; reason?: string } {
  if (polygon.length < 3) {
    return { valid: false, reason: 'Too few points' }
  }

  const quad = polygon.slice(0, 4) as [number, number][]
  const area = computePolygonArea(quad)
  if (area < 1000) {
    return { valid: false, reason: 'Too small area' }
  }

  return { valid: true }
}

function computePolygonArea(quad: [number, number][]): number {
  let area = 0
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4
    area += quad[i][0] * quad[j][1]
    area -= quad[j][0] * quad[i][1]
  }
  return Math.abs(area / 2)
}

export function autoLinkPerspectiveToForm(wall: WallData): [number, number][] {
  if (!wall.polygon || wall.polygon.length < 3) {
    return wall.corners
  }

  try {
    return inferPerspectiveCorners(wall.polygon)
  } catch {
    return wall.corners
  }
}
