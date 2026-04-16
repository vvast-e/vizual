import type { WallData } from '@/store/useWallStore'

export interface DetectWallsResponse {
  walls: WallData[]
  image_size: { width: number; height: number }
}

export interface DetectExteriorResponse {
  walls: WallData[]
  image_size: { width: number; height: number }
  masks: {
    wall: string
    holes: string
    wall_minus_holes: string
  }
  debug: {
    building_bbox: number[] | null
    windows_bboxes: number[][]
    doors_bboxes: number[][]
    walls_count: number
    split_method?: string
    seam_debug?: Record<string, unknown>
    split_arbiter?: Record<string, unknown>
  }
}

/**
 * Запросить автоматический расчет перспективы с бэкенда (Вариант Б)
 */
export async function estimateHomographyFromBackend(
  polygon: [number, number][]
): Promise<[number, number][]> {
  const res = await fetch('/api/estimate-homography', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(polygon),
  })
  
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  }
  
  const data = await res.json()
  return data.corners
}

export interface DetectExteriorResponse {
  walls: WallData[]
  image_size: { width: number; height: number }
  masks: {
    wall: string
    holes: string
    wall_minus_holes: string
  }
  debug: {
    building_bbox: number[] | null
    windows_bboxes: number[][]
    doors_bboxes: number[][]
    walls_count: number
    split_method?: string
    seam_debug?: Record<string, unknown>
    split_arbiter?: Record<string, unknown>
  }
}

/**
 * Отправляет фото на бэкенд, возвращает определённые стены и размер обработанного изображения.
 * Координаты walls в системе координат image_size (фронт масштабирует под канвас при отрисовке).
 */
export async function detectWalls(photoFile: File): Promise<DetectWallsResponse> {
  const formData = new FormData()
  formData.append('file', photoFile)

  const res = await fetch('/api/detect-walls', {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  const data = (await res.json()) as DetectWallsResponse
  if (!data.walls || !Array.isArray(data.walls) || !data.image_size) {
    throw new Error('Invalid response: walls and image_size required')
  }
  return data
}

/**
 * Отправляет фото на бэкенд (экстерьер‑пайплайн: GroundingDINO + SAM).
 * Возвращает маски (wall, holes, wall_minus_holes) в base64 PNG.
 */
export async function detectExterior(photoFile: File): Promise<DetectExteriorResponse> {
  const formData = new FormData()
  formData.append('file', photoFile)

  const res = await fetch('/api/detect-exterior', {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  const data = (await res.json()) as DetectExteriorResponse
  if (!data.masks || !data.image_size) {
    throw new Error('Invalid response: masks and image_size required')
  }
  return data
}

export interface SplitExteriorResponse {
  walls: import('@/store/useWallStore').WallData[]
}

export interface SplitExteriorTargetWall {
  targetWallId: number
  walls: WallData[]
}

/**
 * Разделить фасад маской по линии (два клика). Координаты в image_size.
 * Если передан splitTarget — режется только полигон выбранной стены, остальные стены сохраняются.
 */
export async function splitExteriorWalls(
  maskBase64: string,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  imageWidth: number,
  imageHeight: number,
  splitTarget?: SplitExteriorTargetWall
): Promise<SplitExteriorResponse> {
  const formData = new FormData()
  formData.append('mask_base64', maskBase64)
  formData.append('x1', String(x1))
  formData.append('y1', String(y1))
  formData.append('x2', String(x2))
  formData.append('y2', String(y2))
  formData.append('image_width', String(imageWidth))
  formData.append('image_height', String(imageHeight))
  if (splitTarget != null) {
    formData.append('target_wall_id', String(splitTarget.targetWallId))
    formData.append('walls_json', JSON.stringify(splitTarget.walls))
  }

  const res = await fetch('/api/exterior/split', {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  const data = (await res.json()) as SplitExteriorResponse
  if (!data.walls || !Array.isArray(data.walls)) {
    throw new Error('Invalid response: walls required')
  }
  return data
}
