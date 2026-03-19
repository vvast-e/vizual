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

/**
 * Разделить фасад маской по линии (два клика). Координаты в image_size.
 */
export async function splitExteriorWalls(
  maskBase64: string,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  imageWidth: number,
  imageHeight: number
): Promise<SplitExteriorResponse> {
  const formData = new FormData()
  formData.append('mask_base64', maskBase64)
  formData.append('x1', String(x1))
  formData.append('y1', String(y1))
  formData.append('x2', String(x2))
  formData.append('y2', String(y2))
  formData.append('image_width', String(imageWidth))
  formData.append('image_height', String(imageHeight))

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
