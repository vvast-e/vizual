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
