import type { WallData } from '@/store/useWallStore'

export interface DetectWallsResponse {
  walls: WallData[]
  image_size: { width: number; height: number }
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
