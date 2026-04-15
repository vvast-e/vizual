import type { WallWarpRegion } from '@/store/useWallStore'

export interface WarpWallTextureArgs {
  textureUrl: string
  corners: [number, number][]
  polygon?: [number, number][]
  /** Два региона с бэкенда (нижний quad + верхний треугольник/quad). */
  regions?: WallWarpRegion[]
  imageSize: { width: number; height: number }
  textureScale: number
  opacity?: number
  /** Растровая маска (Base64 PNG) с вырезанными окнами/дверьми */
  maskBase64?: string | null
}

export async function warpWallTexture({
  textureUrl,
  corners,
  polygon,
  regions,
  imageSize,
  textureScale,
  opacity = 1,
  maskBase64,
}: WarpWallTextureArgs): Promise<Blob> {
  const texRes = await fetch(textureUrl)
  if (!texRes.ok) throw new Error(`Texture fetch failed: HTTP ${texRes.status}`)
  const texBlob = await texRes.blob()

  const form = new FormData()
  form.append('texture', texBlob, 'texture.png')
  form.append('corners', JSON.stringify(corners))
  if (polygon && polygon.length >= 3) {
    form.append('polygon', JSON.stringify(polygon))
  }
  if (regions && regions.length === 2) {
    form.append('regions', JSON.stringify(regions))
  }
  if (maskBase64) {
    form.append('mask_base64', maskBase64)
  }
  form.append('image_width', String(imageSize.width))
  form.append('image_height', String(imageSize.height))
  form.append('texture_scale', String(textureScale))
  void opacity
  form.append('opacity', '1')

  const res = await fetch('/api/warp-wall-texture', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `HTTP ${res.status}`)
  }
  return await res.blob()
}

