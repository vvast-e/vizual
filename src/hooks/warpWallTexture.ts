export interface WarpWallTextureArgs {
  textureUrl: string
  corners: [number, number][]
  polygon?: [number, number][]
  imageSize: { width: number; height: number }
  textureScale: number
  opacity?: number
}

export async function warpWallTexture({
  textureUrl,
  corners,
  polygon,
  imageSize,
  textureScale,
  opacity = 0.85,
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
  form.append('image_width', String(imageSize.width))
  form.append('image_height', String(imageSize.height))
  form.append('texture_scale', String(textureScale))
  form.append('opacity', String(opacity))

  const res = await fetch('/api/warp-wall-texture', { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `HTTP ${res.status}`)
  }
  return await res.blob()
}

