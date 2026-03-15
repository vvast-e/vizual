import { memo, useEffect, useState } from 'react'
import type { Texture } from '@/types/material'
import { loadTextureImage } from '@/lib/texture-processor'

export interface TexturePreviewProps {
  texture: Texture
  width?: number
  height?: number
  className?: string
  alt?: string
}

function TexturePreviewComponent({
  texture,
  width = 80,
  height = 80,
  className = '',
  alt,
}: TexturePreviewProps) {
  const [src, setSrc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)

  const url = texture.thumbnailUrl ?? texture.url

  useEffect(() => {
    setFailed(false)
    setSrc(null)
    if (!url) return
    loadTextureImage(url)
      .then((img) => setSrc(img.src))
      .catch(() => setFailed(true))
  }, [url])

  if (failed || !src) {
    return (
      <div
        className={`flex items-center justify-center rounded border border-gray-200 bg-gray-100 text-gray-400 ${className}`}
        style={{ width, height }}
        aria-label={texture.name}
      >
        {failed ? '?' : '…'}
      </div>
    )
  }

  return (
    <img
      src={src}
      alt={alt ?? texture.name}
      width={width}
      height={height}
      className={`rounded border border-gray-200 object-cover ${className}`}
      loading="lazy"
    />
  )
}

export const TexturePreview = memo(TexturePreviewComponent)
