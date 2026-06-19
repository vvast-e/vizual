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
  const [state, setState] = useState<{
    url: string | null
    src: string | null
    failed: boolean
  }>({ url: null, src: null, failed: false })

  const url = texture.thumbnailUrl ?? texture.url

  useEffect(() => {
    if (!url) return
    let cancelled = false
    loadTextureImage(url)
      .then((img) => {
        if (cancelled) return
        setState({ url, src: img.src, failed: false })
      })
      .catch(() => {
        if (cancelled) return
        setState({ url, src: null, failed: true })
      })
    return () => {
      cancelled = true
    }
  }, [url])

  const isLoadingThisUrl = state.url !== url
  if (isLoadingThisUrl || state.failed || !state.src) {
    return (
      <div
        className={`flex items-center justify-center rounded border border-gray-200 bg-gray-100 text-gray-400 ${className}`}
        style={{ width, height }}
        aria-label={texture.name}
      >
        {state.failed ? '?' : '…'}
      </div>
    )
  }

  return (
    <img
      src={state.src}
      alt={alt ?? texture.name}
      width={width}
      height={height}
      className={`rounded border border-gray-200 object-cover ${className}`}
      loading="lazy"
    />
  )
}

export const TexturePreview = memo(TexturePreviewComponent)
