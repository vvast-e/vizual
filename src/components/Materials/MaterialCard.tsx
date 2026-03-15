import { memo } from 'react'
import type { Material } from '@/types/material'
import { TexturePreview } from './TexturePreview'

export interface MaterialCardProps {
  material: Material
  selected?: boolean
  onSelect?: () => void
  className?: string
}

function MaterialCardComponent({ material, selected, onSelect, className = '' }: MaterialCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      aria-label={`Выбрать материал: ${material.name}`}
      className={
        'flex flex-col items-center gap-2 rounded-lg border-2 p-3 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400 ' +
        (selected
          ? 'border-gray-800 bg-gray-100'
          : 'border-gray-200 bg-white hover:border-gray-400 hover:bg-gray-50') +
        ' ' +
        className
      }
    >
      <TexturePreview texture={material.texture} width={72} height={72} />
      <span className="text-sm font-medium text-gray-800">{material.name}</span>
    </button>
  )
}

export const MaterialCard = memo(MaterialCardComponent)
