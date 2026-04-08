import { useEffect, useState } from 'react'
import { useMaterialStore } from '@/store/useMaterialStore'
import { DEFAULT_MATERIALS } from '@/lib/materials-catalog'
import { fetchMaterials } from '@/lib/materials-api'
import { MaterialCard } from './MaterialCard'
import type { Material } from '@/types/material'
import type { MaterialDTO } from '@/lib/materials-api'

/** Convert backend DTO to frontend Material type */
function dtoToMaterial(dto: MaterialDTO): Material {
  return {
    id: dto.id,
    name: dto.name,
    category: (dto.category || 'other') as Material['category'],
    texture: {
      id: `tex-${dto.id}`,
      url: dto.url,
      name: dto.name,
    },
    presetColors: [],
  }
}

export interface MaterialCatalogProps {
  materials?: Material[]
  className?: string
}

export function MaterialCatalog({
  materials: propMaterials,
  className = '',
}: MaterialCatalogProps) {
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const setSelectedMaterial = useMaterialStore((s) => s.setSelectedMaterial)
  const [apiMaterials, setApiMaterials] = useState<Material[] | null>(null)

  useEffect(() => {
    if (propMaterials) return // если переданы пропсом — не загружаем
    let cancelled = false
    fetchMaterials()
      .then((list) => {
        if (!cancelled && list.length > 0) {
          setApiMaterials(list.map(dtoToMaterial))
        }
      })
      .catch(() => {
        // fallback to defaults silently
      })
    return () => { cancelled = true }
  }, [propMaterials])

  const materials = propMaterials ?? apiMaterials ?? DEFAULT_MATERIALS
  const selectedId = selectedMaterial?.id ?? null

  return (
    <div className={className} role="listbox" aria-label="Каталог материалов">
      <div className="grid grid-cols-2 gap-2">
        {materials.map((material) => (
          <MaterialCard
            key={material.id}
            material={material}
            selected={selectedId === material.id}
            onSelect={() => setSelectedMaterial(material)}
          />
        ))}
      </div>
    </div>
  )
}
