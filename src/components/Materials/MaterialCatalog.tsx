import { useMaterialStore } from '@/store/useMaterialStore'
import { DEFAULT_MATERIALS } from '@/lib/materials-catalog'
import { MaterialCard } from './MaterialCard'
import type { Material } from '@/types/material'

export interface MaterialCatalogProps {
  materials?: Material[]
  className?: string
}

export function MaterialCatalog({
  materials = DEFAULT_MATERIALS,
  className = '',
}: MaterialCatalogProps) {
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const setSelectedMaterial = useMaterialStore((s) => s.setSelectedMaterial)

  const selectedId = selectedMaterial?.id ?? null

  return (
    <div className={className} role="listbox" aria-label="Каталог материалов">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
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
