import { useEffect, useMemo, useState } from 'react'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { MOCK_MATERIALS, MATERIAL_CATEGORIES, type MockMaterialCategory } from '@/data/mock-materials'
import { fetchMaterials, materialDtoToMaterial } from '@/lib/materials-api'
import { TexturePreview } from './TexturePreview'
import type { Material } from '@/types/material'

interface MaterialWidgetProps {
  open: boolean
  onClose: () => void
}

export function MaterialWidget({ open, onClose }: MaterialWidgetProps) {
  const sceneMode = useUIStore((s) => s.sceneMode)
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState<string>('all')
  const [apiMaterials, setApiMaterials] = useState<Material[] | null>(null)
  const setSelectedMaterial = useMaterialStore((s) => s.setSelectedMaterial)
  const addQuickAccessMaterial = useMaterialStore((s) => s.addQuickAccessMaterial)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setApiMaterials(null)
    fetchMaterials({
      visible_only: true,
      scene_category: sceneMode,
      page_size: 100,
    })
      .then((list) => {
        if (cancelled) return
        setApiMaterials(list.map(materialDtoToMaterial))
      })
      .catch(() => {
        if (!cancelled) setApiMaterials([])
      })
    return () => {
      cancelled = true
    }
  }, [open, sceneMode])

  const baseMaterials = useMemo(() => {
    if (apiMaterials === null) return null
    if (apiMaterials.length > 0) return apiMaterials
    return MOCK_MATERIALS
  }, [apiMaterials])

  const categoryOptions = useMemo(() => {
    if (apiMaterials && apiMaterials.length > 0) {
      const uniq = [...new Set(apiMaterials.map((m) => m.category))].sort()
      return [{ value: 'all' as const, label: 'Все' }, ...uniq.map((c) => ({ value: c, label: c }))]
    }
    return MATERIAL_CATEGORIES as { value: MockMaterialCategory | 'all'; label: string }[]
  }, [apiMaterials])

  const filtered = useMemo(() => {
    if (!baseMaterials) return []
    return baseMaterials.filter((m) => {
      const matchSearch = !search || m.name.toLowerCase().includes(search.toLowerCase())
      const matchCategory = category === 'all' || m.category === category
      return matchSearch && matchCategory
    })
  }, [baseMaterials, search, category])

  const handleSelect = (material: Material) => {
    setSelectedMaterial(material)
    addQuickAccessMaterial(material)
    onClose()
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-[600px] max-h-[80vh] rounded-xl bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-5 py-4">
          <h2 className="text-lg font-semibold text-gray-900">Материалы</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-5 py-3">
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Поиск материалов..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-lg border border-gray-200 bg-white py-2.5 pl-10 pr-4 text-sm text-gray-800 outline-none placeholder:text-gray-400 focus:border-gray-400 focus:ring-1 focus:ring-gray-200"
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-2 px-5 pb-3">
          {categoryOptions.map((cat) => (
            <button
              key={cat.value}
              type="button"
              onClick={() => setCategory(cat.value)}
              className={`rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
                category === cat.value
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        <div className="max-h-[50vh] overflow-y-auto px-5 pb-5">
          {baseMaterials === null ? (
            <p className="py-8 text-center text-gray-500">Загрузка каталога…</p>
          ) : filtered.length === 0 ? (
            <p className="py-8 text-center text-gray-500">Материалы не найдены</p>
          ) : (
            <div className="grid grid-cols-3 gap-3">
              {filtered.map((material) => (
                <button
                  key={material.id}
                  type="button"
                  onClick={() => handleSelect(material)}
                  className="flex flex-col items-center gap-2 rounded-lg border-2 border-gray-200 p-3 transition-colors hover:border-gray-400 hover:bg-gray-50"
                >
                  <TexturePreview texture={material.texture} width={64} height={64} />
                  <span className="text-xs font-medium text-gray-700">{material.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
