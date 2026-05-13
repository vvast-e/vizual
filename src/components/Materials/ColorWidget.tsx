import { useEffect, useMemo, useRef, useState } from 'react'
import { useColorize } from '@/hooks/useColorize'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useUIStore } from '@/store/useUIStore'
import { colorDtoToColor, fetchColors } from '@/lib/materials-api'
import type { Color } from '@/types/material'

interface ColorWidgetProps {
  open: boolean
  onClose: () => void
}

export function ColorWidget({ open, onClose }: ColorWidgetProps) {
  const sceneMode = useUIStore((s) => s.sceneMode)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [colors, setColors] = useState<Color[]>([])

  const { setSelectedColor } = useColorize()
  const addQuickAccessColor = useMaterialStore((s) => s.addQuickAccessColor)
  const searchRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      const timer = setTimeout(() => searchRef.current?.focus(), 50)
      return () => clearTimeout(timer)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setLoading(true)
    fetchColors({
      visible_only: true,
      scene_category: sceneMode,
      page_size: 100,
    })
      .then(async (list) => {
        if (cancelled) return
        const coloredList = await Promise.all(list.map(colorDtoToColor))
        if (cancelled) return
        setColors(coloredList)
        setLoading(false)
      })
      .catch(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, sceneMode])

  const filtered = useMemo(() => {
    return colors.filter((c) => {
      return !search || c.name?.toLowerCase().includes(search.toLowerCase())
    })
  }, [colors, search])

  const handleSelect = (color: Color) => {
    setSelectedColor(color)
    addQuickAccessColor(color)
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
          <h2 className="text-lg font-semibold text-gray-900">Цвета</h2>
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
              ref={searchRef}
              type="text"
              placeholder="Поиск цветов..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Escape') onClose() }}
              className="w-full rounded-lg border border-gray-200 bg-white py-2.5 pl-10 pr-4 text-sm text-gray-800 outline-none placeholder:text-gray-400 focus:border-gray-400 focus:ring-1 focus:ring-gray-200"
            />
          </div>
        </div>

        <div className="max-h-[50vh] overflow-y-auto px-5 pb-5">
          {loading ? (
            <p className="py-8 text-center text-gray-500">Загрузка каталога…</p>
          ) : filtered.length === 0 ? (
            <p className="py-8 text-center text-gray-500">Цвета не найдены</p>
          ) : (
            <div className="grid grid-cols-5 gap-3">
              {filtered.map((color, idx) => (
                <button
                  key={color.id ?? `${color.hex}-${idx}`}
                  type="button"
                  onClick={() => handleSelect(color)}
                  className="flex flex-col items-center gap-2 rounded-lg border-2 border-gray-200 p-2 transition-colors hover:border-gray-400 hover:bg-gray-50"
                >
                  {color.swatchUrl ? (
                    <img
                      src={color.swatchUrl}
                      alt=""
                      className="h-16 w-16 rounded-lg border border-gray-300 object-cover shadow-sm"
                    />
                  ) : (
                    <div
                      className="h-16 w-16 rounded-lg border border-gray-300 shadow-sm"
                      style={{ backgroundColor: color.hex }}
                    />
                  )}
                  <span className="text-xs font-medium text-gray-700">{color.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
