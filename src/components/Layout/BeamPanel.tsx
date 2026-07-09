import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, LayoutList, Sparkles, Layers } from 'lucide-react'
import { useBeamStore } from '@/store/useBeamStore'
import { useWallStore } from '@/store/useWallStore'
import { useUIStore } from '@/store/useUIStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { fetchMaterials, materialDtoToMaterial } from '@/lib/materials-api'
import type { Material } from '@/types/material'

export function BeamPanel() {
  const sceneMode = useUIStore((s) => s.sceneMode)
  const walls = useWallStore((s) => s.walls)
  const ceiling = walls.find((w) => w.surface === 'ceiling')

  const enabled = useBeamStore((s) => s.enabled)
  const setEnabled = useBeamStore((s) => s.setEnabled)
  const count = useBeamStore((s) => s.count)
  const setCount = useBeamStore((s) => s.setCount)
  const halfWidth = useBeamStore((s) => s.halfWidth)
  const setHalfWidth = useBeamStore((s) => s.setHalfWidth)
  const direction = useBeamStore((s) => s.direction)
  const setDirection = useBeamStore((s) => s.setDirection)
  const autoLayout = useBeamStore((s) => s.autoLayout)
  const clear = useBeamStore((s) => s.clear)
  const beamTextureUrl = useBeamStore((s) => s.textureUrl)
  const setTextureUrl = useBeamStore((s) => s.setTextureUrl)

  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)

  const [collapsed, setCollapsed] = useState(false)
  const [beamCatalog, setBeamCatalog] = useState<Material[]>([])

  // Загружаем материалы с material_category='beam' для выбора в панели
  useEffect(() => {
    if (sceneMode !== 'interior') return
    let cancelled = false
    fetchMaterials({ scene_category: 'interior', material_category: 'beam', visible_only: true })
      .then((list) => {
        if (!cancelled) setBeamCatalog(list.map(materialDtoToMaterial))
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [sceneMode])

  /** Подтянуть текущий выбранный материал как материал балок */
  const handlePickMaterial = useCallback(() => {
    if (selectedMaterial?.texture.url) {
      setTextureUrl(selectedMaterial.texture.url)
    }
  }, [selectedMaterial, setTextureUrl])

  const ensureTexture = useCallback(() => {
    if (!beamTextureUrl && selectedMaterial?.texture.url) {
      setTextureUrl(selectedMaterial.texture.url)
    }
  }, [beamTextureUrl, selectedMaterial, setTextureUrl])

  const handleAutoLayout = useCallback(() => {
    if (!ceiling) return
    ensureTexture()
    autoLayout(ceiling)
    if (!enabled) setEnabled(true)
  }, [ceiling, autoLayout, enabled, setEnabled, ensureTexture])

  // Живой пересчёт раскладки при изменении ползунков/направления (debounce 80мс)
  const liveLayoutTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!enabled || !ceiling) return
    if (liveLayoutTimerRef.current) clearTimeout(liveLayoutTimerRef.current)
    liveLayoutTimerRef.current = setTimeout(() => {
      autoLayout(ceiling)
    }, 80)
    return () => {
      if (liveLayoutTimerRef.current) clearTimeout(liveLayoutTimerRef.current)
    }
  }, [enabled, count, halfWidth, direction, ceiling, autoLayout])

  const handleToggleEnabled = useCallback(() => {
    if (enabled) {
      setEnabled(false)
      clear()
    } else {
      ensureTexture()
      if (ceiling) autoLayout(ceiling)
      setEnabled(true)
    }
  }, [enabled, setEnabled, clear, autoLayout, ceiling, ensureTexture])

  if (sceneMode !== 'interior' || !ceiling) return null

  const hasTexture = !!beamTextureUrl
  const canPickMaterial = !!selectedMaterial?.texture.url

  return (
    <div className="border-t border-gray-100">
      <button
        type="button"
        className="flex w-full items-center justify-between px-4 py-3 hover:bg-gray-50"
        onClick={() => setCollapsed((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <LayoutList className="h-3.5 w-3.5 text-amber-500" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Фальшбалки
          </h2>
        </div>
        {collapsed ? (
          <ChevronRight className="h-3.5 w-3.5 text-gray-400" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-gray-400" />
        )}
      </button>

      {!collapsed && (
        <div className="px-4 pb-4 space-y-3">
          {/* Материал балок */}
          <div className="space-y-1">
            <span className="text-xs text-gray-600">Материал балок</span>
            <div className="flex items-center gap-2">
              {hasTexture ? (
                <img
                  src={beamTextureUrl!}
                  alt="материал балок"
                  className="h-8 w-8 flex-shrink-0 rounded border border-gray-200 object-cover"
                />
              ) : (
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded border border-dashed border-gray-300">
                  <Layers className="h-4 w-4 text-gray-300" />
                </div>
              )}
              <button
                type="button"
                disabled={!canPickMaterial}
                onClick={handlePickMaterial}
                className="flex-1 rounded border border-gray-200 px-2 py-1 text-left text-xs text-gray-600 transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {canPickMaterial
                  ? `Взять: ${selectedMaterial!.name}`
                  : 'Выберите материал в каталоге'}
              </button>
            </div>
          </div>

          {/* Быстрый выбор текстуры из каталога балок */}
          {beamCatalog.length > 0 && (
            <div className="space-y-1">
              <span className="text-xs text-gray-600">Готовые текстуры</span>
              <div className="flex flex-wrap gap-1.5">
                {beamCatalog.map((mat) => (
                  <button
                    key={mat.id}
                    type="button"
                    title={mat.name}
                    onClick={() => setTextureUrl(mat.texture.url)}
                    className={`h-10 w-10 flex-shrink-0 overflow-hidden rounded border-2 transition-colors ${
                      beamTextureUrl === mat.texture.url
                        ? 'border-amber-500'
                        : 'border-gray-200 hover:border-amber-300'
                    }`}
                  >
                    <img
                      src={mat.texture.url}
                      alt={mat.name}
                      className="h-full w-full object-cover"
                    />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Вкл/выкл */}
          <label className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Показать балки</span>
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              onClick={handleToggleEnabled}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                enabled ? 'bg-amber-500' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
                  enabled ? 'translate-x-4' : 'translate-x-1'
                }`}
              />
            </button>
          </label>

          {/* Количество */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Количество</span>
              <span className="text-xs font-medium text-gray-800">{count}</span>
            </div>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              className="w-full accent-amber-500"
            />
          </div>

          {/* Ширина */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-600">Ширина</span>
              <span className="text-xs font-medium text-gray-800">
                {Math.round(halfWidth * 200)}%
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={25}
              step={1}
              value={Math.round(halfWidth * 200)}
              onChange={(e) => setHalfWidth(Number(e.target.value) / 200)}
              className="w-full accent-amber-500"
            />
          </div>

          {/* Направление */}
          <div className="space-y-1">
            <span className="text-xs text-gray-600">Направление</span>
            <div className="mt-1 flex rounded-lg border border-gray-200 bg-gray-50 p-0.5">
              <button
                type="button"
                onClick={() => setDirection('depth')}
                className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                  direction === 'depth'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Вдоль
              </button>
              <button
                type="button"
                onClick={() => setDirection('width')}
                className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                  direction === 'width'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Поперёк
              </button>
              <button
                type="button"
                onClick={() => setDirection('grid')}
                className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                  direction === 'grid'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Сетка
              </button>
            </div>
          </div>

          {/* Авто-раскладка */}
          <button
            type="button"
            onClick={handleAutoLayout}
            disabled={!hasTexture && !canPickMaterial}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg bg-amber-500 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-amber-600 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Авто-раскладка
          </button>

          {!hasTexture && !canPickMaterial && (
            <p className="text-[10px] text-amber-600">
              Выберите материал в каталоге, затем нажмите «Взять».
            </p>
          )}
          <p className="text-[10px] text-gray-400">
            Кликните на балку для её удаления.
          </p>
        </div>
      )}
    </div>
  )
}
