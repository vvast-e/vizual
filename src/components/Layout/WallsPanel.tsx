import { useCallback } from 'react'
import { Eye, EyeOff, Trash2, Plus, Square } from 'lucide-react'
import { useWallStore } from '@/store/useWallStore'
import { useUIStore } from '@/store/useUIStore'
import { useMaterialStore } from '@/store/useMaterialStore'

const WALL_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
]

interface WallsPanelProps {
  onStartCustomMask: () => void
  isDrawingCustomMask: boolean
  onCancelCustomMask: () => void
}

export function WallsPanel({ onStartCustomMask, isDrawingCustomMask, onCancelCustomMask }: WallsPanelProps) {
  const walls = useWallStore((s) => s.walls)
  const selectedWallId = useWallStore((s) => s.selectedWallId)
  const selectWall = useWallStore((s) => s.selectWall)
  const wallTextures = useWallStore((s) => s.wallTextures)
  const removeWall = useWallStore((s) => s.removeWall)
  const wallVisibility = useUIStore((s) => s.wallVisibility)
  const toggleWallVisibility = useUIStore((s) => s.toggleWallVisibility)
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)

  const handleWallClick = useCallback(
    (wallId: number) => {
      selectWall(selectedWallId === wallId ? null : wallId)
    },
    [selectedWallId, selectWall]
  )

  return (
    <aside className="flex w-[250px] flex-shrink-0 flex-col border-r border-gray-200 bg-white">
      <div className="border-b border-gray-100 px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400">Стены</h2>
      </div>

      <div className="tour-walls-list flex-1 overflow-y-auto">
        {walls.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <Square className="mx-auto mb-2 h-8 w-8 text-gray-300" />
            <p className="text-xs text-gray-400">Стены не обнаружены</p>
          </div>
        ) : (
          <ul className="py-1">
            {walls.map((wall, i) => {
              const color = WALL_COLORS[i % WALL_COLORS.length]
              const isSelected = selectedWallId === wall.id
              const isVisible = wallVisibility[wall.id] !== false
              const hasTexture = wallTextures[wall.id] != null

              return (
                <li key={wall.id}>
                  <div
                    className={`flex cursor-pointer items-center gap-2 px-4 py-2.5 transition-colors ${
                      isSelected
                        ? 'bg-blue-50'
                        : 'hover:bg-gray-50'
                    }`}
                    onClick={() => handleWallClick(wall.id)}
                  >
                    <div
                      className="h-3 w-3 flex-shrink-0 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className={`flex-1 text-sm ${isSelected ? 'font-medium text-blue-900' : 'text-gray-700'}`}>
                      Стена {i + 1}
                    </span>
                    {hasTexture && (
                      <div className="h-2 w-2 rounded-full bg-green-400" title="Текстура применена" />
                    )}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        toggleWallVisibility(wall.id)
                      }}
                      className="rounded p-1 text-gray-400 transition-colors hover:bg-gray-200 hover:text-gray-600"
                      title={isVisible ? 'Скрыть' : 'Показать'}
                    >
                      {isVisible ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        removeWall(wall.id)
                      }}
                      className="rounded p-1 text-gray-400 transition-colors hover:bg-red-100 hover:text-red-500"
                      title="Удалить"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      <div className="border-t border-gray-100 p-3">
        {isDrawingCustomMask ? (
          <div className="space-y-2">
            <p className="text-xs text-gray-500">Кликните 4 точки на холсте для задания перспективы</p>
            <button
              type="button"
              onClick={onCancelCustomMask}
              className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-xs text-gray-600 transition-colors hover:bg-gray-50"
            >
              Отмена
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={onStartCustomMask}
            disabled={walls.length === 0}
            className="tour-add-mask flex w-full items-center justify-center gap-1.5 rounded-md border border-dashed border-gray-300 px-3 py-2 text-xs text-gray-500 transition-colors hover:border-gray-400 hover:bg-gray-50 hover:text-gray-700 disabled:opacity-40"
            title={walls.length === 0 ? 'Загрузите фото, чтобы добавить область' : 'Нарисовать область'}
          >
            <Plus className="h-3.5 w-3.5" />
            Нарисовать область
          </button>
        )}
      </div>

      {selectedWallId != null && selectedMaterial && (
        <div className="border-t border-gray-100 bg-gray-50 px-4 py-2">
          <p className="text-xs text-gray-500">
            Нажмите «Применить профиль» для наложения на стену
          </p>
        </div>
      )}
    </aside>
  )
}
