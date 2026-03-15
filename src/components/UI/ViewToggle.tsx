import { useVisualizerStore } from '@/store/useVisualizerStore'
import { LayoutGrid, Box } from 'lucide-react'

export interface ViewToggleProps {
  className?: string
}

export function ViewToggle({ className = '' }: ViewToggleProps) {
  const viewMode = useVisualizerStore((s) => s.viewMode)
  const setViewMode = useVisualizerStore((s) => s.setViewMode)

  return (
    <div
      className={`flex rounded-lg border border-gray-200 bg-gray-50 p-1 ${className}`}
      role="tablist"
      aria-label="Режим просмотра"
    >
      <button
        type="button"
        role="tab"
        aria-selected={viewMode === '2d'}
        aria-label="2D — фото и текстура"
        onClick={() => setViewMode('2d')}
        className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
          viewMode === '2d' ? 'bg-white text-gray-900 shadow' : 'text-gray-600 hover:text-gray-900'
        }`}
      >
        <LayoutGrid className="h-4 w-4" aria-hidden />
        2D
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={viewMode === '3d'}
        aria-label="3D — сцена"
        onClick={() => setViewMode('3d')}
        className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
          viewMode === '3d' ? 'bg-white text-gray-900 shadow' : 'text-gray-600 hover:text-gray-900'
        }`}
      >
        <Box className="h-4 w-4" aria-hidden />
        3D
      </button>
    </div>
  )
}
