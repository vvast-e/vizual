import { useNavigate } from 'react-router-dom'
import { useUIStore } from '@/store/useUIStore'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useWallStore } from '@/store/useWallStore'
import { downloadDataUrl } from '@/lib/export-utils'
import { Home, Building2, Download, RotateCcw } from 'lucide-react'

export function EditorHeader() {
  const navigate = useNavigate()
  const sceneMode = useUIStore((s) => s.sceneMode)
  const setSceneMode = useUIStore((s) => s.setSceneMode)

  const handleExport = () => {
    const canvas = document.querySelector('canvas') as HTMLCanvasElement | null
    if (!canvas) return
    const dataUrl = canvas.toDataURL('image/png')
    downloadDataUrl(dataUrl, `vizual-${Date.now()}.png`)
  }

  const handleReset = () => {
    useVisualizerStore.getState().setPhotoDataUrl(null)
    useWallStore.getState().setWalls([], null)
    useWallStore.getState().setExteriorMaskBase64(null)
    useWallStore.getState().setWallTexture(0, null)
    localStorage.removeItem('vizual-state')
    navigate('/upload')
  }

  return (
    <header className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-2.5">
      <div className="flex items-center gap-4">
        <h1
          className="cursor-pointer text-lg font-medium tracking-tight text-gray-900"
          onClick={handleReset}
        >
          Vizual
        </h1>
        <div className="flex items-center gap-0.5 rounded-lg border border-gray-200 bg-gray-50 p-0.5">
          <button
            type="button"
            onClick={() => setSceneMode('interior')}
            className={`flex items-center gap-1 rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              sceneMode === 'interior'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Home className="h-3.5 w-3.5" />
            Интерьер
          </button>
          <button
            type="button"
            onClick={() => setSceneMode('exterior')}
            className={`flex items-center gap-1 rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              sceneMode === 'exterior'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Building2 className="h-3.5 w-3.5" />
            Экстерьер
          </button>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={handleExport}
          className="flex items-center gap-1.5 rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-50"
        >
          <Download className="h-3.5 w-3.5" />
          Экспорт PNG
        </button>
        <button
          type="button"
          onClick={handleReset}
          className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
          title="Начать заново"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      </div>
    </header>
  )
}
