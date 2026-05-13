import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { useUIStore } from '@/store/useUIStore'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useWallStore } from '@/store/useWallStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { buildExportImageDataUrl, downloadDataUrl } from '@/lib/export-utils'
import { Home, Building2, Download, RotateCcw, HelpCircle, Upload } from 'lucide-react'
import { ConfirmDialog } from '@/components/UI/ConfirmDialog'

interface EditorHeaderProps {
  onRequestChangePhoto?: () => void
}

export function EditorHeader({ onRequestChangePhoto }: EditorHeaderProps = {}) {
  const navigate = useNavigate()
  const sceneMode = useUIStore((s) => s.sceneMode)
  const setSceneMode = useUIStore((s) => s.setSceneMode)
  const setTourActive = useUIStore((s) => s.setTourActive)
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const selectedColor = useMaterialStore((s) => s.selectedColor)
  const [confirmResetOpen, setConfirmResetOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)

  const handleChangePhoto = () => {
    onRequestChangePhoto?.()
  }

  const handleExport = async () => {
    const canvas = document.querySelector('canvas') as HTMLCanvasElement | null
    if (!canvas) return
    const ui = useUIStore.getState()
    const prevHide = ui.hideWallMasks
    const prevEditWallCorners = ui.editWallCorners
    const prevWallVisibility = ui.wallVisibility
    const walls = useWallStore.getState().walls
    try {
      if (!prevHide) {
        ui.setHideWallMasks(true)
      }
      if (prevEditWallCorners) {
        ui.setEditWallCorners(false)
      }
      if (walls.length > 0) {
        const hiddenVisibility = { ...prevWallVisibility }
        for (const wall of walls) {
          hiddenVisibility[wall.id] = false
        }
        useUIStore.setState({ wallVisibility: hiddenVisibility })
      }
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))
      const dataUrl = buildExportImageDataUrl(canvas, {
        material: selectedMaterial,
        color: selectedColor,
      })
      downloadDataUrl(dataUrl, `vizual-${Date.now()}.png`)
      toast.success('Изображение сохранено')
    } finally {
      useUIStore.setState({ wallVisibility: prevWallVisibility })
      if (prevEditWallCorners) {
        ui.setEditWallCorners(true)
      }
      if (!prevHide) {
        ui.setHideWallMasks(false)
      }
    }
  }

  const handleReset = () => {
    useVisualizerStore.getState().setPhotoDataUrl(null)
    useWallStore.getState().setWalls([], null)
    useWallStore.getState().setExteriorMaskBase64(null)
    useWallStore.getState().setWallTexture(0, null)
    localStorage.removeItem('vizual-state')
    navigate('/upload')
  }

  const handleRestartTour = () => {
    setTourActive(true)
    setHelpOpen(false)
  }

  const handleF1 = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLElement && e.target.tagName === 'INPUT') return
    if (e.key === 'F1') {
      e.preventDefault()
      setHelpOpen((v) => !v)
    }
  }

  useEffect(() => {
    window.addEventListener('keydown', handleF1)
    return () => window.removeEventListener('keydown', handleF1)
  }, [])

  return (
    <header className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-2.5">
      <div className="flex items-center gap-4">
        <h1
          className="cursor-pointer text-lg font-medium tracking-tight text-gray-900"
          onClick={() => setConfirmResetOpen(true)}
        >
          Vizual
        </h1>
        <div className="tour-mode-toggle flex items-center gap-0.5 rounded-lg border border-gray-200 bg-gray-50 p-0.5">
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
        {onRequestChangePhoto && (
          <button
            type="button"
            onClick={handleChangePhoto}
            className="flex items-center gap-1.5 rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-50"
          >
            <Upload className="h-3.5 w-3.5" />
            Сменить фото
          </button>
        )}
        <button
          type="button"
          onClick={() => setHelpOpen(true)}
          className="flex items-center justify-center rounded-md p-1.5 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
          title="Справка (F1)"
        >
          <HelpCircle className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={handleExport}
          className="tour-export flex items-center gap-1.5 rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-50"
        >
          <Download className="h-3.5 w-3.5" />
          Экспорт PNG
        </button>
        <button
          type="button"
          onClick={() => setConfirmResetOpen(true)}
          className="tour-reset flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
          title="Начать заново"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      </div>
      <ConfirmDialog
        open={confirmResetOpen}
        onOpenChange={setConfirmResetOpen}
        title="Начать заново?"
        description="Текущий проект будет сброшен. Все наложенные текстуры и цвета будут потеряны."
        confirmLabel="Сбросить"
        onConfirm={handleReset}
        variant="danger"
      />
      {helpOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50" onClick={() => setHelpOpen(false)}>
          <div className="w-[400px] rounded-xl bg-white p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-base font-semibold text-gray-900">Горячие клавиши</h3>
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-600">Показать оригинал</dt>
                <dd className="font-mono text-gray-800">Space</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600">Отменить</dt>
                <dd className="font-mono text-gray-800">Ctrl+Z</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600">Повторить</dt>
                <dd className="font-mono text-gray-800">Ctrl+Y</dd>
              </div>
            </dl>
            <button
              type="button"
              onClick={handleRestartTour}
              className="mt-5 w-full rounded-lg bg-gray-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-black"
            >
              Перезапустить подсказки
            </button>
            <button
              type="button"
              onClick={() => setHelpOpen(false)}
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
            >
              Закрыть
            </button>
          </div>
        </div>
      )}
    </header>
  )
}
