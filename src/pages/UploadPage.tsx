import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Home, Building2 } from 'lucide-react'
import { MAX_PHOTO_SIZE_BYTES, ALLOWED_IMAGE_TYPES } from '@/lib/constants'
import { detectWalls, detectExterior } from '@/hooks/useWallDetection'
import { autoLinkPerspectiveToForm } from '@/lib/perspective-helper'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useWallStore } from '@/store/useWallStore'
import { useUIStore } from '@/store/useUIStore'
import type { SceneMode } from '@/store/useUIStore'
import { Loader } from '@/components/UI/Loader'

const MAX_SIZE_MB = MAX_PHOTO_SIZE_BYTES / (1024 * 1024)
const ACCEPT = ALLOWED_IMAGE_TYPES.join(',')

export function UploadPage() {
  const navigate = useNavigate()
  const [sceneMode, setSceneModeState] = useState<SceneMode>('interior')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback(
    async (file: File | null) => {
      if (!file) return
      if (file.size > MAX_PHOTO_SIZE_BYTES) {
        setError(`Файл слишком большой (макс. ${MAX_SIZE_MB} МБ)`)
        return
      }
      if (!ALLOWED_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_TYPES)[number])) {
        setError('Поддерживаются только JPEG, PNG и WebP')
        return
      }

      setError(null)
      setIsLoading(true)

      try {
        const reader = new FileReader()
        const dataUrl = await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string)
          reader.onerror = reject
          reader.readAsDataURL(file)
        })

        useVisualizerStore.getState().setPhotoDataUrl(dataUrl)
        useUIStore.getState().setSceneMode(sceneMode)

        if (sceneMode === 'exterior') {
          const res = await detectExterior(file)
          useWallStore.getState().setWalls(res.walls ?? [], res.image_size)
          useWallStore.getState().setExteriorMaskBase64(res.masks?.wall_minus_holes ?? null)
          
          // Автоматически привязываем перспективу к форме для каждой стены
          const walls = useWallStore.getState().walls
          walls.forEach((wall) => {
            if (wall.polygon && wall.polygon.length >= 3) {
              const newCorners = autoLinkPerspectiveToForm(wall)
              useWallStore.getState().updateWallCorners(wall.id, newCorners)
            }
          })
        } else {
          const res = await detectWalls(file)
          useWallStore.getState().setWalls(res.walls, res.image_size)
          useWallStore.getState().setExteriorMaskBase64(null)
          
          // Для интерьера тоже привязываем перспективу
          const walls = useWallStore.getState().walls
          walls.forEach((wall) => {
            if (wall.polygon && wall.polygon.length >= 3) {
              const newCorners = autoLinkPerspectiveToForm(wall)
              useWallStore.getState().updateWallCorners(wall.id, newCorners)
            }
          })
        }

        navigate('/editor')
      } catch (err) {
        console.error('[UploadPage] detection failed', err)
        setError('Не удалось определить стены. Попробуйте другое фото.')
      } finally {
        setIsLoading(false)
      }
    },
    [sceneMode, navigate]
  )

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0] ?? null
      handleFile(file)
      e.target.value = ''
    },
    [handleFile]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      if (isLoading) return
      const file = e.dataTransfer.files?.[0] ?? null
      handleFile(file)
    },
    [isLoading, handleFile]
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-50 px-4">
      <h1 className="mb-2 text-3xl font-light tracking-tight text-gray-900">Vizual</h1>
      <p className="mb-8 text-sm text-gray-500">Визуализация отделочных материалов</p>

      <div className="mb-6 flex items-center gap-1 rounded-lg border border-gray-200 bg-white p-1">
        <button
          type="button"
          onClick={() => setSceneModeState('interior')}
          className={`flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            sceneMode === 'interior'
              ? 'bg-gray-800 text-white shadow-sm'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          <Home className="h-4 w-4" />
          Интерьер
        </button>
        <button
          type="button"
          onClick={() => setSceneModeState('exterior')}
          className={`flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            sceneMode === 'exterior'
              ? 'bg-gray-800 text-white shadow-sm'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          <Building2 className="h-4 w-4" />
          Экстерьер
        </button>
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label="Загрузить фото"
        onClick={() => !isLoading && document.getElementById('upload-input')?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onKeyDown={(e) => e.key === 'Enter' && !isLoading && document.getElementById('upload-input')?.click()}
        className={`relative flex w-full max-w-lg flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed border-gray-300 bg-white p-12 text-gray-600 transition-all hover:border-gray-400 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 ${
          isLoading ? 'pointer-events-none opacity-60' : 'cursor-pointer'
        }`}
      >
        <input
          id="upload-input"
          type="file"
          accept={ACCEPT}
          onChange={onInputChange}
          className="hidden"
          aria-hidden
        />
        {isLoading ? (
          <div className="flex flex-col items-center gap-3">
            <Loader ariaLabel="Определение стен" />
            <p className="text-sm text-gray-500">
              {sceneMode === 'exterior' ? 'Определение фасада…' : 'Определение стен…'}
            </p>
          </div>
        ) : (
          <>
            <Upload className="h-12 w-12 text-gray-400" aria-hidden />
            <div className="text-center">
              <p className="text-sm font-medium">Перетащите фото сюда</p>
              <p className="mt-1 text-xs text-gray-400">или нажмите для выбора</p>
            </div>
            <p className="text-xs text-gray-400">JPEG, PNG, WebP до {MAX_SIZE_MB} МБ</p>
          </>
        )}
      </div>

      {error && (
        <p className="mt-4 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">{error}</p>
      )}
    </div>
  )
}
