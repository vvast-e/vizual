import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Canvas2D } from '@/components/2d/Canvas2D'
import { MaterialCatalog } from '@/components/Materials/MaterialCatalog'
import { ColorPicker } from '@/components/Materials/ColorPicker'
import { EditorHeader } from '@/components/Layout/EditorHeader'
import { WallsPanel } from '@/components/Layout/WallsPanel'
import { OnboardingTour } from '@/components/OnboardingTour'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useWallStore } from '@/store/useWallStore'
import { useUIStore } from '@/store/useUIStore'
import { loadState } from '@/lib/persist'

export function EditorPage() {
  const navigate = useNavigate()
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)
  const walls = useWallStore((s) => s.walls)
  const wallImageSize = useWallStore((s) => s.wallImageSize)
  const setHideWallMasks = useUIStore((s) => s.setHideWallMasks)
  const [isDrawingCustomMask, setIsDrawingCustomMask] = useState(false)
  const [restoreChecked, setRestoreChecked] = useState(false)

  useEffect(() => {
    if (restoreChecked) return

    const hasRuntimeData = Boolean(photoDataUrl) && walls.length > 0 && Boolean(wallImageSize)
    if (hasRuntimeData) {
      setHideWallMasks(false)
      setRestoreChecked(true)
      return
    }

    const restored = loadState()
    setRestoreChecked(true)
    if (!restored && !photoDataUrl) {
      navigate('/upload')
    }
  }, [photoDataUrl, walls.length, wallImageSize, restoreChecked, navigate, setHideWallMasks])

  useEffect(() => {
    if (!photoDataUrl) return
    if (walls.length === 0) return
    if (!wallImageSize) return
    setHideWallMasks(false)
  }, [photoDataUrl, walls.length, wallImageSize, setHideWallMasks])

  const handleStartCustomMask = useCallback(() => {
    setIsDrawingCustomMask(true)
  }, [])

  const handleCancelCustomMask = useCallback(() => {
    setIsDrawingCustomMask(false)
  }, [])

  const handleCustomMaskComplete = useCallback(
    (corners: [number, number][]) => {
      const newId = Date.now()
      const cx = corners.reduce((a, c) => a + c[0], 0) / corners.length
      const cy = corners.reduce((a, c) => a + c[1], 0) / corners.length
      const wall = {
        id: newId,
        corners,
        polygon: corners,
        center: [cx, cy] as [number, number],
      }
      useWallStore.getState().addCustomWall(wall)
      setIsDrawingCustomMask(false)
    },
    []
  )

  if (!photoDataUrl) return null

  return (
    <div className="flex h-screen flex-col">
      <OnboardingTour />
      <EditorHeader />
      <div className="flex flex-1 overflow-hidden">
        <WallsPanel
          onStartCustomMask={handleStartCustomMask}
          isDrawingCustomMask={isDrawingCustomMask}
          onCancelCustomMask={handleCancelCustomMask}
        />
        <main className="flex flex-1 flex-col overflow-hidden bg-gray-100 p-4">
          <Canvas2D
            customMaskMode={isDrawingCustomMask}
            onCustomMaskComplete={handleCustomMaskComplete}
          />
        </main>
        <aside className="tour-materials-panel flex w-[320px] flex-shrink-0 flex-col gap-4 overflow-y-auto border-l border-gray-200 bg-white p-5">
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Материалы
            </h2>
            <MaterialCatalog />
          </section>
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Цвет
            </h2>
            <ColorPicker />
          </section>
        </aside>
      </div>
    </div>
  )
}
