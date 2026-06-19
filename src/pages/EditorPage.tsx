import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Canvas2D } from '@/components/2d/Canvas2D'
import { EditorHeader } from '@/components/Layout/EditorHeader'
import { WallsPanel } from '@/components/Layout/WallsPanel'
import { OnboardingTour } from '@/components/OnboardingTour'
import { MaterialWidget } from '@/components/Materials/MaterialWidget'
import { ColorWidget } from '@/components/Materials/ColorWidget'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useWallStore } from '@/store/useWallStore'
import { useUIStore } from '@/store/useUIStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useHistoryStore } from '@/store/useHistoryStore'
import { loadState } from '@/lib/persist'
import { TexturePreview } from '@/components/Materials/TexturePreview'
import type { Material } from '@/types/material'
import type { Color } from '@/types/material'

const QUICK_ACCESS_KEY = 'vizual-quick-access'

interface QuickAccess {
  material: Material | null
  color: Color | null
  materials: Material[]
  colors: Color[]
}

function loadQuickAccess(): QuickAccess {
  try {
    const raw = localStorage.getItem(QUICK_ACCESS_KEY)
    if (raw) {
      return JSON.parse(raw)
    }
  } catch {}
  return { material: null, color: null, materials: [], colors: [] }
}

function saveQuickAccess(data: QuickAccess) {
  try {
    localStorage.setItem(QUICK_ACCESS_KEY, JSON.stringify(data))
  } catch {}
}

export function EditorPage() {
  const navigate = useNavigate()
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)
  const walls = useWallStore((s) => s.walls)
  const wallImageSize = useWallStore((s) => s.wallImageSize)
  const setHideWallMasks = useUIStore((s) => s.setHideWallMasks)
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const setSelectedMaterial = useMaterialStore((s) => s.setSelectedMaterial)
  const selectedColor = useMaterialStore((s) => s.selectedColor)
  const setSelectedColor = useMaterialStore((s) => s.setSelectedColor)
  const quickAccessMaterials = useMaterialStore((s) => s.quickAccessMaterials)
  const quickAccessColors = useMaterialStore((s) => s.quickAccessColors)
  const [isDrawingCustomMask, setIsDrawingCustomMask] = useState(false)
  const [restoreChecked, setRestoreChecked] = useState(false)
  const [materialWidgetOpen, setMaterialWidgetOpen] = useState(false)
  const [colorWidgetOpen, setColorWidgetOpen] = useState(false)

  // Load quick access on mount
  useEffect(() => {
    const saved = loadQuickAccess()
    const history = useHistoryStore.getState()
    history.setApplying(true)
    try {
      if (saved.material) setSelectedMaterial(saved.material)
      if (saved.color) setSelectedColor(saved.color)
      if (saved.materials) {
        saved.materials.forEach((m: Material) => useMaterialStore.getState().addQuickAccessMaterial(m))
      }
      if (saved.colors) {
        saved.colors.forEach((c: Color) => useMaterialStore.getState().addQuickAccessColor(c))
      }
    } finally {
      history.setApplying(false)
      history.clear()
    }
  }, [setSelectedMaterial, setSelectedColor])

  // Save quick access when selection changes
  useEffect(() => {
    saveQuickAccess({ 
      material: selectedMaterial, 
      color: selectedColor,
      materials: quickAccessMaterials,
      colors: quickAccessColors
    })
  }, [selectedMaterial, selectedColor, quickAccessMaterials, quickAccessColors])

  useEffect(() => {
    if (restoreChecked) return

    const hasRuntimeData = Boolean(photoDataUrl) && walls.length > 0 && Boolean(wallImageSize)
    if (hasRuntimeData) {
      const hs = useHistoryStore.getState()
      hs.setApplying(true)
      try { setHideWallMasks(false) } finally { hs.setApplying(false) }
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
    const hs = useHistoryStore.getState()
    hs.setApplying(true)
    try { setHideWallMasks(false) } finally { hs.setApplying(false) }
  }, [photoDataUrl, walls.length, wallImageSize, setHideWallMasks])

  const handleStartCustomMask = useCallback(() => {
    setIsDrawingCustomMask(true)
  }, [])

  const handleCancelCustomMask = useCallback(() => {
    setIsDrawingCustomMask(false)
  }, [])

  const handleChangePhoto = useCallback(() => {
    setHideWallMasks(true)
    setMaterialWidgetOpen(false)
    setColorWidgetOpen(false)
    setSelectedMaterial(null)
    setSelectedColor(null)
    navigate('/upload')
  }, [setHideWallMasks, setSelectedMaterial, setSelectedColor, navigate])

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
      <EditorHeader onRequestChangePhoto={handleChangePhoto} />
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
        <aside className="tour-materials-panel flex w-[360px] flex-shrink-0 flex-col gap-3 border-l border-gray-200 bg-white p-4">
          {/* Кнопка Материалы */}
          <button
            type="button"
            onClick={() => setMaterialWidgetOpen(true)}
            className="flex h-14 w-full items-center justify-center gap-2 rounded-lg border-2 border-gray-200 font-medium text-gray-700 transition-colors hover:border-gray-400 hover:bg-gray-50"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Материалы
          </button>

          {/* Quick Access Materials - под кнопкой */}
          <div className="flex flex-col gap-2">
            {quickAccessMaterials.map((material) => (
              <div key={material.id} className="relative group">
                <button
                  type="button"
                  onClick={() => setSelectedMaterial(material)}
                  className={`flex w-full items-center gap-2 rounded-lg border-2 p-2 ${
                    selectedMaterial?.id === material.id ? 'border-gray-800' : 'border-gray-200'
                  }`}
                >
                  <TexturePreview texture={material.texture} width={48} height={48} />
                  <span className="text-xs font-medium text-gray-700 truncate">{material.name}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    useMaterialStore.getState().removeQuickAccessMaterial(material.id)
                  }}
                  className="absolute right-1 top-1 hidden h-5 w-5 items-center justify-center rounded-full bg-gray-800 text-white group-hover:flex"
                >
                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>

          {/* Кнопка Цвета */}
          <button
            type="button"
            onClick={() => setColorWidgetOpen(true)}
            className="flex h-14 w-full items-center justify-center gap-2 rounded-lg border-2 border-gray-200 font-medium text-gray-700 transition-colors hover:border-gray-400 hover:bg-gray-50"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
            </svg>
            Цвета
          </button>

          {/* Quick Access Colors - под кнопкой */}
          <div className="flex flex-col gap-2">
            {quickAccessColors.map((color) => {
              const ckey = color.id ?? color.hex
              const selKey = selectedColor?.id ?? selectedColor?.hex
              return (
              <div key={ckey} className="relative group">
                <button
                  type="button"
                  onClick={() => setSelectedColor(color)}
                  className={`flex w-full items-center gap-2 rounded-lg border-2 p-2 ${
                    selKey === ckey ? 'border-gray-800' : 'border-gray-200'
                  }`}
                >
                  {color.swatchUrl ? (
                    <img
                      src={color.swatchUrl}
                      alt=""
                      className="h-12 w-12 flex-shrink-0 rounded-lg border border-gray-300 object-cover"
                    />
                  ) : (
                    <div
                      className="h-12 w-12 flex-shrink-0 rounded-lg border border-gray-300"
                      style={{ backgroundColor: color.hex }}
                    />
                  )}
                  <span className="text-xs font-medium text-gray-700 truncate">{color.name || color.hex}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    useMaterialStore.getState().removeQuickAccessColor(ckey)
                  }}
                  className="absolute right-1 top-1 hidden h-5 w-5 items-center justify-center rounded-full bg-gray-800 text-white group-hover:flex"
                >
                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              )
            })}
          </div>
        </aside>
      </div>

      <MaterialWidget open={materialWidgetOpen} onClose={() => setMaterialWidgetOpen(false)} />
      <ColorWidget open={colorWidgetOpen} onClose={() => setColorWidgetOpen(false)} />
    </div>
  )
}
