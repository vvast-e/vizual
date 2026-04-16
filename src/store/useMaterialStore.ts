import { create } from 'zustand'
import type { Material, Color } from '@/types/material'

interface MaterialState {
  selectedMaterial: Material | null
  selectedColor: Color | null
  /** Интенсивность колоризации 0..1 */
  colorizeOpacity: number
  /** Материалы в панели быстрого доступа */
  quickAccessMaterials: Material[]
  /** Цвета в панели быстрого доступа */
  quickAccessColors: Color[]
  setSelectedMaterial: (material: Material | null) => void
  setSelectedColor: (color: Color | null) => void
  setColorizeOpacity: (opacity: number) => void
  addQuickAccessMaterial: (material: Material) => void
  removeQuickAccessMaterial: (materialId: string) => void
  addQuickAccessColor: (color: Color) => void
  removeQuickAccessColor: (hex: string) => void
}

export const useMaterialStore = create<MaterialState>((set) => ({
  selectedMaterial: null,
  selectedColor: null,
  colorizeOpacity: 1,
  quickAccessMaterials: [],
  quickAccessColors: [],
  setSelectedMaterial: (selectedMaterial) => set({ selectedMaterial }),
  setSelectedColor: (selectedColor) => set({ selectedColor }),
  setColorizeOpacity: (colorizeOpacity) => set({ colorizeOpacity }),
  addQuickAccessMaterial: (material) =>
    set((state) => {
      if (state.quickAccessMaterials.find((m) => m.id === material.id)) return state
      return { quickAccessMaterials: [...state.quickAccessMaterials, material] }
    }),
  removeQuickAccessMaterial: (materialId) =>
    set((state) => {
      const newMaterials = state.quickAccessMaterials.filter((m) => m.id !== materialId)
      const wasSelected = state.selectedMaterial?.id === materialId
      return {
        quickAccessMaterials: newMaterials,
        ...(wasSelected ? { selectedMaterial: null, selectedColor: null } : {}),
      }
    }),
  addQuickAccessColor: (color) =>
    set((state) => {
      if (state.quickAccessColors.find((c) => c.hex === color.hex)) return state
      return { quickAccessColors: [...state.quickAccessColors, color] }
    }),
  removeQuickAccessColor: (hex) =>
    set((state) => {
      const newColors = state.quickAccessColors.filter((c) => c.hex !== hex)
      const wasSelected = state.selectedColor?.hex === hex
      return {
        quickAccessColors: newColors,
        ...(wasSelected ? { selectedColor: null } : {}),
      }
    }),
}))
