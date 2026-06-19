import { create } from 'zustand'
import type { Material, Color } from '@/types/material'
import { useHistoryStore } from './useHistoryStore'

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
  removeQuickAccessColor: (key: string) => void
}

export const useMaterialStore = create<MaterialState>((set) => ({
  selectedMaterial: null,
  selectedColor: null,
  colorizeOpacity: 1,
  quickAccessMaterials: [],
  quickAccessColors: [],
  setSelectedMaterial: (selectedMaterial) => {
    useHistoryStore.getState().record('selectedMaterial')
    set({ selectedMaterial })
  },
  setSelectedColor: (selectedColor) => {
    useHistoryStore.getState().record('selectedColor')
    set({ selectedColor })
  },
  setColorizeOpacity: (colorizeOpacity) => {
    useHistoryStore.getState().record('colorizeOpacity')
    set({ colorizeOpacity })
  },
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
      const key = color.id ?? color.hex
      if (state.quickAccessColors.find((c) => (c.id ?? c.hex) === key)) return state
      return { quickAccessColors: [...state.quickAccessColors, color] }
    }),
  removeQuickAccessColor: (key: string) =>
    set((state) => {
      const newColors = state.quickAccessColors.filter((c) => (c.id ?? c.hex) !== key)
      const sel = state.selectedColor
      const wasSelected = (sel?.id ?? sel?.hex) === key
      return {
        quickAccessColors: newColors,
        ...(wasSelected ? { selectedColor: null } : {}),
      }
    }),
}))
