import { create } from 'zustand'
import type { Material, Color } from '@/types/material'

interface MaterialState {
  selectedMaterial: Material | null
  selectedColor: Color | null
  /** Интенсивность колоризации 0..1 */
  colorizeOpacity: number
  setSelectedMaterial: (material: Material | null) => void
  setSelectedColor: (color: Color | null) => void
  setColorizeOpacity: (opacity: number) => void
}

export const useMaterialStore = create<MaterialState>((set) => ({
  selectedMaterial: null,
  selectedColor: null,
  colorizeOpacity: 1,
  setSelectedMaterial: (selectedMaterial) => set({ selectedMaterial }),
  setSelectedColor: (selectedColor) => set({ selectedColor }),
  setColorizeOpacity: (colorizeOpacity) => set({ colorizeOpacity }),
}))
