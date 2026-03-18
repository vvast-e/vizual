import { create } from 'zustand'

export type MaskTool = 'brush' | 'rect' | 'lasso' | null

interface UIState {
  materialCatalogOpen: boolean
  exportModalOpen: boolean
  activeToolbarPanel: string | null
  /** Инструмент маски: кисть, прямоугольник, лассо; null — пан */
  maskTool: MaskTool
  /** Режим правки углов стены (draggable handles) */
  editWallCorners: boolean
  /** Скрывать все синие маски стен (оверлеи) */
  hideWallMasks: boolean
  setMaterialCatalogOpen: (open: boolean) => void
  setExportModalOpen: (open: boolean) => void
  setActiveToolbarPanel: (panel: string | null) => void
  setMaskTool: (tool: MaskTool) => void
  setEditWallCorners: (v: boolean) => void
  setHideWallMasks: (v: boolean) => void
}

export const useUIStore = create<UIState>((set) => ({
  materialCatalogOpen: false,
  exportModalOpen: false,
  activeToolbarPanel: null,
  maskTool: null,
  editWallCorners: false,
  hideWallMasks: false,
  setMaterialCatalogOpen: (materialCatalogOpen) => set({ materialCatalogOpen }),
  setExportModalOpen: (exportModalOpen) => set({ exportModalOpen }),
  setActiveToolbarPanel: (activeToolbarPanel) => set({ activeToolbarPanel }),
  setMaskTool: (maskTool) => set({ maskTool }),
  setEditWallCorners: (editWallCorners) => set({ editWallCorners }),
  setHideWallMasks: (hideWallMasks) => set({ hideWallMasks }),
}))
