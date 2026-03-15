import { create } from 'zustand'

interface UIState {
  /** Открыта ли модалка каталога материалов */
  materialCatalogOpen: boolean
  /** Открыта ли модалка экспорта */
  exportModalOpen: boolean
  /** Активная панель тулбара */
  activeToolbarPanel: string | null
  setMaterialCatalogOpen: (open: boolean) => void
  setExportModalOpen: (open: boolean) => void
  setActiveToolbarPanel: (panel: string | null) => void
}

export const useUIStore = create<UIState>((set) => ({
  materialCatalogOpen: false,
  exportModalOpen: false,
  activeToolbarPanel: null,
  setMaterialCatalogOpen: (materialCatalogOpen) => set({ materialCatalogOpen }),
  setExportModalOpen: (exportModalOpen) => set({ exportModalOpen }),
  setActiveToolbarPanel: (activeToolbarPanel) => set({ activeToolbarPanel }),
}))
