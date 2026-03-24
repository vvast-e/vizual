import { create } from 'zustand'

export type MaskTool = 'brush' | 'rect' | 'lasso' | null
export type SceneMode = 'interior' | 'exterior'
export type EditCornersMode = 'polygon' | 'perspective'

interface UIState {
  materialCatalogOpen: boolean
  exportModalOpen: boolean
  activeToolbarPanel: string | null
  /** Инструмент маски: кисть, прямоугольник, лассо; null — пан */
  maskTool: MaskTool
  /** Режим правки углов стены (draggable handles) */
  editWallCorners: boolean
  /** В каком режиме мы находимся при правке углов? (Зелёные точки vs Синие точки) */
  editCornersMode: EditCornersMode
  /** Скрывать все синие маски стен (оверлеи) */
  hideWallMasks: boolean
  /** Режим сцены: интерьер или экстерьер */
  sceneMode: SceneMode
  /** Режим "Разделить фасад": два клика задают линию разреза */
  splitFacadeMode: boolean
  setMaterialCatalogOpen: (open: boolean) => void
  setExportModalOpen: (open: boolean) => void
  setActiveToolbarPanel: (panel: string | null) => void
  setMaskTool: (tool: MaskTool) => void
  setEditWallCorners: (v: boolean) => void
  setEditCornersMode: (m: EditCornersMode) => void
  setHideWallMasks: (v: boolean) => void
  setSceneMode: (mode: SceneMode) => void
  setSplitFacadeMode: (v: boolean) => void
}

export const useUIStore = create<UIState>((set) => ({
  materialCatalogOpen: false,
  exportModalOpen: false,
  activeToolbarPanel: null,
  maskTool: null,
  editWallCorners: false,
  editCornersMode: 'polygon',
  hideWallMasks: false,
  sceneMode: 'interior',
  splitFacadeMode: false,
  setMaterialCatalogOpen: (materialCatalogOpen) => set({ materialCatalogOpen }),
  setExportModalOpen: (exportModalOpen) => set({ exportModalOpen }),
  setActiveToolbarPanel: (activeToolbarPanel) => set({ activeToolbarPanel }),
  setMaskTool: (maskTool) => set({ maskTool }),
  setEditWallCorners: (editWallCorners) => set({ editWallCorners }),
  setEditCornersMode: (editCornersMode) => set({ editCornersMode }),
  setHideWallMasks: (hideWallMasks) => set({ hideWallMasks }),
  setSceneMode: (sceneMode) => set({ sceneMode }),
  setSplitFacadeMode: (splitFacadeMode) => set({ splitFacadeMode }),
}))
