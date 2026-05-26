import { create } from 'zustand'
import { useHistoryStore } from './useHistoryStore'

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
  /** Видимость каждой стены (id → visible) */
  wallVisibility: Record<number, boolean>
  /** Режим сцены: интерьер или экстерьер */
  sceneMode: SceneMode
  /** Запущен ли обучающий тур */
  tourActive: boolean
  /** Режим двух кликов: линия разреза фасада (экстерьер) */
  exteriorSplitLineActive: boolean
  setMaterialCatalogOpen: (open: boolean) => void
  setExportModalOpen: (open: boolean) => void
  setActiveToolbarPanel: (panel: string | null) => void
  setMaskTool: (tool: MaskTool) => void
  setEditWallCorners: (v: boolean) => void
  setEditCornersMode: (m: EditCornersMode) => void
  setHideWallMasks: (v: boolean) => void
  setSceneMode: (mode: SceneMode) => void
  toggleWallVisibility: (wallId: number) => void
  setTourActive: (v: boolean) => void
  setExteriorSplitLineActive: (v: boolean) => void
}

export const useUIStore = create<UIState>((set) => ({
  materialCatalogOpen: false,
  exportModalOpen: false,
  activeToolbarPanel: null,
  maskTool: null,
  editWallCorners: false,
  editCornersMode: 'polygon',
  hideWallMasks: false,
  wallVisibility: {},
  sceneMode: 'interior',
  tourActive: false,
  exteriorSplitLineActive: false,
  setMaterialCatalogOpen: (materialCatalogOpen) => set({ materialCatalogOpen }),
  setExportModalOpen: (exportModalOpen) => set({ exportModalOpen }),
  setActiveToolbarPanel: (activeToolbarPanel) => set({ activeToolbarPanel }),
  setMaskTool: (maskTool) => set({ maskTool }),
  setEditWallCorners: (editWallCorners) => {
    useHistoryStore.getState().record('editWallCorners')
    set({ editWallCorners })
  },
  setEditCornersMode: (editCornersMode) => {
    useHistoryStore.getState().record('editCornersMode')
    set({ editCornersMode })
  },
  setHideWallMasks: (hideWallMasks) => {
    useHistoryStore.getState().record('hideWallMasks')
    set({ hideWallMasks })
  },
  setSceneMode: (sceneMode) => {
    useHistoryStore.getState().record('sceneMode')
    set({ sceneMode })
  },
  toggleWallVisibility: (wallId) => {
    useHistoryStore.getState().record(`wallVisibility-${wallId}`)
    set((s) => ({
      wallVisibility: {
        ...s.wallVisibility,
        [wallId]: s.wallVisibility[wallId] === false ? true : false,
      },
    }))
  },
  setTourActive: (tourActive) => set({ tourActive }),
  setExteriorSplitLineActive: (exteriorSplitLineActive) => set({ exteriorSplitLineActive }),
}))
