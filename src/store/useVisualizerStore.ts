import { create } from 'zustand'
import type { SceneMode } from '@/types/scene'

interface VisualizerState {
  /** Режим: 2D (фото + текстура) или 3D (сцена) */
  viewMode: '2d' | '3d'
  /** Для 3D: комната или дом */
  sceneMode: SceneMode
  /** Data URL загруженного фото (2D) */
  photoDataUrl: string | null
  setViewMode: (mode: '2d' | '3d') => void
  setSceneMode: (mode: SceneMode) => void
  setPhotoDataUrl: (url: string | null) => void
}

export const useVisualizerStore = create<VisualizerState>((set) => ({
  viewMode: '2d',
  sceneMode: 'room',
  photoDataUrl: null,
  setViewMode: (viewMode) => set({ viewMode }),
  setSceneMode: (sceneMode) => set({ sceneMode }),
  setPhotoDataUrl: (photoDataUrl) => set({ photoDataUrl }),
}))
