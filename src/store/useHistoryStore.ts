import { create } from 'zustand'
import { useWallStore } from './useWallStore'
import { useUIStore } from './useUIStore'
import { useMaterialStore } from './useMaterialStore'
import type { WallData, WallImageSize } from './useWallStore'
import type { SceneMode, EditCornersMode } from './useUIStore'
import type { Material, Color } from '@/types/material'

export interface SerializedState {
  walls: WallData[]
  wallImageSize: WallImageSize | null
  selectedWallId: number | null
  wallTextures: Record<number, string | null>
  wallRawTextures: Record<number, string | null>
  exteriorMaskBase64: string | null
  interiorMaskBase64: string | null
  selectedMaterial: Material | null
  selectedColor: Color | null
  colorizeOpacity: number
  sceneMode: SceneMode
  hideWallMasks: boolean
  wallVisibility: Record<number, boolean>
  editWallCorners: boolean
  editCornersMode: EditCornersMode
}

interface HistoryState {
  past: SerializedState[]
  future: SerializedState[]
  /** Флаг подавления записи во время undo/redo (чтобы не зацикливать). */
  applying: boolean
  /** Ключ последнего коалесцируемого события (для дребезга dragов/слайдеров). */
  lastCoalesceKey: string | null
  /** Время последнего record (ms). */
  lastRecordedAt: number
}

interface HistoryActions {
  /** Записать текущее состояние в past (вызывать ДО мутации). */
  record: (coalesceKey?: string, coalesceWindowMs?: number) => void
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
  /** Полная очистка истории (после rehydrate/reset фото). */
  clear: () => void
  /** Включить/выключить флаг подавления (для безопасных программных правок). */
  setApplying: (v: boolean) => void
}

function snapshot(): SerializedState {
  const w = useWallStore.getState()
  const u = useUIStore.getState()
  const m = useMaterialStore.getState()
  return {
    walls: JSON.parse(JSON.stringify(w.walls)),
    wallImageSize: w.wallImageSize ? { ...w.wallImageSize } : null,
    selectedWallId: w.selectedWallId,
    wallTextures: { ...w.wallTextures },
    wallRawTextures: { ...w.wallRawTextures },
    exteriorMaskBase64: w.exteriorMaskBase64,
    interiorMaskBase64: w.interiorMaskBase64,
    selectedMaterial: m.selectedMaterial ? { ...m.selectedMaterial } : null,
    selectedColor: m.selectedColor ? { ...m.selectedColor } : null,
    colorizeOpacity: m.colorizeOpacity,
    sceneMode: u.sceneMode,
    hideWallMasks: u.hideWallMasks,
    wallVisibility: { ...u.wallVisibility },
    editWallCorners: u.editWallCorners,
    editCornersMode: u.editCornersMode,
  }
}

function apply(state: SerializedState) {
  // Прямой setState минует «обёрнутые» actions — record не сработает.
  useWallStore.setState({
    walls: state.walls,
    wallImageSize: state.wallImageSize,
    selectedWallId: state.selectedWallId,
    wallTextures: state.wallTextures,
    wallRawTextures: state.wallRawTextures,
    exteriorMaskBase64: state.exteriorMaskBase64,
    interiorMaskBase64: state.interiorMaskBase64,
  })
  useUIStore.setState({
    sceneMode: state.sceneMode,
    hideWallMasks: state.hideWallMasks,
    wallVisibility: state.wallVisibility,
    editWallCorners: state.editWallCorners,
    editCornersMode: state.editCornersMode,
  })
  useMaterialStore.setState({
    selectedMaterial: state.selectedMaterial,
    selectedColor: state.selectedColor,
    colorizeOpacity: state.colorizeOpacity,
  })
}

const MAX_HISTORY = 100

export const useHistoryStore = create<HistoryState & HistoryActions>((set, get) => ({
  past: [],
  future: [],
  applying: false,
  lastCoalesceKey: null,
  lastRecordedAt: 0,

  record: (coalesceKey, coalesceWindowMs = 500) => {
    if (get().applying) return
    const now = Date.now()
    if (
      coalesceKey &&
      get().lastCoalesceKey === coalesceKey &&
      now - get().lastRecordedAt < coalesceWindowMs
    ) {
      // Тот же drag/slide — продлеваем окно, но новую запись не добавляем.
      set({ lastRecordedAt: now })
      return
    }
    const snap = snapshot()
    set((s) => {
      const past = [...s.past, snap]
      if (past.length > MAX_HISTORY) past.splice(0, past.length - MAX_HISTORY)
      return {
        past,
        future: [],
        lastCoalesceKey: coalesceKey ?? null,
        lastRecordedAt: now,
      }
    })
  },

  undo: () => {
    const { past, applying } = get()
    if (applying || past.length === 0) return
    const current = snapshot()
    const newPast = [...past]
    const prev = newPast.pop()!
    set((s) => ({
      past: newPast,
      future: [current, ...s.future],
      applying: true,
      lastCoalesceKey: null,
      lastRecordedAt: 0,
    }))
    try {
      apply(prev)
    } finally {
      set({ applying: false })
    }
  },

  redo: () => {
    const { future, applying } = get()
    if (applying || future.length === 0) return
    const current = snapshot()
    const newFuture = [...future]
    const next = newFuture.shift()!
    set((s) => ({
      past: [...s.past, current],
      future: newFuture,
      applying: true,
      lastCoalesceKey: null,
      lastRecordedAt: 0,
    }))
    try {
      apply(next)
    } finally {
      set({ applying: false })
    }
  },

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,
  clear: () =>
    set({ past: [], future: [], lastCoalesceKey: null, lastRecordedAt: 0 }),
  setApplying: (v) => set({ applying: v }),
}))
