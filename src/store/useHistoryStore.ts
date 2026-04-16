import { create } from 'zustand'
import { useWallStore } from './useWallStore'
import { useUIStore } from './useUIStore'

export interface SerializedState {
  walls: ReturnType<typeof useWallStore.getState>['walls']
  selectedWallId: ReturnType<typeof useWallStore.getState>['selectedWallId']
  editWallCorners: ReturnType<typeof useUIStore.getState>['editWallCorners']
  editCornersMode: ReturnType<typeof useUIStore.getState>['editCornersMode']
  wallImageSize: ReturnType<typeof useWallStore.getState>['wallImageSize']
}

interface HistoryState {
  past: SerializedState[]
  future: SerializedState[]
}

interface HistoryActions {
  push: () => void
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
}

export const useHistoryStore = create<HistoryState & HistoryActions>((set, get) => ({
  past: [],
  future: [],

  push: () => {
    const wallStore = useWallStore.getState()
    const uiStore = useUIStore.getState()
    const state: SerializedState = {
      walls: JSON.parse(JSON.stringify(wallStore.walls)),
      selectedWallId: wallStore.selectedWallId,
      editWallCorners: uiStore.editWallCorners,
      editCornersMode: uiStore.editCornersMode,
      wallImageSize: wallStore.wallImageSize,
    }
    set((s) => ({
      past: [...s.past, state],
      future: [],
    }))
  },

  undo: () => {
    const { past } = get()
    if (past.length === 0) return

    const wallStore = useWallStore.getState()
    const uiStore = useUIStore.getState()
    const current: SerializedState = {
      walls: JSON.parse(JSON.stringify(wallStore.walls)),
      selectedWallId: wallStore.selectedWallId,
      editWallCorners: uiStore.editWallCorners,
      editCornersMode: uiStore.editCornersMode,
      wallImageSize: wallStore.wallImageSize,
    }

    const newPast = [...past]
    const prev = newPast.pop()!

    set({ past: newPast, future: [current, ...get().future] })

    useWallStore.getState().setWalls(prev.walls, prev.wallImageSize, false)
    if (prev.selectedWallId != null) {
      useWallStore.getState().selectWall(prev.selectedWallId)
    }
    useUIStore.getState().setEditWallCorners(prev.editWallCorners)
    if (prev.editCornersMode) {
      useUIStore.getState().setEditCornersMode(prev.editCornersMode)
    }
  },

  redo: () => {
    const { future } = get()
    if (future.length === 0) return

    const wallStore = useWallStore.getState()
    const uiStore = useUIStore.getState()
    const current: SerializedState = {
      walls: JSON.parse(JSON.stringify(wallStore.walls)),
      selectedWallId: wallStore.selectedWallId,
      editWallCorners: uiStore.editWallCorners,
      editCornersMode: uiStore.editCornersMode,
      wallImageSize: wallStore.wallImageSize,
    }

    const newFuture = [...future]
    const next = newFuture.shift()!

    set({ past: [...get().past, current], future: newFuture })

    useWallStore.getState().setWalls(next.walls, next.wallImageSize, false)
    if (next.selectedWallId != null) {
      useWallStore.getState().selectWall(next.selectedWallId)
    }
    useUIStore.getState().setEditWallCorners(next.editWallCorners)
    if (next.editCornersMode) {
      useUIStore.getState().setEditCornersMode(next.editCornersMode)
    }
  },

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,
}))
