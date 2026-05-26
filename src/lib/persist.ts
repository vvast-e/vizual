import { useWallStore } from '@/store/useWallStore'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useUIStore } from '@/store/useUIStore'
import { useMaterialStore } from '@/store/useMaterialStore'
import { useHistoryStore } from '@/store/useHistoryStore'

const STORAGE_KEY = 'vizual-state'

interface PersistedState {
  photoDataUrl: string | null
  walls: ReturnType<typeof useWallStore.getState>['walls']
  wallImageSize: ReturnType<typeof useWallStore.getState>['wallImageSize']
  wallTextures: ReturnType<typeof useWallStore.getState>['wallTextures']
  exteriorMaskBase64: string | null
  interiorMaskBase64: string | null
  sceneMode: ReturnType<typeof useUIStore.getState>['sceneMode']
}

export function saveState() {
  try {
    const wallState = useWallStore.getState()
    const visualState = useVisualizerStore.getState()
    const uiState = useUIStore.getState()

    const state: PersistedState = {
      photoDataUrl: visualState.photoDataUrl,
      walls: wallState.walls,
      wallImageSize: wallState.wallImageSize,
      wallTextures: wallState.wallTextures,
      exteriorMaskBase64: wallState.exteriorMaskBase64,
      interiorMaskBase64: wallState.interiorMaskBase64,
      sceneMode: uiState.sceneMode,
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // localStorage может быть недоступен (приватный режим, quota exceeded)
  }
}

export function loadState(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return false
    }

    const state: PersistedState = JSON.parse(raw)
    if (!state.photoDataUrl) return false
    // Защита от "битого" restore: фото есть, а стен/размера нет — в редакторе не появятся ни маски, ни тур.
    const hasRenderableWalls =
      Array.isArray(state.walls) &&
      state.walls.length > 0 &&
      !!state.wallImageSize &&
      Number(state.wallImageSize.width) > 0 &&
      Number(state.wallImageSize.height) > 0
    if (!hasRenderableWalls) {
      localStorage.removeItem(STORAGE_KEY)
      return false
    }

    const history = useHistoryStore.getState()
    history.setApplying(true)
    try {
      useVisualizerStore.getState().setPhotoDataUrl(state.photoDataUrl)
      useUIStore.getState().setSceneMode(state.sceneMode ?? 'interior')

      if (state.walls && state.walls.length > 0 && state.wallImageSize) {
        useWallStore.getState().setWalls(state.walls, state.wallImageSize)
      }
      if (state.exteriorMaskBase64) {
        useWallStore.getState().setExteriorMaskBase64(state.exteriorMaskBase64)
      }
      if (state.interiorMaskBase64) {
        useWallStore.getState().setInteriorMaskBase64(state.interiorMaskBase64)
      }
    } finally {
      history.setApplying(false)
      history.clear()
    }

    return true
  } catch {
    return false
  }
}

export function clearState() {
  localStorage.removeItem(STORAGE_KEY)
  const history = useHistoryStore.getState()
  history.setApplying(true)
  try {
    useVisualizerStore.getState().setPhotoDataUrl(null)
    useWallStore.getState().setWalls([], null)
    useWallStore.getState().setExteriorMaskBase64(null)
    useWallStore.getState().setInteriorMaskBase64(null)
  } finally {
    history.setApplying(false)
    history.clear()
  }
}

export function initPersistence() {
  const history = useHistoryStore.getState()
  history.setApplying(true)
  try {
    useMaterialStore.getState().setSelectedMaterial(null)
    useMaterialStore.getState().setSelectedColor(null)
  } finally {
    history.setApplying(false)
    history.clear()
  }
  useWallStore.subscribe(saveState)
  useVisualizerStore.subscribe(saveState)
}
