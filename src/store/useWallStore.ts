import { create } from 'zustand'

export interface WallData {
  id: number
  corners: [number, number][]
  center: [number, number]
}

export interface WallImageSize {
  width: number
  height: number
}

interface WallState {
  walls: WallData[]
  /** Размер изображения, в координатах которого заданы walls (для масштабирования на канвас) */
  wallImageSize: WallImageSize | null
  selectedWallId: number | null
  isDetecting: boolean
  /** Назначенная текстура (URL) для каждой стены */
  wallTextures: Record<number, string | null>
  setWalls: (walls: WallData[], imageSize: WallImageSize | null) => void
  selectWall: (id: number | null) => void
  setDetecting: (v: boolean) => void
  setWallTexture: (wallId: number, textureUrl: string | null) => void
  updateWallCorners: (wallId: number, corners: [number, number][]) => void
}

export const useWallStore = create<WallState>((set) => ({
  walls: [],
  wallImageSize: null,
  selectedWallId: null,
  isDetecting: false,
  wallTextures: {},
  setWalls: (walls, wallImageSize) => set({ walls, wallImageSize, selectedWallId: null, wallTextures: {} }),
  selectWall: (id) => set({ selectedWallId: id }),
  setDetecting: (isDetecting) => set({ isDetecting }),
  setWallTexture: (wallId, textureUrl) =>
    set((s) => ({ wallTextures: { ...s.wallTextures, [wallId]: textureUrl } })),
  updateWallCorners: (wallId, corners) =>
    set((s) => ({
      walls: s.walls.map((w) => (w.id === wallId ? { ...w, corners } : w)),
    })),
}))
