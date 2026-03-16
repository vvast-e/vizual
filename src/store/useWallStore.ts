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
  setWalls: (walls: WallData[], imageSize: WallImageSize | null) => void
  selectWall: (id: number | null) => void
  setDetecting: (v: boolean) => void
}

export const useWallStore = create<WallState>((set) => ({
  walls: [],
  wallImageSize: null,
  selectedWallId: null,
  isDetecting: false,
  setWalls: (walls, wallImageSize) => set({ walls, wallImageSize, selectedWallId: null }),
  selectWall: (id) => set({ selectedWallId: id }),
  setDetecting: (isDetecting) => set({ isDetecting }),
}))
