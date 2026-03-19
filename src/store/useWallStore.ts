import { create } from 'zustand'

export interface WallData {
  id: number
  corners: [number, number][]
  /** Точный контур стены (для клиппинга/оверлея). Для интерьера может отсутствовать. */
  polygon?: [number, number][]
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
  /** Base64 PNG маски фасада (wall_minus_holes) для ручного split */
  exteriorMaskBase64: string | null
  setWalls: (walls: WallData[], imageSize: WallImageSize | null) => void
  setExteriorMaskBase64: (mask: string | null) => void
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
  exteriorMaskBase64: null,
  setWalls: (walls, wallImageSize) =>
    set((s) => ({
      walls,
      wallImageSize,
      selectedWallId: null,
      wallTextures: {},
      exteriorMaskBase64: wallImageSize == null ? null : s.exteriorMaskBase64,
    })),
  setExteriorMaskBase64: (exteriorMaskBase64) => set({ exteriorMaskBase64 }),
  selectWall: (id) => set({ selectedWallId: id }),
  setDetecting: (isDetecting) => set({ isDetecting }),
  setWallTexture: (wallId, textureUrl) =>
    set((s) => ({ wallTextures: { ...s.wallTextures, [wallId]: textureUrl } })),
  updateWallCorners: (wallId, corners) =>
    set((s) => ({
      walls: s.walls.map((w) => {
        if (w.id !== wallId) return w
        const cx = corners.length ? corners.reduce((acc, c) => acc + c[0], 0) / corners.length : w.center[0]
        const cy = corners.length ? corners.reduce((acc, c) => acc + c[1], 0) / corners.length : w.center[1]
        // Пока пользователь двигает углы, контур разумнее держать равным quad из corners.
        return {
          ...w,
          corners,
          polygon: w.polygon ? (corners as [number, number][]) : w.polygon,
          center: [Number(cx.toFixed(2)), Number(cy.toFixed(2))] as [number, number],
        }
      }),
    })),
}))
