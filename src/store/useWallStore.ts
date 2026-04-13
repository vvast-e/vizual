import { create } from 'zustand'

/** Два региона варпа (нижний + фронтон), с бэкенда экстерьера при len(polygon) >= 5. */
export interface WallWarpRegion {
  corners: [number, number][]
  polygon: [number, number][]
}

export interface WallData {
  id: number
  corners: [number, number][]
  /** Точный контур стены (для клиппинга/оверлея). Для интерьера может отсутствовать. */
  polygon?: [number, number][]
  /** Двухучастковый варп (стена + фронтон). Сбрасывается при ручном движении углов. */
  regions?: WallWarpRegion[]
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
  setWalls: (walls: WallData[], imageSize: WallImageSize | null, resetTextures?: boolean) => void
  setExteriorMaskBase64: (mask: string | null) => void
  selectWall: (id: number | null) => void
  setDetecting: (v: boolean) => void
  setWallTexture: (wallId: number, textureUrl: string | null) => void
  updateWallCorners: (wallId: number, corners: [number, number][]) => void
  /** Обновить одну вершину polygon по индексу (для экстерьера). */
  updateWallPolygonVertex: (wallId: number, vertexIndex: number, point: [number, number]) => void
  /** Удалить стену из списка */
  removeWall: (wallId: number) => void
  /** Добавить кастомную маску (нарисованную пользователем) */
  addCustomWall: (wall: WallData) => void
}

export const useWallStore = create<WallState>((set) => ({
  walls: [],
  wallImageSize: null,
  selectedWallId: null,
  isDetecting: false,
  wallTextures: {},
  exteriorMaskBase64: null,
  setWalls: (walls, wallImageSize, resetTextures = true) =>
    set((s) => ({
      walls,
      wallImageSize,
      selectedWallId: null,
      wallTextures: resetTextures
        ? {}
        : Object.fromEntries(
            Object.entries(s.wallTextures).filter(([id]) => walls.some((w) => w.id === Number(id)))
          ),
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
        // Для интерьера (без polygon) — corners и polygon совпадают.
        // Для экстерьера polygon остаётся неизменным (редактируется через updateWallPolygonVertex).
        return {
          ...w,
          corners,
          // Не перезаписываем polygon corners'ами — polygon может иметь больше точек
          regions: undefined,
          center: [Number(cx.toFixed(2)), Number(cy.toFixed(2))] as [number, number],
        }
      }),
    })),
  updateWallPolygonVertex: (wallId, vertexIndex, point) =>
    set((s) => ({
      walls: s.walls.map((w) => {
        if (w.id !== wallId) return w
        const poly = w.polygon ? [...w.polygon] : [...w.corners]
        if (vertexIndex < 0 || vertexIndex >= poly.length) return w
        poly[vertexIndex] = point
        const cx = poly.reduce((acc, c) => acc + c[0], 0) / poly.length
        const cy = poly.reduce((acc, c) => acc + c[1], 0) / poly.length
        return {
          ...w,
          polygon: poly as [number, number][],
          // Важно: оставляем corners без изменений, чтобы синий прямоугольник перспективы не слетал!
          regions: undefined,
          center: [Number(cx.toFixed(2)), Number(cy.toFixed(2))] as [number, number],
        }
      }),
    })),
  removeWall: (wallId) =>
    set((s) => {
      const newTextures = { ...s.wallTextures }
      delete newTextures[wallId]
      return {
        walls: s.walls.filter((w) => w.id !== wallId),
        wallTextures: newTextures,
        selectedWallId: s.selectedWallId === wallId ? null : s.selectedWallId,
      }
    }),
  addCustomWall: (wall) =>
    set((s) => ({
      walls: [...s.walls, wall],
      selectedWallId: wall.id,
    })),
}))
