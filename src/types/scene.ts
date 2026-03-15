/**
 * Типы для 3D сцены (интерьер / экстерьер).
 */

import type { MaterialId } from './material'

export type SceneMode = 'room' | 'house'

export interface SurfaceMaterialBinding {
  /** Имя меша/поверхности в модели */
  meshName: string
  materialId: MaterialId | null
}

export interface SceneState {
  mode: SceneMode
  bindings: SurfaceMaterialBinding[]
}
