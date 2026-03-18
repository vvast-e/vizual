/**
 * Типы для каталога материалов визуализатора.
 */

export type MaterialId = string

export type MaterialCategory = 'vagonka' | 'brus' | 'blockhouse' | 'boards' | 'planken' | 'other'

export interface Texture {
  id: string
  url: string
  name: string
  /** Путь к превью (swatch) */
  thumbnailUrl?: string
}

export interface Color {
  hex: string
  name?: string
}

export interface Material {
  id: MaterialId
  name: string
  category: MaterialCategory
  texture: Texture
  /** Пресетные цвета покраски */
  presetColors?: Color[]
}
