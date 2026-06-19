/**
 * Типы для каталога материалов визуализатора.
 */

export type MaterialId = string

export type MaterialCategory = 'vagonka' | 'brus' | 'blockhouse' | 'boards' | 'planken' | 'other'

export type ColorCategory = 'wood' | 'metal' | 'wallpaper' | 'paint'

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
  /** Идентификатор из API (для быстрого доступа и удаления) */
  id?: string
  /** Превью-изображение (каталог цветов из БД) */
  swatchUrl?: string
}

export interface Material {
  id: MaterialId
  name: string
  category: MaterialCategory
  texture: Texture
  /** Пресетные цвета покраски */
  presetColors?: Color[]
}
