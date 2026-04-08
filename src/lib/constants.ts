/**
 * Константы приложения (размеры, лимиты, цвета).
 */

/** Максимальный размер загружаемого фото (байты). NF5: 10MB */
export const MAX_PHOTO_SIZE_BYTES = 10 * 1024 * 1024

/** Допустимые MIME-типы для фото */
export const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const

/** Базовый путь к текстурам в public */
export const TEXTURES_BASE_PATH = '/textures'

/** Базовый путь к 3D моделям в public */
export const MODELS_BASE_PATH = '/models'

/** Пресетные цвета для колоризации (HSV-сдвиг) */
export const PRESET_COLORS = [
  { hex: '#ffffff', name: 'Белый' },
  { hex: '#f5f5f5', name: 'Слоновая кость' },
  { hex: '#e5e5e5', name: 'Серый светлый' },
  { hex: '#a0a0a0', name: 'Серый' },
  { hex: '#8b4513', name: 'Коричневый' },
  { hex: '#654321', name: 'Тёмное дерево' },
  { hex: '#daa520', name: 'Золотистый' },
  { hex: '#cd853f', name: 'Песочный' },
  { hex: '#a0522d', name: 'Сиена' },
  { hex: '#d2691e', name: 'Шоколадный' },
  { hex: '#b22222', name: 'Красный' },
  { hex: '#2e8b57', name: 'Зелёный' },
  { hex: '#4682b4', name: 'Синий' },
  { hex: '#708090', name: 'Сланцевый' },
] as const
