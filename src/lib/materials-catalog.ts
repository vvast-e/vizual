import type { Material } from '@/types/material'
import { TEXTURES_BASE_PATH } from './constants'

/**
 * Каталог материалов по умолчанию (пути к текстурам в public).
 * Используется как fallback, если GET /api/materials не вернул данных или недоступен.
 */
export const DEFAULT_MATERIALS: Material[] = [
  {
    id: 'vagonka-natural',
    name: 'Вагонка натуральная',
    category: 'vagonka',
    texture: {
      id: 'tex-vagonka-natural',
      url: `${TEXTURES_BASE_PATH}/vagonka/natural.png`,
      name: 'Натуральная',
      thumbnailUrl: `${TEXTURES_BASE_PATH}/vagonka/swatches/natural.png`,
    },
    presetColors: [],
  },
  {
    id: 'vagonka-light',
    name: 'Вагонка светлая',
    category: 'vagonka',
    texture: {
      id: 'tex-vagonka-light',
      url: `${TEXTURES_BASE_PATH}/vagonka/light.png`,
      name: 'Светлая',
      thumbnailUrl: `${TEXTURES_BASE_PATH}/vagonka/swatches/light.png`,
    },
    presetColors: [],
  },
  {
    id: 'brus-natural',
    name: 'Имитация бруса',
    category: 'brus',
    texture: {
      id: 'tex-brus-natural',
      url: `${TEXTURES_BASE_PATH}/brus/natural.png`,
      name: 'Натуральный',
      thumbnailUrl: `${TEXTURES_BASE_PATH}/brus/swatches/natural.png`,
    },
    presetColors: [],
  },
  {
    id: 'blockhouse-natural',
    name: 'Блок-хаус',
    category: 'blockhouse',
    texture: {
      id: 'tex-blockhouse-natural',
      url: `${TEXTURES_BASE_PATH}/blockhouse/natural.png`,
      name: 'Натуральный',
      thumbnailUrl: `${TEXTURES_BASE_PATH}/blockhouse/swatches/natural.png`,
    },
    presetColors: [],
  },
  {
    id: 'boards-oak',
    name: 'Доска дуб',
    category: 'boards',
    texture: {
      id: 'tex-boards-oak',
      url: `${TEXTURES_BASE_PATH}/boards/oak.png`,
      name: 'Дуб',
      thumbnailUrl: `${TEXTURES_BASE_PATH}/boards/swatches/oak.png`,
    },
    presetColors: [],
  },
  {
    id: 'planken-natural',
    name: 'Планкен натуральный',
    category: 'planken',
    texture: {
      id: 'tex-planken-natural',
      url: `${TEXTURES_BASE_PATH}/planken/natural.png`,
      name: 'Натуральный',
      thumbnailUrl: `${TEXTURES_BASE_PATH}/planken/swatches/natural.png`,
    },
    presetColors: [],
  },
]
