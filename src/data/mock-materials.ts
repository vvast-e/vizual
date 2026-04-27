/** Локальные пресеты — fallback, если API материалов пустой или недоступен. */
import type { Material } from '@/types/material'

export type MockMaterialCategory = 'vagonka' | 'brus' | 'blockhouse' | 'boards' | 'planken'

export const MOCK_MATERIALS: Material[] = [
  // Планкен (1)
  { id: 'm21', name: 'Планкен Натуральный', category: 'planken', texture: { id: 't21', url: '/textures/planken/natural.png', name: 'Натуральный' } },

  // Вагонка (1)
  { id: 'm1', name: 'Вагонка Натуральная', category: 'vagonka', texture: { id: 't1', url: '/textures/vagonka/natural.png', name: 'Натуральная' } },

  // Брус (1)
  { id: 'm6', name: 'Брус Натуральный', category: 'brus', texture: { id: 't6', url: '/textures/brus/natural.png', name: 'Натуральный' } },

  // Блок-хаус (1)
  { id: 'm11', name: 'Блок-хаус Натуральный', category: 'blockhouse', texture: { id: 't11', url: '/textures/blockhouse/natural.png', name: 'Натуральный' } },

  // Доски (1)
  { id: 'm16', name: 'Доска Дуб', category: 'boards', texture: { id: 't16', url: '/textures/boards/oak.png', name: 'Дуб' } },
]

export const MATERIAL_CATEGORIES: { value: MockMaterialCategory | 'all'; label: string }[] = [
  { value: 'all', label: 'Все' },
  { value: 'planken', label: 'Планкен' },
  { value: 'vagonka', label: 'Вагонка' },
  { value: 'brus', label: 'Брус' },
  { value: 'blockhouse', label: 'Блок-хаус' },
  { value: 'boards', label: 'Доски' },
]
