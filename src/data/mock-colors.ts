/**
 * Локальные пресеты — fallback, если GET /api/colors пустой или недоступен.
 */
import type { Color } from '@/types/material'

export type MockColorCategory = 'wood' | 'metal' | 'wallpaper' | 'paint'

export interface ColorWithCategory extends Color {
  category: MockColorCategory
}

export const MOCK_COLORS: ColorWithCategory[] = [
  // Дерево (5)
  { hex: '#8B4513', name: 'Орех', category: 'wood' },
  { hex: '#D2691E', name: 'Вишня', category: 'wood' },
  { hex: '#CD853F', name: 'Золотистый дуб', category: 'wood' },
  { hex: '#DEB887', name: 'Буковый', category: 'wood' },
  { hex: '#A0522D', name: 'Сиена', category: 'wood' },

  // Металл (5)
  { hex: '#C0C0C0', name: 'Серебро', category: 'metal' },
  { hex: '#FFD700', name: 'Золото', category: 'metal' },
  { hex: '#B87333', name: 'Медь', category: 'metal' },
  { hex: '#71797E', name: 'Сталь', category: 'metal' },
  { hex: '#464646', name: 'Графит', category: 'metal' },

  // Обои (5)
  { hex: '#F5F5DC', name: 'Бежевый', category: 'wallpaper' },
  { hex: '#FFFDD0', name: 'Кремовый', category: 'wallpaper' },
  { hex: '#E6E6FA', name: 'Лаванда', category: 'wallpaper' },
  { hex: '#FFF0F5', name: 'Розовый туман', category: 'wallpaper' },
  { hex: '#F0FFF0', name: 'Мятный', category: 'wallpaper' },

  // Краска (5)
  { hex: '#1E90FF', name: 'Додо синий', category: 'paint' },
  { hex: '#32CD32', name: 'Лаймовый', category: 'paint' },
  { hex: '#FF6347', name: 'Помидор', category: 'paint' },
  { hex: '#9370DB', name: 'Фиолетовый', category: 'paint' },
  { hex: '#20B2AA', name: 'Бирюзовый', category: 'paint' },
]

export const COLOR_CATEGORIES: { value: MockColorCategory | 'all'; label: string }[] = [
  { value: 'all', label: 'Все' },
  { value: 'wood', label: 'Дерево' },
  { value: 'metal', label: 'Металл' },
  { value: 'wallpaper', label: 'Обои' },
  { value: 'paint', label: 'Краска' },
]
