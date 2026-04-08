import { useCallback, useMemo } from 'react'
import { useMaterialStore } from '@/store/useMaterialStore'
import { loadTextureImage } from '@/lib/texture-processor'
import { applyHsvColorShift } from '@/lib/color-utils'
import { PRESET_COLORS } from '@/lib/constants'

/**
 * Колоризация текстуры: загрузка + HSV-сдвиг цвета (сохраняет фактуру).
 */
export function useColorize() {
  const selectedColor = useMaterialStore((s) => s.selectedColor)
  const colorizeOpacity = useMaterialStore((s) => s.colorizeOpacity)
  const setSelectedColor = useMaterialStore((s) => s.setSelectedColor)
  const setColorizeOpacity = useMaterialStore((s) => s.setColorizeOpacity)

  const presets = useMemo(
    (): Array<{ hex: string; name?: string }> => [...PRESET_COLORS],
    []
  )

  const applyColor = useCallback(
    (
      source: HTMLImageElement | HTMLCanvasElement,
      colorHex?: string,
      opacity?: number
    ): HTMLCanvasElement => {
      const hex = colorHex ?? selectedColor?.hex ?? '#ffffff'
      const op = opacity ?? colorizeOpacity
      return applyHsvColorShift(source, hex, op)
    },
    [selectedColor?.hex, colorizeOpacity]
  )

  const getColorizedTextureUrl = useCallback(
    async (textureUrl: string): Promise<string> => {
      const img = await loadTextureImage(textureUrl)
      const canvas = applyColor(img, selectedColor?.hex, colorizeOpacity)
      return canvas.toDataURL('image/png')
    },
    [applyColor, selectedColor?.hex, colorizeOpacity]
  )

  return {
    selectedColor,
    colorizeOpacity,
    setSelectedColor,
    setColorizeOpacity,
    presets,
    applyColor,
    getColorizedTextureUrl,
  }
}
