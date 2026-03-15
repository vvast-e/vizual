import { useCallback, useState } from 'react'
import { loadTextureImage, getPatternCanvas } from '@/lib/texture-processor'

/**
 * Загрузка и кэширование текстур.
 * Возвращает URL изображения или data URL паттерна для Fabric.
 */
export function useTexture() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadTexture = useCallback(async (url: string): Promise<HTMLImageElement | null> => {
    setLoading(true)
    setError(null)
    try {
      const img = await loadTextureImage(url)
      return img
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка загрузки текстуры')
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const getPattern = useCallback(
    async (
      textureUrl: string,
      repeat: 'repeat' | 'repeat-x' | 'repeat-y' = 'repeat'
    ): Promise<HTMLCanvasElement | null> => {
      setLoading(true)
      setError(null)
      try {
        return await getPatternCanvas(textureUrl, repeat)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Ошибка создания паттерна')
        return null
      } finally {
        setLoading(false)
      }
    },
    []
  )

  return { loadTexture, getPattern, loading, error }
}
