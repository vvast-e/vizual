import { useCallback } from 'react'
import { downloadDataUrl, defaultExportFilename } from '@/lib/export-utils'

/**
 * Хук для экспорта 2D канваса в PNG.
 * Для 3D сцены потребуется отдельная логика (render to texture).
 */
export function useExport() {
  const exportToPng = useCallback((dataUrl: string, filename?: string) => {
    const name = filename ?? defaultExportFilename('visualizer')
    downloadDataUrl(dataUrl, name)
  }, [])

  return { exportToPng }
}
