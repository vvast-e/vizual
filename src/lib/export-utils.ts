/**
 * Экспорт канваса/сцены в PNG (download).
 */

/**
 * Скачивает data URL как файл с заданным именем.
 */
export function downloadDataUrl(dataUrl: string, filename: string): void {
  const link = document.createElement('a')
  link.href = dataUrl
  link.download = filename
  link.rel = 'noopener'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/**
 * Генерирует имя файла с датой.
 */
export function defaultExportFilename(prefix: string = 'visualizer'): string {
  const date = new Date()
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  const h = String(date.getHours()).padStart(2, '0')
  const min = String(date.getMinutes()).padStart(2, '0')
  return `${prefix}-${y}${m}${d}-${h}${min}.png`
}
