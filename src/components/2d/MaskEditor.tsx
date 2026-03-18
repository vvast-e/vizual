import { Paintbrush, Eraser, Square, Lasso } from 'lucide-react'
import type { MaskTool } from '@/store/useUIStore'

export interface MaskEditorProps {
  drawingMode: boolean
  onDrawingModeChange: (enabled: boolean) => void
  brushSize: number
  onBrushSizeChange: (size: number) => void
  onClearMask: () => void
  maskTool: MaskTool
  onMaskToolChange: (tool: MaskTool) => void
  onClearMaskToolState: () => void
  onFinishLasso: () => void
  disabled?: boolean
  className?: string
}

export function MaskEditor({
  drawingMode,
  onDrawingModeChange,
  brushSize,
  onBrushSizeChange,
  onClearMask,
  maskTool,
  onMaskToolChange,
  onClearMaskToolState,
  onFinishLasso,
  disabled,
  className = '',
}: MaskEditorProps) {
  void drawingMode
  const handleTool = (tool: MaskTool) => {
    if (disabled) return
    if (tool === maskTool) {
      onClearMaskToolState()
      onMaskToolChange(null)
      onDrawingModeChange(false)
      return
    }
    onClearMaskToolState()
    onMaskToolChange(tool)
    onDrawingModeChange(tool === 'brush')
  }

  return (
    <div
      className={`flex flex-wrap items-center gap-3 ${className}`}
      role="toolbar"
      aria-label="Инструменты маски"
    >
      <div className="flex gap-1">
        <button
          type="button"
          title="Кисть"
          aria-pressed={maskTool === 'brush'}
          onClick={() => handleTool('brush')}
          disabled={disabled}
          className={`rounded border p-2 ${maskTool === 'brush' ? 'border-gray-800 bg-gray-200' : 'border-gray-300'} ${disabled ? 'opacity-50' : ''}`}
        >
          <Paintbrush className="h-4 w-4" aria-hidden />
        </button>
        <button
          type="button"
          title="Прямоугольник"
          aria-pressed={maskTool === 'rect'}
          onClick={() => handleTool('rect')}
          disabled={disabled}
          className={`rounded border p-2 ${maskTool === 'rect' ? 'border-gray-800 bg-gray-200' : 'border-gray-300'} ${disabled ? 'opacity-50' : ''}`}
        >
          <Square className="h-4 w-4" aria-hidden />
        </button>
        <button
          type="button"
          title="Лассо"
          aria-pressed={maskTool === 'lasso'}
          onClick={() => handleTool('lasso')}
          disabled={disabled}
          className={`rounded border p-2 ${maskTool === 'lasso' ? 'border-gray-800 bg-gray-200' : 'border-gray-300'} ${disabled ? 'opacity-50' : ''}`}
        >
          <Lasso className="h-4 w-4" aria-hidden />
        </button>
        <button
          type="button"
          title="Очистить маску"
          onClick={onClearMask}
          disabled={disabled}
          className="rounded border border-gray-300 p-2 hover:bg-gray-100 disabled:opacity-50"
        >
          <Eraser className="h-4 w-4" aria-hidden />
        </button>
      </div>
      {maskTool === 'lasso' && (
        <button
          type="button"
          onClick={onFinishLasso}
          disabled={disabled}
          className="rounded border border-gray-800 bg-gray-100 px-2 py-1 text-xs hover:bg-gray-200 disabled:opacity-50"
        >
          Завершить лассо
        </button>
      )}
      {maskTool === 'brush' && (
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-600">Размер кисти:</label>
          <input
            type="range"
            min={5}
            max={80}
            value={brushSize}
            onChange={(e) => onBrushSizeChange(Number(e.target.value))}
            disabled={disabled}
            className="w-24"
            aria-label="Размер кисти"
          />
          <span className="text-xs text-gray-500">{brushSize}</span>
        </div>
      )}
    </div>
  )
}
