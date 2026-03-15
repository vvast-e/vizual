import { Paintbrush, Eraser } from 'lucide-react'

export interface MaskEditorProps {
  drawingMode: boolean
  onDrawingModeChange: (enabled: boolean) => void
  brushSize: number
  onBrushSizeChange: (size: number) => void
  onClearMask: () => void
  disabled?: boolean
  className?: string
}

export function MaskEditor({
  drawingMode,
  onDrawingModeChange,
  brushSize,
  onBrushSizeChange,
  onClearMask,
  disabled,
  className = '',
}: MaskEditorProps) {
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
          aria-pressed={drawingMode}
          onClick={() => onDrawingModeChange(!drawingMode)}
          disabled={disabled}
          className={`rounded border p-2 ${drawingMode ? 'border-gray-800 bg-gray-200' : 'border-gray-300'} ${disabled ? 'opacity-50' : ''}`}
        >
          <Paintbrush className="h-4 w-4" aria-hidden />
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
    </div>
  )
}
