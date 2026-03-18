import { useColorize } from '@/hooks/useColorize'
import type { Color } from '@/types/material'

export interface ColorPickerProps {
  value?: Color | null
  onChange?: (color: Color) => void
  opacity?: number
  onOpacityChange?: (opacity: number) => void
  className?: string
}

export function ColorPicker({
  value,
  onChange,
  opacity = 1,
  onOpacityChange,
  className = '',
}: ColorPickerProps) {
  const {
    selectedColor,
    colorizeOpacity,
    setSelectedColor,
    setColorizeOpacity,
    presets,
  } = useColorize()

  const currentColor: Color | null | undefined = value ?? selectedColor
  const currentOpacity = onOpacityChange !== undefined ? opacity : colorizeOpacity

  const handlePreset = (color: { hex: string; name?: string }) => {
    const c: Color = { hex: color.hex, name: color.name }
    if (onChange) onChange(c)
    else setSelectedColor(c)
  }

  const handleInputColor = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hex = e.target.value
    handlePreset({ hex })
  }

  const handleOpacity = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value)
    if (onOpacityChange) onOpacityChange(v)
    else setColorizeOpacity(v)
  }

  return (
    <div className={`flex flex-col gap-3 ${className}`} role="group" aria-label="Выбор цвета покраски">
      <div className="flex flex-wrap items-center gap-2">
        {presets.map((c) => (
          <button
            key={c.hex}
            type="button"
            title={c.name}
            aria-label={c.name ?? c.hex}
            onClick={() => handlePreset(c)}
            className={`h-8 w-8 rounded-full border-2 transition-transform hover:scale-110 ${
              currentColor?.hex === c.hex ? 'border-gray-800 ring-2 ring-gray-400' : 'border-gray-300'
            }`}
            style={{ backgroundColor: c.hex }}
          />
        ))}
      </div>
      <div className="flex items-center gap-2">
        <input
          type="color"
          value={currentColor?.hex ?? '#ffffff'}
          onChange={handleInputColor}
          aria-label="Произвольный цвет"
          className="h-8 w-12 cursor-pointer rounded border border-gray-300"
        />
        <span className="text-sm text-gray-600">Свой цвет</span>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-600">Интенсивность:</label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={currentOpacity}
          onChange={handleOpacity}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(currentOpacity * 100)}
          className="w-24"
        />
        <span className="text-xs text-gray-500">{Math.round(currentOpacity * 100)}%</span>
      </div>
    </div>
  )
}
