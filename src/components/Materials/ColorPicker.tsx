import { useColorize } from '@/hooks/useColorize'
import type { Color } from '@/types/material'

export interface ColorPickerProps {
  value?: Color | null
  onChange?: (color: Color) => void
  opacity?: number
  onOpacityChange?: (opacity: number) => void
  className?: string
}

export function ColorPicker({ value, onChange, className = '' }: ColorPickerProps) {
  const { selectedColor, setSelectedColor, presets } = useColorize()

  const currentColor: Color | null | undefined = value ?? selectedColor
  const handlePreset = (color: { hex: string; name?: string }) => {
    const c: Color = { hex: color.hex, name: color.name }
    if (onChange) onChange(c)
    else setSelectedColor(c)
  }

  const handleInputColor = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hex = e.target.value
    handlePreset({ hex })
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
    </div>
  )
}
