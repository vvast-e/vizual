import { Download } from 'lucide-react'

export interface ExportButtonProps {
  onClick: () => void
  disabled?: boolean
  label?: string
  className?: string
}

export function ExportButton({
  onClick,
  disabled,
  label = 'Экспорт PNG',
  className = '',
}: ExportButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={`inline-flex items-center gap-2 rounded bg-gray-800 px-3 py-1.5 text-sm text-white hover:bg-gray-700 disabled:opacity-50 ${className}`}
    >
      <Download className="h-4 w-4" aria-hidden />
      {label}
    </button>
  )
}
