export interface LoaderProps {
  className?: string
  ariaLabel?: string
}

export function Loader({ className = '', ariaLabel = 'Загрузка' }: LoaderProps) {
  return (
    <div
      className={`flex items-center justify-center p-4 ${className}`}
      role="status"
      aria-label={ariaLabel}
    >
      <span className="h-8 w-8 animate-spin rounded-full border-2 border-gray-300 border-t-gray-800" />
    </div>
  )
}
