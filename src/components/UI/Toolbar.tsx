import { type ReactNode } from 'react'

export interface ToolbarProps {
  children: ReactNode
  className?: string
  ariaLabel?: string
}

export function Toolbar({ children, className = '', ariaLabel = 'Панель инструментов' }: ToolbarProps) {
  return (
    <div
      role="toolbar"
      aria-label={ariaLabel}
      className={`flex flex-wrap items-center gap-2 ${className}`}
    >
      {children}
    </div>
  )
}
