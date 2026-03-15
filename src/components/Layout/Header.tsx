import { type ReactNode } from 'react'

export interface HeaderProps {
  children?: ReactNode
  className?: string
}

export function Header({ children, className = '' }: HeaderProps) {
  return (
    <header className={`border-b border-gray-200 px-4 py-3 ${className}`}>
      {children}
    </header>
  )
}
