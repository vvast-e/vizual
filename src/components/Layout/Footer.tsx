import { type ReactNode } from 'react'

export interface FooterProps {
  children?: ReactNode
  className?: string
}

export function Footer({ children, className = '' }: FooterProps) {
  return (
    <footer className={`border-t border-gray-200 px-4 py-3 text-sm text-gray-500 ${className}`}>
      {children}
    </footer>
  )
}
