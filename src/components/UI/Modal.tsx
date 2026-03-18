import * as Dialog from '@radix-ui/react-dialog'
import { type ReactNode } from 'react'

export interface ModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  children: ReactNode
  className?: string
}

export function Modal({ open, onOpenChange, title, children, className = '' }: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <Dialog.Content
          className={`fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-lg bg-white p-6 shadow-xl focus:outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 ${className}`}
          aria-describedby={undefined}
        >
          <Dialog.Title className="text-lg font-semibold text-gray-900">{title}</Dialog.Title>
          <div className="mt-4">{children}</div>
          <Dialog.Close asChild>
            <button
              type="button"
              className="absolute right-4 top-4 rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700"
              aria-label="Закрыть"
            >
              ×
            </button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
