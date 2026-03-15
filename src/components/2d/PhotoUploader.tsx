import { useCallback, useRef } from 'react'
import { Upload } from 'lucide-react'
import { MAX_PHOTO_SIZE_BYTES, ALLOWED_IMAGE_TYPES } from '@/lib/constants'

const MAX_SIZE_MB = MAX_PHOTO_SIZE_BYTES / (1024 * 1024)
const ACCEPT = ALLOWED_IMAGE_TYPES.join(',')

export interface PhotoUploaderProps {
  onFileSelect: (file: File) => void
  disabled?: boolean
  className?: string
}

export function PhotoUploader({ onFileSelect, disabled, className = '' }: PhotoUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File | null) => {
      if (!file) return
      if (file.size > MAX_PHOTO_SIZE_BYTES) return
      if (!ALLOWED_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_TYPES)[number])) return
      onFileSelect(file)
    },
    [onFileSelect]
  )

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      handleFile(file ?? null)
      e.target.value = ''
    },
    [handleFile]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      if (disabled) return
      const file = e.dataTransfer.files?.[0]
      handleFile(file ?? null)
    },
    [disabled, handleFile]
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const onClick = useCallback(() => {
    if (disabled) return
    inputRef.current?.click()
  }, [disabled])

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Загрузить фото"
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      className={
        className ||
        'flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-8 text-gray-600 transition-colors hover:border-gray-400 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400' +
        (disabled ? ' pointer-events-none opacity-60' : ' cursor-pointer')
      }
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        onChange={onInputChange}
        className="hidden"
        aria-hidden
        capture="environment"
      />
      <Upload className="h-12 w-12" aria-hidden />
      <p className="text-center text-sm">
        Перетащите фото сюда или нажмите для выбора
        <br />
        <span className="text-gray-400">JPEG, PNG, WebP до {MAX_SIZE_MB} МБ</span>
      </p>
    </div>
  )
}
