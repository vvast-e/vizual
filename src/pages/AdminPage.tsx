import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchMaterials, uploadMaterial, deleteMaterial } from '@/lib/materials-api'
import type { MaterialDTO } from '@/lib/materials-api'

const ADMIN_LOGIN = 'admin'
const ADMIN_PASSWORD = '1Qst2la#'

export function AdminPage() {
  const [isAuthed, setIsAuthed] = useState(false)
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [authError, setAuthError] = useState('')

  const [materials, setMaterials] = useState<MaterialDTO[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Upload state
  const [uploadName, setUploadName] = useState('')
  const [uploadCategory, setUploadCategory] = useState('other')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Drag-drop
  const [dragOver, setDragOver] = useState(false)

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    if (login === ADMIN_LOGIN && password === ADMIN_PASSWORD) {
      setIsAuthed(true)
      setAuthError('')
    } else {
      setAuthError('Неверный логин или пароль')
    }
  }

  const loadMaterials = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const list = await fetchMaterials()
      setMaterials(list)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isAuthed) loadMaterials()
  }, [isAuthed, loadMaterials])

  const handleUpload = async (file: File) => {
    if (!file) return
    setUploading(true)
    setError('')
    try {
      await uploadMaterial(file, uploadName || file.name, uploadCategory, login, password)
      setUploadName('')
      setUploadFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
      await loadMaterials()
    } catch (err) {
      setError(String(err))
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (filename: string) => {
    if (!confirm(`Удалить текстуру ${filename}?`)) return
    setError('')
    try {
      await deleteMaterial(filename, login, password)
      await loadMaterials()
    } catch (err) {
      setError(String(err))
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setUploadFile(file)
      handleUpload(file)
    }
  }

  // ─── Login form ───
  if (!isAuthed) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <form
          onSubmit={handleLogin}
          className="w-full max-w-sm rounded-lg border border-gray-200 bg-white p-6 shadow-md"
        >
          <h1 className="mb-4 text-xl font-bold text-gray-800">Админ-панель</h1>
          {authError && (
            <div className="mb-3 rounded bg-red-50 p-2 text-sm text-red-600">{authError}</div>
          )}
          <label className="mb-1 block text-sm text-gray-600">Логин</label>
          <input
            type="text"
            value={login}
            onChange={(e) => setLogin(e.target.value)}
            className="mb-3 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
            autoFocus
          />
          <label className="mb-1 block text-sm text-gray-600">Пароль</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mb-4 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
          <button
            type="submit"
            className="w-full rounded bg-gray-800 py-2 text-sm font-medium text-white hover:bg-gray-700"
          >
            Войти
          </button>
        </form>
      </div>
    )
  }

  // ─── Admin panel ───
  return (
    <div className="flex h-screen flex-col bg-gray-50">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-3">
        <h1 className="text-lg font-bold text-gray-800">Управление текстурами</h1>
        <button
          onClick={() => setIsAuthed(false)}
          className="rounded border border-gray-300 px-3 py-1 text-sm text-gray-600 hover:bg-gray-50"
        >
          Выйти
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-4 rounded bg-red-50 p-3 text-sm text-red-600">{error}</div>
        )}

        {/* Upload area */}
        <div
          className={`mb-6 rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
            dragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 bg-white'
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <p className="mb-3 text-sm text-gray-500">
            Перетащите файл сюда или выберите вручную
          </p>
          <div className="flex flex-wrap items-end justify-center gap-3">
            <div>
              <label className="mb-1 block text-xs text-gray-500">Название</label>
              <input
                type="text"
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                placeholder="Название текстуры"
                className="rounded border border-gray-300 px-2 py-1 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-gray-500">Категория</label>
              <input
                type="text"
                value={uploadCategory}
                onChange={(e) => setUploadCategory(e.target.value)}
                placeholder="vagonka, brus, other..."
                className="rounded border border-gray-300 px-2 py-1 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-gray-500">Файл</label>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null
                  setUploadFile(f)
                }}
                className="text-sm"
              />
            </div>
            <button
              onClick={() => uploadFile && handleUpload(uploadFile)}
              disabled={!uploadFile || uploading}
              className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
            >
              {uploading ? 'Загрузка...' : 'Загрузить'}
            </button>
          </div>
        </div>

        {/* Materials list */}
        {loading ? (
          <p className="text-sm text-gray-500">Загрузка...</p>
        ) : materials.length === 0 ? (
          <p className="text-sm text-gray-500">Текстуры не найдены</p>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
            {materials.map((m) => (
              <div
                key={m.id}
                className="group relative overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm"
              >
                <img
                  src={m.url}
                  alt={m.name}
                  className="aspect-square w-full object-cover"
                  loading="lazy"
                />
                <div className="p-2">
                  <p className="truncate text-sm font-medium text-gray-700">{m.name}</p>
                  <p className="truncate text-xs text-gray-400">{m.category} / {m.filename}</p>
                </div>
                <button
                  onClick={() => handleDelete(m.filename)}
                  className="absolute right-1 top-1 rounded bg-red-600/80 px-2 py-0.5 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100"
                  title="Удалить"
                >
                  Удалить
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
