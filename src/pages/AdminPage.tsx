import { useCallback, useEffect, useState } from 'react'
import {
  adminLogin,
  bootstrapFirstAdmin,
  createAdmin,
  deleteAdmin,
  deleteColor,
  deleteMaterial,
  fetchAdmins,
  fetchColorsAdmin,
  fetchMaterialsAdmin,
  getAdminToken,
  patchAdmin,
  patchColor,
  patchMaterial,
  setAdminToken,
  uploadColor,
  uploadMaterial,
  type AdminDTO,
  type ColorDTO,
  type MaterialDTO,
} from '@/lib/materials-api'

type MainSection = 'add-catalog' | 'list-catalog' | 'add-admin' | 'list-admin'
type CatalogListTab = 'materials' | 'colors'

export function AdminPage() {
  const [isAuthed, setIsAuthed] = useState(false)
  const [authChecked, setAuthChecked] = useState(false)
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [authError, setAuthError] = useState('')
  const [setupSecret, setSetupSecret] = useState('')
  const [bootstrapUser, setBootstrapUser] = useState('')
  const [bootstrapPass, setBootstrapPass] = useState('')
  const [showBootstrap, setShowBootstrap] = useState(false)

  const [section, setSection] = useState<MainSection>('add-catalog')
  const [listTab, setListTab] = useState<CatalogListTab>('materials')

  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Materials list filters
  const [matScene, setMatScene] = useState<string>('')
  const [matCat, setMatCat] = useState('')
  const [matQ, setMatQ] = useState('')
  const [matVisibleOnly, setMatVisibleOnly] = useState(false)
  const [materials, setMaterials] = useState<MaterialDTO[]>([])
  const [matTotal, setMatTotal] = useState(0)
  const [matPage, setMatPage] = useState(1)

  // Colors list
  const [colScene, setColScene] = useState<string>('')
  const [colQ, setColQ] = useState('')
  const [colVisibleOnly, setColVisibleOnly] = useState(false)
  const [colors, setColors] = useState<ColorDTO[]>([])
  const [colTotal, setColTotal] = useState(0)
  const [colPage, setColPage] = useState(1)

  const [admins, setAdmins] = useState<AdminDTO[]>([])

  // Add material form
  const [mFile, setMFile] = useState<File | null>(null)
  const [mName, setMName] = useState('')
  const [mScene, setMScene] = useState('interior')
  const [mMatCat, setMMatCat] = useState('other')
  const [mPrice, setMPrice] = useState('')
  const [mSize, setMSize] = useState('')
  const [mVis, setMVis] = useState(true)

  // Add color form
  const [cFile, setCFile] = useState<File | null>(null)
  const [cName, setCName] = useState('')
  const [cScene, setCScene] = useState('interior')
  const [cPrice, setCPrice] = useState('')
  const [cVis, setCVis] = useState(true)

  // Add admin
  const [newAdminUser, setNewAdminUser] = useState('')
  const [newAdminPass, setNewAdminPass] = useState('')

  useEffect(() => {
    const t = getAdminToken()
    if (!t) {
      setAuthChecked(true)
      return
    }
    fetchAdmins()
      .then(() => setIsAuthed(true))
      .catch(() => {
        setAdminToken(null)
        setIsAuthed(false)
      })
      .finally(() => setAuthChecked(true))
  }, [])

  const loadMaterials = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetchMaterialsAdmin({
        page: matPage,
        page_size: 20,
        scene_category: matScene || undefined,
        material_category: matCat || undefined,
        q: matQ || undefined,
        visible_only: matVisibleOnly,
      })
      setMaterials(res.materials)
      setMatTotal(res.total)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [matPage, matScene, matCat, matQ, matVisibleOnly])

  const loadColors = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetchColorsAdmin({
        page: colPage,
        page_size: 20,
        scene_category: colScene || undefined,
        q: colQ || undefined,
        visible_only: colVisibleOnly,
      })
      setColors(res.colors)
      setColTotal(res.total)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [colPage, colScene, colQ, colVisibleOnly])

  const loadAdmins = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setAdmins(await fetchAdmins())
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!isAuthed || section !== 'list-catalog' || listTab !== 'materials') return
    void loadMaterials()
  }, [isAuthed, section, listTab, loadMaterials])

  useEffect(() => {
    if (!isAuthed || section !== 'list-catalog' || listTab !== 'colors') return
    void loadColors()
  }, [isAuthed, section, listTab, loadColors])

  useEffect(() => {
    if (!isAuthed || section !== 'list-admin') return
    void loadAdmins()
  }, [isAuthed, section, loadAdmins])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError('')
    try {
      await adminLogin(login, password)
      setIsAuthed(true)
    } catch (err) {
      setAuthError(String(err))
    }
  }

  const handleBootstrap = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError('')
    try {
      await bootstrapFirstAdmin(bootstrapUser, bootstrapPass, setupSecret)
      await adminLogin(bootstrapUser, bootstrapPass)
      setLogin(bootstrapUser)
      setPassword(bootstrapPass)
      setIsAuthed(true)
      setShowBootstrap(false)
    } catch (err) {
      setAuthError(String(err))
    }
  }

  const logout = () => {
    setAdminToken(null)
    setIsAuthed(false)
  }

  const submitMaterial = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!mFile) {
      setError('Выберите файл текстуры')
      return
    }
    setError('')
    try {
      await uploadMaterial(mFile, {
        name: mName || mFile.name,
        scene_category: mScene,
        material_category: mMatCat,
        price: mPrice,
        size: mSize || undefined,
        is_visible_in_visualizer: mVis,
      })
      setMFile(null)
      setMName('')
      setMPrice('')
      setMSize('')
    } catch (err) {
      setError(String(err))
    }
  }

  const submitColor = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!cFile) {
      setError('Выберите файл (превью цвета)')
      return
    }
    setError('')
    try {
      await uploadColor(cFile, {
        name: cName || cFile.name,
        scene_category: cScene,
        price: cPrice,
        is_visible_in_visualizer: cVis,
      })
      setCFile(null)
      setCName('')
      setCPrice('')
    } catch (err) {
      setError(String(err))
    }
  }

  const submitNewAdmin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      await createAdmin(newAdminUser, newAdminPass)
      setNewAdminUser('')
      setNewAdminPass('')
    } catch (err) {
      setError(String(err))
    }
  }

  if (!authChecked) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 text-gray-600">
        Проверка сессии…
      </div>
    )
  }

  if (!isAuthed) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50 py-8">
        <div className="w-full max-w-sm">
          <form
            onSubmit={handleLogin}
            className="rounded-xl border border-gray-200 bg-white p-8 shadow-lg"
          >
            <div className="mb-6 text-center">
              <h1 className="text-2xl font-semibold tracking-tight text-gray-900">Админ-панель</h1>
              <p className="mt-1 text-sm text-gray-500">Вход в систему управления</p>
            </div>
            {authError && !showBootstrap && (
              <div className="mb-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">{authError}</div>
            )}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Логин</label>
                <input
                  type="text"
                  value={login}
                  onChange={(e) => setLogin(e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
                  autoFocus
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Пароль</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
                />
              </div>
              <button
                type="submit"
                className="w-full rounded-lg bg-gray-800 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
              >
                Войти
              </button>
            </div>
          </form>

          <div className="mt-4 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
            <button
              type="button"
              onClick={() => setShowBootstrap(!showBootstrap)}
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              {showBootstrap ? 'Скрыть настройку' : 'Первичная настройка (первый админ)'}
            </button>
            {showBootstrap && (
              <form onSubmit={handleBootstrap} className="mt-4 space-y-3">
                {authError && (
                  <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">{authError}</div>
                )}
                <p className="text-xs text-gray-500">
                  Нужен секрет <code className="rounded bg-gray-100 px-1.5 py-0.5">FIRST_ADMIN_SETUP_SECRET</code> на сервере.
                </p>
                <input
                  type="password"
                  placeholder="Секрет настройки"
                  value={setupSecret}
                  onChange={(e) => setSetupSecret(e.target.value)}
                  className="block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
                />
                <input
                  type="text"
                  placeholder="Логин нового админа"
                  value={bootstrapUser}
                  onChange={(e) => setBootstrapUser(e.target.value)}
                  className="block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
                />
                <input
                  type="password"
                  placeholder="Пароль"
                  value={bootstrapPass}
                  onChange={(e) => setBootstrapPass(e.target.value)}
                  className="block w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
                />
                <button
                  type="submit"
                  className="w-full rounded-lg bg-gray-800 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
                >
                  Создать первого админа
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    )
  }

  const tabBtn = (id: MainSection, label: string) => (
    <button
      type="button"
      onClick={() => setSection(id)}
      className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${
        section === id
          ? 'bg-gray-800 text-white shadow-sm'
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
      }`}
    >
      {label}
    </button>
  )

  const matPages = Math.max(1, Math.ceil(matTotal / 20))
  const colPages = Math.max(1, Math.ceil(colTotal / 20))

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-200 bg-white px-6 py-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-gray-900">Админ-панель</h1>
          <p className="text-sm text-gray-500">Управление каталогом материалов</p>
        </div>
        <button
          type="button"
          onClick={logout}
          className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-900"
        >
          Выйти
        </button>
      </header>

      <nav className="flex flex-wrap gap-2 border-b border-gray-200 bg-white px-6 py-3">
        {tabBtn('add-catalog', 'Добавить')}
        {tabBtn('list-catalog', 'Каталог')}
        {tabBtn('add-admin', 'Админы')}
        {tabBtn('list-admin', 'Список админов')}
      </nav>

      <div className="flex-1 overflow-y-auto p-6">
        {error && <div className="mb-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>}

        {section === 'add-catalog' && (
          <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-2">
            <form onSubmit={submitMaterial} className="space-y-4 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Новая текстура</h2>
              <div>
                <label className="block w-full cursor-pointer rounded-lg border-2 border-dashed border-gray-300 px-4 py-6 text-center transition-colors hover:border-gray-400">
                  <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setMFile(e.target.files?.[0] ?? null)} className="hidden" />
                  {mFile ? (
                    <div className="flex flex-col items-center gap-2">
                      <img src={URL.createObjectURL(mFile)} alt="" className="h-16 w-16 rounded-lg object-cover" />
                      <span className="text-sm text-gray-500">{mFile.name}</span>
                    </div>
                  ) : (
                    <span className="text-sm text-gray-500">Нажмите для выбора файла</span>
                  )}
                </label>
              </div>
              <input
                placeholder="Название"
                value={mName}
                onChange={(e) => setMName(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
              />
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Сцена</label>
                  <select value={mScene} onChange={(e) => setMScene(e.target.value)} className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2.5 text-sm focus:border-gray-500 focus:outline-none">
                    <option value="interior">interior</option>
                    <option value="exterior">exterior</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Категория</label>
                  <input
                    value={mMatCat}
                    onChange={(e) => setMMatCat(e.target.value)}
                    placeholder="planken, brus..."
                    className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2.5 text-sm focus:border-gray-500 focus:outline-none"
                  />
                </div>
              </div>
              <input placeholder="Цена (необязательно)" value={mPrice} onChange={(e) => setMPrice(e.target.value)} className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200" />
              <input placeholder="Размер (необязательно)" value={mSize} onChange={(e) => setMSize(e.target.value)} className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200" />
              <label className="flex items-center gap-3 text-sm text-gray-700">
                <input type="checkbox" checked={mVis} onChange={(e) => setMVis(e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400" />
                Видно в визуализаторе
              </label>
              <button type="submit" className="w-full rounded-lg bg-gray-800 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2">
                Загрузить текстуру
              </button>
            </form>

            <form onSubmit={submitColor} className="space-y-4 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Новый цвет</h2>
              <div>
                <label className="block w-full cursor-pointer rounded-lg border-2 border-dashed border-gray-300 px-4 py-6 text-center transition-colors hover:border-gray-400">
                  <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setCFile(e.target.files?.[0] ?? null)} className="hidden" />
                  {cFile ? (
                    <div className="flex flex-col items-center gap-2">
                      <img src={URL.createObjectURL(cFile)} alt="" className="h-16 w-16 rounded-lg object-cover" />
                      <span className="text-sm text-gray-500">{cFile.name}</span>
                    </div>
                  ) : (
                    <span className="text-sm text-gray-500">Нажмите для выбора файла</span>
                  )}
                </label>
              </div>
              <input
                placeholder="Название"
                value={cName}
                onChange={(e) => setCName(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
              />
              <div>
                <label className="block text-sm font-medium text-gray-700">Сцена</label>
                <select value={cScene} onChange={(e) => setCScene(e.target.value)} className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2.5 text-sm focus:border-gray-500 focus:outline-none">
                  <option value="interior">interior</option>
                  <option value="exterior">exterior</option>
                </select>
              </div>
              <input placeholder="Цена (необязательно)" value={cPrice} onChange={(e) => setCPrice(e.target.value)} className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200" />
              <label className="flex items-center gap-3 text-sm text-gray-700">
                <input type="checkbox" checked={cVis} onChange={(e) => setCVis(e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400" />
                Видно в визуализаторе
              </label>
              <button type="submit" className="w-full rounded-lg bg-gray-800 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2">
                Загрузить цвет
              </button>
            </form>
          </div>
        )}

        {section === 'list-catalog' && (
          <div className="mx-auto max-w-6xl space-y-4">
            <div className="flex gap-2 rounded-lg bg-white p-1 shadow-sm border border-gray-200 w-fit">
              <button
                type="button"
                onClick={() => setListTab('materials')}
                className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                  listTab === 'materials'
                    ? 'bg-gray-800 text-white shadow-sm'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Текстуры
              </button>
              <button
                type="button"
                onClick={() => setListTab('colors')}
                className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                  listTab === 'colors'
                    ? 'bg-gray-800 text-white shadow-sm'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Цвета
              </button>
            </div>

            {listTab === 'materials' && (
              <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
                <div className="flex flex-wrap gap-3 border-b border-gray-200 p-4">
                  <select value={matScene} onChange={(e) => { setMatScene(e.target.value); setMatPage(1) }} className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-gray-500 focus:outline-none">
                    <option value="">Все сцены</option>
                    <option value="interior">interior</option>
                    <option value="exterior">exterior</option>
                  </select>
                  <input
                    placeholder="Категория материала"
                    value={matCat}
                    onChange={(e) => { setMatCat(e.target.value); setMatPage(1) }}
                    className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-gray-500 focus:outline-none"
                  />
                  <input
                    placeholder="Поиск по названию"
                    value={matQ}
                    onChange={(e) => { setMatQ(e.target.value); setMatPage(1) }}
                    className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-gray-500 focus:outline-none"
                  />
                  <label className="flex items-center gap-2 text-sm text-gray-600">
                    <input type="checkbox" checked={matVisibleOnly} onChange={(e) => { setMatVisibleOnly(e.target.checked); setMatPage(1) }} className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400" />
                    Только видимые
                  </label>
                  <button type="button" onClick={() => void loadMaterials()} className="rounded-lg bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 transition-colors">
                    Обновить
                  </button>
                </div>
                {loading ? (
                  <div className="flex items-center justify-center py-12 text-gray-500">Загрузка...</div>
                ) : (
                  <>
                    <p className="px-4 py-2 text-sm text-gray-500">Всего: {matTotal}</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-gray-200 bg-gray-50">
                            <th className="px-4 py-3 font-medium text-gray-600">Превью</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Название</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Сцена</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Категория</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Видимость</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Действия</th>
                          </tr>
                        </thead>
                        <tbody>
                          {materials.map((m) => (
                            <tr key={m.id} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                              <td className="px-4 py-3">
                                <img src={m.url} alt="" className="h-12 w-12 rounded-lg object-cover" />
                              </td>
                              <td className="px-4 py-3 font-medium text-gray-900">{m.name}</td>
                              <td className="px-4 py-3 text-gray-600">{m.scene_category}</td>
                              <td className="px-4 py-3 text-gray-600">{m.category}</td>
                              <td className="px-4 py-3">
                                <input
                                  type="checkbox"
                                  checked={m.is_visible_in_visualizer}
                                  onChange={async (e) => {
                                    try {
                                      const upd = await patchMaterial(m.id, { is_visible_in_visualizer: e.target.checked })
                                      setMaterials((prev) => prev.map((x) => (x.id === m.id ? upd : x)))
                                    } catch (err) {
                                      setError(String(err))
                                    }
                                  }}
                                  className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400"
                                />
                              </td>
                              <td className="px-4 py-3">
                                <button
                                  type="button"
                                  className="rounded-lg px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                                  onClick={async () => {
                                    if (!confirm('Удалить текстуру?')) return
                                    try {
                                      await deleteMaterial(m.id)
                                      await loadMaterials()
                                    } catch (err) {
                                      setError(String(err))
                                    }
                                  }}
                                >
                                  Удалить
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="flex items-center justify-between border-t border-gray-200 px-4 py-3">
                      <button
                        type="button"
                        disabled={matPage <= 1}
                        onClick={() => setMatPage((p) => Math.max(1, p - 1))}
                        className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Назад
                      </button>
                      <span className="text-sm text-gray-500">
                        {matPage} / {matPages}
                      </span>
                      <button
                        type="button"
                        disabled={matPage >= matPages}
                        onClick={() => setMatPage((p) => p + 1)}
                        className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Вперёд
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {listTab === 'colors' && (
              <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
                <div className="flex flex-wrap gap-3 border-b border-gray-200 p-4">
                  <select value={colScene} onChange={(e) => { setColScene(e.target.value); setColPage(1) }} className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-gray-500 focus:outline-none">
                    <option value="">Все сцены</option>
                    <option value="interior">interior</option>
                    <option value="exterior">exterior</option>
                  </select>
                  <input
                    placeholder="Поиск"
                    value={colQ}
                    onChange={(e) => { setColQ(e.target.value); setColPage(1) }}
                    className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-gray-500 focus:outline-none"
                  />
                  <label className="flex items-center gap-2 text-sm text-gray-600">
                    <input type="checkbox" checked={colVisibleOnly} onChange={(e) => { setColVisibleOnly(e.target.checked); setColPage(1) }} className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400" />
                    Только видимые
                  </label>
                  <button type="button" onClick={() => void loadColors()} className="rounded-lg bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 transition-colors">
                    Обновить
                  </button>
                </div>
                {loading ? (
                  <div className="flex items-center justify-center py-12 text-gray-500">Загрузка...</div>
                ) : (
                  <>
                    <p className="px-4 py-2 text-sm text-gray-500">Всего: {colTotal}</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-gray-200 bg-gray-50">
                            <th className="px-4 py-3 font-medium text-gray-600">Превью</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Название</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Сцена</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Видимость</th>
                            <th className="px-4 py-3 font-medium text-gray-600">Действия</th>
                          </tr>
                        </thead>
                        <tbody>
                          {colors.map((c) => (
                            <tr key={c.id} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                              <td className="px-4 py-3">
                                <img src={c.url} alt="" className="h-12 w-12 rounded-lg object-cover" />
                              </td>
                              <td className="px-4 py-3 font-medium text-gray-900">{c.name}</td>
                              <td className="px-4 py-3 text-gray-600">{c.scene_category}</td>
                              <td className="px-4 py-3">
                                <input
                                  type="checkbox"
                                  checked={c.is_visible_in_visualizer}
                                  onChange={async (e) => {
                                    try {
                                      const upd = await patchColor(c.id, { is_visible_in_visualizer: e.target.checked })
                                      setColors((prev) => prev.map((x) => (x.id === c.id ? upd : x)))
                                    } catch (err) {
                                      setError(String(err))
                                    }
                                  }}
                                  className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400"
                                />
                              </td>
                              <td className="px-4 py-3">
                                <button
                                  type="button"
                                  className="rounded-lg px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                                  onClick={async () => {
                                    if (!confirm('Удалить цвет?')) return
                                    try {
                                      await deleteColor(c.id)
                                      await loadColors()
                                    } catch (err) {
                                      setError(String(err))
                                    }
                                  }}
                                >
                                  Удалить
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="flex items-center justify-between border-t border-gray-200 px-4 py-3">
                      <button
                        type="button"
                        disabled={colPage <= 1}
                        onClick={() => setColPage((p) => Math.max(1, p - 1))}
                        className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Назад
                      </button>
                      <span className="text-sm text-gray-500">
                        {colPage} / {colPages}
                      </span>
                      <button
                        type="button"
                        disabled={colPage >= colPages}
                        onClick={() => setColPage((p) => p + 1)}
                        className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Вперёд
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {section === 'add-admin' && (
          <form onSubmit={submitNewAdmin} className="mx-auto max-w-md space-y-4 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-gray-900">Новый администратор</h2>
            <input
              type="text"
              placeholder="Логин"
              value={newAdminUser}
              onChange={(e) => setNewAdminUser(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
            />
            <input
              type="password"
              placeholder="Пароль"
              value={newAdminPass}
              onChange={(e) => setNewAdminPass(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-200"
            />
            <button type="submit" className="w-full rounded-lg bg-gray-800 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2">
              Создать админа
            </button>
          </form>
        )}

        {section === 'list-admin' && (
          <div className="mx-auto max-w-xl rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center py-12 text-gray-500">Загрузка...</div>
            ) : (
              <ul className="divide-y divide-gray-200">
                {admins.map((a) => (
                  <li key={a.id} className="flex flex-wrap items-center justify-between gap-3 px-6 py-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-100 text-sm font-medium text-gray-600">
                        {a.username[0].toUpperCase()}
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{a.username}</p>
                        <p className="text-sm text-gray-500">{a.is_active ? 'активен' : 'неактивен'}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-2 text-sm text-gray-600">
                        <input
                          type="checkbox"
                          checked={a.is_active}
                          onChange={async (e) => {
                            try {
                              const upd = await patchAdmin(a.id, e.target.checked)
                              setAdmins((prev) => prev.map((x) => (x.id === a.id ? upd : x)))
                            } catch (err) {
                              setError(String(err))
                            }
                          }}
                          className="h-4 w-4 rounded border-gray-300 text-gray-800 focus:ring-gray-400"
                        />
                        активен
                      </label>
                      <button
                        type="button"
                        className="rounded-lg px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                        onClick={async () => {
                          if (!confirm('Удалить админа?')) return
                          try {
                            await deleteAdmin(a.id)
                            await loadAdmins()
                          } catch (err) {
                            setError(String(err))
                          }
                        }}
                      >
                        Удалить
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
