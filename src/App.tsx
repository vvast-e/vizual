import { lazy, Suspense } from 'react'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { Canvas2D } from '@/components/2d/Canvas2D'
import { MaterialCatalog } from '@/components/Materials/MaterialCatalog'
import { ColorPicker } from '@/components/Materials/ColorPicker'
import { ViewToggle } from '@/components/UI/ViewToggle'
import { Loader } from '@/components/UI/Loader'

const Scene3D = lazy(() => import('@/components/3d/Scene3D').then((m) => ({ default: m.Scene3D })))

function App() {
  const viewMode = useVisualizerStore((s) => s.viewMode)
  const sceneMode = useVisualizerStore((s) => s.sceneMode)
  const setSceneMode = useVisualizerStore((s) => s.setSceneMode)

  return (
    <div className="min-h-screen bg-white text-gray-900">
      <header className="border-b border-gray-200 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-xl font-medium">Визуализатор материалов</h1>
          <ViewToggle />
        </div>
      </header>
      <main className="p-4">
        {viewMode === '2d' && (
          <div className="flex flex-col gap-6">
            <section aria-label="Каталог материалов">
              <h2 className="mb-2 text-sm font-medium text-gray-600">Материалы</h2>
              <MaterialCatalog />
            </section>
            <section aria-label="Цвет покраски">
              <h2 className="mb-2 text-sm font-medium text-gray-600">Цвет</h2>
              <ColorPicker />
            </section>
            <Canvas2D width={800} height={500} />
          </div>
        )}
        {viewMode === '3d' && (
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Сцена:</span>
              <button
                type="button"
                onClick={() => setSceneMode('room')}
                className={`rounded px-3 py-1 text-sm ${sceneMode === 'room' ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-700'}`}
              >
                Комната
              </button>
              <button
                type="button"
                onClick={() => setSceneMode('house')}
                className={`rounded px-3 py-1 text-sm ${sceneMode === 'house' ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-700'}`}
              >
                Дом
              </button>
            </div>
            <section aria-label="Каталог материалов">
              <h2 className="mb-2 text-sm font-medium text-gray-600">Материалы</h2>
              <MaterialCatalog />
            </section>
            <Suspense fallback={<Loader ariaLabel="Загрузка 3D сцены" />}>
              <Scene3D sceneMode={sceneMode} />
            </Suspense>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
