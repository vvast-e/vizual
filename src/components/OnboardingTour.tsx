import { useEffect } from 'react'
import { Joyride, STATUS, EVENTS, LIFECYCLE } from 'react-joyride'
import type { Step, EventData } from 'react-joyride'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useUIStore } from '@/store/useUIStore'
import { useWallStore } from '@/store/useWallStore'

export function OnboardingTour() {
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)
  const tourActive = useUIStore((s) => s.tourActive)
  const setTourActive = useUIStore((s) => s.setTourActive)
  const setEditWallCorners = useUIStore((s) => s.setEditWallCorners)
  const walls = useWallStore((s) => s.walls)

  useEffect(() => {
    if (!photoDataUrl || walls.length === 0) return
    if (tourActive) return
    if (localStorage.getItem('vizual_tour_completed')) return

    const neededTargets = [
      '.tour-canvas-container',
      '.tour-mode-toggle',
      '.tour-walls-list',
      '.tour-materials-panel',
      '.tour-apply-texture',
      '.tour-before-after',
      '.tour-hide-masks',
      '.tour-edit-mask',
      '.tour-texture-scale',
      '.tour-undo-redo',
      '.tour-export',
      '.tour-reset',
    ]

    let raf = 0
    let attempts = 0
    const maxAttempts = 24 // ~400ms на ожидание DOM после restore/рендера

    const checkTargets = () => {
      const ready = neededTargets.every((sel) => document.querySelector(sel))
      if (ready) {
        setTourActive(true)
        return
      }
      attempts += 1
      if (attempts < maxAttempts) {
        raf = window.requestAnimationFrame(checkTargets)
      }
    }

    raf = window.requestAnimationFrame(checkTargets)
    return () => {
      if (raf) window.cancelAnimationFrame(raf)
    }
  }, [photoDataUrl, walls.length, tourActive, setTourActive])

  const handleJoyrideEvent = (data: EventData) => {
    const { status, index, lifecycle, type } = data

    // При достижении последнего шага или пропуске — отключаем тур
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status as any)) {
      setTourActive(false)
      localStorage.setItem('vizual_tour_completed', 'true')
      setEditWallCorners(false) // На всякий случай сворачиваем панельку
    }

    // Логика для подсветки раскрытого меню "Править маску"
    // Шаг 6 имеет индекс 5
    if (index === 8) {
      if (lifecycle === LIFECYCLE.TOOLTIP || type === EVENTS.STEP_AFTER) {
        setEditWallCorners(true)
      }
    } else {
      // На всех остальных шагах прячем меню правки углов
      setEditWallCorners(false)
    }
  }

  const steps: Step[] = [
    {
      target: '.tour-canvas-container',
      content: 'Нейросеть автоматически нашла стены или фасады. Кликните на область, чтобы выбрать её для наложения материала.',
      placement: 'center',
    },
    {
      target: '.tour-mode-toggle',
      content: 'Выберите режим работы: «Интерьер» — для комнат (перспективный квад из 4 точек), «Экстерьер» — для фасадов, где нейросеть автоматически вырезает окна и двери.',
      placement: 'bottom',
    },
    {
      target: '.tour-walls-list',
      content: 'Здесь список всех найденных областей. Выбирайте нужную для работы, скрывайте ненужные иконкой «глазика», удаляйте или очищайте текстуру с помощью кнопок рядом с каждой стеной.',
      placement: 'right',
    },
    {
      target: '.tour-add-mask',
      content: 'Если нейросеть пропустила часть стены, нажмите сюда и кликните точки на холсте — минимум 3, замкните контур кликом рядом с первой точкой.',
      placement: 'right',
    },
    {
      target: '.tour-materials-panel',
      content: 'Откройте «Материалы» для выбора профиля (имитация дерева, кирпича и т.д.) или «Цвета» для выбора оттенка краски. Выбранные элементы сохраняются в быстром доступе.',
      placement: 'left',
    },
    {
      target: '.tour-apply-texture',
      content: 'Выделив область и выбрав материал или цвет, нажмите «Применить профиль» — текстура наложится с учётом перспективы и освещения фото.',
      placement: 'bottom',
    },
    {
      target: '.tour-before-after',
      content: 'Удерживайте кнопку «Оригинал» (или клавишу Пробел), чтобы мгновенно сравнить результат с исходным фото.',
      placement: 'bottom',
    },
    {
      target: '.tour-hide-masks',
      content: '«Скрыть области» убирает синие контуры стен с холста — удобно для финальной оценки результата без лишних линий.',
      placement: 'bottom',
    },
    {
      target: '.tour-edit-mask',
      content: 'Если материал лёг неровно, нажмите «Редактировать». Потяните за зелёные кружки, чтобы скорректировать точный контур области.',
      placement: 'bottom',
    },
    {
      target: '.tour-texture-scale',
      content: 'Слайдер «Масштаб» изменяет размер рисунка текстуры на выбранной области — уменьшайте значение для более мелкого узора.',
      placement: 'bottom',
    },
    {
      target: '.tour-undo-redo',
      content: '«Отменить» (Ctrl+Z) и «Вернуть» (Ctrl+Y) откатывают изменения границ областей: угловые точки, разрезы и вручную добавленные маски.',
      placement: 'bottom',
    },
    {
      target: '.tour-export',
      content: 'Когда результат готов, нажмите «Экспорт PNG» для сохранения итогового изображения.',
      placement: 'bottom',
    },
    {
      target: '.tour-reset',
      content: 'Кнопка сброса возвращает на экран загрузки фото. Все наложенные текстуры и области будут потеряны.',
      placement: 'bottom',
    },
  ]

  return (
    <Joyride
      steps={steps}
      run={tourActive}
      continuous
      scrollToFirstStep
      // @ts-expect-error React Joyride v2/v3 type mismatch in callback prop
      callback={handleJoyrideEvent}
      styles={{
        buttonPrimary: {
          backgroundColor: '#1f2937', // gray-800
          borderRadius: '6px',
        },
        buttonBack: {
          color: '#4b5563',
          marginRight: '8px',
        },
        tooltip: {
          textAlign: 'left',
          fontSize: '14px',
        }
      }}
      locale={{
        back: 'Назад',
        close: 'Закрыть',
        last: 'Понятно',
        next: 'Далее',
        skip: 'Пропустить',
      }}
    />
  )
}
