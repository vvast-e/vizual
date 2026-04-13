import { useEffect } from 'react'
import { Joyride, STATUS, EVENTS, LIFECYCLE } from 'react-joyride'
import type { Step, EventData } from 'react-joyride'
import { useVisualizerStore } from '@/store/useVisualizerStore'
import { useUIStore } from '@/store/useUIStore'

export function OnboardingTour() {
  const photoDataUrl = useVisualizerStore((s) => s.photoDataUrl)
  const tourActive = useUIStore((s) => s.tourActive)
  const setTourActive = useUIStore((s) => s.setTourActive)
  const setEditWallCorners = useUIStore((s) => s.setEditWallCorners)

  useEffect(() => {
    if (photoDataUrl && !localStorage.getItem('vizual_tour_completed')) {
      const timer = setTimeout(() => {
        setTourActive(true)
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [photoDataUrl, setTourActive])

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
    if (index === 5) {
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
      content: 'Нейросеть автоматически нашла стены или фасады. Кликните на область, чтобы выбрать её для выбора материала.',
      placement: 'center',
    },
    {
      target: '.tour-walls-list',
      content: 'Здесь список всех поверхностей. Вы можете выбирать стены для работы, а также скрывать их с помощью иконки "глазика", если область мешает обзору.',
      placement: 'right',
    },
    {
      target: '.tour-add-mask',
      content: 'Если нейросеть пропустила часть стены, нажмите сюда и поставьте 4 точки на фото, чтобы создать новую область вручную.',
      placement: 'right',
    },
    {
      target: '.tour-materials-panel',
      content: 'Выберите профиль (например, имитацию бруса) и желаемый оттенок краски. Цвет автоматически подстроится под освещение на фото.',
      placement: 'left',
    },
    {
      target: '.tour-apply-texture',
      content: 'Выделив стену и настроив материал, нажмите "Применить профиль", чтобы примерить обшивку.',
      placement: 'bottom',
    },
    {
      target: '.tour-edit-mask',
      content: 'Если материал лег неровно, нажмите сюда. Вы сможете потянуть за углы области, чтобы исправить перспективу, а в режиме "Форма" — подкорректировать точное заполнение границ.',
      placement: 'bottom',
    },
  ]

  return (
    <Joyride
      steps={steps}
      run={tourActive}
      continuous
      scrollToFirstStep
      onEvent={handleJoyrideEvent}
      styles={{
        buttonPrimary: {
          backgroundColor: '#2563eb', // синий (blue-600)
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
