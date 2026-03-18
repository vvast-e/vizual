# План: opacity, пан, текстура, прямоугольник

## 1) Что такое opacity

**Opacity** (непрозрачность) — свойство слоя от 0 до 1: 0 = полностью прозрачный, 1 = полностью непрозрачный. Значение `0.85` в коде текстуры означает 85% непрозрачности (15% просвечивает фото). Никаких изменений по пункту не требуется — только пояснение.

---

## 2) Исправление панорамы (двигать фото нельзя)

**Причина:** Фоновому изображению задано `evented: false`, из‑за чего в Fabric.js v7 события мыши над объектом могут не доходить до канваса или обрабатываться так, что пан не запускается.

**Решение (по документации Fabric):** События мыши приходят и на канвас, и на объект; у события канваса есть `target` — объект под курсором (или `undefined`). Нужно:
- Вернуть фоновому изображению **`evented: true`** (оставить **`selectable: false`**, чтобы его нельзя было выделять и таскать).
- Пометить фоновое изображение, чтобы в обработчиках отличать его от масок (например, `data.isBackground = true` или ref на объект).
- В `canvas.on('mouse:down')` при **`tool === null`**: запускать пан только если **`opt.target == null`** ИЛИ **`opt.target` — это фоновое изображение** (проверка по метке или ref). Тогда клик по пустому месту и по фото будет начинать пан.

**Файлы и места:**
- `src/hooks/useCanvas2D.ts`
  - Добавить ref для фонового изображения, например `backgroundImageRef = useRef<FabricImage | null>(null)`.
  - В `loadPhotoFromDataUrl`: не задавать `evented: false`; задать `selectable: false` и пометку в `data`, например `data: { isBackground: true }`; после `canvas.add(img)` записать `backgroundImageRef.current = img`.
  - В `clearCanvas` обнулить `backgroundImageRef.current`.
  - В `initCanvas` в обработчике `mouse:down`: в блоке, где `tool === null`, перед `panStartRef.current = { ... }` добавить проверку: если `opt.target != null` и `opt.target !== backgroundImageRef.current` (или проверка по `(opt.target as any).data?.isBackground`), то `return` (пан не начинать). Иначе — выполнять текущую логику установки `panStartRef`.

---

## 3) Текстура «в зуме» — наложение не совпадает с фото

**Причина:** Rect с паттерном создаётся с `left: 0, top: 0` и размером `canvas.width` x `canvas.height` (размер элемента канваса в пикселях). Это экранные/размеры канваса, а не координаты сцены. При зуме/пане объекты рисуются в координатах сцены; кроме того, в Fabric 7 у Rect по умолчанию `originX`/`originY = 'center'`, поэтому интерпретация `left`/`top` может быть не «левый верхний угол».

**Решение:**
- Слой текстуры должен покрывать в **координатах сцены** ту же область, что и фоновое изображение. То есть нужно вычислить bounding box фонового изображения в координатах сцены (или использовать первый объект, помеченный как фон) и задать Rect по этим bounds.
- В `applyPatternToCanvas`: не принимать только `canvas`; принимать также **опционально bounds** `{ left, top, width, height }` в координатах сцены. Если bounds не переданы — использовать текущее поведение (0, 0, canvas.width, canvas.height) для обратной совместимости.
- В `useCanvas2D.applyTexture`: перед вызовом `applyPatternToCanvas` найти фоновое изображение (первый объект с `data.isBackground` или по ref), вычислить его aabb в координатах сцены (например, через `obj.getBoundingRect(true)` или аналог в Fabric 7 с учётом viewport; при необходимости использовать `canvas.getScenePoint` для углов объекта). Передать эти bounds в `applyPatternToCanvas`.
- При создании Rect текстуры в `applyPatternToCanvas` явно задать **`originX: 'left'`, `originY: 'top'`**, чтобы `left`/`top`/`width`/`height` трактовались как левый верхний угол и размеры.
- При необходимости (если паттерн всё ещё выглядит «в зуме») рассмотреть масштаб паттерна: задать у Pattern или Rect масштаб так, чтобы плотность повторения текстуры соответствовала размеру фото в сцене (детали — после проверки текущего поведения).

**Файлы:**
- `src/lib/texture-processor.ts`: сигнатура `applyPatternToCanvas(canvas, textureUrl, repeat, clipPath?, sceneBounds?)`; при наличии `sceneBounds` использовать их для `left`, `top`, `width`, `height`; для Rect задать `originX: 'left'`, `originY: 'top'`.
- `src/hooks/useCanvas2D.ts`: в `applyTexture` получить bounds фонового изображения в координатах сцены и передать в `applyPatternToCanvas`.

---

## 4) Прямоугольное выделение: первый клик = центр прямоугольника

**Текущее поведение:** Первый клик трактуется как левый верхний угол, второй (при отпускании) — как противоположный угол. В Fabric 7 у Rect по умолчанию `originX`/`originY = 'center'`, поэтому задание `left`/`top` как угла даёт визуально неверное положение.

**Требуемое поведение:** Первый клик — **центр** прямоугольника; при перетаскивании прямоугольник рисуется от центра (расширяется во все стороны от первой точки).

**Решение:**
- В `mouse:down` при `tool === 'rect'`: сохранять не «угол», а **центр**: `rectCenterRef.current = { x: scenePoint.x, y: scenePoint.y }` (можно переименовать ref с `rectStartRef` на `rectCenterRef` или оставить имя, но хранить центр).
- В `mouse:move` и `mouse:up` при отрисовке Rect (превью и финальный):  
  - `cx = rectCenterRef.current.x`, `cy = rectCenterRef.current.y`.  
  - `width = 2 * Math.abs(scenePoint.x - cx)`, `height = 2 * Math.abs(scenePoint.y - cy)`.  
  - `left = cx - width / 2`, `top = cy - height / 2`.  
  - Создавать Rect с **`originX: 'left'`, `originY: 'top'`** и `left`, `top`, `width`, `height`, чтобы координаты однозначно соответствовали левому верхнему углу и размерам.
- Применить это и к превью прямоугольника, и к финальному объекту маски (в `mouse:up`).
- В `removeRectPreview` сбрасывать `rectStartRef` (или `rectCenterRef`) так же, как сейчас.

**Файлы:**
- `src/hooks/useCanvas2D.ts`: логика rect в `mouse:down` (сохранять центр), `mouse:move` (превью от центра), `mouse:up` (финальный Rect от центра); везде для Rect задавать `originX: 'left'`, `originY: 'top'`.

---

## Реализационный чек-лист

1. **Opacity:** Оставить как есть; при необходимости добавить в UI подсказку или комментарий, что opacity — непрозрачность слоя (0–1).
2. **Пан:** В `useCanvas2D` добавить `backgroundImageRef`, в `loadPhotoFromDataUrl` помечать изображение (`data.isBackground`), не ставить `evented: false`, сохранять ref; в `clearCanvas` обнулять ref.
3. **Пан:** В `mouse:down` при `tool === null` перед установкой `panStartRef` проверять: если `opt.target != null` и target не фоновое изображение — `return`; иначе выполнять текущую логику пана.
4. **Текстура:** В `texture-processor.applyPatternToCanvas` добавить опциональный параметр `sceneBounds`; при создании Rect задать `originX: 'left'`, `originY: 'top'`; при наличии `sceneBounds` использовать их для позиции и размера Rect.
5. **Текстура:** В `useCanvas2D.applyTexture` находить фоновое изображение (по `data.isBackground` или по ref), вычислять его bounds в координатах сцены, передавать в `applyPatternToCanvas`.
6. **Прямоугольник:** В `mouse:down` при rect сохранять центр (одна точка); в `mouse:move` и `mouse:up` вычислять width/height как удвоенное расстояние от центра до текущей точки, left/top = center - halfSize; для всех Rect (превью и маска) задавать `originX: 'left'`, `originY: 'top'`.
