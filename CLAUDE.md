# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1) Назначение проекта

- `vizual` — full-stack приложение для визуализации материалов/цветов на фото.
- Frontend: React + TypeScript + Vite (`src/`).
- Backend: FastAPI + ML пайплайн + PostgreSQL для админ-каталога (`backend/`).
- Прод-окружение: Docker Compose (`docker-compose.yml`) с `frontend` (nginx), `backend`, `postgres`.

## 2) Базовые принципы для изменений

- Делать минимальные, точные изменения под задачу.
- Не выполнять несогласованный рефакторинг "по пути".
- Сохранять обратную совместимость API/DTO, если это не оговорено явно.
- Не редактировать плановые/служебные файлы пользователя без прямого запроса.
- Не удалять существующий функционал детекции/warp, если задача не про это.

## 3) Структура и зоны ответственности

- `src/`:
  - UI, виджеты, страницы, клиент API.
  - Алиас `@` настроен в `vite.config.ts`.
- `backend/`:
  - `main.py` — FastAPI app + ML endpoints.
  - `catalog_routes.py` — CRUD админов/материалов/цветов.
  - `catalog_auth.py` — bcrypt/JWT.
  - `db/` — SQLAlchemy (`database.py`, `models.py`, `deps.py`).
  - `alembic/` — миграции.
- `public/`:
  - `textures/`, `colors/` — файлы, на которые ссылается БД.
- `docker/`:
  - `backend.Dockerfile`, `nginx.conf`.

## 3а) Frontend-архитектура

### Страницы и маршруты

`/upload` → `/editor` → `/admin`. При отсутствии сессии в localStorage приложение редиректит на `/upload`.

### Zustand-сторы (4 штуки)

| Стор | Ответственность |
|---|---|
| `useVisualizerStore` | `photoDataUrl`, `viewMode` (2d/3d) |
| `useWallStore` | Определённые стены, текстуры по стенам, `wallImageSize` |
| `useMaterialStore` | Выбранный материал/цвет, списки быстрого доступа |
| `useUIStore` | Инструменты тулбара, maskTool, `sceneMode` (interior/exterior), режим правки углов |

Состояние сессии (фото, стены, текстуры, sceneMode) сохраняется в `localStorage` через `src/lib/persist.ts`. Быстрый доступ к материалам/цветам персистится отдельно под ключом `vizual-quick-access` в `EditorPage`.

### Canvas2D и useCanvas2D

`Canvas2D.tsx` рендерит Fabric.js-канвас; вся логика — в `useCanvas2D.ts`:

- Фото загружается как заблокированный `FabricImage` (фон).
- Каждая стена получает кнопку-оверлей `Rect`, центрированную по `wall.center` (`wallOverlaysRef`).
- Слои текстур хранятся в `textureLayersRef`, ключ — `wallId` или `"background"`.
- Хэндлы правки углов: `Rect` (синие, perspective mode) или `Circle` (зелёные, polygon mode) в `cornerHandlesRef`.

### Два пайплайна применения текстур

**Интерьер** (`sceneMode === 'interior'`): клиентский рендер в `src/lib/texture-processor.ts`.
- `renderPerspectiveWallTexture` разбивает quad стены на 2 треугольника и отображает тайловую текстуру через `drawTexturedTriangle` (аффинное преобразование через `ctx.setTransform`).
- Результат — `FabricImage` с Fabric `clipPath` из полигона стены.

**Экстерьер** (`sceneMode === 'exterior'`): серверный warp.
- `warpWallTexture` (`src/hooks/warpWallTexture.ts`) постит текстуру + corners + polygon + маску на `POST /api/warp-wall-texture`, получает обратно PNG blob и добавляет его как `FabricImage`, выровненный по трансформу фона.
- `wall.regions` (два региона варпа: тело стены + фронтон) передаются бэкенду.

### Колоризация текстур

`useColorize` → `applyHsvColorShift` (`src/lib/color-utils.ts`): алгоритм HSL Color Transfer.
Сохраняет фактуру текстуры, вычисляя отклонение яркости каждого пикселя от средней, затем применяет тон/насыщенность целевого цвета, сохраняя «детали» (сучки, волокна). `preserveFactor` адаптируется к яркости краски.

Hex цветов из API извлекается в рантайме через усреднение 5×5-пикселей swatch-превью (`extractHexFromImage` в `src/lib/materials-api.ts`).

### Модель данных стены

```ts
WallData {
  id: number
  corners: [number, number][]   // 4-точечный quad [TL, BL, BR, TR] — для перспективы
  polygon?: [number, number][]  // точный контур (произвольное кол-во точек) — для клиппинга
  regions?: WallWarpRegion[]    // только экстерьер: [wallRegion, gableRegion]
  center: [number, number]      // в системе координат image_size
}
```

Координаты всегда в системе координат бэкенда (`wallImageSize`). Канвас масштабирует через `bounds.width / wallImageSize.width` в рантайме.

## 4) Стандарты кода

### 4.1 TypeScript / React

- Использовать функциональные компоненты и хуки.
- Не использовать `any`, если можно вывести тип.
- DTO и domain-типы держать согласованными (`src/lib/materials-api.ts`, `src/types/*`).
- Для запросов к API использовать единый слой клиента (`src/lib/materials-api.ts`), а не `fetch` из случайных компонентов.
- Ошибки API показывать пользователю в понятном виде.
- Для состояния использовать существующие store-паттерны (Zustand).

### 4.2 Python / FastAPI

- Писать маршруты с явными схемами запроса/ответа.
- Валидацию данных выполнять на уровне Pydantic + сервисных функций.
- Не хранить пароли в plaintext; только bcrypt.
- Закрывать DB сессию через dependency (`get_db`).
- Для операций "БД + файл" учитывать частичную неуспешность и делать best-effort cleanup.

### 4.3 Общие

- Имена переменных/функций — семантичные.
- Комментарии — только там, где действительно есть неочевидная логика.
- Не добавлять "временные заглушки" и отладочные принты в финальный код.

## 5) PostgreSQL и миграции (обязательно)

- Источник строки подключения: `DATABASE_URL`.
- Для локального backend (вне контейнера) использовать host `localhost`.
- Для backend внутри docker-compose использовать host `postgres`.
- Строка должна быть ASCII/UTF-8 без скрытых символов (NBSP/`Â`), иначе возможны ошибки `UnicodeDecodeError` в `psycopg2`.

Рекомендуемые примеры:

- Локально (backend на хосте, postgres в docker):
  - `postgresql://vizual:vizual@localhost:5432/vizual`
- В compose:
  - `postgresql://vizual:vizual@postgres:5432/vizual`

Миграции:

1. Проверить корректный `DATABASE_URL` в текущем окружении.
2. Выполнить в `backend/`:
   - `alembic upgrade head`
3. Новые схемные изменения вносить только через миграции в `backend/alembic/versions/`.

## 6) Запуск и окружения

### 6.1 Локальная разработка

- Frontend:
  - `npm install`
  - `npm run dev`
- Backend:
  - активировать `backend/venv`
  - `pip install -r requirements.txt`
  - `uvicorn main:app --host 0.0.0.0 --port 8000`
- Vite проксирует `/api` на `http://localhost:8000` (см. `vite.config.ts`).

### 6.2 Docker Compose

- `docker compose up --build`
- Сервисы:
  - frontend: `:80`
  - backend: `:8000` и `:8001`
  - postgres: `:5432`
- nginx проксирует:
  - `/api/*` -> backend:8000
  - `/textures/*` -> backend:8000/textures/*
  - `/colors/*` -> backend:8000/colors/*
  - `/gdino`, `/sam` -> backend:8001

## 7) API и контрактные правила

- Публичные выдачи в визуализатор должны фильтровать `is_visible_in_visualizer=true`.
- Админские CRUD операции защищать токеном Bearer (JWT).
- Не ломать существующие ключи ответа без миграционного слоя на frontend.
- При добавлении нового поля:
  1) модель БД,
  2) миграция,
  3) pydantic-схемы/роуты,
  4) клиент API,
  5) UI.

## 8) Работа с файлами материалов/цветов

- Upload хранить в `public/textures` и `public/colors`; в БД хранить относительный путь.
- Пути очищать от небезопасных сегментов (`..`, обратные слеши и т.д.).
- Удаление записи должно удалять и файл (best effort), с безопасной обработкой ошибок.

## 9) Проверки перед завершением задачи

- Frontend:
  - `npm run lint`
  - `npm run build` (или минимум `tsc -b`)
- Backend:
  - отсутствие синтаксических ошибок
  - маршруты импортируются без падений
  - миграции применяются (`alembic upgrade head`)
- Если менялся каталог:
  - проверить `GET /api/materials` и/или `GET /api/colors`
  - проверить auth-роуты и CRUD сценарии в админке

## 10) Чего делать нельзя

- Не хардкодить админ-пароли/секреты в коде.
- Не коммитить реальные креды, `.env` и приватные ключи.
- Не менять одновременно архитектуру frontend и backend без необходимости задачи.
- Не переписывать большие части `main.py`, если задача затрагивает только каталог или UI.

## 11) Рекомендации по рабочему процессу для Claude

1. Сначала определить зону изменения (frontend/backend/db/docker).
2. Проверить существующие паттерны в соседних файлах и следовать им.
3. Делать изменения малыми шагами, сохраняя контракт между слоями.
4. После изменений проверить типизацию/линт/миграции.
5. В отчёте указывать:
   - что изменено,
   - почему,
   - какие команды/проверки выполнены,
   - что осталось сделать вручную (если есть).

