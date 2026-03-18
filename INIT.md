# Инициализация проекта визуализатора материалов

## 1. Создать Vite-проект (в родительской папке или с указанием имени)

Если папка `vizual` уже есть и вы находитесь в ней — инициализируйте **в текущей папке**:

```powershell
cd d:\vizual
npm create vite@latest . -- --template react-ts
```

**Что выбирать:**
- **Project name:** если спросит — можно оставить `vizual` или ввести `material-visualizer`.
- **Template:** обязательно **react-ts** (React + TypeScript).

Если команда спрашивает про перезапись (папка не пуста): ответить **y** (yes).

---

## 2. Установить зависимости

```powershell
cd d:\vizual
npm install
```

---

## 3. Tailwind CSS

```powershell
npm install -D tailwindcss postcss autoprefixer
 
```

В **tailwind.config.js** в `content` добавить пути к исходникам, например:

```js
content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
```

В **src/index.css** в начало файла добавить:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

## 4. Остальные зависимости (одной командой)

```powershell
npm install three @react-three/fiber @react-three/drei three-projected-material fabric zustand react-router-dom clsx tailwind-merge lucide-react
npm install @radix-ui/react-slider @radix-ui/react-dialog @radix-ui/react-tabs
npm install -D @types/fabric
```

---

## 5. Path aliases

В **tsconfig.json** (и при наличии — в **tsconfig.app.json**) в `compilerOptions` добавить:

```json
"baseUrl": ".",
"paths": {
  "@/*": ["./src/*"]
}
```

В **vite.config.ts** в `resolve.alias` добавить:

```ts
import path from "path"

// внутри defineConfig:
resolve: {
  alias: {
    "@": path.resolve(__dirname, "./src"),
  },
},
```

---

## 6. ESLint + Prettier (по желанию)

```powershell
npm install -D eslint @typescript-eslint/eslint-plugin @typescript-eslint/parser eslint-plugin-react-hooks eslint-plugin-react-refresh prettier
```

Далее настроить `.eslintrc.cjs` и `.prettierrc` по стандартам проекта.

---

## Краткая последовательность (копируй и выполняй по шагам)

```powershell
cd d:\vizual
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

После этого вручную поправить `tailwind.config.js` (content) и `src/index.css` (директивы @tailwind), затем:

```powershell
npm install three @react-three/fiber @react-three/drei three-projected-material fabric zustand react-router-dom clsx tailwind-merge lucide-react @radix-ui/react-slider @radix-ui/react-dialog @radix-ui/react-tabs
npm install -D @types/fabric
```

Алиасы и структуру папок добавить по спецификации плана.
