import { Routes, Route, Navigate } from 'react-router-dom'
import { UploadPage } from '@/pages/UploadPage'
import { EditorPage } from '@/pages/EditorPage'
import { AdminPage } from '@/pages/AdminPage'

function App() {
  return (
    <Routes>
      <Route path="/upload" element={<UploadPage />} />
      <Route path="/editor" element={<EditorPage />} />
      <Route path="/admin" element={<AdminPage />} />
      <Route path="*" element={<Navigate to="/upload" replace />} />
    </Routes>
  )
}

export default App
