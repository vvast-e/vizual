import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'sonner'
import { UploadPage } from '@/pages/UploadPage'
import { EditorPage } from '@/pages/EditorPage'
import { AdminPage } from '@/pages/AdminPage'

function App() {
  return (
    <>
      <Toaster
        position="bottom-center"
        toastOptions={{
          style: {
            background: '#1f2937',
            color: '#f9fafb',
            border: '1px solid #374151',
            borderRadius: '8px',
            fontSize: '13px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          },
        }}
        duration={2500}
      />
      <Routes>
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/editor" element={<EditorPage />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="*" element={<Navigate to="/upload" replace />} />
      </Routes>
    </>
  )
}

export default App
