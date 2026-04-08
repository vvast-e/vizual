/**
 * API-клиент для работы с материалами (текстурами).
 * GET /api/materials — список
 * POST /api/materials — загрузка (admin)
 * DELETE /api/materials/:filename — удаление (admin)
 */

export interface MaterialDTO {
  id: string
  name: string
  filename: string
  category: string
  url: string
}

function basicAuth(login: string, password: string): string {
  return btoa(`${login}:${password}`)
}

/** Получить список всех материалов */
export async function fetchMaterials(): Promise<MaterialDTO[]> {
  const res = await fetch('/api/materials')
  if (!res.ok) throw new Error(`Failed to fetch materials: ${res.status}`)
  const data = await res.json()
  return data.materials ?? []
}

/** Загрузить новый материал (требует авторизации) */
export async function uploadMaterial(
  file: File,
  name: string,
  category: string,
  login: string,
  password: string
): Promise<MaterialDTO> {
  const form = new FormData()
  form.append('file', file)
  form.append('name', name)
  form.append('category', category)
  form.append('authorization', basicAuth(login, password))
  const res = await fetch('/api/materials', { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Upload failed: ${res.status}`)
  }
  return res.json()
}

/** Удалить материал (требует авторизации) */
export async function deleteMaterial(
  filename: string,
  login: string,
  password: string
): Promise<void> {
  const res = await fetch(`/api/materials/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: {
      Authorization: `Basic ${basicAuth(login, password)}`,
    },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Delete failed: ${res.status}`)
  }
}
