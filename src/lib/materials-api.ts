/**
 * API-клиент: материалы, цвета, админы, JWT.
 */

import type { Color, Material, MaterialCategory } from '@/types/material'

const ADMIN_TOKEN_KEY = 'vizual-admin-token'

export interface MaterialDTO {
  id: string
  name: string
  filename: string
  category: string
  scene_category: string
  url: string
  price: number | null
  size: string | null
  is_visible_in_visualizer: boolean
}

export interface ColorDTO {
  id: string
  name: string
  scene_category: string
  url: string
  price: number | null
  is_visible_in_visualizer: boolean
}

export interface AdminDTO {
  id: number
  username: string
  is_active: boolean
}

export function getAdminToken(): string | null {
  return localStorage.getItem(ADMIN_TOKEN_KEY)
}

export function setAdminToken(token: string | null): void {
  if (token) localStorage.setItem(ADMIN_TOKEN_KEY, token)
  else localStorage.removeItem(ADMIN_TOKEN_KEY)
}

function authHeaders(): HeadersInit {
  const t = getAdminToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

export async function adminLogin(username: string, password: string): Promise<void> {
  const res = await fetch('/api/admin/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Login failed: ${res.status}`)
  }
  const data = (await res.json()) as { access_token: string }
  setAdminToken(data.access_token)
}

export async function bootstrapFirstAdmin(
  username: string,
  password: string,
  setupSecret: string
): Promise<void> {
  const res = await fetch('/api/admin/bootstrap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, setup_secret: setupSecret }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Bootstrap failed: ${res.status}`)
  }
}

export interface ListMaterialsParams {
  page?: number
  page_size?: number
  scene_category?: string
  material_category?: string
  q?: string
  visible_only?: boolean
}

export async function fetchMaterials(params: ListMaterialsParams = {}): Promise<MaterialDTO[]> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.scene_category) sp.set('scene_category', params.scene_category)
  if (params.material_category) sp.set('material_category', params.material_category)
  if (params.q) sp.set('q', params.q)
  if (params.visible_only != null) sp.set('visible_only', String(params.visible_only))
  const q = sp.toString()
  const res = await fetch(`/api/materials${q ? `?${q}` : ''}`)
  if (!res.ok) throw new Error(`Failed to fetch materials: ${res.status}`)
  const data = (await res.json()) as { materials?: MaterialDTO[] }
  return data.materials ?? []
}

export async function fetchMaterialsAdmin(
  params: ListMaterialsParams = {}
): Promise<{ materials: MaterialDTO[]; total: number }> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.scene_category) sp.set('scene_category', params.scene_category)
  if (params.material_category) sp.set('material_category', params.material_category)
  if (params.q) sp.set('q', params.q)
  if (params.visible_only != null) sp.set('visible_only', String(params.visible_only))
  const q = sp.toString()
  const res = await fetch(`/api/materials${q ? `?${q}` : ''}`, { headers: { ...authHeaders() } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Failed: ${res.status}`)
  }
  return res.json()
}

export async function uploadMaterial(
  file: File,
  fields: {
    name: string
    scene_category: string
    material_category: string
    price?: string
    size?: string
    is_visible_in_visualizer: boolean
  }
): Promise<MaterialDTO> {
  const form = new FormData()
  form.append('file', file)
  form.append('name', fields.name)
  form.append('scene_category', fields.scene_category)
  form.append('material_category', fields.material_category)
  if (fields.price != null && fields.price !== '') form.append('price', fields.price)
  if (fields.size) form.append('size', fields.size)
  form.append('is_visible_in_visualizer', fields.is_visible_in_visualizer ? 'true' : 'false')
  const res = await fetch('/api/materials', { method: 'POST', body: form, headers: { ...authHeaders() } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Upload failed: ${res.status}`)
  }
  return res.json()
}

export async function patchMaterial(
  id: string,
  body: Partial<{
    name: string
    scene_category: string
    material_category: string
    price: number | null
    size: string | null
    is_visible_in_visualizer: boolean
  }>
): Promise<MaterialDTO> {
  const res = await fetch(`/api/materials/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Patch failed: ${res.status}`)
  }
  return res.json()
}

export async function deleteMaterial(id: string): Promise<void> {
  const res = await fetch(`/api/materials/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    headers: { ...authHeaders() },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Delete failed: ${res.status}`)
  }
}

export interface ListColorsParams {
  page?: number
  page_size?: number
  scene_category?: string
  q?: string
  visible_only?: boolean
}

export async function fetchColors(params: ListColorsParams = {}): Promise<ColorDTO[]> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.scene_category) sp.set('scene_category', params.scene_category)
  if (params.q) sp.set('q', params.q)
  if (params.visible_only != null) sp.set('visible_only', String(params.visible_only))
  const q = sp.toString()
  const res = await fetch(`/api/colors${q ? `?${q}` : ''}`)
  if (!res.ok) throw new Error(`Failed to fetch colors: ${res.status}`)
  const data = (await res.json()) as { colors?: ColorDTO[] }
  return data.colors ?? []
}

export async function fetchColorsAdmin(
  params: ListColorsParams = {}
): Promise<{ colors: ColorDTO[]; total: number }> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.scene_category) sp.set('scene_category', params.scene_category)
  if (params.q) sp.set('q', params.q)
  if (params.visible_only != null) sp.set('visible_only', String(params.visible_only))
  const q = sp.toString()
  const res = await fetch(`/api/colors${q ? `?${q}` : ''}`, { headers: { ...authHeaders() } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Failed: ${res.status}`)
  }
  return res.json()
}

export async function uploadColor(
  file: File,
  fields: {
    name: string
    scene_category: string
    price?: string
    is_visible_in_visualizer: boolean
  }
): Promise<ColorDTO> {
  const form = new FormData()
  form.append('file', file)
  form.append('name', fields.name)
  form.append('scene_category', fields.scene_category)
  if (fields.price != null && fields.price !== '') form.append('price', fields.price)
  form.append('is_visible_in_visualizer', fields.is_visible_in_visualizer ? 'true' : 'false')
  const res = await fetch('/api/colors', { method: 'POST', body: form, headers: { ...authHeaders() } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Upload failed: ${res.status}`)
  }
  return res.json()
}

export async function patchColor(
  id: string,
  body: Partial<{
    name: string
    scene_category: string
    price: number | null
    is_visible_in_visualizer: boolean
  }>
): Promise<ColorDTO> {
  const res = await fetch(`/api/colors/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Patch failed: ${res.status}`)
  }
  return res.json()
}

export async function deleteColor(id: string): Promise<void> {
  const res = await fetch(`/api/colors/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    headers: { ...authHeaders() },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Delete failed: ${res.status}`)
  }
}

export async function fetchAdmins(): Promise<AdminDTO[]> {
  const res = await fetch('/api/admins', { headers: { ...authHeaders() } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Failed: ${res.status}`)
  }
  return res.json()
}

export async function createAdmin(username: string, password: string): Promise<AdminDTO> {
  const res = await fetch('/api/admins', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Failed: ${res.status}`)
  }
  return res.json()
}

export async function patchAdmin(id: number, is_active: boolean): Promise<AdminDTO> {
  const res = await fetch(`/api/admins/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ is_active }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Failed: ${res.status}`)
  }
  return res.json()
}

export async function deleteAdmin(id: number): Promise<void> {
  const res = await fetch(`/api/admins/${id}`, { method: 'DELETE', headers: { ...authHeaders() } })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail || `Failed: ${res.status}`)
  }
}

/** DTO → визуализатор Material */
export function materialDtoToMaterial(dto: MaterialDTO): Material {
  const cat = (dto.category || 'other') as MaterialCategory
  return {
    id: dto.id,
    name: dto.name,
    category: cat,
    texture: {
      id: `tex-${dto.id}`,
      url: dto.url,
      name: dto.name,
    },
    presetColors: [],
  }
}

export async function extractHexFromImage(url: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = 5
      canvas.height = 5
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        resolve('#9d8b70')
        return
      }
      ctx.drawImage(img, 0, 0, 5, 5)
      const data = ctx.getImageData(0, 0, 5, 5).data
      let r = 0, g = 0, b = 0
      for (let i = 0; i < data.length; i += 4) {
        r += data[i]
        g += data[i + 1]
        b += data[i + 2]
      }
      const count = data.length / 4
      const hex =
        '#' +
        [Math.round(r / count), Math.round(g / count), Math.round(b / count)]
          .map((x) => x.toString(16).padStart(2, '0'))
          .join('')
      resolve(hex)
    }
    img.onerror = () => resolve('#9d8b70')
    img.src = url
  })
}

/** DTO → визуализатор Color (оттенок для HSV вычисляется из превью) */
export async function colorDtoToColor(dto: ColorDTO): Promise<Color> {
  const hex = dto.url ? await extractHexFromImage(dto.url) : '#9d8b70'
  return {
    id: String(dto.id),
    name: dto.name,
    hex,
    swatchUrl: dto.url,
  }
}
