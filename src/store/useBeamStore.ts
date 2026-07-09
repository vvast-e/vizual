import { create } from 'zustand'
import { autoLayoutBeams, type BeamDirection, type BeamQuad } from '@/lib/beam-layout'
import type { WallData } from '@/store/useWallStore'

export type { BeamDirection, BeamQuad }

interface BeamState {
  enabled: boolean
  count: number
  halfWidth: number
  direction: BeamDirection
  textureUrl: string | null
  beams: BeamQuad[]

  setEnabled: (v: boolean) => void
  setCount: (n: number) => void
  setHalfWidth: (w: number) => void
  setDirection: (d: BeamDirection) => void
  setTextureUrl: (url: string | null) => void
  autoLayout: (ceiling: WallData) => void
  addBeam: (beam: BeamQuad) => void
  removeBeam: (id: number) => void
  clear: () => void
}

export const useBeamStore = create<BeamState>((set, get) => ({
  enabled: false,
  count: 3,
  halfWidth: 0.05,
  direction: 'grid',
  textureUrl: null,
  beams: [],

  setEnabled: (enabled) => set({ enabled }),
  setCount: (count) => set({ count }),
  setHalfWidth: (halfWidth) => set({ halfWidth }),
  setDirection: (direction) => set({ direction }),
  setTextureUrl: (textureUrl) => set({ textureUrl }),

  autoLayout: (ceiling) => {
    const { count, halfWidth, direction } = get()
    const beams = autoLayoutBeams(ceiling, count, halfWidth, direction)
    set({ beams })
  },

  addBeam: (beam) =>
    set((s) => ({ beams: [...s.beams, beam] })),

  removeBeam: (id) =>
    set((s) => ({ beams: s.beams.filter((b) => b.id !== id) })),

  clear: () => set({ beams: [] }),
}))
