/**
 * Типы для 2D канваса (Fabric.js, фото, маска).
 */

export type CanvasTool = 'pan' | 'zoom' | 'brush' | 'eraser' | 'select'

export interface CanvasDimensions {
  width: number
  height: number
}

export interface MaskData {
  /** Data URL или blob для маски */
  dataUrl: string
  width: number
  height: number
}
