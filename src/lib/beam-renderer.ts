/**
 * Рендер псевдо-3D фальшбалок на Fabric.js канвас.
 * Каждая балка = нижняя грань (текстура) + две боковые грани (затемнённая текстура).
 */

import type { MutableRefObject } from 'react'
import { FabricImage } from 'fabric'
import type { Canvas } from 'fabric'
import type { BeamQuad } from '@/lib/beam-layout'
import { renderPerspectiveWallTexture } from '@/lib/texture-processor'

/** Максимальная высота боковой грани (пиксели экрана); масштабируется по ширине балки */
const SIDE_HEIGHT_MAX_PX = 14

/** Степень затемнения боковой грани (0=чёрный, 1=оригинал) */
const SIDE_DARKEN = 0.45

function edgeLen2(a: [number, number], b: [number, number]): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

/**
 * Затемняет offscreen-canvas и возвращает его data URL.
 */
function darkenCanvas(src: HTMLCanvasElement, factor: number): string {
  const dst = document.createElement('canvas')
  dst.width = src.width
  dst.height = src.height
  const ctx = dst.getContext('2d')
  if (!ctx) return src.toDataURL()
  ctx.drawImage(src, 0, 0)
  ctx.globalCompositeOperation = 'multiply'
  ctx.fillStyle = `rgb(${Math.round(factor * 255)},${Math.round(factor * 255)},${Math.round(factor * 255)})`
  ctx.fillRect(0, 0, dst.width, dst.height)
  ctx.globalCompositeOperation = 'source-over'
  return dst.toDataURL()
}

export interface RenderedBeam {
  /** Fabric-объекты, которые нужно добавить на канвас (нижняя грань + боковины) */
  objects: FabricImage[]
  beamId: number
}

/**
 * Рендерит одну балку и возвращает Fabric-объекты.
 * scaleX/scaleY — масштаб от wallImageSize к экранным координатам.
 * offsetX/offsetY — сдвиг фона на канвасе (bounds.left/top).
 */
export async function renderBeam(
  beam: BeamQuad,
  textureUrl: string,
  canvasWidth: number,
  canvasHeight: number,
  scaleX: number,
  scaleY: number,
  offsetX: number,
  offsetY: number,
): Promise<RenderedBeam> {
  // Переводим координаты балки в экранные
  const screenQuad = beam.quad.map(
    ([x, y]) => [offsetX + x * scaleX, offsetY + y * scaleY] as [number, number],
  )

  // Нижняя грань балки
  const { canvas: faceCanvas, offsetX: faceOX, offsetY: faceOY } =
    await renderPerspectiveWallTexture(
      textureUrl,
      screenQuad,
      canvasWidth,
      canvasHeight,
      0.25,       // textureScale
      undefined,  // polygon
      null,       // maskImage
      null,       // fullCanvasSize
      'beam',     // sourceMode — одна доска, волокно вдоль балки
    )

  const faceDataUrl = faceCanvas.toDataURL()

  const objects: FabricImage[] = []

  // Нижняя грань
  const faceImg = await FabricImage.fromURL(faceDataUrl, { crossOrigin: 'anonymous' })
  faceImg.set({ left: faceOX, top: faceOY, originX: 'left', originY: 'top', selectable: false, evented: false })
  objects.push(faceImg)

  // ── Боковые грани ──────────────────────────────────────────────────────────
  // Определяем, какая пара противоположных рёбер ДЛИННЕЕ → это «длинные стороны» балки.
  // Quad = [0,1,2,3]:
  //   Пара A: рёбра 0→1 и 3→2  (для depth-балки это длинные стороны)
  //   Пара B: рёбра 0→3 и 1→2  (для width-балки это длинные стороны)
  const pairALen = (edgeLen2(screenQuad[0], screenQuad[1]) + edgeLen2(screenQuad[3], screenQuad[2])) / 2
  const pairBLen = (edgeLen2(screenQuad[0], screenQuad[3]) + edgeLen2(screenQuad[1], screenQuad[2])) / 2

  let longEdgePairs: [[number, number], [number, number]][]
  let crossLen: number
  if (pairALen >= pairBLen) {
    // depth-стиль: длинные рёбра 0→1 и 3→2
    longEdgePairs = [
      [screenQuad[0], screenQuad[1]],
      [screenQuad[3], screenQuad[2]],
    ]
    crossLen = pairBLen
  } else {
    // width-стиль: длинные рёбра 0→3 и 1→2
    longEdgePairs = [
      [screenQuad[0], screenQuad[3]],
      [screenQuad[1], screenQuad[2]],
    ]
    crossLen = pairALen
  }

  // Высота боковины: 25% от ширины балки, но не более SIDE_HEIGHT_MAX_PX и не менее 2px.
  // Узкие/дальние балки получают тонкую боковину — перспективно корректно.
  const sideH = Math.max(2, Math.min(SIDE_HEIGHT_MAX_PX, crossLen * 0.25))

  for (const [pTop, pBot] of longEdgePairs) {
    // Quad боковины: верхнее ребро = длинное ребро балки, нижнее = сдвинуто вниз на sideH.
    const sideQuad: [number, number][] = [
      pTop,
      [pTop[0], pTop[1] + sideH],
      [pBot[0], pBot[1] + sideH],
      pBot,
    ]

    const { canvas: sideCanvas, offsetX: sOX, offsetY: sOY } =
      await renderPerspectiveWallTexture(
        textureUrl,
        sideQuad,
        canvasWidth,
        canvasHeight,
        0.25,       // textureScale
        undefined,  // polygon
        null,       // maskImage
        null,       // fullCanvasSize
        'beam',     // sourceMode — одна доска, без поперечного тайлинга
      )

    const darkUrl = darkenCanvas(sideCanvas, SIDE_DARKEN)
    const sideImg = await FabricImage.fromURL(darkUrl, { crossOrigin: 'anonymous' })
    sideImg.set({ left: sOX, top: sOY, originX: 'left', originY: 'top', selectable: false, evented: false })
    objects.push(sideImg)
  }

  return { objects, beamId: beam.id }
}

/**
 * Удаляет слой балок с канваса.
 */
export function clearBeamLayers(
  canvas: Canvas,
  beamLayersRef: MutableRefObject<Record<number, FabricImage[]>>,
) {
  for (const objs of Object.values(beamLayersRef.current)) {
    objs.forEach((o) => canvas.remove(o))
  }
  beamLayersRef.current = {}
}
