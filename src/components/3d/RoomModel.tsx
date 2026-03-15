import { useRef } from 'react'
import type * as THREE from 'three'
import { useMaterialStore } from '@/store/useMaterialStore'

/**
 * Комната: примитивы-заглушка (пол + 4 стены).
 * Когда будет room-base.glb — можно загружать через useGLTF и вешать текстуры.
 */
export function RoomModel() {
  const floorRef = useRef<THREE.Mesh>(null)
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const color = selectedMaterial?.texture.url ? '#8b7355' : '#a0a0a0'

  return (
    <group>
      <mesh ref={floorRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[6, 6]} />
        <meshStandardMaterial color={color} roughness={0.8} metalness={0} />
      </mesh>
      {[
        { pos: [0, 1.5, -3] as [number, number, number], rot: [0, 0, 0] as [number, number, number], size: [6, 3, 0.2] as [number, number, number] },
        { pos: [0, 1.5, 3] as [number, number, number], rot: [0, Math.PI, 0] as [number, number, number], size: [6, 3, 0.2] as [number, number, number] },
        { pos: [-3, 1.5, 0] as [number, number, number], rot: [0, Math.PI / 2, 0] as [number, number, number], size: [6, 3, 0.2] as [number, number, number] },
        { pos: [3, 1.5, 0] as [number, number, number], rot: [0, -Math.PI / 2, 0] as [number, number, number], size: [6, 3, 0.2] as [number, number, number] },
      ].map(({ pos, rot, size }, i) => (
        <mesh key={i} position={pos} rotation={rot} castShadow receiveShadow>
          <boxGeometry args={size} />
          <meshStandardMaterial color={color} roughness={0.8} metalness={0} />
        </mesh>
      ))}
    </group>
  )
}
