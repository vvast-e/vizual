import { useMaterialStore } from '@/store/useMaterialStore'

/**
 * Дом (экстерьер): заглушка из примитивов.
 * Когда будет house-exterior.glb — загрузка через useGLTF.
 */
export function HouseModel() {
  const selectedMaterial = useMaterialStore((s) => s.selectedMaterial)
  const color = selectedMaterial?.texture.url ? '#8b7355' : '#c4a574'

  return (
    <group>
      <mesh position={[0, 1.5, 0]} castShadow receiveShadow>
        <boxGeometry args={[4, 3, 3]} />
        <meshStandardMaterial color={color} roughness={0.8} metalness={0} />
      </mesh>
      <mesh position={[0, 3.2, 0]} rotation={[0, 0, 0]} castShadow>
        <coneGeometry args={[2.5, 1.5, 4]} />
        <meshStandardMaterial color="#6b5344" roughness={0.9} metalness={0} />
      </mesh>
    </group>
  )
}
