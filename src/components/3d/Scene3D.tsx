import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { RoomModel } from './RoomModel'
import { HouseModel } from './HouseModel'

export interface Scene3DProps {
  className?: string
  /** Режим: комната или дом */
  sceneMode?: 'room' | 'house'
}

function SceneFallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#888" />
    </mesh>
  )
}

export function Scene3D({ className = '', sceneMode = 'room' }: Scene3DProps) {
  return (
    <div className={className} style={{ width: '100%', height: 400, background: '#e5e7eb' }}>
      <Canvas
        camera={{ position: [4, 3, 4], fov: 50 }}
        gl={{ antialias: true }}
        dpr={[1, 2]}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 8, 5]} intensity={1} castShadow />
        <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
        <Suspense fallback={<SceneFallback />}>
          {sceneMode === 'room' && <RoomModel />}
          {sceneMode === 'house' && <HouseModel />}
        </Suspense>
      </Canvas>
    </div>
  )
}
