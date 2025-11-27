// viz-frontend/src/components/Vis3D.tsx

import React, { useRef, useMemo } from 'react'; // Import useMemo
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { TokenData } from '../services/api';

interface Vis3DProps {
  tokens: TokenData[];
}

// Points component for efficient rendering of many tokens
function TokenParticles({ tokens }: { tokens: TokenData[] }) {
  const pointsRef = useRef<THREE.Points>(null);

  const colorScheme = useMemo(() => ({
    masked: new THREE.Color(0xff0000), // Red for masked
    decodedHighConf: new THREE.Color(0x00ff00), // Green for decoded, high confidence
    decodedLowConf: new THREE.Color(0xffff00), // Yellow for decoded, low confidence
  }), []);

  const [positions, colors] = useMemo(() => {
    const positionsArray = new Float32Array(tokens.length * 3);
    const colorsArray = new Float32Array(tokens.length * 3);

    tokens.forEach((token, i) => {
      // Scale up coordinates for better visibility in the 3D scene
      positionsArray[i * 3] = token.pos[0] * 5;
      positionsArray[i * 3 + 1] = token.pos[1] * 5;
      positionsArray[i * 3 + 2] = token.pos[2] * 5;

      let displayColor = new THREE.Color();
      if (token.is_masked) {
        displayColor.copy(colorScheme.masked);
      } else {
        if (token.conf > 0.9) { // High confidence threshold
          displayColor.copy(colorScheme.decodedHighConf);
        } else {
          displayColor.copy(colorScheme.decodedLowConf);
        }
      }
      displayColor.toArray(colorsArray, i * 3);
    });
    return [positionsArray, colorsArray];
  }, [tokens, colorScheme]); // Re-calculate only when tokens change

  return (
    <points ref={pointsRef}>
      <bufferGeometry attach="geometry">
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={positions.length / 3}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          array={colors}
          count={colors.length / 3}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial vertexColors={true} size={0.1} sizeAttenuation={true} transparent opacity={0.7} />
    </points>
  );
};


const Vis3D: React.FC<Vis3DProps> = ({ tokens }) => {
  return (
    <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls />
      <axesHelper args={[5]} /> {/* Helps with orientation */}
      {tokens && tokens.length > 0 ? (
        <TokenParticles tokens={tokens} />
      ) : (
        <Text
          position={[0, 0, 0]}
          fontSize={0.5}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          Waiting for generation...
        </Text>
      )}
    </Canvas>
  );
};

export default Vis3D;