"use client";

import { useMemo } from "react";
import * as THREE from "three";
import {
  CENTRE_CIRCLE_RADIUS, GROUND_MARGIN, PENALTY_SPOTS, PITCH_LENGTH,
  PITCH_LINES, PITCH_WIDTH, toWorld,
} from "./pitchGeometry";

const LINE_COLOR = "#8FA3B8";

/** The pitch: turf, mown stripes, markings, goals. */
export function Pitch({ stripes = 12 }: { stripes?: number }) {
  const lineGeometries = useMemo(
    () =>
      PITCH_LINES.map(([x1, y1, x2, y2]) => {
        const [ax, az] = toWorld(x1, y1);
        const [bx, bz] = toWorld(x2, y2);
        return new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(ax, 0.02, az),
          new THREE.Vector3(bx, 0.02, bz),
        ]);
      }),
    [],
  );

  const circle = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 64; i++) {
      const a = (i / 64) * Math.PI * 2;
      pts.push(
        new THREE.Vector3(
          Math.cos(a) * CENTRE_CIRCLE_RADIUS,
          0.02,
          Math.sin(a) * CENTRE_CIRCLE_RADIUS,
        ),
      );
    }
    return new THREE.BufferGeometry().setFromPoints(pts);
  }, []);

  const stripeWidth = PITCH_LENGTH / stripes;

  return (
    <group>
      {/* ground plane, extended so real off-pitch tracking positions land on grass */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry
          args={[PITCH_LENGTH + GROUND_MARGIN * 2, PITCH_WIDTH + GROUND_MARGIN * 2]}
        />
        <meshStandardMaterial color="#0C2418" roughness={0.95} />
      </mesh>

      {/* mown stripes */}
      {Array.from({ length: stripes }).map((_, i) => (
        <mesh
          key={i}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[-PITCH_LENGTH / 2 + stripeWidth * (i + 0.5), 0, 0]}
          receiveShadow
        >
          <planeGeometry args={[stripeWidth, PITCH_WIDTH]} />
          <meshStandardMaterial
            color={i % 2 === 0 ? "#123D29" : "#0F3323"}
            roughness={0.9}
          />
        </mesh>
      ))}

      {lineGeometries.map((g, i) => (
        <primitive key={i} object={new THREE.Line(g, new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.55 }))} />
      ))}
      <primitive object={new THREE.Line(circle, new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.55 }))} />

      {PENALTY_SPOTS.map(([x, y], i) => {
        const [wx, wz] = toWorld(x, y);
        return (
          <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[wx, 0.03, wz]}>
            <circleGeometry args={[0.35, 16]} />
            <meshBasicMaterial color={LINE_COLOR} transparent opacity={0.7} />
          </mesh>
        );
      })}

      <Goal x={-PITCH_LENGTH / 2} />
      <Goal x={PITCH_LENGTH / 2} />
    </group>
  );
}

/** A goal frame: 8 yards wide, 2.67 yards high, in StatsBomb units. */
function Goal({ x }: { x: number }) {
  const halfWidth = 4;
  const height = 2.67;
  const bar = 0.16;
  return (
    <group position={[x, 0, 0]}>
      <mesh position={[0, height / 2, -halfWidth]} castShadow>
        <cylinderGeometry args={[bar, bar, height, 8]} />
        <meshStandardMaterial color="#D8E2EC" roughness={0.4} />
      </mesh>
      <mesh position={[0, height / 2, halfWidth]} castShadow>
        <cylinderGeometry args={[bar, bar, height, 8]} />
        <meshStandardMaterial color="#D8E2EC" roughness={0.4} />
      </mesh>
      <mesh position={[0, height, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[bar, bar, halfWidth * 2, 8]} />
        <meshStandardMaterial color="#D8E2EC" roughness={0.4} />
      </mesh>
    </group>
  );
}
