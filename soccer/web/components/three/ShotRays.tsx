"use client";

import { useMemo } from "react";
import * as THREE from "three";
import type { Shot } from "@/lib/types";
import { PITCH_LENGTH, PITCH_WIDTH, toWorld } from "./pitchGeometry";

/**
 * Shots as rays from the strike point to where the ball actually finished.
 *
 * StatsBomb gives shots a three-component `end_location` — x, y and HEIGHT in
 * the goal mouth — so a shot placed in the top corner renders in the top corner
 * rather than being flattened onto the ground. Marker radius scales with xG,
 * and goals are emissive so they read instantly.
 */
export function ShotRays({ shots }: { shots: Shot[] }) {
  const rays = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const goal = new THREE.Color("#09AF51");
    const onTarget = new THREE.Color("#C0851F");
    const off = new THREE.Color("#5C6B7E");

    for (const s of shots) {
      const [sx, sz] = toWorld(s.x, s.y);
      const ex = s.end_x != null ? s.end_x - PITCH_LENGTH / 2 : sx;
      const ez = s.end_y != null ? s.end_y - PITCH_WIDTH / 2 : sz;
      const ey = s.end_z ?? 0.2;
      const c =
        s.outcome === "Goal" ? goal
        : ["Saved", "Saved To Post"].includes(s.outcome) ? onTarget
        : off;
      positions.push(sx, 0.25, sz, ex, Math.max(ey, 0.1), ez);
      colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(colors), 3));
    return geo;
  }, [shots]);

  return (
    <group>
      <primitive
        object={
          new THREE.LineSegments(
            rays,
            new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 }),
          )
        }
      />
      {shots.map((s) => {
        const [sx, sz] = toWorld(s.x, s.y);
        const isGoal = s.outcome === "Goal";
        // Radius encodes xG. sqrt keeps AREA proportional to the value rather
        // than radius, so a 0.4 xG chance does not look four times a 0.1 one.
        const r = 0.5 + Math.sqrt(s.xg ?? 0.01) * 2.6;
        return (
          <mesh key={s.event_id} position={[sx, 0.3, sz]}>
            <sphereGeometry args={[r, 16, 12]} />
            <meshStandardMaterial
              color={isGoal ? "#09AF51" : "#94A3B4"}
              emissive={isGoal ? "#09AF51" : "#000000"}
              emissiveIntensity={isGoal ? 0.85 : 0}
              transparent
              opacity={isGoal ? 0.95 : 0.55}
            />
          </mesh>
        );
      })}
    </group>
  );
}
