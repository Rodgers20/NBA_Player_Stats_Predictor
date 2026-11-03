"use client";

import { useMemo } from "react";
import type { FreezeFrame } from "@/lib/types";
import { toWorld } from "./pitchGeometry";

/**
 * A single 360 freeze-frame rendered as positioned players.
 *
 * This is real positional tracking: the ingested Euro 2024 data holds 2,698,999
 * player positions across 51 matches. Teammates, opponents, the keeper and the
 * acting player are each distinguished by shape and colour, not colour alone.
 */
export function FreezeFramePlayers({ frame }: { frame: FreezeFrame | null }) {
  const players = useMemo(() => frame?.players ?? [], [frame]);
  if (!frame) return null;

  return (
    <group>
      {players.map((p, i) => {
        const [x, z] = toWorld(p.x, p.y);
        const color = p.keeper
          ? "#C0851F"
          : p.actor
            ? "#00D26A"
            : p.teammate
              ? "#3792FD"
              : "#FC4540";
        const height = p.actor ? 2.6 : 2.0;
        return (
          <group key={i} position={[x, 0, z]}>
            <mesh position={[0, height / 2, 0]} castShadow>
              <capsuleGeometry args={[0.55, height * 0.55, 4, 10]} />
              <meshStandardMaterial
                color={color}
                emissive={p.actor ? color : "#000000"}
                emissiveIntensity={p.actor ? 0.6 : 0}
                roughness={0.5}
              />
            </mesh>
            {/* ground disc keeps position legible from a low camera angle */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
              <ringGeometry args={[0.85, 1.15, 20]} />
              <meshBasicMaterial color={color} transparent opacity={0.5} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}
