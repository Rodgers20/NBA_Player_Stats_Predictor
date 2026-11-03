"use client";

import { useMemo } from "react";
import * as THREE from "three";
import type { Pass } from "@/lib/types";
import { arcPoints, passArcHeight } from "./pitchGeometry";

/**
 * Passes as 3D arcs.
 *
 * The arc height is NOT invented for visual effect — it comes from StatsBomb's
 * own `pass.height` classification (Ground / Low / High), which the ingested
 * Euro 2024 data supplies for all 53,888 passes. That is what makes this a
 * data visualisation rather than decoration: a lofted ball genuinely arcs
 * higher than a ground pass.
 */
export function PassArcs({
  passes, showIncomplete = true, maxPasses = 900,
}: {
  passes: Pass[];
  showIncomplete?: boolean;
  maxPasses?: number;
}) {
  const { completed, incomplete } = useMemo(() => {
    const subset = passes.slice(0, maxPasses);
    const build = (list: Pass[]) => {
      const positions: number[] = [];
      for (const p of list) {
        if (p.end_x == null || p.end_y == null) continue;
        const pts = arcPoints(p.x, p.y, p.end_x, p.end_y, passArcHeight(p.height), 18);
        for (let i = 0; i < pts.length - 1; i++) {
          positions.push(...pts[i], ...pts[i + 1]);
        }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(new Float32Array(positions), 3),
      );
      return geo;
    };
    return {
      completed: build(subset.filter((p) => p.completed)),
      incomplete: build(subset.filter((p) => !p.completed)),
    };
  }, [passes, maxPasses]);

  return (
    <group>
      <primitive
        object={
          new THREE.LineSegments(
            completed,
            new THREE.LineBasicMaterial({
              color: new THREE.Color("#09AF51"),
              transparent: true,
              opacity: 0.5,
            }),
          )
        }
      />
      {showIncomplete && (
        <primitive
          object={
            new THREE.LineSegments(
              incomplete,
              new THREE.LineBasicMaterial({
                color: new THREE.Color("#FC4540"),
                transparent: true,
                opacity: 0.28,
              }),
            )
          }
        />
      )}
    </group>
  );
}
