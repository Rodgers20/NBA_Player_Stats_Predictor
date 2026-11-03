/**
 * Shared pitch geometry.
 *
 * StatsBomb uses a 120 x 80 pitch with the origin at a corner. We map that into
 * a centred 3D space: X runs along the length, Z across the width, Y is height.
 *
 * GROUND_MARGIN is not arbitrary. The ingested 360 tracking data contains real
 * player positions from x -9.5 to 125.1 and y -9.35 to 89.5 — players legitimately
 * stand off the pitch for throw-ins and behind the goal line. The ground plane is
 * sized so those positions land on grass rather than in the void.
 */
export const PITCH_LENGTH = 120;
export const PITCH_WIDTH = 80;
export const GROUND_MARGIN = 14;

/** StatsBomb (x, y) -> three.js (x, z), centred on the origin. */
export const toWorld = (x: number, y: number): [number, number] => [
  x - PITCH_LENGTH / 2,
  y - PITCH_WIDTH / 2,
];

/** Arc apex height for a pass, taken from StatsBomb's own height classification. */
export const passArcHeight = (height: string | null): number => {
  switch (height) {
    case "High Pass": return 7.5;
    case "Low Pass": return 2.6;
    default: return 0.35; // Ground Pass
  }
};

/** Quadratic bezier sampled into points, used for pass tubes. */
export function arcPoints(
  x1: number, y1: number, x2: number, y2: number, apex: number, segments = 24,
): [number, number, number][] {
  const [ax, az] = toWorld(x1, y1);
  const [bx, bz] = toWorld(x2, y2);
  const mx = (ax + bx) / 2;
  const mz = (az + bz) / 2;
  const pts: [number, number, number][] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const it = 1 - t;
    const px = it * it * ax + 2 * it * t * mx + t * t * bx;
    const pz = it * it * az + 2 * it * t * mz + t * t * bz;
    const py = 2 * it * t * apex;
    pts.push([px, py + 0.06, pz]);
  }
  return pts;
}

/** Pitch line segments as [x1,y1,x2,y2] in StatsBomb coordinates. */
export const PITCH_LINES: [number, number, number, number][] = [
  // boundary
  [0, 0, 120, 0], [120, 0, 120, 80], [120, 80, 0, 80], [0, 80, 0, 0],
  // halfway
  [60, 0, 60, 80],
  // left penalty area (18 yard)
  [0, 18, 18, 18], [18, 18, 18, 62], [18, 62, 0, 62],
  // left six yard
  [0, 30, 6, 30], [6, 30, 6, 50], [6, 50, 0, 50],
  // right penalty area
  [120, 18, 102, 18], [102, 18, 102, 62], [102, 62, 120, 62],
  // right six yard
  [120, 30, 114, 30], [114, 30, 114, 50], [114, 50, 120, 50],
];

export const CENTRE_CIRCLE_RADIUS = 10;
export const PENALTY_SPOTS: [number, number][] = [[12, 40], [108, 40]];
