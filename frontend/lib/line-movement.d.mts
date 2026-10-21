import type { LineSnapshot } from './types';
export function bookSeries(snapshots: LineSnapshot[]): Map<string, LineSnapshot[]>;
export function primaryBook(snapshots: LineSnapshot[]): string | null;
export function movement(points: LineSnapshot[]): { opening: number; current: number; change: number; samples: number } | null;
