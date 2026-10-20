import type { Side, SideResult } from './hit-rate.mjs';

export type RibbonCell = { value: number; result: SideResult; heightPct: number };
export type WindowSummary = { label: string; n: number; hits: number; rate: number | null };

export function ribbonCells(values: number[], line: number, side?: Side): RibbonCell[];
export function windowSummaries(values: number[], line: number, side?: Side, windows?: number[]): WindowSummary[];
