import type { Side, SideResult } from './hit-rate.mjs';

export type RibbonCell = { value: number; result: SideResult; heightPct: number };
export type RibbonSummary = { hits: number; decided: number; pushes: number };

export function ribbonSummary(values: number[], line: number, side?: Side): RibbonSummary;
export function ribbonCells(values: number[], line: number, side?: Side): RibbonCell[];
