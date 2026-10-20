import type { Side, SideResult } from './hit-rate.mjs';

export type RibbonCell = { value: number; result: SideResult; heightPct: number };
export type WindowSummary = { label: string; n: number; hits: number; decided: number; rate: number | null };
export type RibbonSummary = { hits: number; decided: number; pushes: number };

export function ribbonSummary(values: number[], line: number, side?: Side): RibbonSummary;

export function ribbonCells(values: number[], line: number, side?: Side): RibbonCell[];
export function windowSummaries(values: number[], line: number, side?: Side, windows?: number[]): WindowSummary[];
