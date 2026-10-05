import type { PropsResponse, PlayerChartData } from './types';
export function playerKey(name: string): string;
export function filterProps(data: PropsResponse, params?: Record<string, string | number | boolean>): PropsResponse;
export function windowChart(data: PlayerChartData, count: number): PlayerChartData;
export function seriesChart(player: string, series: { games: Record<string, unknown>[] }, stat: string, count?: number): PlayerChartData;
