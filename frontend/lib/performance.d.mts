import type {CalibrationBin, PerformanceResponse} from './types';
export function metric(value?: number | null, digits?: number): string;
export function percentage(value?: number | null, digits?: number): string;
export function seasonLabel(year: number, league: string): string;
export function calibrationPoints(bins?: CalibrationBin[]): CalibrationBin[];
export function selectedSeason(live: PerformanceResponse['live'], requested?: number): PerformanceResponse['live']['seasons'][number] | undefined;
