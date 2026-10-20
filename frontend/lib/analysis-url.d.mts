import type { WindowKey } from './analysis-windows.mjs';

export type AnalysisView = { stat: string; period: WindowKey; line: number | null };

export function parseAnalysisView(search: string, defaults: AnalysisView): AnalysisView;
export function analysisViewToSearch(view: AnalysisView, defaults: AnalysisView): string;
