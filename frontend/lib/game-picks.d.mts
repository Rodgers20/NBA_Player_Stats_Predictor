export interface SpreadLean { side: "home" | "away"; p: number }
export interface TotalLean { side: "over" | "under"; p: number }
export interface ModelLeans { spread: SpreadLean | null; total: TotalLean | null }

export function normalCdf(z: number): number;
export function sigmaFromInterval(interval: readonly number[] | null | undefined, kind: "margin" | "total"): number;
export function modelLeans(
  game: { spread?: number | null; total?: number | null } | null | undefined,
  prediction: { spread?: number | null; total?: number | null; prediction_intervals?: { margin?: readonly number[]; total?: readonly number[] } } | null | undefined,
): ModelLeans;
