export type StatId = "goals" | "assists" | "shots" | "sot" | "fouls" | "cards";
export type WindowId = "L5" | "L10" | "L20" | "Season" | "Home" | "Away" | "VS";
export type Side = "over" | "under";

export interface StatSpec {
  label: string;
  lines: number[];
  defaultLine: number;
  market: "anytime_goal" | "assist" | "shots_on_target" | null;
}
export interface GameLike {
  opponent?: string;
  is_home?: boolean;
  goals?: number | null; assists?: number | null; shots?: number | null; sot?: number | null;
  fouls_committed?: number | null; yellow?: number | null; red?: number | null;
}
export interface HitRate { hits: number; pushes: number; total: number; rate: number | null }
export interface PropLike { market: string; line: number | null; prob: number }

export const STATS: Record<StatId, StatSpec>;
export const STAT_IDS: StatId[];
export const WINDOWS: WindowId[];

export function statValue(game: GameLike, id: StatId): number | null;
export function filterGames<T extends GameLike>(games: T[], win: WindowId, opponent?: string | null): T[];
export function valuesOf(games: GameLike[], id: StatId): number[];
export function hitRate(values: number[], line: number, side?: Side): HitRate;
export function average(values: number[]): number | null;
export function median(values: number[]): number | null;
export function fmtNum(v: number | null | undefined, dp?: number): string;
export function hitHeadline(hr: HitRate, line: number, side?: Side): string;
export function fairOdds(prob: number): { decimal: number; american: string } | null;
export function modelProbFor<P extends PropLike>(
  props: P[] | null | undefined, id: StatId, line: number, side?: Side,
): { prob: number; prop: P } | null;
export function nearestLine(id: StatId, line: number): number;
