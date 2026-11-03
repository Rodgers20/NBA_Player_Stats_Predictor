export type MarketId = "anytime_goal" | "shots_on_target" | "assist";
export type Confidence = "high" | "medium" | "low";

export interface PropLike {
  player: string; team: string; is_home: boolean; market: string; line: number | null;
  prob: number; expected?: number | null; per_game?: number | null; confidence?: string;
}
export interface FixtureLike<P extends PropLike = PropLike> {
  fixture_key: string; home_team: string; away_team: string; props: P[];
}
export interface Filters {
  market: "all" | MarketId;
  line: 0.5 | 1.5 | null;
  venue: "all" | "home" | "away";
  conf: "all" | "medium" | "high";
  minProb: number;
  teams: string[];
  fixture: string;
  sort: "prob" | "expected" | "avg";
}

export const MARKETS: MarketId[];
export const MARKET_LABEL: Record<MarketId, string>;
export const DEFAULT_FILTERS: Readonly<Filters>;
export const PAGE_SIZE: number;

export function parseFilters(
  input: URLSearchParams | Record<string, string | string[] | undefined> | null | undefined,
): Filters;
export function toQuery(filters: Filters): string;
export function applyFilters<F extends FixtureLike>(
  fixtures: F[], filters: Filters, opts?: { skipMarket?: boolean },
): F[];
export function marketCounts(
  fixtures: FixtureLike[], filters: Filters,
): { all: number; anytime_goal: number; shots_on_target: number; assist: number };
export function takeRows<F extends FixtureLike>(fixtures: F[], limit: number): F[];
export function totalRows(fixtures: FixtureLike[]): number;
export function teamsOf(fixtures: FixtureLike[]): string[];
export function probTone(prob: number): "hot" | "warm" | "cool";
