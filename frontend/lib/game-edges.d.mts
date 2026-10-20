import type { Game, GamePrediction } from './types';

export type GameEdge = {
  game: Game;
  side: 'home' | 'away';
  team: string;
  price: number;
  modelProb: number;
  ev: number;
  marketProb: number | null;
};

export const EDGE_MIN: number;
export function evPerUnit(prob: number, price: number | null | undefined): number | null;
export function gameEdges(games: Game[], predictions: Pick<GamePrediction, 'matchup' | 'home_win_probability'>[]): GameEdge[];
export function bettableEdges<T extends { ev: number }>(rows: T[]): T[];
