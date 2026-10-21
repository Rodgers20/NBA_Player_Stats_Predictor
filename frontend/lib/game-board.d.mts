import type { Game, GamePrediction } from './types';
export function gameState(game: Game): 'pre' | 'in' | 'post';
export function scoreboard(game: Game, prediction?: GamePrediction): {away: number | null; home: number | null; label: string};
export function marketProbability(game: Game): number | null;
export function gamePicks(game: Game, prediction?: GamePrediction): {winner: string; spread: string; total: string};
export function shiftDate(day: string, offset: number): string;
