import type { Game, Prop } from './types';

export function isStarted(game: Pick<Game, 'status_state' | 'start_time'> | null | undefined, now?: number): boolean;
export function popularBoard(
  props: Prop[],
  games: Game[] | undefined,
  options?: { by?: 'hit' | 'edge'; now?: number; limit?: number },
): Prop[];
