export type BoardFilters = {
  stat: string;
  direction: string;
  location: string;
  sort: string;
  game: string;
  locksOnly: boolean;
  minStars: number;
  window: string;
  /** Only picks with a positive expected value. */
  ev: boolean;
  /** Picked players, pipe-separated; empty means everyone. */
  players: string;
};
export type BoardState = BoardFilters & { q: string };

export function parseFilters(search: string, defaults: BoardFilters): BoardState;
export function filtersToSearch(filters: BoardState, defaults: BoardFilters): string;
export function isStaleGame(game: string, matchups: string[] | undefined): boolean;
