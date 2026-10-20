export type BoardFilters = {
  stat: string;
  direction: string;
  location: string;
  sort: string;
  game: string;
  locksOnly: boolean;
  minStars: number;
};
export type BoardState = BoardFilters & { q: string };

export function parseFilters(search: string, defaults: BoardFilters): BoardState;
export function filtersToSearch(filters: BoardState, defaults: BoardFilters): string;
