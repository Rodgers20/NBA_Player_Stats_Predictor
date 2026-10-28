export function boxScoreIndex(teamsResponse: { teams?: { abbrev: string; games?: { id: string; date: string }[] }[] } | null | undefined): Map<string, string>;
export function boxScoreHref(index: Map<string, string>, game: { date: string; opponent: string }, league: string): string | null;
