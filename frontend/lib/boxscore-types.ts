export interface BoxPlayer {
  id: string;
  name: string;
  image: string | null;
  short: string;
  position: string;
  jersey: string;
  starter: boolean;
  dnp: boolean;
  reason: string | null;
  stats: Record<string, string>;
}

export interface BoxTeam {
  side: "away" | "home";
  abbrev: string;
  name: string;
  logo: string | null;
  color: string | null;
  score: number | null;
  winner: boolean;
  record: string | null;
  linescore: number[];
  columns: string[];
  players: BoxPlayer[];
  totals: Record<string, string>;
  team_stats: { label: string; value: string }[];
  leaders: { label: string; title: string; name: string; value: number; id: string; image: string | null }[];
}

export interface BoxScore {
  game_id: string;
  state: "pre" | "in" | "post" | "unknown";
  status: string;
  period?: number | null;
  clock?: string | null;
  available: boolean;
  message?: string;
  teams: BoxTeam[];
}
