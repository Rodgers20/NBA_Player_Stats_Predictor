export interface StandingRow {
  abbrev: string;
  name: string;
  nickname: string;
  logo: string | null;
  conference: string;
  wins: number;
  losses: number;
  win_pct: number | null;
  games_behind: string;
  points_for: number | null;
  points_against: number | null;
  point_diff: number | null;
  home: string | null;
  road: string | null;
  conf: string | null;
  div: string | null;
  last10: string | null;
  streak: string | null;
  clincher: string | null;
  seed: number;
  games: number;
}

export interface PowerRow {
  rank: number;
  abbrev: string;
  name: string;
  nickname: string;
  logo: string | null;
  wins: number;
  losses: number;
  power: number;
  point_diff: number;
  last10: string | null;
  streak: string | null;
  movement: number;
}

export interface PicturePick {
  abbrev: string;
  nickname: string;
  logo: string | null;
  seed: number;
  wins: number;
  losses: number;
  clincher: string | null;
  zone: "direct" | "play-in" | "out";
  games_back_of_direct: number | null;
}

export interface StandingsResponse {
  league: string;
  season: number;
  fallback: boolean;
  message: string | null;
  conferences: { name: string; teams: StandingRow[] }[];
  power_rankings: PowerRow[];
  playoff_picture: { direct: number; play_in: number; conferences: { name: string; teams: PicturePick[] }[] } | null;
}
