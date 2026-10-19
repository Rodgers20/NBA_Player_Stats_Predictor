export interface TeamGame {
  id: string;
  date: string;
  opponent: string;
  is_home: boolean;
  points_for?: number | null;
  points_against?: number | null;
  result?: string | null;
}

export interface BasketballTeam {
  abbrev: string;
  name: string;
  nickname: string;
  city: string;
  conference: string;
  division: string;
  color: string;
  logo: string;
  seasons: (number | string)[];
  season: number | string | null;
  wins: number | null;
  losses: number | null;
  points_per_game: number | null;
  points_allowed_per_game: number | null;
  rank?: number | null;
  history_through: string | null;
  stats_available: boolean;
  games: TeamGame[];
  upcoming: TeamGame[];
}

export interface TeamsResponse {
  league: string;
  season?: number | string;
  seasons?: (number | string)[];
  available_seasons?: (number | string)[];
  teams: BasketballTeam[];
  message?: string | null;
}
