export interface Prop {
  player:       string;
  team:         string;
  opponent:     string;
  stat:         string;
  stat_label:   string;
  line:         number | null;
  avg:          number | null;
  l5_avg:       number | null;
  direction:    string;
  hit_rate:     number;
  hits:         number | null;
  total:        number | null;
  ev:           number;
  is_lock:      boolean;
  is_combo:     boolean;
  game_matchup: string;
  blowout_risk: boolean;
  insight:      string;
  def_rank:     number | null;
  has_live_odds:boolean;
  live_line:    number | null;
}

export interface PropsResponse {
  count:        number;
  target_date:  string | null;
  game_matchups:string[];
  props:        Prop[];
}

export interface Game {
  matchup:    string;
  home_team:  string;
  away_team:  string;
  game_time:  string;
  spread:     number | null;
  total:      number | null;
  home_ml:    number | null;
  away_ml:    number | null;
}

export interface GamesResponse {
  target_date: string | null;
  games:       Game[];
}

export interface ChartRecord {
  date:     string;
  opponent: string;
  value:    number | null;
  hit:      boolean | null;
}

export interface PlayerChartData {
  player:  string;
  stat:    string;
  line:    number | null;
  avg:     number | null;
  l5_avg:  number | null;
  games:   ChartRecord[];
}

export interface PlayerStats {
  player:       string;
  team:         string;
  position:     string;
  season_avgs:  Record<string, number>;
  l5_avgs:      Record<string, number>;
  games_played: number;
}
