export interface Prop {
  player:       string;
  headshot_url?: string | null;
  team:         string;
  opponent:     string;
  stat:         string;
  stat_label:   string;
  line:         number | null;
  avg:          number | null;
  l5_avg:       number | null;
  direction:    string;
  hit_rate:     number | null;
  model_projection?: number | null;
  hits:         number | null;
  total:        number | null;
  ev:           number | null;
  recommendation_eligible?: boolean;
  quality_reason?: string;
  probability_source?: string;
  model_prob?: number | null;
  price?: number | null;
  is_lock:      boolean;
  is_combo:     boolean;
  game_matchup: string;
  blowout_risk: boolean;
  insight:      string;
  def_rank:     number | null;
  has_live_odds:boolean;
  live_line:    number | null;
  is_home_today?: boolean | null;
  l5_values?: (number | null)[];
  chart_windows?: Record<string, { values: (number | null)[]; labels: string[] }>;
}

export interface PropsResponse {
  status?: string;
  message?: string | null;
  count:        number;
  target_date:  string | null;
  game_matchups:string[];
  stat_counts?: Record<string, number>;
  props:        Prop[];
}

export interface Game {
  matchup:    string;
  home_team:  string;
  away_team:  string;
  game_time:  string;
  game_id?: string | null;
  status_text?: string | null;
  home_name?: string | null;
  away_name?: string | null;
  home_wins?: number | null;
  home_losses?: number | null;
  away_wins?: number | null;
  away_losses?: number | null;
  home_score?: number | null;
  away_score?: number | null;
  home_injuries?: GameInjury[];
  away_injuries?: GameInjury[];
  spread:     number | null;
  total:      number | null;
  home_ml:    number | null;
  away_ml:    number | null;
  odds_source?: string | null;
  odds_provider?: string | null;
  odds_updated_at?: string | null;
}

export interface GameInjury {
  name: string;
  status: string;
  reason: string;
}

export interface GamesResponse {
  target_date: string | null;
  games:       Game[];
}

export interface GamePrediction {
  matchup: string;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  spread: number | null;
  total: number | null;
  confidence: string;
  predicted_home_score?: number | null;
  predicted_away_score?: number | null;
  intel?: string[];
  winner_reason?: string | null;
  spread_reason?: string | null;
  total_reason?: string | null;
  market_spread?: number | null;
  market_total?: number | null;
  spread_pick?: "HOME" | "AWAY" | null;
  spread_team?: string | null;
  spread_confidence?: string | null;
  total_pick?: "OVER" | "UNDER" | null;
  total_confidence?: string | null;
}

export interface GamePredictionsResponse {
  target_date: string | null;
  predictions: GamePrediction[];
  errors: { matchup: string; message: string }[];
  message?: string | null;
}

export interface ChartRecord {
  date:     string;
  opponent: string;
  value:    number | null;
  hit:      boolean | null;
  season?:  string | null;
  is_home?: boolean | null;
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
  headshot_url?: string | null;
  fg_pct?: number | null;
  injury_status?: string | null;
  injury_reason?: string | null;
  history_through?: string | null;
  history_age_days?: number | null;
  projections?: Record<string, number>;
  projection_context?: { opponent: string; is_home: boolean; game_date: string } | null;
  projection_message?: string | null;
  player:       string;
  team:         string;
  position:     string;
  season_avgs:  Record<string, number>;
  l5_avgs:      Record<string, number>;
  games_played: number;
}
