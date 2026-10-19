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
  pick_type?: 'pick' | 'lean' | 'research';
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
  status_state?: "pre" | "in" | "post" | null;
  start_time?: string | null;
  home_logo?: string | null;
  away_logo?: string | null;
  home_color?: string | null;
  away_color?: string | null;
  venue?: string | null;
  broadcast?: string | null;
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
  winner_probability?: number | null;
  home_win_probability?: number | null;
  model_metadata?: {name?: string; validated?: boolean; version?: string; trained_through?: string;
    context_warning?: string | null; preseason_validated?: boolean;
    injury_scenario_applied?: boolean; injury_adjustment_validated?: boolean};
  data_quality?: {as_of?: string; injury_player_history_stale?: boolean;
    teams?: Record<string, {latest_game_date?: string; history_age_days?: number | null; history_older_than_14_days?: boolean}>;
    limitations?: string[]};
  prediction_intervals?: {coverage?: number; margin?: [number,number]; total?: [number,number]};
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

export interface HighlightClip {
  id: string;
  title: string;
  description?: string;
  thumbnail?: string | null;
  url: string;
  video_url?: string | null;
  duration?: number | null;
  source?: string;
  external_url?: string | null;
}
export interface HighlightGame {
  game_id: string;
  matchup: string;
  date: string;
  status?: string;
  clips: HighlightClip[];
}
export interface HighlightsResponse {
  league: string;
  checked_at: string;
  games: HighlightGame[];
  message?: string | null;
}
export interface ScheduleIndex {
  current_date: string;
  target_date: string;
  dates: string[];
}

export interface CalibrationBin { lower: number; count: number; predicted: number; actual: number }
export interface EvaluationMetrics {
  margin: { mae: number; rmse: number };
  total: { mae: number; rmse: number };
  winner: { accuracy: number; log_loss: number; brier: number; calibration_bins: CalibrationBin[] };
  interval_coverage_80?: {margin: number; total: number};
}
export interface ModelEvaluation {
  version: string; created_at: string; league: string; promoted: boolean;
  games: number; development_games: number; holdout_games: number;
  development_through: string; holdout_from: string; holdout_through: string;
  selected: {margin: string; total: string};
  baseline_names?: {margin: string; total: string};
  holdout: EvaluationMetrics; baseline_holdout: EvaluationMetrics;
  candidates: Record<string, {margin: {mae: number; rmse: number}; total: {mae: number; rmse: number}}>;
  features: string[]; limits: string[];
}
export interface ModelRecord {
  games: number; winner: {n: number; correct: number; rate: number | null};
  margin_mae: number | null; total_mae: number | null;
}
export interface SavedGameForecast {
  date: string; season: number; game_id: string;
  home_team: string; away_team: string; home_score: number; away_score: number;
  predicted_home_score: number | null; predicted_away_score: number | null;
  predicted_winner: string | null; winner_correct: boolean | null;
  margin_error: number | null; total_error: number | null;
  generated_at: string; tipoff: string;
}
export interface PerformanceResponse {
  league: string; checked_at: string; evaluation: ModelEvaluation | null;
  live: {current_season: number; excluded_unverified: number; message: string;
    seasons: {season: number; record: ModelRecord; games: SavedGameForecast[];
      by_month: (ModelRecord & {month: string})[]}[]};
  prospective?: {model_version: string | null; saved: number; pending: number; record?: ModelRecord; message: string};
  market_record: {available: boolean; message: string};
}
