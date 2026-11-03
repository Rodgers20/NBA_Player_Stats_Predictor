export interface Envelope<T> {
  success: boolean;
  data: T | null;
  error: string | null;
  meta: Record<string, unknown>;
}

export interface Match {
  match_id: number;
  competition: string;
  country: string;
  season: string;
  match_date: string;
  kick_off: string | null;
  stadium: string | null;
  referee: string | null;
  competition_stage: string;
  match_week: number | null;
  home_team_id: number;
  home_team: string;
  away_team_id: number;
  away_team: string;
  home_score: number;
  away_score: number;
  result: "H" | "D" | "A";
  home_manager: string | null;
  away_manager: string | null;
}

export interface Competition {
  competition_id: number;
  competition: string;
  country: string;
  season_id: number;
  season: string;
  match_count: number;
  first_match: string;
  last_match: string;
}

export interface StatRow {
  label: string;
  home: number;
  away: number;
  unit: string;
}

export type EventKind =
  | "goal" | "own_goal" | "shootout_goal"
  | "yellow_card" | "red_card" | "second_yellow"
  | "substitution";

export interface MatchEvent {
  minute: number;
  second: number;
  period: number;
  team: string;
  player: string | null;
  kind: EventKind;
  replacement: string | null;
  xg: number | null;
}

export interface MomentumPoint {
  minute: number;
  home: number;
  away: number;
  net: number;
}

export interface Pass {
  event_id: string;
  minute: number;
  second: number;
  period: number;
  team: string;
  player: string;
  position: string | null;
  x: number; y: number;
  end_x: number; end_y: number;
  height: "Ground Pass" | "Low Pass" | "High Pass" | null;
  length: number | null;
  angle: number | null;
  recipient: string | null;
  body_part: string | null;
  pass_type: string | null;
  completed: boolean;
  cross: boolean;
  switch: boolean;
  goal_assist: boolean;
  shot_assist: boolean;
}

export interface Shot {
  event_id: string;
  minute: number; second: number; period: number;
  team: string; player: string; position: string | null;
  x: number; y: number;
  end_x: number | null; end_y: number | null; end_z: number | null;
  xg: number | null;
  outcome: string;
  body_part: string | null;
  technique: string | null;
  shot_type: string | null;
  first_time: boolean;
  distance: number;
  players_in_frame: number;
}

export interface HeatmapCell { bx: number; by: number; n: number; intensity: number; }
export interface Heatmap { bins_x: number; bins_y: number; peak: number; cells: HeatmapCell[]; }

export interface FramePlayer {
  x: number; y: number;
  teammate: boolean; actor: boolean; keeper: boolean; off_pitch: boolean;
}
export interface FreezeFrame {
  event_id: string;
  minute: number; second: number; period: number;
  type: string; team: string; player: string | null;
  players: FramePlayer[];
}

export interface TopPlayer {
  player: string; team: string; position: string | null;
  actions: number; passes: number; passes_completed: number;
  shots: number; xg: number; goals: number; assists: number;
  dribbles: number; defensive: number; rating: number;
}

export interface StandingRow {
  position: number; team: string;
  played: number; won: number; drawn: number; lost: number;
  goals_for: number; goals_against: number; goal_difference: number;
  points: number; form: ("W" | "D" | "L")[];
  /** The club's next scheduled fixture in this competition, if any. */
  next?: { opponent: string; at_home: boolean; date: string; time: string | null } | null;
  /** e.g. "UEFA Champions League", "Relegation" — null outside any band. */
  qualification?: string | null;
  /** Semantic tone for the band's colour bar, not a brand colour. */
  qualification_tone?: "primary" | "secondary" | "tertiary" | "danger" | "" | null;
}

export interface ScorerRow { player: string; team: string; goals: number; xg: number; }
export interface AssistRow { player: string; team: string; assists: number; }
export interface ContributionRow {
  player: string; team: string; goals: number; assists: number; contributions: number;
}

export interface PassNetworkNode {
  player: string; position: string | null; x: number; y: number; touches: number;
}
export interface PassNetworkEdge { source: string; target: string; passes: number; }
export interface PassNetwork { nodes: PassNetworkNode[]; edges: PassNetworkEdge[]; }

export interface LiveMessage {
  seq: number;
  kind: "clock" | "event" | "stats" | "momentum" | "state";
  match_id: number;
  payload: Record<string, unknown>;
}

export interface PredictCompetition {
  competition_code: string;
  competition: string;
  country: string;
  matches: number;
  last_match: string;
  latest_season: string;
}

export interface Prediction {
  competition_code: string;
  home_team: string;
  away_team: string;
  basis: string;
  win: { home: number; draw: number; away: number };
  expected_goals: { home: number; away: number; total: number };
  total_goals: {
    over_1_5: number; over_2_5: number; over_3_5: number; btts: number;
    distribution: { goals: number; probability: number }[];
  };
  most_likely_score: { home: number; away: number; probability: number };
  result_pick?: "home" | "draw" | "away";
  top_scores?: { home: number; away: number; probability: number }[];
  context?: {
    forecast_date: string; model_trained_through: string | null;
    home_last_result: string | null; away_last_result: string | null;
    home_prior_matches: number; away_prior_matches: number;
    home_season_matches: number | null; away_season_matches: number | null;
  };
  corners?: CountMarket;
  sot?: CountMarket;
  calibration?: Partial<Record<"goals" | "corners" | "sot", RateCorrection | null>>;
  reasons?: { market: string; label: string; text: string }[];
  limitations?: string[];
  fixture?: { referee?: string | null; odds?: { home?: number | null; draw?: number | null; away?: number | null } };
}

/** Corners or shots on target: per-team rates plus the match total. */
export type CountMarket =
  | { available: false; note: string }
  | {
      available: true;
      home: CountSide;
      away: CountSide;
      total: { expected: number; line: number; over: number; under: number };
      both_teams: { threshold: number; probability: number } | null;
      calibration: RateCorrection | null;
    };

export interface CountSide {
  team: string;
  expected: number;
  line: number;
  at_least: number;
  distribution: { count: number; probability: number }[];
}

/** How far the model's own settled forecasts say a rate should move. */
export interface RateCorrection {
  market: string;
  competition_code: string;
  n: number;
  predicted_mean: number;
  actual_mean: number;
  factor: number;
  clipped: boolean;
  source: string;
  through: string | null;
}

/** Every fixture in one round, each with a compact forecast. */
export interface MatchdayBoard {
  competition_code: string;
  competition: string;
  matchday: number | null;
  matchdays: number[];
  fixtures: MatchdayFixture[];
  note: string | null;
}

export interface MatchdayFixture {
  provider_id: number | null;
  kickoff_utc: string | null;
  date: string;
  status: string;
  home_team: string | null;
  away_team: string | null;
  home_name: string | null;
  away_name: string | null;
  home_goals: number | null;
  away_goals: number | null;
  prediction: MatchdayForecast | null;
  /** Why there is no forecast: "played", "unknown club", "no history". */
  unavailable: string | null;
}

export interface MatchdayForecast {
  result_pick: "home" | "draw" | "away";
  win: { home: number; draw: number; away: number };
  expected_goals: { home: number; away: number; total: number };
  over_2_5: number;
  btts: number;
  most_likely_score: { home: number; away: number; probability: number };
  corners: { expected: number; line: number; over: number } | null;
  sot: { expected: number; line: number; over: number } | null;
}

export interface PredictMetrics {
  competition_metrics?: { competition_code: string; n: number; accuracy: number | null;
    logloss: number | null; brier_score: number | null; from: string | null; through: string | null };
  methodology_note?: string; last_training_result?: string; baseline_logloss?: number;
  brier_score?: number; market_comparison_n?: number; served_logloss_odds_subset?: number;
  calibration?: {from: number; n: number; confidence: number; accuracy: number}[];
  n_train: number; n_test: number; test_from: string;
  feature_groups?: string[];
  n_features?: number;
  dixon_coles_rho?: number;
  result_accuracy: number; result_logloss: number;
  market_accuracy: number; market_logloss: number;
  blend_weight: number; blend_accuracy: number; blend_logloss: number;
  formonly_accuracy: number; formonly_logloss: number;
  home_goals_mae: number; away_goals_mae: number; total_goals_mae: number;
  over25_accuracy: number; over25_logloss: number;
  verdict: string;
}

export interface BacktestRow {
  date: string; home_team: string; away_team: string;
  actual_home: number; actual_away: number;
  actual_result: string; predicted: string; correct: boolean;
  win: { home: number; draw: number; away: number };
  expected_total: number; actual_total: number;
  over_2_5: number; actual_over_2_5: boolean;
}

// ---------------------------------------------------------------- leagues ---

export interface League {
  code: string;
  name: string;
  country: string;
  kind: "league" | "cup";
  total_matches: number;
  current_season_year: number;
  current_season_matches: number;
  displayed_season_year: number | null;
  displayed_season: string | null;
  seasons: number[];
  last_match: string | null;
  /** Official competition badge, served by football-data.org's crest CDN. */
  emblem?: string | null;
}

export interface LeagueSeason {
  season_year: number;
  season: string;
  matches: number;
  last_match: string;
}

export interface StandingRowLeague {
  position: number;
  team: string;
  played: number; won: number; drawn: number; lost: number;
  goals_for: number; goals_against: number;
  goal_difference: number; points: number;
  form: ("W" | "D" | "L")[];
}

export interface LeagueStandings {
  competition_code: string;
  competition: string;
  season_year: number | null;
  season: string | null;
  is_fallback: boolean;
  matches_played: number;
  rows: StandingRowLeague[];
  note: string | null;
}

export interface LeagueMatch {
  date: string; time: string | null;
  home_team: string; away_team: string;
  home_goals: number; away_goals: number;
  result: "H" | "D" | "A";
  total_goals: number;
  odds_home: number | null; odds_draw: number | null; odds_away: number | null;
  ht_home_goals: number | null; ht_away_goals: number | null;
  home_shots: number | null; away_shots: number | null;
  home_sot: number | null; away_sot: number | null;
  home_corners: number | null; away_corners: number | null;
  home_fouls: number | null; away_fouls: number | null;
  home_yellow: number | null; away_yellow: number | null;
  home_red: number | null; away_red: number | null;
  home_xg: number | null; away_xg: number | null;
}

export interface LeagueMatches {
  competition_code: string;
  competition: string;
  season_year: number | null;
  season: string | null;
  is_fallback: boolean;
  rows: LeagueMatch[];
}

export interface Fixture {
  competition_code: string;
  date: string;
  time: string | null;
  kickoff_utc: string | null;
  home_team: string;
  away_team: string;
  referee: string | null;
  odds_home: number | null; odds_draw: number | null; odds_away: number | null;
  live_status: "scheduled" | "live";
  fixture_key: string;
}

export interface LiveMatch {
  competition_code: string;
  date: string | null; time: string | null; kickoff_utc: string | null;
  home_team: string; away_team: string;
  fixture_key: string;
  minutes_elapsed: number | null;
  /** Running score, null when the provider has not published one yet. */
  home_goals?: number | null;
  away_goals?: number | null;
  /** The provider's own word for the state: IN_PLAY, PAUSED. */
  provider_status?: string | null;
}

export interface LiveCounts {
  total: number;
  by_competition: Record<string, number>;
  matches: LiveMatch[];
}

export interface PlayerLeaders {
  competition_code: string;
  competition: string;
  season_year: number;
  article?: string;
  scorers: {
    player: string; team: string | null; goals: number;
    assists?: number | null; played?: number | null;
    crest?: string | null; player_id?: string;
  }[];
  /**
   * Per TEAM, computed from results — no provider publishes a player board.
   * `keeper` is the club's number 1 where the squad settles it: the shirt, not
   * the team sheet, and null when it does not.
   */
  clean_sheets: {
    team?: string; player?: string; clean_sheets: number;
    played?: number; goals_conceded?: number; position?: number;
    keeper?: {
      name: string; player_id: number | null; shirt: number | null;
      nationality: string | null; club: string | null; crest: string | null;
    } | null;
  }[];
  clean_sheets_kind?: "team" | "player";
  assists: {
    player: string; team: string | null; assists: number;
    goals?: number; played?: number | null;
    crest?: string | null; player_id?: string;
  }[];
  available: boolean;
  is_fallback: boolean;
  note: string | null;
  assists_note: string | null;
  clean_sheets_note?: string | null;
}

export interface TeamColors {
  primary: string;
  on_primary: string;
  accent: string;
}

export interface PlayerSearchHit {
  player_id: string;
  name: string;
  /** Always the club, never the national team. */
  club: string | null;
  crest: string | null;
  colors: TeamColors;
  nationality: string | null;
  age: number | null;
  position: string | null;
  /** Every competition this one person appears in. */
  competitions: string[];
  has_full_stats: boolean;
  goals: number | null;
}

export interface PlayerShot {
  match_id: number; minute: number; x: number; y: number;
  xg: number | null; outcome: string | null; body_part: string | null;
  technique: string | null; situation: string | null; is_goal: boolean;
  home_team: string; away_team: string; match_date: string;
}

export interface PlayerMatch {
  match_id: number;
  match_date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  stage: string | null;
  team: string | null;
  goals: number; assists: number; shots: number;
  xg: number; chances: number;
  yellows: number; reds: number; touches: number;
}

export interface RadarMetric {
  key: string; label: string; value: number; percentile: number;
}

/** One competition's worth of a player's data. */
export interface PlayerSpell {
  competition: string;
  season?: string | null;
  competition_code: string;
  emblem: string | null;
  source: string;
  detail: "full" | "basic";
  /** Which panels this spell can fill. */
  sections: string[];
  represented?: string | null;
  position?: string | null;
  stats: Record<string, number | null>;
  per90?: Record<string, number | null>;
  shots_detail?: PlayerShot[];
  heatmap?: { gx: number; gy: number; touches: number }[];
  radar?: { role: string | null; metrics: RadarMetric[]; peers: number };
  matches_detail?: PlayerMatch[];
  limitations: string[];
}

export interface PositionBadge {
  code: string;
  label: string;
  /** "specific" = a named role, "coarse" = only a band (Offence/Midfield/…). */
  tier: "specific" | "coarse";
  /** Pitch coordinates, 0-1, attacking upward. */
  x: number;
  y: number;
}

export interface PositionDetail {
  primary: PositionBadge | null;
  /** Other roles used in, ranked; `share` is the fraction of primary appearances. */
  secondary: (PositionBadge & { matches: number; share: number })[];
  /** Which competition the secondary roles describe. */
  source: string | null;
}

export interface PlayerProfile {
  player_id: string;
  name: string;
  club: string | null;
  crest: string | null;
  colors: TeamColors;
  nationality: string | null;
  age: number | null;
  date_of_birth?: string | null;
  shirt_number?: number | null;
  position: string | null;
  position_detail?: PositionDetail | null;
  spells: PlayerSpell[];
  default_spell: number;
  sources: string[];
}

export interface NewsItem {
  title: string;
  link: string;
  source: string;
  source_key: string;
  published: string | null;
  summary: string | null;
  club?: string;
}

export interface CompetitionNews {
  competition_code: string;
  matched: NewsItem[];
  recent: NewsItem[];
  note: string;
  sources: string[];
}

export interface TeamOfTheWeekSlot {
  slot: string;
  x: number;
  y: number;
  player: {
    player_id: number | null;
    name: string;
    /** Surname only, for the pitch graphic. */
    short_name: string;
    team: string | null;
    /** The club whose crest sits on the portrait. */
    club: string | null;
    position: string | null;
    rating: number;
    minutes: number | null;
  } | null;
}

export interface TeamOfTheWeekData {
  competition_code: string;
  competition: string;
  match_week: number | null;
  rounds: number[];
  players: TeamOfTheWeekSlot[];
  note: string;
}

export interface MetricSpec {
  key: string;
  label: string;
  unit: string;
  precision: number;
  higher_is_better?: boolean;
}

export interface LeaderBoardRow {
  team?: string | null;
  player?: string;
  player_id?: string;
  crest?: string | null;
  value: number;
  played?: number;
  assists_unpublished?: boolean;
}

export interface LeaderBoard {
  competition_code: string;
  competition: string;
  metric: string;
  label: string;
  unit: string;
  precision: number;
  higher_is_better?: boolean;
  rows: LeaderBoardRow[];
  note: string | null;
}

export interface SeasonHonour {
  season: string;
  season_year: number | null;
  winner: string | null;
  runner_up: string | null;
  /** How it was resolved: a cup final, a league table, or why neither. */
  decided_by: "final" | "table" | "in_progress" | "unavailable" | "undecided" | null;
  note: string | null;
  matches: number;
}

export interface SquadMember {
  player_id: number;
  name: string;
  position: string | null;
  date_of_birth: string | null;
  nationality: string | null;
  shirt?: number | null;
  crest?: string | null;
}

export interface TeamFormMatch {
  date: string;
  competition_code: string;
  competition: string;
  season: string;
  opponent: string;
  at_home: boolean;
  goals_for: number;
  goals_against: number;
  outcome: "W" | "D" | "L";
}

export interface FixtureDifficulty {
  next: {
    opponent: string; at_home: boolean; date: string; time: string | null;
    kickoff_utc?: string | null; competition_code: string;
    difficulty: number; label: string; tone: string;
  }[];
  summary: string | null;
  note: string;
}

export interface TeamSeason {
  season: string; season_year: number; position: number;
  played: number; won: number; drawn: number; lost: number;
  goals_for: number; goals_against: number; points: number;
}

export interface TeamProfile {
  team: string;
  lake_name: string;
  crest: string | null;
  country: string | null;
  country_flag: string | null;
  founded: number | null;
  venue: string | null;
  website: string | null;
  club_colors: string | null;
  colors: TeamColors;
  coach: {
    name: string | null; nationality: string | null; date_of_birth: string | null;
    season?: string | null; played?: number; points?: number;
    points_per_game?: number | null; win_rate?: number | null; note?: string;
  } | null;
  competition_code: string;
  competition: string;
  competitions: string[];
  standing: StandingRow | null;
  form: TeamFormMatch[];
  next_match: Fixture | null;
  fixtures: Fixture[];
  difficulty: FixtureDifficulty;
  squad: SquadMember[];
  players: {
    scorers: { player: string; team: string | null; goals: number; assists: number | null; player_id?: string }[];
    assists: { player: string; team: string | null; goals: number; assists: number | null; player_id?: string }[];
  };
  season_stats: TeamSeasonStats;
  season_xi: SeasonXi;
  history: TeamSeason[];
  limitations: string[];
}

/** One club's season read down a column, each metric ranked in its division. */
export interface TeamSeasonStats {
  competition_code: string;
  competition: string;
  team: string;
  played: number;
  metrics: TeamSeasonMetric[];
  note: string | null;
}

/** Contributors placed by line; a slot with no player stays empty. */
export interface SeasonXi {
  ratings_note?: string;
  ratings_source_url?: string;
  stats_competition?: string;
  stats_season?: string;
  stats_note?: string;
  team: string;
  competition_code: string;
  competition: string;
  slots: SeasonXiSlot[];
  filled: number;
  note: string | null;
  match_date?: string;
  match_label?: string;
  formation?: string;
  source_url?: string;
}

export interface SeasonXiSlot {
  slot: string;
  line: string;
  abbreviation: string;
  x: number;
  y: number;
  player: {
    player_id: number | null;
    name: string;
    short_name: string;
    shirt: number | string | null;
    goals: number | null;
    assists: number | null;
    played: number;
    rating: number | null;
    rated_matches?: number | null;
    rating_source?: string;
    rating_source_url?: string;
    stats_scope?: string;
  } | null;
}

export interface TeamSeasonMetric {
  key: string;
  label: string;
  unit: string;
  value: number;
  precision: number;
  higher_is_better: boolean;
  rank: number;
  of: number;
}

export interface ClubDirectoryEntry {
  name: string;
  short_name: string | null;
  tla: string | null;
  crest: string | null;
  team_id: number;
  competitions: string[];
  /** Every spelling the backend's club reconciliation resolves to this entry. */
  aliases?: string[];
}

export interface H2HMatch {
  date: string; season: string;
  home_team: string; away_team: string;
  home_goals: number; away_goals: number;
  result: "H" | "D" | "A";
}

export interface FormMatch {
  date: string; opponent: string; at_home: boolean;
  goals_for: number; goals_against: number;
  outcome: "W" | "D" | "L";
}

export interface PlayerDetails {
  provider_id: number | null;
  source: string;
  height: string | null;
  weight: string | null;
  date_of_birth: string | null;
  nationality: string | null;
  preferred_foot: string | null;
  contract_end: string | null;
  market_value: number | null;
  value_history: { date: string; value: number }[];
  career: { team_id: number; team: string; crest: string | null; seasons: number[] }[];
  fetched_at: number | null;
  status: string;
}
