/** Shapes of the static JSON in public/data/nhl, as documented in docs/nhl-contract.md. */
export type NhlSide = "home" | "away";
export type NhlConfidence = "low" | "medium" | "high";
export type NhlMarketKey = "goal" | "sog" | "assists" | "points";

export interface NhlTeamRef { id: number; abbr: string; name: string; short: string; logo: string; logo_dark: string }
export interface NhlGameTeam extends NhlTeamRef { record: string | null; score: number | null }
export interface NhlPick {
  source: "live" | "pre"; winner: NhlSide; winner_abbr: string; p_home: number; p_away: number; confidence: NhlConfidence;
  proj_total: number; proj_home: number; proj_away: number; correct: boolean | null;
}
export interface NhlGoalie { id: number; name: string; sv_pct: number | null; projected: boolean }
export interface NhlGame {
  id: number; date: string; start: string; status: "scheduled" | "live" | "final"; game_type: string; venue: string | null;
  home: NhlGameTeam; away: NhlGameTeam; period_type: "REG" | "OT" | "SO" | null; pick: NhlPick | null;
  goalies: { home: NhlGoalie | null; away: NhlGoalie | null } | null; props_count: number;
}
export interface NhlScheduleIndex { generated_at: string; days: { date: string; games: number; final: number; game_ids: number[] }[] }
export interface NhlSchedule { date: string; generated_at: string; games: NhlGame[] }

export interface NhlFormSide { last10: string[]; record_l10: string; gf: number | null; ga: number | null; sf: number | null; sa: number | null; win_pct: number | null }
export interface NhlFactor { key: string; label: string; home: number | string | null; away: number | string | null; edge: NhlSide | "even"; pct: number }
export interface NhlTopProp {
  player_id: number; name: string; team: string; pos: string; headshot: string; market: NhlMarketKey;
  line: number; side: "over" | "under"; p: number; fair_odds: number; projection: number; confidence: NhlConfidence;
}
export interface NhlBoxSkater { player_id: number; name: string; pos: string; toi: string; goals: number; assists: number; points: number; sog: number; pp_goals: number }
export interface NhlBoxGoalie { player_id: number; name: string; saves: number; shots_against: number; sv_pct: number | null; toi: string; decision: string | null; starter: boolean }
export interface NhlBox {
  periods: { period: number; home: number; away: number }[]; sog: { home: number; away: number };
  skaters: Record<NhlSide, NhlBoxSkater[]>; goalies: Record<NhlSide, NhlBoxGoalie[]>;
}
export interface NhlGameDetail {
  game: NhlGame; form: Record<NhlSide, NhlFormSide>; rest: Record<NhlSide, { days: number | null; b2b: boolean }>;
  elo: Record<NhlSide, number | null>;
  h2h: { home_wins: number; away_wins: number; games: { date: string; home: string; away: string; home_score: number; away_score: number }[] };
  factors: NhlFactor[]; top_props: NhlTopProp[]; box: NhlBox | null;
}

export interface NhlHealth {
  generated_at: string; season: number; season_label: string; latest_final_date: string | null;
  counts: Record<string, number>; model: { game_version: string; props_version: string; trained_at: string; train_games: number; seasons: string[] };
  schedule_days: { date: string; games: number }[]; lineup_note: string | null;
}

export interface NhlLine {
  line: number; p_over: number; p_under: number; fair_over: number; fair_under: number;
  hit_l10: number | null; hit_l20: number | null; hit_season: number | null;
}
export interface NhlMarket {
  projection: number; season_avg: number | null; l10_avg: number | null; confidence: NhlConfidence; lines: NhlLine[];
}
export type NhlMarkets = Partial<Record<NhlMarketKey, NhlMarket>>;
export interface NhlPropPlayer {
  player_id: number; name: string; team: string; pos: string; headshot: string; opp: string; home: boolean; game_id: number;
  date: string; start: string; lineup: string; proj_toi: number | null; markets: NhlMarkets;
}
export interface NhlPropsBoard { generated_at: string; dates: string[]; lineup_note: string | null; model: { version: string }; players: NhlPropPlayer[] }

export interface NhlPlayerRow { id: number; name: string; team: string; pos: string; headshot: string; gp: number; goals: number; assists: number; points: number; sog: number; toi: number | null }
export interface NhlPlayersList { generated_at: string; season: number; players: NhlPlayerRow[] }

export interface NhlRolling { goals: number | null; assists: number | null; points: number | null; sog: number | null; toi: number | null }
export interface NhlLogRow { game_id: number; date: string; opp: string; home: boolean; toi: string; goals: number; assists: number; points: number; sog: number; pp_goals: number; result: string | null }
export interface NhlHitRow { line: number; l5: number | null; l10: number | null; l20: number | null; season: number | null }
export interface NhlPlayerDetail {
  player: { id: number; name: string; team: string; pos: string; headshot: string }; generated_at: string;
  season: { gp: number; goals: number; assists: number; points: number; sog: number; toi: number | null; pp_goals: number; shooting_pct: number | null };
  rolling: Record<"l5" | "l10" | "l20" | "season", NhlRolling>;
  hit_rates: Partial<Record<NhlMarketKey, NhlHitRow[]>>; log: NhlLogRow[];
  next: null | { game_id: number; date: string; opp: string; home: boolean; lineup: string; markets: NhlMarkets;
    opp_context: { sa_pg: number | null; ga_pg: number | null; opp_goalie: string | null; goalie_sv_pct: number | null; rank_sa: number | null; rank_ga: number | null } | null };
  splits: Record<"home" | "away", { gp: number; goals: number; assists: number; points: number; sog: number } | null>;
}

export interface NhlStandingTeam { team: NhlTeamRef; rank: number; gp: number; w: number; l: number; otl: number; pts: number; pct: number; gf: number; ga: number; diff: number; l10: string; streak: string; elo: number | null }
export interface NhlStandings { generated_at: string; conferences: { name: string; divisions: { name: string; teams: NhlStandingTeam[] }[] }[] }

export interface NhlCalBin { predicted: number; actual: number; n: number }
export interface NhlPropPerf {
  n: number; model: { log_loss: number; brier: number }; baselines: Record<string, { log_loss: number; brier: number }>;
  calibration: NhlCalBin[]; top_decile?: { n: number; predicted: number; hit_rate: number; base_rate: number };
  lines: { line: number; n: number; log_loss: number; brier: number; hit_rate: number }[];
}
export interface NhlModelPerformance {
  generated_at: string;
  game: { n: number; seasons: string[]; model: { accuracy: number; log_loss: number; brier: number }; baselines: Record<string, { accuracy: number; log_loss: number; brier: number }>;
    calibration: NhlCalBin[]; total_goals: { mae: number; mean_actual: number; mean_pred: number }; note?: string };
  props: Partial<Record<NhlMarketKey, NhlPropPerf>>;
  record: { games: { graded: number; correct: number; rate: number | null };
    props: Partial<Record<NhlMarketKey, { graded: number; top_decile_hit_rate: number | null; avg_p: number | null }>>;
    recent: { date: string; game_id: number; pick: string; p: number; result: string; correct: boolean }[] };
}
