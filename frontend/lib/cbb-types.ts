/** Shapes of the static JSON in public/data/cbb, as documented in docs/cbb-data-contract.md. */
export type CbbConfidence = "high" | "medium" | "low" | "coin_flip";
export type CbbSide = "home" | "away";

export interface CbbTeam {
  id: string; name: string; short: string; abbr: string; logo: string; color: string; alt_color: string;
  conference_id: string | null; conference: string | null; rank: number | null; d1: boolean;
  record: string; conf_record: string | null; streak: string | null;
}
export interface CbbLine {
  provider: string; spread_home: number | null; total: number | null; home_ml: number | null; away_ml: number | null;
  open_spread_home: number | null; open_total: number | null; updated: string | null;
}
export interface CbbPick {
  source: "live" | "backtest"; model_version: string; winner: CbbSide; winner_abbr: string; p_home: number; p_away: number;
  confidence: CbbConfidence; proj_margin: number; proj_total: number; proj_home_pts: number; proj_away_pts: number;
  market_p_home: number | null;
  ats: { side: CbbSide | "push"; text: string; p_cover: number; confidence: CbbConfidence } | null;
  total: { side: "over" | "under"; line: number; p: number; confidence: CbbConfidence } | null;
}
export interface CbbResult {
  home_score: number; away_score: number; overtimes: number; margin: number; total: number; winner: CbbSide;
  winner_correct: boolean | null; market_winner_correct: boolean | null;
  ats_result: CbbSide | "push" | null; ats_correct: boolean | null;
  total_result: "over" | "under" | "push" | null; total_correct: boolean | null; margin_error: number | null;
}
export interface CbbGame {
  id: string; date: string; time: string; kickoff: string; status: "scheduled" | "live" | "final" | "postponed" | "other";
  season: number; neutral: boolean; conf_game: boolean; note: string | null; venue: string | null; broadcast: string | null;
  home: CbbTeam; away: CbbTeam; score: { home: number; away: number; overtimes: number } | null;
  line: CbbLine | null; pick: CbbPick | null; result: CbbResult | null;
}
export interface CbbFormGame { date: string; opp: string; opp_abbr: string; home: boolean; win: boolean; pts_for: number; pts_against: number }
export interface CbbForm {
  games: CbbFormGame[]; n: number; wins: number; losses: number; record: string;
  avg_scored: number | null; avg_allowed: number | null; avg_margin: number | null;
}
export interface CbbRating { off: number; def: number; power: number; off100: number; def100: number; games_played: number; power_rank: number | null }
export interface CbbFactor { label: string; value: number }
export interface CbbGameDetail extends CbbGame {
  form: { home: { overall: CbbForm; split: CbbForm }; away: { overall: CbbForm; split: CbbForm } };
  h2h: { n: number; home_wins: number; away_wins: number; avg_margin_home: number | null;
    games: { date: string; game_id: string; neutral: boolean; at_home: boolean; home_team_pts: number; away_team_pts: number; winner: CbbSide }[] };
  ratings: { home: CbbRating; away: CbbRating } | null;
  line_movement: { at: string; provider: string; spread_home: number | null; total: number | null; home_ml: number | null; away_ml: number | null }[];
  factors: { margin: CbbFactor[]; total: CbbFactor[] };
  why: { label: string; points: number; favors: CbbSide; abbr: string }[];
}
export interface CbbHealth {
  status: string; generated_at: string; season: number; model_version: string; ap_week: string | null;
  schedule: { from: string; to: string; days: { date: string; games: number }[] };
}
export interface CbbSchedule { date: string; generated_at: string; games: CbbGame[] }
export interface CbbRankTeam {
  id: string; name: string; short: string; abbr: string; logo: string; color: string; conference: string | null;
  power_rank: number; ap_rank: number | null; ap_previous: number; record: string; conf_record: string | null; last10: string | null;
  games: number; off: number; def: number; power: number; off100: number; def100: number; net100: number;
}
export interface CbbRankings { generated_at: string; season: number; ap_week: string | null; ap_updated: string | null; ratings_note: string | null; teams: CbbRankTeam[] }
export interface CbbStandingTeam {
  id: string; name: string; short: string; abbr: string; logo: string; color: string; conference: string; ap_rank: number | null;
  conf_wins: number; conf_losses: number; conf_record: string; wins: number; losses: number; overall: string;
  diff: number; avg_diff: number | null; streak: string | null;
}
export interface CbbStandings { generated_at: string; season: number; conferences: { id: string; name: string; short: string; logo: string | null; teams: CbbStandingTeam[] }[] }
export interface CbbRec { wins: number; losses: number; n: number; rate: number | null }
export interface CbbTrackRecord { games: number; winner: CbbRec | null; market_winner: CbbRec | null; ats: CbbRec | null; total: CbbRec | null; margin_mae: number | null; total_mae: number | null }
export interface CbbMetrics { margin_mae: number; margin_rmse: number; win_accuracy: number; log_loss: number; brier: number; total_mae: number }
export interface CbbCalibrationBin { predicted: number; actual: number; n: number }
export interface CbbModelPerformance {
  generated_at: string;
  model: { version: string; trained_at: string; n_games: number; seasons: number[] };
  backtest: { n_games: number; seasons: number[]; model: CbbMetrics; ratings_only: CbbMetrics; home_team_always_wins: number; sigma_margin: number; sigma_total: number;
    holdout: { n_games: number; seasons: number[]; from_season: number; model: CbbMetrics; ratings_only: CbbMetrics; home_team_always_wins: number; calibration: CbbCalibrationBin[] } | null } | null;
  vs_market: { n_games: number; seasons: number[]; model_margin_mae: number; market_margin_mae: number; model_winner: CbbRec; market_winner: CbbRec;
    ats: CbbRec; ats_by_edge: (CbbRec & { min_edge: number })[]; ou: CbbRec; ou_by_edge: (CbbRec & { min_edge: number })[];
    model_total_mae: number; market_total_mae: number; by_season: { season: number; n_games: number; ats: CbbRec }[] } | null;
  by_season: { season: number; n: number; margin_mae: number; win_accuracy: number; total_mae: number }[];
  calibration: CbbCalibrationBin[];
  track_record: { all: CbbTrackRecord; live: CbbTrackRecord; by_season: (CbbTrackRecord & { season: number })[] };
  recent: CbbGame[];
  notes: string[];
}
