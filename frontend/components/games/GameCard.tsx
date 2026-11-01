"use client";

import type { Game, GamePrediction } from "@/lib/types";
import { TeamLogo } from "@/components/teams/TeamLogo";
import { gameState, scoreboard } from "@/lib/game-board.mjs";
import { fmtLine, fmtPrice } from "@/lib/game-markets.mjs";
import { safeTeamColor } from "@/lib/team-glow.mjs";
import { teamInfo, type TeamMap } from "./MatchupCards";
import { useGameSlip } from "./useGameSlip";
import { BoardCard, BoardCell, BoardRow, BoardWinBar, ModelCell, TeamCell, lineText } from "./shared/Board";
import styles from "./GameCard.module.css";

/** Card-shaped placeholders so the grid keeps its geometry while data loads. */
export function GameCardSkeletons({ count = 3 }: { count?: number }) {
  return (
    <div className={styles.grid} role="status" aria-label="Loading games">
      {Array.from({ length: count }, (_, index) => <div key={index} className={styles.skeletonCard} />)}
    </div>
  );
}

const ET = "America/New_York";
const when = (game: Game) => { const d = new Date(game.start_time || game.game_time); return Number.isNaN(d.getTime()) ? null : d; };
/** "7:30 PM" in Eastern time. */
export const tipClock = (game: Game) => when(game)?.toLocaleTimeString("en-US", { timeZone: ET, hour: "numeric", minute: "2-digit" }) ?? "TBA";
/** The hour a game tips in, for grouping the slate ("7:00 PM"). */
export const tipHour = (game: Game) => { const d = when(game); return d ? d.toLocaleTimeString("en-US", { timeZone: ET, hour: "numeric" }).replace(/\s+/, ":00 ") : "TBA"; };

const record = (wins?: number | null, losses?: number | null) => (wins != null && losses != null ? `${wins}–${losses}` : "");

/** Slate card: both teams against the book's spread, moneyline and total (tap a price to add it to the slip) and the model's own line and win chance. */
export function GameCard({ game, prediction, teams, league, date }: { game: Game; prediction?: GamePrediction; teams: TeamMap; league: string; date?: string }) {
  const away = teamInfo(game, "away", teams);
  const home = teamInfo(game, "home", teams);
  const state = gameState(game);
  const actual = state !== "pre";
  const score = scoreboard(game, prediction);
  const href = `/games/detail?${new URLSearchParams({ league, ...(game.game_id ? { id: game.game_id } : { matchup: game.matchup }), ...(date ? { date } : {}) })}`;
  const slip = useGameSlip(game, league as "nba" | "wnba", date, prediction);
  const pHome = prediction?.home_win_probability;
  const chance = typeof pHome === "number" && pHome > 0 && pHome < 1 ? pHome : null;
  const margin = prediction?.spread ?? null; // model home margin
  const status = actual ? game.status_text || (state === "post" ? "Final" : "Live") : tipClock(game);

  const row = (side: "away" | "home") => {
    const t = side === "away" ? away : home;
    const mine = score[side], theirs = score[side === "away" ? "home" : "away"];
    const lean = prediction?.predicted_winner === t.abbrev || prediction?.predicted_winner === game[`${side}_team`];
    const winP = chance == null ? null : side === "home" ? chance : 1 - chance;
    const spread = game.spread == null ? null : side === "home" ? game.spread : -game.spread;
    const ml = side === "home" ? game.home_ml : game.away_ml;
    const over = side === "away";
    const totalAdd = over ? slip.total.over : slip.total.under;
    return (
      <BoardRow key={side}>
        <TeamCell logo={<TeamLogo team={t} size={30} />} name={t.abbrev} sub={record(game[`${side}_wins`], game[`${side}_losses`]) || t.nickname}
          score={actual ? (mine ?? "—") : null} dim={state === "post" && mine != null && theirs != null && mine < theirs} />
        <BoardCell text={spread == null ? "—" : fmtLine(spread)} inSlip={slip.spread[side]?.inSlip} onToggle={slip.spread[side]?.onToggle} label={slip.spread[side] && `${slip.spread[side]!.inSlip ? "Remove" : "Add"} ${slip.spread[side]!.label} ${slip.spread[side]!.inSlip ? "from" : "to"} slip`} />
        <BoardCell text={ml == null ? "—" : fmtPrice(ml)} pick={!actual && lean ? winP : null} inSlip={slip.moneyline[side]?.inSlip} onToggle={slip.moneyline[side]?.onToggle} label={slip.moneyline[side] && `${slip.moneyline[side]!.inSlip ? "Remove" : "Add"} ${slip.moneyline[side]!.label} ${slip.moneyline[side]!.inSlip ? "from" : "to"} slip`} />
        <BoardCell text={game.total == null ? "—" : `${over ? "o" : "u"}${game.total}`} inSlip={totalAdd?.inSlip} onToggle={totalAdd?.onToggle} label={totalAdd && `${totalAdd.inSlip ? "Remove" : "Add"} ${totalAdd.label} ${totalAdd.inSlip ? "from" : "to"} slip`} />
        <ModelCell win={winP} line={margin == null ? null : lineText(side === "home" ? -margin : margin)} />
      </BoardRow>
    );
  };

  return (
    <BoardCard href={href} label={`Open ${away.name} at ${home.name}`}
      status={<span className={state === "in" ? "font-bold text-lock-gold" : ""}>{status}</span>}
      foot={chance != null && !actual ? <BoardWinBar away={{ abbr: away.abbrev, p: 1 - chance, color: safeTeamColor(away.color) }} home={{ abbr: home.abbrev, p: chance, color: safeTeamColor(home.color) }} /> : undefined}>
      {row("away")}{row("home")}
    </BoardCard>
  );
}
