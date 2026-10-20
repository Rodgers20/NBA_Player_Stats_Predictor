import Link from "next/link";
import type { Game, GamePrediction } from "@/lib/types";
import { TeamLogo } from "@/components/teams/TeamLogo";
import { gameState, marketProbability, scoreboard } from "@/lib/game-board.mjs";
import { gameMarkets } from "@/lib/game-markets.mjs";
import { quoteAge } from "@/lib/tonight.mjs";
import { teamGlow } from "@/lib/team-glow.mjs";
import { teamInfo, tipLabel, type TeamMap } from "./MatchupCards";
import { MarketBlock } from "./MarketBlock";
import styles from "./GameCard.module.css";

/** Card-shaped placeholders so the grid keeps its geometry while data loads. */
export function GameCardSkeletons({ count = 3 }: { count?: number }) {
  return (
    <div className={styles.grid} role="status" aria-label="Loading games">
      {Array.from({ length: count }, (_, index) => <div key={index} className={styles.skeletonCard} />)}
    </div>
  );
}

const record = (wins?: number | null, losses?: number | null) => (wins != null && losses != null ? `${wins}–${losses}` : "");

/** Slate card: team-colour glow, tip/score, win outlook and the three main markets with real recent form. */
export function GameCard({ game, prediction, teams, league, date, index = 0 }: { game: Game; prediction?: GamePrediction; teams: TeamMap; league: string; date?: string; index?: number }) {
  const away = teamInfo(game, "away", teams);
  const home = teamInfo(game, "home", teams);
  const state = gameState(game);
  const actual = state !== "pre";
  const score = scoreboard(game, prediction);
  const outlook = marketProbability(game);
  const homePct = outlook == null ? null : Math.round(outlook * 100);
  const { moneyline, spread, total, totalTitle } = gameMarkets(game, away.abbrev, home.abbrev, teams[game.away_team]?.games ?? [], teams[game.home_team]?.games ?? []);
  const href = `/games/detail?${new URLSearchParams({ league, ...(game.game_id ? { id: game.game_id } : { matchup: game.matchup }), ...(date ? { date } : {}) })}`;

  return (
    <Link href={href} className={`${styles.card} card-depth enter`} style={{ "--i": index } as React.CSSProperties}>
      <div className={styles.glow} aria-hidden style={{ backgroundImage: `${teamGlow(away.color, "left")}, ${teamGlow(home.color, "right")}` }} />
      <div className={styles.head}>
        <div className={styles.top}>
          <span>{tipLabel(game)}</span>
          <span className={`${styles.status}${state === "in" ? ` ${styles.live}` : ""}`}>{actual ? game.status_text || (state === "post" ? "Final" : "Live") : "Pregame"}</span>
        </div>
        <div className={styles.matchup}>
          <div className={styles.team}>
            <TeamLogo team={away} size={46} />
            <span className={styles.teamName}>{away.nickname || away.name}</span>
            <span className={styles.record}>{record(game.away_wins, game.away_losses) || away.abbrev}</span>
          </div>
          {actual
            ? <span className={styles.score}>{score.away ?? "—"}<span className={styles.at}> – </span>{score.home ?? "—"}</span>
            : <span className={styles.at}>@</span>}
          <div className={styles.team}>
            <TeamLogo team={home} size={46} />
            <span className={styles.teamName}>{home.nickname || home.name}</span>
            <span className={styles.record}>{record(game.home_wins, game.home_losses) || home.abbrev}</span>
          </div>
        </div>
      </div>
      {homePct != null ? (
        <div className={styles.outlook}>
          <div className={styles.track} role="img" aria-label={`Sportsbook win chance: ${away.abbrev} ${100 - homePct} percent, ${home.abbrev} ${homePct} percent`}>
            <div className={styles.trackAway} style={{ width: `${100 - homePct}%` }} />
            <div className={styles.trackHome} style={{ width: `${homePct}%` }} />
          </div>
          <div className={styles.outlookLabels}><span>{away.abbrev} {100 - homePct}%</span><span>{homePct}% {home.abbrev}</span></div>
        </div>
      ) : null}
      <div className={styles.blocks}>
        <MarketBlock title="Money line" sides={moneyline} />
        <MarketBlock title="Spread" sides={spread} />
        <MarketBlock title={totalTitle} sides={total} />
      </div>
      <div className={styles.foot}>
        <span>Open game</span>
        <span className={styles.sub}>{game.odds_source || game.odds_provider || "ESPN lines"}{quoteAge(game.odds_updated_at) ? ` · ${quoteAge(game.odds_updated_at)?.label}` : ""}</span>
      </div>
    </Link>
  );
}
