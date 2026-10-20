import Link from "next/link";
import type { Game, GamePrediction } from "@/lib/types";
import { TeamLogo } from "@/components/teams/TeamLogo";
import { gameState, marketProbability, scoreboard } from "@/lib/game-board.mjs";
import { marketForm } from "@/lib/market-form.mjs";
import { teamGlow } from "@/lib/team-glow.mjs";
import { teamInfo, tipLabel, type TeamMap } from "./MatchupCards";
import { MarketBlock, fmtLine, fmtPrice, type MarketSide } from "./MarketBlock";
import styles from "./GameCard.module.css";

const record = (wins?: number | null, losses?: number | null) => (wins != null && losses != null ? `${wins}–${losses}` : "");

/** Slate card: team-colour glow, tip/score, win outlook and the three main markets with real recent form. */
export function GameCard({ game, prediction, teams, league, index = 0 }: { game: Game; prediction?: GamePrediction; teams: TeamMap; league: string; index?: number }) {
  const away = teamInfo(game, "away", teams);
  const home = teamInfo(game, "home", teams);
  const state = gameState(game);
  const actual = state !== "pre";
  const score = scoreboard(game, prediction);
  const outlook = marketProbability(game);
  const homePct = outlook == null ? null : Math.round(outlook * 100);
  const form = marketForm(game, teams[game.away_team]?.games ?? [], teams[game.home_team]?.games ?? []);
  const href = game.game_id ? `/games/detail?id=${game.game_id}&league=${league}` : `/games/detail?league=${league}&matchup=${encodeURIComponent(game.matchup)}`;

  const moneyline: [MarketSide, MarketSide] | null = game.away_ml == null && game.home_ml == null ? null : [
    { label: away.abbrev, value: fmtPrice(game.away_ml), form: { ...form.moneyline.away, caption: "L10 wins", flat: true, note: "straight up" } },
    { label: home.abbrev, value: fmtPrice(game.home_ml), form: { ...form.moneyline.home, caption: "L10 wins", flat: true, note: "straight up" } },
  ];
  const spread: [MarketSide, MarketSide] | null = game.spread == null || !form.spread ? null : [
    { label: away.abbrev, value: fmtLine(-game.spread), form: { ...form.spread.away, caption: "L10 covers", flat: true, note: `vs ${fmtLine(-game.spread)}` } },
    { label: home.abbrev, value: fmtLine(game.spread), form: { ...form.spread.home, caption: "L10 covers", flat: true, note: `vs ${fmtLine(game.spread)}` } },
  ];
  const total: [MarketSide, MarketSide] | null = game.total == null || !form.total ? null : [
    { label: away.abbrev, form: { values: form.total.away, line: game.total, caption: "L10 over", note: `vs ${game.total}` } },
    { label: home.abbrev, form: { values: form.total.home, line: game.total, caption: "L10 over", note: `vs ${game.total}` } },
  ];

  return (
    <Link href={href} className={`${styles.card} card-depth enter`} style={{ "--i": index } as React.CSSProperties} aria-label={`${away.name} at ${home.name}, open game`}>
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
        <MarketBlock title={game.total == null ? "Total" : `Total ${game.total}`} sides={total} />
      </div>
      <div className={styles.foot}>
        <span>Open game</span>
        <span className={styles.sub}>{game.odds_source || game.odds_provider || "ESPN lines"}</span>
      </div>
    </Link>
  );
}
