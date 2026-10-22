"use client";

import { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { gameState } from "@/lib/game-board.mjs";
import { gameMarkets } from "@/lib/game-markets.mjs";
import { quoteAge } from "@/lib/tonight.mjs";
import { modelContext } from "@/lib/model-context.mjs";
import { PageWithSidebar } from "@/components/layout/PageWithSidebar";
import { BoxScore } from "./BoxScore";
import { DefenseStrip } from "./DefenseStrip";
import { GameHighlights } from "./GameHighlights";
import { MarketBlock } from "./MarketBlock";
import { MarketGauge, MatchupBanner, teamInfo } from "./MatchupCards";
import { Injuries, MatchupDetails } from "./MatchupDetails";
import { GameProps } from "./GameProps";
import { GameInsights } from "./GameInsights";
import { MatchupPanel } from "./MatchupPanel";
import { Tabs, type TabDef } from "./GameTabs";
import { useTeamMap } from "./useTeamMap";
import { useGameSlip, withAdds } from "./useGameSlip";
import type { Game, GamePrediction } from "@/lib/types";
import gameStyles from "./games.module.css";
import styles from "./GameDetail.module.css";

const MAIN_TABS = [{ id: "lines", label: "Gamelines" }, { id: "props", label: "Props" }, { id: "insights", label: "Insights" }, { id: "matchup", label: "Matchup" }, { id: "box", label: "Box score" }, { id: "clips", label: "Highlights" }] as const satisfies readonly TabDef<string>[];
const SIDE_TABS = [{ id: "matchup", label: "Defense" }, { id: "injuries", label: "Injuries" }, { id: "insights", label: "Model notes" }] as const satisfies readonly TabDef<string>[];
type MainTab = (typeof MAIN_TABS)[number]["id"];
type SideTab = (typeof SIDE_TABS)[number]["id"];

function remember(tab: MainTab) {
  const url = new URL(window.location.href);
  url.searchParams.set("tab", tab);
  window.history.replaceState(null, "", url);
}

/** The three markets with their slip buttons. Its own component so the slip hook runs only once the game is loaded. */
function GameMarkets({ game, markets, league, date, prediction }: { game: Game; markets: ReturnType<typeof gameMarkets>; league: "nba" | "wnba"; date?: string; prediction?: GamePrediction }) {
  const slip = useGameSlip(game, league, date, prediction);
  const totalActions = [
    slip.total.over && { text: `Over ${game.total}`, add: slip.total.over },
    slip.total.under && { text: `Under ${game.total}`, add: slip.total.under },
  ].filter((item): item is NonNullable<typeof item> => Boolean(item));
  return (
    <div className={styles.markets}>
      <MarketBlock title="Money line" sides={withAdds(markets.moneyline, [slip.moneyline.away, slip.moneyline.home])} />
      <MarketBlock title="Spread" sides={withAdds(markets.spread, [slip.spread.away, slip.spread.home])} />
      <MarketBlock title={markets.totalTitle} sides={withAdds(markets.total, [undefined, undefined])} actions={totalActions} />
    </div>
  );
}

export function GameDetailPage() {
  const params = useSearchParams();
  const preferred = usePrefs(state => state.league);
  const requested = params.get("league");
  const league = requested === "wnba" || requested === "nba" ? requested : preferred;
  const id = params.get("id") ?? "";
  const matchup = params.get("matchup") ?? "";
  const date = /^\d{4}-\d{2}-\d{2}$/.test(params.get("date") ?? "") ? params.get("date")! : undefined;
  const validId = /^\d{6,12}$/.test(id);
  const initial = MAIN_TABS.find(tab => tab.id === params.get("tab"))?.id ?? "lines";
  const [tab, setTab] = useState<MainTab>(initial);
  const [sideTab, setSideTab] = useState<SideTab>("matchup");

  const games = useQuery({ queryKey: ["games", league, date], queryFn: () => api.games(league, date), enabled: validId || !!matchup });
  const predictions = useQuery({ queryKey: ["game-predictions", league, date], queryFn: () => api.predictions(league, date), staleTime: 300_000 });
  const { teamMap: teams, ready } = useTeamMap(league);

  const game = games.data?.games.find(item => (validId ? item.game_id === id : item.matchup === matchup));
  const prediction = game && predictions.data?.target_date === games.data?.target_date ? predictions.data?.predictions.find(item => item.matchup === game.matchup) : undefined;
  const status = predictions.isPending ? "Loading model estimate…" : predictions.data?.message || predictions.data?.errors.find(item => item.matchup === game?.matchup)?.message || "Model estimate unavailable for this game.";

  const back = <Link href="/games" className={styles.back}><ArrowLeft size={16} /> All games</Link>;
  if (!validId && !matchup) return <div className={gameStyles.page}>{back}<p role="alert" className={gameStyles.empty}>That game link is not valid.</p></div>;
  if (games.isPending || (!games.isError && !ready)) return <div className={gameStyles.page}>{back}<div className={gameStyles.skeleton} aria-label="Loading game" /></div>;
  if (games.isError) return <div className={gameStyles.page}>{back}<div role="alert" className={gameStyles.empty}><h2>Game unavailable</h2><p>We couldn&apos;t load this game.</p><button onClick={() => void games.refetch()}>Try again</button></div></div>;
  if (!game) return <div className={gameStyles.page}>{back}<div role="alert" className={gameStyles.empty}><h2>Game not found</h2><p>It may be on a different date than this link expects.</p><Link href="/games">Back to the slate ↗</Link></div></div>;

  const away = teamInfo(game, "away", teams);
  const home = teamInfo(game, "home", teams);
  const markets = gameMarkets(game, away.abbrev, home.abbrev, teams[game.away_team]?.games ?? [], teams[game.home_team]?.games ?? []);
  const state = gameState(game);
  const context = modelContext(prediction);

  const main = (
    <div className={styles.stack}>
      {back}
      <MatchupBanner game={game} prediction={prediction} teams={teams} league={league} />
      <Tabs tabs={[...MAIN_TABS]} value={tab} onChange={id => { setTab(id); remember(id); }} label="Game sections" idPrefix="game" />
      <div role="tabpanel" id="game-panel" aria-labelledby={`game-tab-${tab}`} className={styles.panel}>
        {tab === "lines" && <>
          <GameMarkets game={game} markets={markets} league={league} date={date ?? games.data?.target_date ?? undefined} prediction={prediction} />
          <MarketGauge game={game} />
          <MatchupDetails game={game} prediction={prediction} predictionStatus={status} league={league} showDefense={false} showAvailability={false} />
        </>}
        {tab === "props" && <GameProps game={game} league={league} date={date ?? games.data?.target_date ?? undefined} />}
        {tab === "insights" && <GameInsights game={game} league={league} />}
        {tab === "matchup" && <MatchupPanel game={game} league={league} />}
        {tab === "box" && (game.game_id && state !== "pre"
          ? <BoxScore key={`${league}-${game.game_id}`} league={league} gameId={game.game_id} />
          : <p className={styles.note}>The box score appears once the game tips off.</p>)}
        {tab === "clips" && <GameHighlights league={league} gameId={game.game_id} />}
      </div>
    </div>
  );

  const side = (
    <div className={`${styles.side} card-depth`}>
      <Tabs tabs={[...SIDE_TABS]} value={sideTab} onChange={setSideTab} label="Game context" className={styles.sideTabs} idPrefix="side" />
      <div role="tabpanel" id="side-panel" aria-labelledby={`side-tab-${sideTab}`} className={styles.sidePanel}>
      {sideTab === "matchup" && <>
        <DefenseStrip game={game} league={league} detailed />
        <dl className={styles.facts}>
          {game.venue && <><dt>Venue</dt><dd>{game.venue}</dd></>}
          {game.broadcast && <><dt>Watch</dt><dd>{game.broadcast}</dd></>}
          <dt>Lines</dt><dd>{game.odds_source ? `${game.odds_source} · ${game.odds_provider || "Sportsbook"}${quoteAge(game.odds_updated_at) ? ` · ${quoteAge(game.odds_updated_at)?.label}` : ""}` : "No sportsbook quote available."}</dd>
        </dl>
      </>}
      {sideTab === "injuries" && <div className={gameStyles.injuries}>
        <Injuries team={game.away_team} injuries={game.away_injuries ?? []} league={league} />
        <Injuries team={game.home_team} injuries={game.home_injuries ?? []} league={league} />
      </div>}
      {sideTab === "insights" && (prediction
        ? <>
          <p className={gameStyles.source}>{context.label}</p>
          {prediction.intel?.length ? <ul className={styles.intel}>{prediction.intel.slice(0, 6).map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className={gameStyles.source}>No extra model notes for this game.</p>}
        </>
        : <p className={gameStyles.source}>{status}</p>)}
      </div>
    </div>
  );

  return <div className={gameStyles.page}><PageWithSidebar main={main} side={side} sideLabel="Game context" /></div>;
}
