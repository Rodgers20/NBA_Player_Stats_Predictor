"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { tonightBoard } from "@/lib/tonight.mjs";
import { tipLabel } from "@/components/games/MatchupCards";
import { TonightRow } from "@/components/tonight/TonightRow";
import { ResearchList } from "@/components/tonight/ResearchList";
import { GameEdges } from "@/components/tonight/GameEdges";
import { bettableEdges, gameEdges } from "@/lib/game-edges.mjs";
import { isPastSlate } from "@/lib/slate-freshness.mjs";

const BOARD_LIMIT = 500;  // the whole slate, so "N plays from M priced markets" counts every market, not the first page
const TOP_PLAYS = 8;      // nobody reads past the first screen of plays; the rest are one tap away in Props
const SHOWN_PASSES = 3;   // what the model turned down is what makes what it kept believable
const SLATE_PREVIEW = 8;

function slateDate(value: string | null | undefined) {
  if (!value) return "Upcoming games";
  const date = new Date(`${value}T12:00:00`);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });
}

export default function TonightPage() {
  const league = usePrefs(state => state.league);
  const props = useQuery({
    queryKey: ["tonight", league],
    queryFn: () => api.props({ league, include_research: true, sort: "ev", limit: BOARD_LIMIT }),
    refetchInterval: 120_000,
  });
  const games = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league) });
  const predictions = useQuery({ queryKey: ["game-predictions", league], queryFn: () => api.predictions(league), staleTime: 300_000 });

  const board = tonightBoard(props.data?.props ?? []);
  // Game-line value needs no props feed: the model's win chance against each moneyline price. Only use predictions for the same slate.
  const sameSlate = predictions.data && predictions.data.target_date === games.data?.target_date;
  // The static snapshot is rebuilt once a day; until then it holds yesterday's slate, which must not read as tonight's.
  const stale = props.data?.status === "stale" || isPastSlate(games.data?.target_date);
  const edges = stale ? [] : gameEdges(games.data?.games ?? [], sameSlate ? predictions.data?.predictions ?? [] : []);
  const lineEdges = bettableEdges(edges).length;
  const date = props.data?.target_date ?? games.data?.target_date ?? "";
  const slate = stale ? [] : games.data?.games ?? [];
  const priced = board.picks.length + board.leans.length;
  const plays = board.picks.length + lineEdges;

  return <main className="ql-page mx-auto max-w-[1100px] px-4 pb-14 pt-7 sm:px-8">
    <header className="mb-5">
      <p className="ql-kicker">TONIGHT / {league.toUpperCase()}</p>
      <h1 className="ql-heading mt-1">Tonight&apos;s plays</h1>
      <p className="ql-subtitle mt-2">{slateDate(date)} · a play needs a fresh sportsbook price and at least +2% edge after the model weighs the market.</p>
    </header>

    {props.data && !stale && <p className="mb-5 flex flex-wrap items-baseline gap-x-3 gap-y-1 border-y border-white/10 py-3 text-sm text-text-sec" role="status">
      <span><strong className="text-lg font-extrabold tabular-nums text-text-pri">{plays}</strong> {plays === 1 ? "play" : "plays"} from <strong className="tabular-nums text-text-pri">{priced}</strong> priced markets</span>
      <Link href="/performance" className="ml-auto text-xs font-semibold text-[#59e0c8] hover:underline">Model record ↗</Link>
    </p>}

    {props.isError && <div role="alert" className="ql-panel p-8 text-center text-sm text-red-300">
      <p>Tonight&apos;s board could not be loaded.</p>
      <button type="button" className="mt-2 underline" onClick={() => void props.refetch()}>Try again</button>
    </div>}

    {props.isPending && <div className="ql-panel overflow-hidden" aria-busy="true" aria-label="Loading tonight's plays">
      {Array.from({ length: 4 }).map((_, index) => <div key={index} className="h-20 animate-pulse border-b border-white/5" />)}
    </div>}

    {props.data && board.picks.length > 0 && <section aria-label="Bettable plays">
      <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
        <h2 className="ql-section-title">Ranked by edge</h2>
        {board.picks.length > TOP_PLAYS && <Link href="/props?ev=true" className="ql-data-label hover:text-text-pri">ALL {board.picks.length} IN PROPS ↗</Link>}
      </div>
      <div className="ql-panel overflow-hidden">
        {board.picks.slice(0, TOP_PLAYS).map(prop => <TonightRow key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`} prop={prop} league={league} date={date} />)}
      </div>
    </section>}

    {!props.isPending && <GameEdges edges={edges} league={league} date={date} />}

    {stale && <section className="ql-panel px-6 py-10 text-center" role="status">
      <h2 className="ql-section-title">Today&apos;s picks aren&apos;t published yet</h2>
      <p className="mx-auto mt-2 max-w-[60ch] text-sm text-text-sec">The board last refreshed for {slateDate(date)}. New picks appear after the morning data refresh, around 7:15 AM ET.</p>
    </section>}

    {!stale && props.data && board.picks.length === 0 && !lineEdges && <section className="ql-panel px-6 py-10 text-center" role="status">
      <h2 className="ql-section-title">No bettable edges right now</h2>
      <p className="mx-auto mt-2 max-w-[60ch] text-sm text-text-sec">{priced > 0 ? `The model priced ${priced} markets and none cleared +2% once the sportsbook's view is weighed in. Passing is a result.` : slate.length > 0 ? `There ${slate.length === 1 ? "is 1 game" : `are ${slate.length} games`} on the slate, but no sportsbook player props are posted yet. Props usually appear the morning of the game.` : "No priced market passed the price, history and model checks."}</p>
      <div className="mt-5 flex flex-wrap justify-center gap-2">
        <Link href="/props" className="ql-chip text-[#59e0c8] hover:border-[#59e0c8]/50">All props</Link>
        <Link href="/games" className="ql-chip hover:border-white/30">Tonight&apos;s games</Link>
      </div>
    </section>}

    {board.leans.length > 0 && <section className="mt-7" aria-label="Plays the model passed on">
      <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
        <h2 className="ql-section-title">The model passed</h2>
        <span className="ql-data-label">UNDER +2% EDGE · {board.leans.length}</span>
      </div>
      <div className="ql-panel overflow-hidden opacity-80">
        {board.leans.slice(0, SHOWN_PASSES).map(prop => <TonightRow key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`} prop={prop} league={league} date={date} />)}
      </div>
      {board.leans.length > SHOWN_PASSES && <details className="mt-2">
        <summary className="ql-data-label cursor-pointer py-2">SHOW {board.leans.length - SHOWN_PASSES} MORE PASSES</summary>
        <div className="ql-panel mt-2 overflow-hidden opacity-80">
          {board.leans.slice(SHOWN_PASSES).map(prop => <TonightRow key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`} prop={prop} league={league} date={date} />)}
        </div>
      </details>}
    </section>}

    {board.research.length > 0 && <ResearchList rows={board.research} league={league} />}

    {slate.length > 0 && <section className="mt-8" aria-label="Slate">
      <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
        <h2 className="ql-section-title">The slate</h2>
        <Link href="/games" className="ql-data-label hover:text-text-pri">ALL GAMES</Link>
      </div>
      <ul className="grid gap-2 sm:grid-cols-2">
        {slate.slice(0, SLATE_PREVIEW).map(game => <li key={game.game_id || game.matchup} className="ql-panel flex items-center justify-between gap-3 px-4 py-3 text-sm">
          <span className="font-bold text-text-pri">{game.matchup}</span>
          <span className="text-xs tabular-nums text-text-sec">{tipLabel(game)}</span>
        </li>)}
      </ul>
    </section>}

    <p className="mt-8 text-center text-xs text-text-sec">Edge is the model&apos;s estimated expected value at the quoted price, after weighing the sportsbook&apos;s own view. Hit counts are history, not a forecast.</p>
  </main>;
}
