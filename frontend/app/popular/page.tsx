"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { popularBoard } from "@/lib/popular.mjs";
import { PopularCard } from "@/components/popular/PopularCard";

const TABS = [
  { key: "hit", label: "Hottest", hint: "Priced plays that cleared their line most often over the last 10 games." },
  { key: "edge", label: "Best edge", hint: "Priced plays where the model sees the most value against the sportsbook price." },
] as const;

function slateDate(value: string | null | undefined) {
  const date = value ? new Date(`${value}T12:00:00`) : null;
  return date && !Number.isNaN(date.getTime())
    ? date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })
    : "Upcoming games";
}

export default function PopularPage() {
  const league = usePrefs(state => state.league);
  const [by, setBy] = useState<(typeof TABS)[number]["key"]>("hit");
  const props = useQuery({
    queryKey: ["popular", league],
    queryFn: () => api.props({ league, sort: "ev", limit: 500 }),
    refetchInterval: 120_000,
  });
  const games = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league), refetchInterval: 120_000 });

  const board = popularBoard(props.data?.props ?? [], games.data?.games, { by });
  const byMatchup = new Map((games.data?.games ?? []).map(game => [game.matchup, game]));
  const date = props.data?.target_date ?? games.data?.target_date ?? "";
  const hint = TABS.find(tab => tab.key === by)?.hint;

  return <main className="ql-page mx-auto max-w-[1100px] px-4 pb-14 pt-7 sm:px-8">
    <header className="mb-6">
      <p className="ql-kicker">POPULAR / {league.toUpperCase()}</p>
      <h1 className="ql-heading mt-1">Popular today</h1>
      <p className="ql-subtitle mt-2">{slateDate(date)} · Today&apos;s priced plays only. Games already under way drop off, and the board refreshes on its own.</p>
    </header>

    <div className="mb-4 flex flex-wrap items-center gap-2" role="group" aria-label="Rank popular plays by">
      {TABS.map(tab => <button key={tab.key} type="button" aria-pressed={by === tab.key} onClick={() => setBy(tab.key)}
        className={`ql-chip min-h-9 px-4 py-2 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${by === tab.key ? "ql-chip-active" : "text-[#a9bec0] hover:border-[#59e0c8]/45 hover:text-white"}`}>{tab.label}</button>)}
      <p className="ml-1 text-xs text-text-sec">{hint}</p>
    </div>

    {props.isError && <div role="alert" className="ql-panel p-8 text-center text-sm text-red-300">
      <p>The popular board could not be loaded.</p>
      <button type="button" className="mt-2 underline" onClick={() => void props.refetch()}>Try again</button>
    </div>}

    {props.isPending && <div className="ql-panel overflow-hidden" aria-busy="true" aria-label="Loading popular plays">
      {Array.from({ length: 5 }).map((_, index) => <div key={index} className="h-24 animate-pulse border-b border-white/5" />)}
    </div>}

    {props.data && board.length > 0 && <section aria-label="Popular plays" className="ql-panel overflow-hidden">
      {board.map((prop, index) => <PopularCard key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`}
        prop={prop} rank={index + 1} league={league} date={date} game={byMatchup.get(prop.game_matchup)} />)}
    </section>}

    {props.data && board.length === 0 && <section className="ql-panel px-6 py-10 text-center" role="status">
      <h2 className="ql-section-title">{props.data.status === "stale" ? "Today's picks aren't published yet" : "Nothing popular right now"}</h2>
      <p className="mx-auto mt-2 max-w-[60ch] text-sm text-text-sec">{props.data.message || "No priced play for a game that hasn't started has a big enough sample yet. Props usually appear the morning of the game."}</p>
    </section>}
  </main>;
}
