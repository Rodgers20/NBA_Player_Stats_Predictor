"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { popularBoard } from "@/lib/popular.mjs";
import { PopularCard } from "@/components/popular/PopularCard";

/** Priced plays that cleared their line most often over the last 10 games: the Props page's "Hot" view. */
export function PopularBoard({ league }: { league: "nba" | "wnba" }) {
  const props = useQuery({
    queryKey: ["popular", league],
    queryFn: () => api.props({ league, sort: "ev", limit: 500 }),
    refetchInterval: 120_000,
  });
  const games = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league), refetchInterval: 120_000 });

  const board = popularBoard(props.data?.props ?? [], games.data?.games, { by: "hit" });
  const byMatchup = new Map((games.data?.games ?? []).map(game => [game.matchup, game]));
  const date = props.data?.target_date ?? games.data?.target_date ?? "";

  return <>
    <p className="mb-3 text-xs text-text-sec">Priced plays that cleared their line most often over the last 10 games. Games already under way drop off.</p>

    {props.isError && <div role="alert" className="ql-panel p-8 text-center text-sm text-red-300">
      <p>The hot list could not be loaded.</p>
      <button type="button" className="mt-2 underline" onClick={() => void props.refetch()}>Try again</button>
    </div>}

    {props.isPending && <div className="ql-panel overflow-hidden" aria-busy="true" aria-label="Loading hot plays">
      {Array.from({ length: 5 }).map((_, index) => <div key={index} className="h-24 animate-pulse border-b border-white/5" />)}
    </div>}

    {props.data && board.length > 0 && <section aria-label="Hot plays" className="ql-panel overflow-hidden">
      {board.map((prop, index) => <PopularCard key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`}
        prop={prop} rank={index + 1} league={league} date={date} game={byMatchup.get(prop.game_matchup)} />)}
    </section>}

    {props.data && board.length === 0 && <section className="ql-panel px-6 py-10 text-center" role="status">
      <h2 className="ql-section-title">{props.data.status === "stale" ? "Today's picks aren't published yet" : "Nothing hot right now"}</h2>
      <p className="mx-auto mt-2 max-w-[60ch] text-sm text-text-sec">{props.data.message || "No priced play for a game that hasn't started has a big enough sample yet. Props usually appear the morning of the game."}</p>
    </section>}
  </>;
}
