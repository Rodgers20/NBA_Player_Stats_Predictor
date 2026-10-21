"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { tonightBoard } from "@/lib/tonight.mjs";
import { TonightRow } from "@/components/tonight/TonightRow";
import { ResearchList } from "@/components/tonight/ResearchList";
import type { Game } from "@/lib/types";

const rowKey = (prop: { player: string; stat: string; direction: string; line: number | null }) => `${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`;

/** The model's props for one game: priced plays ranked by edge, priced no-edge leans, then projections that have no line yet. */
export function GameProps({ game, league, date }: { game: Game; league: "nba" | "wnba"; date?: string }) {
  const props = useQuery({
    queryKey: ["game-props", league, game.matchup],
    queryFn: () => api.props({ league, game: game.matchup, include_research: true, sort: "ev", limit: 100 }),
    staleTime: 120_000,
  });
  const record = useQuery({ queryKey: ["props-record"], queryFn: api.propsRecord });

  if (props.isPending) return <div className="ql-panel h-48 animate-pulse" aria-busy="true" aria-label="Loading props" />;
  if (props.isError) return <div role="alert" className="ql-panel p-5 text-sm text-text-sec">This game&apos;s props could not be loaded. <button type="button" className="underline" onClick={() => void props.refetch()}>Try again</button></div>;

  const board = tonightBoard(props.data.props);
  const slipDate = props.data.target_date ?? date ?? "";
  if (props.data.props.length === 0) return (
    <div role="status" className="ql-panel p-5 text-sm text-text-sec">
      <p className="font-bold text-text-pri">No props for {game.matchup} yet.</p>
      <p className="mt-1">Props cover the current slate only, and appear once a sportsbook posts lines or the model has recent games for the players. <Link href="/props" className="text-teal-300 underline">Best Props board ↗</Link></p>
    </div>
  );

  return <div className="space-y-4">
    <section aria-label={`${game.matchup} plays`}>
      <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
        <h3 className="ql-section-title">Model plays · ranked by edge</h3>
        <span className="ql-data-label">{board.picks.length} {board.picks.length === 1 ? "PLAY" : "PLAYS"}</span>
      </div>
      {board.picks.length > 0
        ? <div className="ql-panel overflow-hidden">{board.picks.map(prop => <TonightRow key={rowKey(prop)} prop={prop} league={league} date={slipDate} record={record.data} />)}</div>
        : <p role="status" className="ql-panel px-4 py-5 text-sm text-text-sec">No priced prop in this game has a positive edge. Passing is a result.</p>}
    </section>
    {board.leans.length > 0 && <details>
      <summary className="ql-data-label cursor-pointer py-2">PRICED, NO EDGE · {board.leans.length}</summary>
      <div className="ql-panel mt-2 overflow-hidden">{board.leans.map(prop => <TonightRow key={rowKey(prop)} prop={prop} league={league} date={slipDate} record={record.data} />)}</div>
    </details>}
    {board.research.length > 0 && <ResearchList rows={board.research} league={league} />}
  </div>;
}
