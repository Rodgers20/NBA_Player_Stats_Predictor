"use client";

import { useQuery } from "@tanstack/react-query";
import { Check, Plus } from "lucide-react";
import { api } from "@/lib/api";
import { gameState } from "@/lib/game-board.mjs";
import { gameEdges } from "@/lib/game-edges.mjs";
import { fmtLine, fmtPrice } from "@/lib/game-markets.mjs";
import { ASSUMED_PRICE } from "@/lib/team-legs.mjs";
import { GameEdges } from "@/components/tonight/GameEdges";
import { useGameSlip } from "@/components/games/useGameSlip";
import type { Game, GamePrediction } from "@/lib/types";
import type { LineMarket } from "./PropFilters";

type LinePick = {
  game: Game; prediction: GamePrediction; side: "home" | "away" | "over" | "under";
  title: string; detail: string; gap: number; confidence?: string | null; reason?: string | null;
};

const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}`;

/** The model's spread cover pick against today's home handicap. `gap` is how far the model's margin sits from the market's. */
function spreadPick(game: Game, prediction: GamePrediction): LinePick | null {
  if (!prediction.spread_pick || !prediction.spread_team || game.spread == null || prediction.spread == null) return null;
  const home = prediction.spread_pick === "HOME";
  return {
    game, prediction, side: home ? "home" : "away",
    title: `${prediction.spread_team} ${fmtLine(home ? game.spread : -game.spread)}`,
    detail: `Model home margin ${signed(prediction.spread)} · market home line ${fmtLine(game.spread)}`,
    gap: Math.abs(prediction.spread + game.spread), confidence: prediction.spread_confidence, reason: prediction.spread_reason,
  };
}

function totalPick(game: Game, prediction: GamePrediction): LinePick | null {
  if (!prediction.total_pick || game.total == null || prediction.total == null) return null;
  const over = prediction.total_pick === "OVER";
  return {
    game, prediction, side: over ? "over" : "under",
    title: `${over ? "Over" : "Under"} ${game.total}`,
    detail: `Model total ${prediction.total.toFixed(1)} · market total ${game.total}`,
    gap: Math.abs(prediction.total - game.total), confidence: prediction.total_confidence, reason: prediction.total_reason,
  };
}

function LineRow({ pick, market, league, date }: { pick: LinePick; market: "SPREAD" | "TOTAL"; league: "nba" | "wnba"; date: string }) {
  const slip = useGameSlip(pick.game, league, date || undefined, pick.prediction);
  const add = market === "SPREAD" ? (pick.side === "home" ? slip.spread.home : slip.spread.away) : (pick.side === "over" ? slip.total.over : slip.total.under);
  return (
    <li className="flex items-start gap-3 border-b border-white/[0.07] px-4 py-3 last:border-b-0">
      <button type="button" disabled={!add} aria-pressed={add?.inSlip ?? false} onClick={add?.onToggle}
        aria-label={add ? `${add.inSlip ? "Remove" : "Add"} ${add.label} ${add.inSlip ? "from" : "to"} slip` : "No line to add"}
        className={`mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-full border transition-colors disabled:opacity-30 ${add?.inSlip ? "border-[#59e0c8] bg-[#59e0c8] text-[#0d151b]" : "border-[#59e0c8]/45 text-[#59e0c8] hover:bg-[#59e0c8]/10"}`}>
        {add?.inSlip ? <Check size={16} aria-hidden="true" /> : <Plus size={16} aria-hidden="true" />}
      </button>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-bold text-[#eaf8f6]">{pick.title} <span className="ml-1 font-mono text-[#f5ba64]" title="ESPN publishes no spread or total price, so the standard price is assumed">{fmtPrice(ASSUMED_PRICE)}*</span></p>
        <p className="mt-0.5 text-[11px] text-[#a9bec0]">{pick.game.matchup} · {pick.detail}</p>
        {pick.reason && <p className="mt-1 text-[11px] leading-snug text-[#a9bec0]">{pick.reason}</p>}
      </div>
      <p className="shrink-0 text-right font-display text-2xl font-bold tabular-nums text-[#59e0c8]">{pick.gap.toFixed(1)}
        <span className="block font-mono text-[10px] font-medium text-[#a9bec0]">PTS FROM MARKET{pick.confidence ? ` · ${pick.confidence.toUpperCase()}` : ""}</span></p>
    </li>
  );
}

/** Whole-game lines for the slate: moneyline value, or the game model's spread and total calls against the posted number. */
export function GameLinesBoard({ league, market }: { league: "nba" | "wnba"; market: LineMarket }) {
  const games = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league) });
  const predictions = useQuery({ queryKey: ["game-predictions", league], queryFn: () => api.predictions(league), staleTime: 300_000 });

  if (games.isPending) return <div className="ql-panel h-40 animate-pulse" aria-busy="true" aria-label="Loading game lines" />;
  if (games.isError) return <div role="alert" className="ql-panel p-6 text-center text-sm text-[#fa837b]">Game lines could not be loaded. <button type="button" className="underline" onClick={() => void games.refetch()}>Try again</button></div>;

  const date = games.data.target_date ?? "";
  const slate = games.data.games.filter(game => gameState(game) === "pre");
  // Only use model calls published for the same slate as the lines.
  const forecasts = predictions.data && predictions.data.target_date === games.data.target_date ? predictions.data.predictions : [];

  if (market === "ML") {
    const edges = gameEdges(slate, forecasts);
    return edges.length > 0
      ? <GameEdges edges={edges} league={league} date={date} />
      : <p role="status" className="ql-panel px-6 py-10 text-center text-sm text-[#a9bec0]">{predictions.isPending ? "Loading the game model…" : "No game has both a moneyline price and a model win chance right now."}</p>;
  }

  const build = market === "SPREAD" ? spreadPick : totalPick;
  const picks = slate.flatMap(game => {
    const prediction = forecasts.find(item => item.matchup === game.matchup);
    const pick = prediction ? build(game, prediction) : null;
    return pick ? [pick] : [];
  }).sort((a, b) => b.gap - a.gap);
  const label = market === "SPREAD" ? "Spread" : "Totals";

  return (
    <section aria-label={`${label} picks`} className="mb-6">
      <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
        <h2 className="ql-section-title">{label}</h2>
        <span className="ql-data-label">{picks.length} {picks.length === 1 ? "PICK" : "PICKS"} · RANKED BY DISTANCE FROM THE MARKET</span>
      </div>
      {picks.length > 0
        ? <ul className="ql-panel overflow-hidden">{picks.map(pick => <LineRow key={pick.game.matchup} pick={pick} market={market} league={league} date={date} />)}</ul>
        : <p role="status" className="ql-panel px-6 py-10 text-center text-sm text-[#a9bec0]">{predictions.isPending ? "Loading the game model…" : `No game has a model ${market === "SPREAD" ? "spread" : "total"} pick against a posted line right now. Passing is a result.`}</p>}
      <p className="mt-2 text-[11px] leading-snug text-[#a9bec0]">* ESPN publishes spread and total lines but no price, so {fmtPrice(ASSUMED_PRICE)} is assumed. Check your book before betting. The gap is the model&apos;s number against the market&apos;s, an estimate, not a guarantee.</p>
    </section>
  );
}
