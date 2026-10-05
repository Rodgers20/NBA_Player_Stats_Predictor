"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { GamesResponse, GamePrediction } from "@/lib/types";
import { usePrefs } from "@/store/prefs";
import { fmtOdds } from "@/lib/utils";

export default function GamesPage() {
  const league = usePrefs(s => s.league);
  const queryClient = useQueryClient();
  const [selectedMatchup, setSelectedMatchup] = useState("");
  const isStatic = process.env.NEXT_PUBLIC_DATA_MODE === "static";
  const refreshLines = useMutation({ mutationFn: api.refreshGameLines, onSuccess: async (_result, refreshedLeague) => {
    await Promise.all([queryClient.invalidateQueries({ queryKey: ["games", refreshedLeague] }), queryClient.invalidateQueries({ queryKey: ["game-predictions", refreshedLeague] })]);
  } });
  const { data, isPending: isLoading, error, refetch } = useQuery<GamesResponse>({
    queryKey: ["games", league],
    queryFn: () => api.games(league),
    refetchInterval: 300_000,
  });

  const { data: predictions, isPending: predictionsLoading, isError: predictionsError, refetch: retryPredictions } = useQuery({
    queryKey: ["game-predictions", league],
    queryFn: () => api.predictions(league),
    staleTime: 300_000,
  });
  const selectedGame = data?.games.find(game => game.matchup === selectedMatchup) ?? data?.games[0];

  return (
    <main className="ql-page px-4 sm:px-8 pt-8 max-w-[1440px] mx-auto pb-12">
      <header className="mb-7 flex flex-wrap items-end justify-between gap-4">
        <div><p className="ql-kicker">02 / {league.toUpperCase()} SLATE ROOM</p>
        <h1 className="ql-heading">Today&apos;s Games</h1>
        {data?.target_date && (
          <p className="ql-subtitle mt-1">{data.target_date} · Choose a matchup for the full model and market breakdown.</p>
        )}</div>
        <div className="flex flex-wrap items-center gap-3">
          {data && <span className="ql-chip font-mono text-xs tabular-nums">{data.games.length} {data.games.length === 1 ? "MATCHUP" : "MATCHUPS"}</span>}
          {!isStatic && <button type="button" disabled={refreshLines.isPending} onClick={() => refreshLines.mutate(league)} className="ql-control min-h-10 px-3 text-xs font-semibold disabled:opacity-40">{refreshLines.isPending ? "Refreshing lines…" : "Refresh ESPN lines"}</button>}
        </div>
      </header>

      {(refreshLines.data || refreshLines.error) && <p role="status" className="mb-3 text-xs text-text-sec">{refreshLines.data?.message || refreshLines.error?.message}</p>}
      {predictionsError && <p role="alert" className="glass p-4 text-sm text-danger mb-4">Game predictions are unavailable. <button onClick={() => void retryPredictions()} className="underline">Try again</button></p>}
      {predictions?.message && <p role="status" className="text-sm text-text-sec mb-4">{predictions.message}</p>}
      {predictions?.target_date && data?.target_date && predictions.target_date !== data.target_date && <p role="status" className="text-sm text-amber-300 mb-4">Predictions are for {predictions.target_date}; they are not shown on this schedule.</p>}

      {isLoading && (
        <div className="grid md:grid-cols-3 gap-3" aria-label="Loading games">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="ql-panel h-28 animate-pulse" />
          ))}
        </div>
      )}

      {error && (
        <div className="ql-panel p-4 text-center text-sm text-[#fa837b]">
          <p>We couldn’t load the game schedule.</p>
          <button className="mt-3 underline" onClick={() => void refetch()}>Try again</button>
        </div>
      )}

      {!isLoading && data?.games?.length === 0 && (
        <div className="ql-panel py-14 text-center">
          <h2 className="text-lg font-semibold text-[#eaf8f6]">No games scheduled</h2>
          <p className="mt-2 text-sm text-[#a9bec0]">No games are available in the current data snapshot.</p>
        </div>
      )}

      {!isLoading && data && selectedGame && (
        <div>
          <div className="mb-3 flex items-center justify-between gap-3"><h2 className="ql-section-title">Select matchup</h2><span className="ql-data-label">MODEL / MARKET SNAPSHOT</span></div>
          <div className="grid gap-3 pb-4 mb-3 sm:grid-cols-2 xl:grid-cols-3" role="group" aria-label="Select game">
            {data.games.map((game) => {
              const prediction = predictions?.target_date === data.target_date ? predictions.predictions.find(p => p.matchup === game.matchup) : undefined;
              const active = selectedGame.matchup === game.matchup;
              return <button type="button" key={game.matchup} onClick={() => setSelectedMatchup(game.matchup)} aria-pressed={active} className={`ql-panel group min-h-32 p-4 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#59e0c8] ${active ? "border-[#59e0c8] bg-[#20363a]" : "hover:border-[#59e0c8]/60"}`}>
                <span className="flex items-center justify-between gap-2"><span className="ql-data-label">{game.status_text || game.game_time || "Scheduled"}</span><span className="text-sm text-[#59e0c8]" aria-hidden="true">{active ? "●" : "↗"}</span></span>
                <span className="mt-3 block text-xl font-extrabold tracking-tight text-text-pri">{game.away_team} <span className="text-sm font-medium text-text-sec">at</span> {game.home_team}</span>
                <span className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-white/10 pt-2 text-xs"><span className="text-text-sec">{prediction?.predicted_away_score == null || prediction.predicted_home_score == null ? "Projection pending" : `Projected ${Math.round(prediction.predicted_away_score)}–${Math.round(prediction.predicted_home_score)}`}</span><span className={prediction?.predicted_winner ? "font-semibold text-[#59e0c8]" : "text-text-sec"}>{prediction?.predicted_winner ? `${prediction.predicted_winner} favored` : "Model pending"}</span></span>
              </button>;
            })}
          </div>
          <GameCard key={selectedGame.matchup} game={selectedGame}
            prediction={predictions?.target_date === data.target_date ? predictions.predictions.find(p => p.matchup === selectedGame.matchup) : undefined}
              predictionStatus={predictionsLoading ? "Loading model estimate…" : predictions?.errors.find(e => e.matchup === selectedGame.matchup)?.message || "Model estimate unavailable"}
          />
        </div>
      )}
    </main>
  );
}

function GameCard({ game, prediction, predictionStatus }: { game: GamesResponse["games"][0]; prediction?: GamePrediction; predictionStatus: string }) {
  const homeLine = game.spread;
  const spreadLine = prediction?.spread_pick === "HOME" ? homeLine : prediction?.spread_pick === "AWAY" && homeLine != null ? -homeLine : null;
  const spreadCall = prediction?.spread_team && spreadLine != null
    ? `${prediction.spread_team} ${spreadLine > 0 ? "+" : ""}${spreadLine.toFixed(1)}`
    : game.spread == null ? "No market line" : "No model edge";
  const totalCall = prediction?.total_pick && game.total != null
    ? `${prediction.total_pick === "OVER" ? "Over" : "Under"} ${game.total.toFixed(1)}`
    : game.total == null ? "No market line" : "No model edge";
  return (
    <section className="flex flex-col gap-3" aria-label={`${game.away_team} at ${game.home_team} matchup analysis`}>
      {/* Teams */}
      <div className="ql-panel overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-5 py-3 sm:px-6"><span className="ql-kicker">MATCHUP / MODEL SCOREBOARD</span><span className="ql-data-label">{game.status_text || game.game_time || "Scheduled"}</span></div>
        <div className="grid grid-cols-[minmax(0,1fr)_30px_minmax(0,1fr)] items-center gap-2 px-4 py-6 text-center sm:grid-cols-[minmax(0,1fr)_56px_minmax(0,1fr)] sm:px-8">
        <div className="min-w-0">
          <span className="ql-data-label">AWAY{game.away_wins != null && game.away_losses != null ? ` · ${game.away_wins}–${game.away_losses}` : ""}</span>
          <span className="mt-2 block truncate text-4xl font-black tracking-tight text-text-pri sm:text-6xl">{game.away_team}</span>
          <span className="mt-1 block truncate text-xs text-text-sec">{game.away_name || "Away"}</span>
          <span className="mt-3 block font-mono text-2xl font-bold tabular-nums text-[#59e0c8]">{prediction?.predicted_away_score == null ? "—" : Math.round(prediction.predicted_away_score)}</span>
        </div>
        <span className="text-xs font-bold uppercase tracking-widest text-text-sec">at</span>
        <div className="min-w-0">
          <span className="ql-data-label">HOME{game.home_wins != null && game.home_losses != null ? ` · ${game.home_wins}–${game.home_losses}` : ""}</span>
          <span className="mt-2 block truncate text-4xl font-black tracking-tight text-text-pri sm:text-6xl">{game.home_team}</span>
          <span className="mt-1 block truncate text-xs text-text-sec">{game.home_name || "Home"}</span>
          <span className="mt-3 block font-mono text-2xl font-bold tabular-nums text-[#59e0c8]">{prediction?.predicted_home_score == null ? "—" : Math.round(prediction.predicted_home_score)}</span>
        </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/10 px-5 py-3 text-xs sm:px-6"><span className="text-text-sec">Projected score · Model estimate</span><span className="font-mono tabular-nums text-[#59e0c8]">{prediction?.predicted_away_score != null && prediction.predicted_home_score != null ? `${game.away_team} ${Math.round(prediction.predicted_away_score)} – ${Math.round(prediction.predicted_home_score)} ${game.home_team}` : predictionStatus}</span></div>
      </div>

      <div>
        <p className="ql-section-title mb-3">Model decisions</p>
        {prediction ? <>
          <div className="grid gap-3 md:grid-cols-3">
            <PickPanel label="Projected winner" value={prediction.predicted_winner} detail={`Home margin ${prediction.spread == null ? "—" : `${prediction.spread > 0 ? "+" : ""}${prediction.spread.toFixed(1)}`} · ${prediction.confidence.toLowerCase()} confidence`} reason={prediction.winner_reason} />
            <PickPanel label="Spread cover" value={spreadCall} detail={prediction.spread == null ? "Model spread unavailable" : `Model home margin ${prediction.spread > 0 ? "+" : ""}${prediction.spread.toFixed(1)}${game.spread != null ? ` · market ${game.home_team} ${game.spread > 0 ? "+" : ""}${game.spread.toFixed(1)}` : ""}`} reason={prediction.spread_reason} />
            <PickPanel label="Over / under" value={totalCall} detail={`Model total ${prediction.total?.toFixed(1) ?? "—"}${game.total != null ? ` · market ${game.total.toFixed(1)}` : ""}`} reason={prediction.total_reason} />
          </div>
          {prediction.intel && prediction.intel.length > 0 && <div className="ql-panel mt-3 p-5"><p className="ql-kicker">MODEL CONTEXT</p><ul className="mt-3 grid gap-2 text-sm text-text-sec sm:grid-cols-2">{prediction.intel.slice(0, 4).map((item, index) => <li key={`${index}-${item}`} className="border-l-2 border-[#59e0c8]/40 pl-3">{item}</li>)}</ul></div>}
        </> : <p role="status" className="text-xs text-text-sec">{predictionStatus}</p>}
      </div>
      {/* Odds */}
      <div className="ql-panel p-5 sm:p-6"><div className="mb-4 flex flex-wrap items-center justify-between gap-2"><h2 className="ql-section-title">Market & availability</h2><span className="ql-data-label text-amber-300">{game.spread != null || game.total != null || game.home_ml != null || game.away_ml != null ? "AVAILABLE QUOTES" : "MARKET LINES UNAVAILABLE"}</span></div>
      {(game.spread != null || game.total != null || game.home_ml != null || game.away_ml != null) ? (
        <>
          <div className="grid gap-2 sm:grid-cols-3">
            <OddsPill label="Home spread" value={game.spread != null ? (game.spread > 0 ? `+${game.spread}` : String(game.spread)) : "—"} />
            <OddsPill label="Total" value={game.total != null ? `O/U ${game.total}` : "—"} />
            <OddsPill label="ML · Away / Home" value={`${fmtOdds(game.away_ml)} / ${fmtOdds(game.home_ml)}`} />
          </div>
          {(game.odds_source || game.odds_provider || game.odds_updated_at) && <p className="mt-3 text-xs text-text-sec">Market quote · {[game.odds_source, game.odds_provider, game.odds_updated_at && `checked ${formatOddsUpdatedAt(game.odds_updated_at)}`].filter(Boolean).join(" · ")}</p>}
        </>
      ) : <p className="text-sm text-text-sec">Market odds are unavailable; cover and over/under picks need a line.</p>}<p className="mt-4 text-xs text-text-sec">Market-dependent picks appear only when ESPN supplies a line for this matchup.</p></div>
      <div className="grid gap-3 text-xs text-text-sec sm:grid-cols-2">
        <Injuries team={game.away_team} injuries={game.away_injuries ?? []} />
        <Injuries team={game.home_team} injuries={game.home_injuries ?? []} />
      </div>
    </section>
  );
}

function PickPanel({ label, value, detail, reason }: { label: string; value: string; detail: string; reason?: string | null }) {
  const marketContext = label === "Over / under";
  return <div className={`ql-panel flex min-h-44 flex-col border-t-2 p-5 ${marketContext ? "border-t-amber-400" : "border-t-[#59e0c8]"}`}>
    <p className="ql-data-label">{label}</p><p className="mt-5 text-2xl font-extrabold leading-tight tracking-tight text-text-pri">{value}</p>
    <p className={`mt-auto pt-4 font-mono text-xs tabular-nums ${marketContext ? "text-amber-300" : "text-[#59e0c8]"}`}>{detail}</p>{reason && <p className="mt-3 border-t border-white/10 pt-3 text-xs leading-relaxed text-text-sec">{reason}</p>}
  </div>;
}

function Injuries({ team, injuries }: { team: string; injuries: NonNullable<GamesResponse["games"][0]["home_injuries"]> }) {
  return <section className="ql-panel p-5"><h2 className="ql-section-title">{team} injury report</h2>{injuries.length ? <ul className="mt-3 space-y-2 text-xs text-text-sec">{injuries.slice(0, 4).map((item, index) => <li key={`${item.name}-${index}`} className="flex flex-wrap justify-between gap-x-3 border-t border-white/10 pt-2"><span className="font-semibold text-text-pri">{item.name}</span><span>{item.status}</span></li>)}</ul> : <p className="mt-3 text-xs text-text-sec">No injury report available.</p>}</section>;
}

function OddsPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="ql-metric flex min-h-20 flex-col justify-between gap-2 p-3">
      <span className="ql-data-label">{label}</span>
      <span className="font-mono text-lg font-bold tabular-nums text-amber-300">{value}</span>
    </div>
  );
}

function formatOddsUpdatedAt(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : `${parsed.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}
