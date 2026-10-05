"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Image from "next/image";
import { Star } from "lucide-react";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { STAT_COLORS } from "@/lib/utils";
import type { ChartRecord } from "@/lib/types";
import { PlayerBarChart } from "@/components/charts/PlayerBarChart";

const STATS = [
  ["PTS", "PTS"], ["AST", "AST"], ["REB", "REB"], ["FG3M", "3PM"],
  ["BLK", "BLK"], ["STL", "STL"], ["PTS+AST", "P+A"],
  ["PTS+REB", "P+R"], ["REB+AST", "A+R"], ["PTS+REB+AST", "PRA"], ["STL+BLK", "S+B"],
] as const;
const PERIODS = [5, 10, 20] as const;
const fmt = (value: number | null | undefined) => value == null ? "—" : value.toFixed(1);
const card = "ql-panel p-4 sm:p-5";

export default function AnalysisPage() {
  return <Suspense fallback={<div className="mx-auto max-w-[1700px] p-8 text-text-sec">Loading player analysis…</div>}><AnalysisSelection /></Suspense>;
}

function AnalysisSelection() {
  const params = useSearchParams();
  const savedLeague = usePrefs(s => s.league);
  const setLeague = usePrefs(s => s.setLeague);
  const requestedLeague = params.get("league");
  const league = requestedLeague === "nba" || requestedLeague === "wnba" ? requestedLeague : savedLeague;
  useEffect(() => {
    if (requestedLeague === "nba" || requestedLeague === "wnba") setLeague(requestedLeague);
  }, [requestedLeague, setLeague]);
  const linkedPlayer = params.get("player") ?? "";
  return <PlayerAnalysis key={`${league}:${linkedPlayer}`} linkedPlayer={linkedPlayer} league={league} />;
}

function PlayerAnalysis({ linkedPlayer, league }: { linkedPlayer: string; league: "nba" | "wnba" }) {
  const { followedPlayers, followPlayer, unfollowPlayer } = usePrefs();
  const [chosenPlayer, setChosenPlayer] = useState("");
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [photoFailed, setPhotoFailed] = useState(false);
  const [stat, setStat] = useState("PTS");
  const [period, setPeriod] = useState<5 | 10 | 20 | "h2h" | "home" | "away" | "current" | "previous">(10);
  const [thresholdOverride, setThresholdOverride] = useState<number | null>(null);
  const players = useQuery({ queryKey: ["players", "all", league], queryFn: () => api.players("", league) });
  const names = players.data?.players ?? [];
  const player = chosenPlayer || linkedPlayer || (league === "wnba" && names.includes("A'ja Wilson") ? "A'ja Wilson" : names[0]) || "";
  const options = (search.trim() ? names.filter(name => name.toLowerCase().includes(search.trim().toLowerCase())) : names).slice(0, 8);
  const chart = useQuery({ queryKey: ["chart", player, stat, 200, league], queryFn: () => api.playerChart(player, stat, 200, league), enabled: !!player });
  const summary = useQuery({ queryKey: ["stats", player, league], queryFn: () => api.playerStats(player, league), enabled: !!player });
  const schedule = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league), enabled: !!player });
  const profile = summary.data;
  const nextGame = schedule.data?.games.find(game => game.home_team === profile?.team || game.away_team === profile?.team);
  const nextOpponent = nextGame
    ? nextGame.home_team === profile?.team ? nextGame.away_team : nextGame.home_team
    : null;
  const seasons = [...new Set(chart.data?.games.map(game => game.season).filter((season): season is string => !!season) ?? [])].sort();
  const currentSeason = seasons.at(-1);
  const previousSeason = seasons.at(-2);
  const games = period === "h2h"
    ? chart.data?.games.filter(game => game.opponent.split(" ").at(-1) === nextOpponent).slice(-10) ?? []
    : period === "home"
    ? chart.data?.games.filter(game => game.is_home === true || game.is_home == null && game.opponent.includes("vs.")).slice(-10) ?? []
    : period === "away"
      ? chart.data?.games.filter(game => game.is_home === false || game.is_home == null && game.opponent.includes("@")).slice(-10) ?? []
      : period === "current"
        ? chart.data?.games.filter(game => game.season === currentSeason) ?? []
        : period === "previous"
          ? chart.data?.games.filter(game => game.season === previousSeason) ?? []
          : chart.data?.games.slice(-period) ?? [];
  const valid = games.filter((game): game is ChartRecord & { value: number } => game.value != null);
  const average = valid.length ? valid.reduce((sum, game) => sum + game.value, 0) / valid.length : null;
  const recentTwenty = chart.data?.games.slice(-20).map(game => game.value).filter((value): value is number => value != null) ?? [];
  const defaultThreshold = recentTwenty.length ? Math.round(recentTwenty.reduce((sum, value) => sum + value, 0) / recentTwenty.length * 2) / 2 : 10;
  const threshold = thresholdOverride ?? Math.max(0, defaultThreshold);
  const hits = valid.filter(game => game.value >= threshold).length;
  const hitRate = valid.length ? Math.round(hits / valid.length * 100) : null;
  const chartGames = games.map(game => ({ ...game, hit: game.value == null ? null : game.value >= threshold }));
  const followed = followedPlayers.includes(player);
  const accent = STAT_COLORS[stat] ?? "#59e0c8";
  const seasonParts = stat.split("+").map(part => summary.data?.season_avgs[part]);
  const seasonAverage = seasonParts.length && seasonParts.every(value => value != null)
    ? seasonParts.reduce<number>((sum, value) => sum + (value ?? 0), 0) : null;
  const recentAverage = chart.data?.l5_avg ?? null;
  const trendDelta = seasonAverage && recentAverage != null
    ? Math.round((recentAverage - seasonAverage) / seasonAverage * 100) : null;
  const historyValues = chart.data?.games.map(game => game.value).filter((value): value is number => value != null) ?? [];
  const historyAverage = historyValues.length ? historyValues.reduce((sum, value) => sum + value, 0) / historyValues.length : null;
  const lastTen = chart.data?.games.slice(-10).filter(game => game.value != null).reverse() ?? [];

  function choosePlayer(name: string) {
    setChosenPlayer(name);
    setSearch("");
    setSearchOpen(false);
    setPhotoFailed(false);
    setThresholdOverride(null);
  }

  return <div className="ql-page mx-auto max-w-[1700px] px-4 py-6 sm:px-8">
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <p className="ql-kicker">01 / PLAYER INTELLIGENCE</p>
        <h1 className="ql-heading mt-1">Player Analysis</h1>
        <p className="ql-subtitle mt-1">Form, model outlook, and matchup context in one reading path.</p>
      </div>
      <span className="ql-data-label border border-white/10 px-3 py-2">{league.toUpperCase()} / PLAYER FILE</span>
    </div>
    <div className="ql-panel mb-4 flex flex-wrap items-center justify-between gap-5 px-4 py-4 sm:px-5">
      <div className="min-w-0 flex-1">
        {player ? <div className="flex flex-wrap items-center gap-3">
          {profile?.headshot_url && !photoFailed
            ? <Image src={profile.headshot_url} alt={player} width={64} height={64} unoptimized
                onError={() => setPhotoFailed(true)}
                className="h-[72px] w-[72px] shrink-0 rounded-md border border-[#345058] bg-[#20363a] object-cover object-top" />
            : <div className="flex h-[72px] w-[72px] shrink-0 items-center justify-center rounded-md border border-teal-400/30 bg-teal-400/10 text-xl font-bold text-teal-400" aria-hidden="true">
                {player.split(" ").map(part => part[0]).slice(0, 2).join("")}
              </div>}
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-2xl font-extrabold tracking-tight text-text-pri sm:text-3xl">{player}</h2>
              {profile?.team && <span className="ql-chip border-teal-400/30 text-teal-300">{profile.team}</span>}
              {profile?.injury_status && !["ACTIVE", "UNKNOWN"].includes(profile.injury_status.toUpperCase()) &&
                <span className="ql-chip border-amber-400/30 bg-amber-400/10 text-amber-300">
                  {profile.injury_status}{profile.injury_reason ? ` · ${profile.injury_reason}` : ""}
                </span>}
              <button aria-label={followed ? `Unfollow ${player}` : `Follow ${player}`} aria-pressed={followed}
                onClick={() => followed ? unfollowPlayer(player) : followPlayer(player)}>
                <Star size={18} color={followed ? "#f5ba64" : "#a9bec0"} fill={followed ? "#f5ba64" : "none"} />
              </button>
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-sec">
              {summary.data?.position && <span className="ql-data-label">{summary.data.position}</span>}
              {profile?.team && <span className="ql-data-label">{league.toUpperCase()}</span>}
            </div>
          </div>
          <div className="ml-auto grid w-full grid-cols-2 gap-4 border-t border-white/10 pt-3 sm:w-auto sm:grid-cols-4 sm:gap-6 sm:border-l sm:border-t-0 sm:py-1 sm:pl-6">
            {(["PTS", "REB", "AST"] as const).map(key => <div key={key} className="ql-metric"><span className="ql-data-label">{key} / GAME</span><strong className="block text-xl font-bold tabular-nums text-text-pri sm:text-2xl">{fmt(summary.data?.season_avgs[key])}</strong></div>)}
            <div className="ql-metric"><span className="ql-data-label">FG%</span><strong className="block text-xl font-bold tabular-nums text-text-pri sm:text-2xl">{fmt(profile?.fg_pct)}</strong></div>
          </div>
        </div> : <h2 className="text-xl font-extrabold text-text-pri">Choose a player</h2>}
      </div>
      <div className="relative w-full sm:w-[300px]">
        <label htmlFor="player-search" className="sr-only">Search {league.toUpperCase()} players</label>
        <input id="player-search" value={search} autoComplete="off" onFocus={() => setSearchOpen(true)}
          onChange={event => { setSearch(event.target.value); setSearchOpen(true); }}
          placeholder={`Search ${league.toUpperCase()} players…`}
          className="ql-control w-full px-4 py-3 text-sm text-text-pri outline-none placeholder:text-text-sec focus:border-teal-400" />
        {searchOpen && <div className="ql-panel absolute inset-x-0 top-full z-30 mt-1 max-h-72 overflow-y-auto shadow-xl">
          {options.length ? options.map(name => <button key={name} onClick={() => choosePlayer(name)}
            className="block w-full px-4 py-2.5 text-left text-sm text-text-pri hover:bg-white/5">{name}</button>)
            : <p className="px-4 py-3 text-sm text-text-sec">{players.isPending ? "Loading players…" : "No players found"}</p>}
        </div>}
      </div>
    </div>

    {players.isError && <p role="alert" className="mb-4 rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">Player list is unavailable. Try reloading the page.</p>}
    {!player && !players.isError && <div className={card}>Loading player data…</div>}
    {player && <>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-y border-white/10 py-3">
        <div className="flex max-w-full gap-1.5 overflow-x-auto pb-1">
          {STATS.map(([value, label]) => <button key={value} aria-pressed={stat === value}
            onClick={() => { setStat(value); setThresholdOverride(null); }}
            className={`ql-chip shrink-0 ${stat === value ? "ql-chip-active" : ""}`}>{label}</button>)}
        </div>
        <div className="flex max-w-full gap-1.5 overflow-x-auto pb-1">
          {PERIODS.map(value => <button key={value} aria-pressed={period === value} onClick={() => setPeriod(value)}
            className={`ql-chip shrink-0 ${period === value ? "ql-chip-active" : ""}`}>L{value}</button>)}
          <button aria-pressed={period === "h2h"} disabled={!nextOpponent} title={nextOpponent ? `Versus ${nextOpponent}` : "Upcoming opponent unavailable"} onClick={() => setPeriod("h2h")}
            className={`ql-chip shrink-0 disabled:opacity-40 ${period === "h2h" ? "ql-chip-active" : ""}`}>H2H</button>
          <button aria-pressed={period === "home" || period === "away"} onClick={() => setPeriod(period === "home" ? "away" : period === "away" ? 10 : "home")}
            className={`ql-chip shrink-0 ${period === "home" || period === "away" ? "ql-chip-active" : ""}`}>{period === "home" ? "Home" : period === "away" ? "Away" : "H/W"}</button>
          <button aria-pressed={period === "current"} disabled={!currentSeason} onClick={() => setPeriod("current")}
            className={`ql-chip shrink-0 disabled:opacity-40 ${period === "current" ? "ql-chip-active" : ""}`}>{currentSeason?.split("-")[0] ?? "Current"}</button>
          <button aria-pressed={period === "previous"} disabled={!previousSeason} onClick={() => setPeriod("previous")}
            className={`ql-chip shrink-0 disabled:opacity-40 ${period === "previous" ? "ql-chip-active" : ""}`}>{previousSeason?.split("-")[0] ?? "Previous"}</button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_300px] lg:items-start">
        <div className="min-w-0 space-y-4">
          <section className={card}>
            <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="ql-kicker">PERFORMANCE ANALYSIS</p>
                <h2 className="ql-section-title mt-2">{stat} · {period === "h2h" ? `vs. ${nextOpponent}` : period === "home" ? "Home" : period === "away" ? "Away" : period === "current" ? currentSeason : period === "previous" ? previousSeason : `Last ${games.length} Games`}</h2>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-sm text-text-sec"><strong className="tabular-nums text-text-pri">{hits} of {valid.length}</strong> games at {threshold}+</span>
                </div>
              </div>
              <div className="flex items-end gap-5">
                <p className="ql-metric text-right"><span className="ql-data-label">{period === "h2h" ? "H2H" : period === "home" ? "HOME" : period === "away" ? "AWAY" : period === "current" ? currentSeason : period === "previous" ? previousSeason : `L${period}`} AVG</span><strong className="block text-xl tabular-nums text-text-pri">{fmt(average)}</strong></p>
                <p className="ql-metric text-right"><span className="ql-data-label">HIT RATE</span><strong className="block text-4xl font-black tabular-nums" style={{ color: accent }}>{hitRate == null ? "—" : `${hitRate}%`}</strong></p>
              </div>
            </div>
            <div className="mb-5 flex flex-wrap items-center gap-3 border-y border-white/10 bg-white/[.025] px-4 py-3">
              <label htmlFor="threshold" className="ql-data-label">THRESHOLD</label>
              <input id="threshold" type="range" min="0" max={Math.max(50, Math.ceil(Math.max(...valid.map(game => game.value), 0) + 5))}
                step="0.5" value={threshold} onChange={event => setThresholdOverride(Number(event.target.value))}
                className="min-w-[150px] flex-1 accent-teal-400" />
              <output htmlFor="threshold" className="min-w-12 text-right font-mono text-sm font-bold tabular-nums text-teal-400">{threshold}+</output>
            </div>
            <p className="ql-data-label mb-3">LAST {games.length} GAMES / PERFORMANCE VS. ANALYSIS THRESHOLD</p>
            {chart.isPending && <div className="h-64 animate-pulse rounded-xl bg-white/5" />}
            {chart.isError && <p role="alert" className="text-sm text-red-300">Game history is unavailable. <button className="underline" onClick={() => void chart.refetch()}>Try again</button></p>}
            {!chart.isPending && chart.data && games.length === 0 && <p className="py-12 text-center text-sm text-text-sec">No completed games are available for this player.</p>}
            {games.length > 0 && <div className="overflow-x-auto"><PlayerBarChart games={chartGames} stat={stat} line={threshold} height={280} /></div>}
            {games.length > 0 && <p className="mt-3 text-xs text-text-sec">{games[0].date} to {games.at(-1)?.date} · Historical results · Threshold is for analysis, not a live sportsbook line.</p>}
          </section>

          <section className={`${card} border-l-[3px] border-l-teal-400`}>
            <p className="ql-kicker">MODEL OUTLOOK</p>
            <h2 className="ql-section-title mt-1">{profile?.projection_context ? "Next Game Prediction" : "Model Projection"}</h2>
            {summary.data?.projections && Object.keys(summary.data.projections).length > 0
              ? <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden border border-white/10 bg-white/10 sm:grid-cols-3">{Object.entries(summary.data.projections).map(([key, value]) => <div key={key} className="bg-[#152128] p-3">
                  <p className="text-2xl font-bold tabular-nums text-teal-400">{fmt(value)}</p><p className="ql-data-label mt-1">{key}</p>
                </div>)}</div>
              : <p className="mt-3 text-sm text-text-sec">No model projection is available.</p>}
            <p className="mt-4 border-t border-white/10 pt-3 text-xs text-text-sec">{summary.data?.projection_message || "Matchup context is unavailable."}</p>
          </section>

          <section className={card}>
            <p className="ql-kicker">SUPPORTING EVIDENCE</p>
            <h2 className="ql-section-title mt-1">Supporting Stats</h2>
            <p className="mb-4 mt-1 text-xs text-text-sec">Season averages vs. recent five-game form · Select a stat to update the chart</p>
            {summary.isError && <p role="alert" className="text-sm text-red-300">Player summary is unavailable.</p>}
            {summary.data && <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
              {(["PTS", "REB", "AST", "FG3M", "STL", "BLK"] as const).map(key => <button key={key} type="button" aria-pressed={stat === key} onClick={() => { setStat(key); setThresholdOverride(null); }} className={`ql-metric border bg-white/[.025] p-3 text-left hover:border-teal-400/40 ${stat === key ? "border-teal-400/50" : "border-white/10"}`}>
                <p className="ql-data-label">{key === "FG3M" ? "3PM" : key} / GAME</p>
                <p className="mt-2 text-xl font-bold tabular-nums" style={{ color: STAT_COLORS[key] ?? "#59e0c8" }}>{fmt(summary.data?.season_avgs[key])}</p>
                <p className="mt-1 text-xs tabular-nums text-text-sec">L5 {fmt(summary.data?.l5_avgs[key])}</p>
              </button>)}
            </div>}
            <div className="mt-5 border-l-2 border-teal-400 bg-teal-400/[.06] p-4">
              <h3 className="ql-data-label text-teal-300">TREND INSIGHT</h3>
              {trendDelta == null
                ? <p className="mt-2 text-sm text-text-sec">A season comparison is unavailable for this stat.</p>
                : <p className="mt-2 text-sm text-text-sec">Over the last 5 games, {player} averaged <strong className="text-text-pri">{fmt(recentAverage)} {stat}</strong> per game, {Math.abs(trendDelta)}% {trendDelta >= 0 ? "above" : "below"} the season average of {fmt(seasonAverage)}.</p>}
            </div>
          </section>
          <section className={card}>
            <p className="ql-kicker">FORM COMPARISON</p>
            <h2 className="ql-section-title mt-1">Season Trends</h2>
            <p className="mb-5 mt-1 text-xs text-text-sec">Season averages vs. the most recent 5-game average.</p>
            {summary.data ? <div className="space-y-5">
              {(["PTS", "REB", "AST", "FG3M", "STL", "BLK"] as const).map(key => {
                const season = summary.data?.season_avgs[key];
                const recent = summary.data?.l5_avgs[key];
                if (season == null || recent == null) return null;
                const maximum = Math.max(season, recent, 1);
                return <div key={key}>
                  <div className="mb-2 flex justify-between text-xs"><span className="font-semibold text-text-pri">{key === "FG3M" ? "3PM" : key}</span><span className="text-text-sec">Season {fmt(season)} · L5 {fmt(recent)}</span></div>
                  <div className="space-y-1.5" role="img" aria-label={`${key}: season ${fmt(season)}, last 5 ${fmt(recent)}`}>
                    <div className="h-2 bg-white/5"><div className="h-2 bg-slate-500" style={{ width: `${season / maximum * 100}%` }} /></div>
                    <div className="h-2 bg-white/5"><div className="h-2" style={{ width: `${recent / maximum * 100}%`, background: STAT_COLORS[key] ?? "#59e0c8" }} /></div>
                  </div>
                </div>;
              })}
              <div className="flex gap-5 text-xs text-text-sec"><span>● Season</span><span className="text-teal-400">● Last 5</span></div>
            </div> : <p className="text-sm text-text-sec">Season data is unavailable.</p>}
          </section>
        </div>

        <aside className="space-y-3 lg:sticky lg:top-24">
          <section className={card}>
            <p className="ql-kicker">NEXT MATCHUP</p>
            <h2 className="ql-section-title mt-1">Matchup</h2>
            {nextGame
              ? <p className="mt-3 text-sm text-text-sec"><strong className="text-text-pri">{profile?.team}</strong> vs. <strong className="text-text-pri">{nextOpponent}</strong>{nextGame.game_time ? ` · ${nextGame.game_time}` : ""}</p>
              : <p className="mt-3 text-sm text-text-sec">The next opponent has not been established for this player.</p>}
          </section>
          <section className={card}>
            <p className="ql-kicker">AVAILABILITY</p>
            <h2 className="ql-section-title mt-1">Injury Context</h2>
            {profile?.injury_status
              ? <p className="mt-3 text-sm text-text-sec">{profile.injury_status}{profile.injury_reason ? ` · ${profile.injury_reason}` : ""}</p>
              : <p className="mt-3 text-sm text-text-sec">No verified injury update is available.</p>}
          </section>
          <section className={card}>
            <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
              <div><p className="ql-kicker">GAME LOG</p><h2 className="ql-section-title mt-1">Bet History</h2></div>
              <span className="ql-data-label">AVG {fmt(historyAverage)} {stat}</span>
            </div>
            {lastTen.length ? <div className="divide-y divide-white/10">{lastTen.map((game, index) => {
              const over = game.value != null && historyAverage != null && game.value >= historyAverage;
              return <div key={`${game.date}-${index}`} className="flex items-center justify-between gap-3 py-2.5 text-xs">
                <span className="font-mono text-text-sec">{game.date.slice(5)} · {game.opponent.split(" ").at(-1)}</span>
                <span className="flex items-center gap-2"><strong className={`min-w-7 text-right font-mono tabular-nums ${over ? "text-teal-400" : "text-rose-400"}`}>{fmt(game.value)}</strong>
                  <span className={`rounded border px-1.5 py-0.5 text-[10px] font-bold ${over ? "border-teal-400/25 bg-teal-400/10 text-teal-400" : "border-rose-400/25 bg-rose-400/10 text-rose-400"}`}>{over ? "OVER" : "UNDER"}</span>
                </span>
              </div>;
            })}</div> : <p className="text-sm text-text-sec">No game history available.</p>}
            {lastTen.length > 0 && <p className="mt-3 text-[11px] text-text-sec">Over/under compares each result with the available game-history average; it does not represent placed bets.</p>}
          </section>
          <p className="px-1 text-xs text-text-sec">History through {summary.data?.history_through ?? "—"}{summary.data?.history_age_days != null ? ` · ${summary.data.history_age_days} days ago` : ""}</p>
        </aside>
      </div>
    </>}
  </div>;
}
