"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Image from "next/image";
import { Percent, Star } from "lucide-react";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { STAT_COLORS } from "@/lib/utils";
import { formatOdds } from "@/lib/prop-research.mjs";
import type { ChartRecord } from "@/lib/types";
import { PlayerBarChart } from "@/components/charts/PlayerBarChart";
import { WindowChips } from "@/components/analysis/WindowChips";
import { GameLogTable } from "@/components/analysis/GameLogTable";
import { gradeSide, hitSummary } from "@/lib/hit-rate.mjs";
import { selectWindow, windowCounts, type WindowKey } from "@/lib/analysis-windows.mjs";
import { analysisViewToSearch, parseAnalysisView, type AnalysisView } from "@/lib/analysis-url.mjs";
import { PlayerPropLines } from "@/components/props/PlayerPropLines";
import { MatchupDefense } from "@/components/analysis/MatchupDefense";
import { PlayerResearch } from "@/components/analysis/PlayerResearch";
import { GameFilterPanel } from "@/components/analysis/GameFilters";
import { DEFAULT_GAME_FILTERS, applyGameFilters, type GameFilters } from "@/lib/game-filters.mjs";
import { MinutesBars } from "@/components/analysis/MinutesBars";
import { scopeLabel, streakInsight } from "@/lib/insights.mjs";
import { tipLabel } from "@/components/games/MatchupCards";

export const STATS = [
  ["PTS", "PTS"], ["AST", "AST"], ["REB", "REB"], ["FG3M", "3PM"],
  ["BLK", "BLK"], ["STL", "STL"], ["PTS+AST", "P+A"],
  ["PTS+REB", "P+R"], ["REB+AST", "A+R"], ["PTS+REB+AST", "PRA"], ["STL+BLK", "S+B"],
] as const;
const STAT_NAMES: Record<string, string> = {
  PTS: "Points", AST: "Assists", REB: "Rebounds", FG3M: "3-Pointers", BLK: "Blocks", STL: "Steals",
  "PTS+AST": "Points + Assists", "PTS+REB": "Points + Rebounds", "REB+AST": "Rebounds + Assists",
  "PTS+REB+AST": "Pts + Reb + Ast", "STL+BLK": "Steals + Blocks",
};
const DEFAULT_VIEW: AnalysisView = { stat: "PTS", period: 10, line: null };
const fmt = (value: number | null | undefined) => value == null ? "—" : value.toFixed(1);
const card = "ql-panel p-4 sm:p-5";
// Our teal and gold washed over the site's card colour, in place of Outlier's team-colour banner.
const HERO_BG = "radial-gradient(120% 140% at 0% 0%, rgba(89,224,200,.22) 0%, rgba(21,33,40,0) 55%), radial-gradient(120% 140% at 100% 0%, rgba(245,186,100,.14) 0%, rgba(21,33,40,0) 50%), var(--color-bg-card)";

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** The prop the analysis was opened from: its line and side become the starting threshold. */
export interface AnalysisFocus {
  stat: string;
  line: number | null;
  side: "Over" | "Under";
  price?: number | null;
  book?: string | null;
  matchup?: string | null;
}

interface Props {
  linkedPlayer: string;
  league: "nba" | "wnba";
  focus?: AnalysisFocus;
  /** Inside the props sheet: no page title, no player search, and the URL belongs to the board underneath. */
  embedded?: boolean;
}

export function PlayerAnalysis({ linkedPlayer, league, focus, embedded = false }: Props) {
  const { followedPlayers, followPlayer, unfollowPlayer } = usePrefs();
  const [chosenPlayer, setChosenPlayer] = useState("");
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchBox = useRef<HTMLDivElement>(null);
  // A click anywhere outside the search (or Escape) closes the suggestions.
  useEffect(() => {
    if (!searchOpen) return;
    const away = (event: PointerEvent) => { if (!searchBox.current?.contains(event.target as Node)) setSearchOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setSearchOpen(false); };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", escape); };
  }, [searchOpen]);
  const [photoFailed, setPhotoFailed] = useState(false);
  const params = useSearchParams();
  // On the page, stat, window and line come from the URL so a view can be shared; in the sheet they come from the prop.
  const initial = embedded ? { ...DEFAULT_VIEW, stat: focus?.stat ?? "PTS", line: focus?.line ?? null } : parseAnalysisView(params.toString(), DEFAULT_VIEW);
  const [stat, setStat] = useState(initial.stat);
  const [periodChoice, setPeriod] = useState<WindowKey>(initial.period);
  const [side, setSide] = useState<"Over" | "Under">(focus?.side ?? "Over");
  const syncTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(syncTimer.current), []);
  const [thresholdOverride, setThresholdOverride] = useState<number | null>(initial.line);
  const [gameFilters, setGameFilters] = useState<GameFilters>(DEFAULT_GAME_FILTERS);
  const [sidePanel, setSidePanel] = useState<"matchup" | "availability" | "log">("matchup");
  const players = useQuery({ queryKey: ["players", "all", league], queryFn: () => api.players("", league), enabled: !embedded || !linkedPlayer });
  const names = players.data?.players ?? [];
  const player = chosenPlayer || linkedPlayer || (league === "wnba" && names.includes("A'ja Wilson") ? "A'ja Wilson" : names[0]) || "";
  const options = (search.trim() ? names.filter(name => name.toLowerCase().includes(search.trim().toLowerCase())) : names).slice(0, 8);
  const chart = useQuery({ queryKey: ["chart", player, stat, 200, league], queryFn: () => api.playerChart(player, stat, 200, league), enabled: !!player });
  const summary = useQuery({ queryKey: ["stats", player, league], queryFn: () => api.playerStats(player, league), enabled: !!player });
  const schedule = useQuery({ queryKey: ["games", league], queryFn: () => api.games(league), enabled: !!player });
  const profile = summary.data;
  const team = /^(|nan|none|null)$/i.test(profile?.team?.trim() ?? "") ? "" : profile?.team ?? "";
  const nextGame = schedule.data?.games.find(game => team && (game.home_team === team || game.away_team === team));
  const nextOpponent = nextGame
    ? nextGame.home_team === team ? nextGame.away_team : nextGame.home_team
    : null;
  const seasons = [...new Set(chart.data?.games.map(game => game.season).filter((season): season is string => !!season) ?? [])].sort();
  const currentSeason = seasons.at(-1);
  const previousSeason = seasons.at(-2);
  // A shared link can ask for a window this player cannot have (no next opponent, no previous season): fall back instead of showing an empty page.
  const period: WindowKey = (periodChoice === "h2h" && schedule.isFetched && !nextOpponent)
    || (periodChoice === "previous" && chart.isFetched && !previousSeason)
    || (periodChoice === "current" && chart.isFetched && !currentSeason) ? 10 : periodChoice;
  const windowContext = { nextOpponent, currentSeason, previousSeason };
  // Game filters narrow the sample first, so "L10" means the last 10 games that match them.
  const allGames = chart.data?.games ?? [];
  const filteredGames = applyGameFilters(allGames, gameFilters);
  const games = chart.data ? selectWindow(filteredGames, period, windowContext) : [];
  const valid = games.filter((game): game is ChartRecord & { value: number } => game.value != null);
  const average = valid.length ? valid.reduce((sum, game) => sum + game.value, 0) / valid.length : null;
  const middle = median(valid.map(game => game.value));
  const recentTwenty = chart.data?.games.slice(-20).map(game => game.value).filter((value): value is number => value != null) ?? [];
  const defaultThreshold = recentTwenty.length ? Math.round(recentTwenty.reduce((sum, value) => sum + value, 0) / recentTwenty.length * 2) / 2 : 10;
  const threshold = thresholdOverride ?? Math.max(0, defaultThreshold);
  // A game exactly on the line is a push and leaves the denominator.
  const tally = hitSummary(valid.map(game => game.value), threshold, side);
  const decided = tally.hits + tally.misses;
  const hitRate = tally.rate;
  const insight = player ? streakInsight({ player, stat, line: threshold, games: filteredGames, scope: scopeLabel(gameFilters), side }) : null;
  const grade = (value: number | null) => (value == null ? null : { hit: true, miss: false, push: null }[gradeSide(value, threshold, side)]);
  const chartGames = games.map(game => ({ ...game, hit: grade(game.value) }));
  const counts = windowCounts(filteredGames, threshold, windowContext, side);
  const windowLabels: Record<string, string> = {
    5: "L5", 10: "L10", 20: "L20", h2h: "H2H", home: "Home", away: "Away",
    current: currentSeason?.split("-")[0] ?? "Season", previous: previousSeason?.split("-")[0] ?? "Prev",
  };
  const windowTitle = period === "h2h" ? `vs ${nextOpponent}` : period === "home" ? "Home games" : period === "away" ? "Away games"
    : period === "current" ? `${currentSeason} season` : period === "previous" ? `${previousSeason} season` : `Last ${games.length}`;
  const followed = followedPlayers.includes(player);
  const accent = STAT_COLORS[stat] ?? "#59e0c8";
  const rateColor = hitRate == null ? "#a9bec0" : hitRate >= 60 ? "var(--color-over)" : hitRate <= 40 ? "var(--color-under)" : "#f5ba64";
  const seasonParts = stat.split("+").map(part => summary.data?.season_avgs[part]);
  const seasonAverage = seasonParts.length && seasonParts.every(value => value != null)
    ? seasonParts.reduce<number>((sum, value) => sum + (value ?? 0), 0) : null;
  const recentAverage = chart.data?.l5_avg ?? null;
  const trendDelta = seasonAverage && recentAverage != null
    ? Math.round((recentAverage - seasonAverage) / seasonAverage * 100) : null;
  const historyValues = chart.data?.games.map(game => game.value).filter((value): value is number => value != null) ?? [];
  const historyAverage = historyValues.length ? historyValues.reduce((sum, value) => sum + value, 0) / historyValues.length : null;
  const lastTen = chart.data?.games.slice(-10).filter(game => game.value != null).reverse() ?? [];
  const onFocusLine = focus && stat === focus.stat && thresholdOverride === focus.line;
  const matchup = nextGame?.matchup ?? focus?.matchup ?? null;

  function sync(next: Partial<AnalysisView>) {
    if (embedded) return;
    const view = { stat, period, line: thresholdOverride, ...next };
    const url = new URL(window.location.href);
    for (const key of ["stat", "period", "line"]) url.searchParams.delete(key);
    new URLSearchParams(analysisViewToSearch(view, DEFAULT_VIEW)).forEach((value, key) => url.searchParams.set(key, value));
    window.history.replaceState(window.history.state, "", url);
  }
  function selectStat(value: string) {
    // Back on the prop's own market, its line returns; any other stat starts from the player's recent average.
    const line = focus && value === focus.stat ? focus.line : null;
    setStat(value); setThresholdOverride(line); sync({ stat: value, line });
  }
  function selectPeriod(value: WindowKey) { setPeriod(value); sync({ period: value }); }
  function changeThreshold(value: number) {
    setThresholdOverride(value);
    clearTimeout(syncTimer.current); // a slider drag fires many changes; write the URL once it settles
    syncTimer.current = setTimeout(() => sync({ line: value }), 200);
  }

  function choosePlayer(name: string) {
    const url = new URL(window.location.href);
    url.searchParams.set("player", name);
    url.searchParams.delete("line"); // the old player's line does not carry over
    window.history.replaceState(null, "", url);
    setChosenPlayer(name);
    setSearch("");
    setSearchOpen(false);
    setPhotoFailed(false);
    setThresholdOverride(null);
    setGameFilters(DEFAULT_GAME_FILTERS); // teammates, minutes and margins belong to the old player
  }

  return <div className={embedded ? "mx-auto max-w-[1500px] px-3 pb-16 pt-3 sm:px-6" : "ql-page ql-enter mx-auto max-w-[1700px] px-4 py-6 sm:px-8"}>
    {!embedded && <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <p className="ql-kicker">01 / PLAYER INTELLIGENCE</p>
        <h1 className="ql-heading mt-1">Player Analysis</h1>
        <p className="ql-subtitle mt-1">Form, model outlook, and matchup context in one reading path.</p>
      </div>
      <div ref={searchBox} className="relative w-full sm:w-[300px]">
        <label htmlFor="player-search" className="sr-only">Search {league.toUpperCase()} players</label>
        <input id="player-search" value={search} autoComplete="off" onFocus={() => setSearchOpen(true)}
          onChange={event => { setSearch(event.target.value); setSearchOpen(true); }}
          placeholder={`Search ${league.toUpperCase()} players…`}
          className="ql-control w-full px-4 py-3 text-sm text-text-pri outline-none placeholder:text-text-sec focus:border-teal-400" />
        {searchOpen && <div className="ql-panel absolute inset-x-0 top-full z-30 mt-1 max-h-80 overflow-y-auto shadow-xl">
          {options.length ? options.map(name => {
            const photo = players.data?.headshots?.[name];
            return <button key={name} onClick={() => choosePlayer(name)}
              className="flex w-full items-center gap-3 px-3 py-2 text-left text-sm text-text-pri hover:bg-white/5">
              {photo
                // eslint-disable-next-line @next/next/no-img-element -- small static headshot; next/image adds nothing here
                ? <img src={photo} alt="" width={32} height={32} loading="lazy" className="h-8 w-8 shrink-0 rounded-full border border-white/10 bg-[#13232a] object-cover object-top" />
                : <span aria-hidden="true" className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-white/10 bg-[#13232a] text-[11px] font-bold text-[#59e0c8]">{name.trim().split(/\s+/).at(-1)?.[0]}</span>}
              <span className="truncate">{name}</span>
            </button>;
          })
            : <p className="px-4 py-3 text-sm text-text-sec">{players.isPending ? "Loading players…" : "No players found"}</p>}
        </div>}
      </div>
    </div>}

    {/* Hero: who, which game, which market */}
    <section className="ql-panel relative mb-4 overflow-hidden" style={{ background: HERO_BG }} aria-label="Player">
      {/* Phones and tablets: centred. Full desktop: one compact row (photo · who and what · line and averages), like Outlier. */}
      {player ? <div className="flex flex-col items-center px-4 pb-5 pt-7 text-center sm:px-8 xl:grid xl:grid-cols-[auto_minmax(0,1fr)_auto] xl:items-center xl:gap-x-6 xl:px-7 xl:py-6 xl:text-left">
        <div className="relative xl:self-center">
          {profile?.headshot_url && !photoFailed
            ? <Image src={profile.headshot_url} alt={player} width={88} height={88} unoptimized onError={() => setPhotoFailed(true)}
                className="h-[88px] w-[88px] rounded-full border border-white/15 bg-[#13232a] object-cover object-top" />
            : <div className="flex h-[88px] w-[88px] items-center justify-center rounded-full border border-teal-400/30 bg-teal-400/10 text-2xl font-bold text-teal-400" aria-hidden="true">
                {player.split(" ").map(part => part[0]).slice(0, 2).join("")}
              </div>}
          {team && <span className="absolute -bottom-1 -right-3 rounded-full border border-white/15 bg-[#0d151b] px-2 py-0.5 text-[11px] font-bold text-teal-300">{team}</span>}
        </div>
        <div className="contents xl:block xl:min-w-0">
        {matchup && <p className="mt-4 flex flex-wrap items-center justify-center gap-2 text-sm text-text-sec xl:mt-0 xl:justify-start">
          <span className="rounded-full bg-white/10 px-2.5 py-0.5 font-bold text-text-pri">{matchup}</span>
          {nextGame && <span>{tipLabel(nextGame)}</span>}
        </p>}
        <div className="mt-2 flex flex-wrap items-center justify-center gap-2 xl:justify-start">
          <h2 className="text-3xl font-extrabold tracking-[-0.02em] text-text-pri sm:text-4xl" style={{ fontFamily: "var(--font-display)" }}>{player}</h2>
          {summary.data?.position && <span className="text-sm text-text-sec">({summary.data.position})</span>}
          <button aria-label={followed ? `Unfollow ${player}` : `Follow ${player}`} aria-pressed={followed} className="p-1"
            onClick={() => followed ? unfollowPlayer(player) : followPlayer(player)}>
            <Star size={18} color={followed ? "#f5ba64" : "#a9bec0"} fill={followed ? "#f5ba64" : "none"} />
          </button>
        </div>
        <p className="mt-1 text-xl font-bold text-text-sec sm:text-2xl" style={{ fontFamily: "var(--font-display)" }}>
          {STAT_NAMES[stat] ?? stat} · {side} {threshold}
        </p>
        {profile?.injury_status && !["ACTIVE", "UNKNOWN"].includes(profile.injury_status.toUpperCase()) &&
          <span className="ql-chip mt-3 border-amber-400/30 bg-amber-400/10 text-amber-300">{profile.injury_status}{profile.injury_reason ? ` · ${profile.injury_reason}` : ""}</span>}
        </div>
        <div className="mt-5 flex w-full flex-wrap items-center justify-between gap-3 xl:mt-0 xl:w-auto xl:flex-col xl:items-end xl:justify-center">
          {focus?.line != null
            ? <button type="button" onClick={() => selectStat(focus.stat)} aria-pressed={!!onFocusLine}
                className={`inline-flex items-center gap-3 rounded-2xl border px-4 py-2.5 text-sm font-bold ${onFocusLine ? "border-[#59e0c8] text-text-pri" : "border-white/15 text-text-sec"} bg-black/30`}>
                <span className="flex h-4 items-end gap-0.5" aria-hidden="true"><i className="h-2 w-1 bg-current" /><i className="h-3 w-1 bg-current" /><i className="h-4 w-1 bg-current" /></span>
                {focus.side} {focus.line}{focus.price != null && <span className="text-text-pri">{formatOdds(focus.price)}</span>}
                {focus.book && <span className="text-[11px] font-semibold uppercase tracking-wider text-[#f5ba64]">{focus.book}</span>}
              </button>
            : <span />}
          <div className="grid grid-cols-4 gap-4 text-left sm:gap-6">
            {(["PTS", "REB", "AST"] as const).map(key => <div key={key}><span className="ql-data-label">{key}</span><strong className="block text-lg font-bold tabular-nums text-text-pri">{fmt(summary.data?.season_avgs[key])}</strong></div>)}
            <div><span className="ql-data-label">FG%</span><strong className="block text-lg font-bold tabular-nums text-text-pri">{fmt(profile?.fg_pct)}</strong></div>
          </div>
        </div>
      </div> : <h2 className="p-8 text-center text-xl font-extrabold text-text-pri">Choose a player</h2>}
    </section>

    {players.isError && <p role="alert" className="mb-4 rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">Player list is unavailable. Try reloading the page.</p>}
    {!player && !players.isError && <div className={card}>Loading player data…</div>}
    {player && <>
      {/* Stat tabs, Outlier style: a scrolling underline row */}
      <nav className="mb-4 flex gap-1 overflow-x-auto border-b border-white/10" aria-label="Statistic">
        {STATS.map(([value, label]) => <button key={value} type="button" aria-pressed={stat === value} onClick={() => selectStat(value)}
          className={`relative shrink-0 px-3 pb-3 pt-1 text-sm font-bold tracking-wide transition-colors ${stat === value ? "text-text-pri" : "text-text-sec hover:text-text-pri"}`}>
          {label}
          {stat === value && <span className="absolute inset-x-3 bottom-0 h-0.5 rounded-full bg-[#59e0c8]" />}
        </button>)}
      </nav>

      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <WindowChips counts={counts} value={period} labels={windowLabels} onSelect={selectPeriod}
          disabled={nextOpponent ? {} : { h2h: "Upcoming opponent unavailable" }} />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_300px] lg:items-start">
        <div className="min-w-0 space-y-4">
          {allGames.length > 0 && <GameFilterPanel games={allGames} value={gameFilters} onChange={setGameFilters} matching={filteredGames.length} />}
          <section className="ql-panel overflow-hidden">
            <div className="px-4 pt-5 sm:px-6">
              <h2 className="flex items-center gap-2 text-lg font-bold text-text-pri"><Percent size={18} className="text-[#59e0c8]" aria-hidden="true" />{player} · {STAT_NAMES[stat] ?? stat}</h2>
              <div className="mt-4 flex flex-wrap items-end justify-between gap-4 border-b border-white/10 pb-4">
                <div>
                  <p className="text-sm font-semibold text-text-pri">{windowTitle}</p>
                  <p className="mt-0.5 flex items-baseline gap-2">
                    <strong className="text-3xl font-black tabular-nums" style={{ color: rateColor, fontFamily: "var(--font-display)" }}>{hitRate == null ? "—" : `${hitRate}%`}</strong>
                    <span className="text-sm text-text-sec">{tally.hits} of {decided} {side.toLowerCase()} {threshold}{tally.pushes ? ` · ${tally.pushes} push` : ""}</span>
                  </p>
                </div>
                <div className="flex items-center gap-1" role="group" aria-label="Side">
                  {(["Over", "Under"] as const).map(value => <button key={value} type="button" aria-pressed={side === value} onClick={() => setSide(value)}
                    className={`ql-chip ${side === value ? "ql-chip-active" : ""}`}>{value}</button>)}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-x-5 gap-y-2 py-3 text-sm text-text-sec">
                <span>Average: <strong className="tabular-nums text-text-pri">{fmt(average)}</strong></span>
                <span>Median: <strong className="tabular-nums text-text-pri">{fmt(middle)}</strong></span>
                <label className="ml-auto flex min-w-[220px] flex-1 items-center gap-3 sm:max-w-[360px]">
                  <span className="ql-data-label">LINE</span>
                  <input type="range" min="0" max={Math.max(50, Math.ceil(Math.max(...valid.map(game => game.value), 0) + 5))}
                    step="0.5" value={threshold} onChange={event => changeThreshold(Number(event.target.value))} className="flex-1 accent-teal-400" aria-label="Line" />
                  <output className="min-w-10 text-right font-mono text-sm font-bold tabular-nums text-teal-400">{threshold}</output>
                </label>
              </div>
            </div>
            <div className="border-t border-white/10 px-2 pb-4 pt-3 sm:px-4">
              {chart.isPending && <div className="h-72 animate-pulse rounded-xl bg-white/5" />}
              {chart.isError && <p role="alert" className="p-4 text-sm text-red-300">Game history is unavailable. <button className="underline" onClick={() => void chart.refetch()}>Try again</button></p>}
              {!chart.isPending && chart.data && games.length === 0 && <p className="py-12 text-center text-sm text-text-sec">{allGames.length > 0 ? "No games match these filters. Loosen or reset them." : "No completed games are available for this player."}</p>}
              {games.length > 0 && <div className="overflow-x-auto"><PlayerBarChart games={chartGames} stat={stat} line={threshold} side={side} height={320} /></div>}
            </div>
            {games.length > 0 && <div className="px-4 pb-5 sm:px-6">
              <p className="mb-3 text-xs text-text-sec">{games[0].date} to {games.at(-1)?.date} · Historical results{onFocusLine ? "" : " · The line is for analysis, not a live sportsbook line"}.</p>
              <GameLogTable games={chartGames} line={threshold} stat={stat} />
            </div>}
          </section>

          {insight && <section className={card}>
            <h2 className="text-lg font-bold text-text-pri">Insights</h2>
            <p className="mt-3 rounded-2xl border border-white/10 bg-white/[.03] p-4 text-base font-semibold text-text-pri">{insight}</p>
          </section>}

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

          {games.length > 0 && <PlayerPropLines key={`${player}:${stat}`} league={league} player={player} stat={stat}
            statLabel={STATS.find(([value]) => value === stat)?.[1] ?? stat} stats={STATS} onStatChange={selectStat} games={games} defaultLine={Math.max(0, defaultThreshold)}
            date={profile?.projection_context?.game_date ?? schedule.data?.target_date ?? new Date().toISOString().slice(0, 10)}
            matchup={nextGame?.matchup ?? (nextOpponent ? `${team} vs ${nextOpponent}` : `${team || player}`)} accent={accent} />}

          <PlayerResearch key={`research:${player}:${stat}`} league={league} player={player} stat={stat} />

          <section className={card}>
            <p className="ql-kicker">SUPPORTING EVIDENCE</p>
            <h2 className="ql-section-title mt-1">Supporting Stats</h2>
            <p className="mb-4 mt-1 text-xs text-text-sec">Season averages vs. recent five-game form · Select a stat to update the chart</p>
            {summary.isError && <p role="alert" className="text-sm text-red-300">Player summary is unavailable.</p>}
            {summary.data && <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
              {(["PTS", "REB", "AST", "FG3M", "STL", "BLK"] as const).map(key => <button key={key} type="button" aria-pressed={stat === key} onClick={() => selectStat(key)} className={`ql-metric border bg-white/[.025] p-3 text-left hover:border-teal-400/40 ${stat === key ? "border-teal-400/50" : "border-white/10"}`}>
                <p className="ql-data-label">{key === "FG3M" ? "3PM" : key} / GAME</p>
                <p className="mt-2 text-xl font-bold tabular-nums" style={{ color: STAT_COLORS[key] ?? "#59e0c8" }}>{fmt(summary.data?.season_avgs[key])}</p>
                <p className="mt-1 text-xs tabular-nums text-text-sec">L5 {fmt(summary.data?.l5_avgs[key])}</p>
              </button>)}
            </div>}
            <MinutesBars games={games} />
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

        <aside className={`space-y-3 lg:sticky ${embedded ? "lg:top-20" : "lg:top-24"}`}>
          {/* One tabbed panel, like Outlier's Matchup / Lineup / Insights column */}
          <div className="ql-panel grid grid-cols-3 gap-1 p-1" role="tablist" aria-label="Context">
            {([["matchup", "Matchup"], ["availability", "Availability"], ["log", "Game log"]] as const).map(([key, label]) =>
              <button key={key} type="button" role="tab" aria-selected={sidePanel === key} onClick={() => setSidePanel(key)}
                className={`rounded-md px-2 py-2 text-xs font-bold transition-colors ${sidePanel === key ? "bg-[#59e0c8]/15 text-text-pri" : "text-text-sec hover:text-text-pri"}`}>{label}</button>)}
          </div>
          {sidePanel === "matchup" && <section className={card} role="tabpanel">
            <p className="ql-kicker">NEXT MATCHUP</p>
            <h2 className="ql-section-title mt-1">Matchup</h2>
            {nextGame
              ? <p className="mt-3 text-sm text-text-sec"><strong className="text-text-pri">{team}</strong> vs. <strong className="text-text-pri">{nextOpponent}</strong> · {tipLabel(nextGame)}</p>
              : <p className="mt-3 text-sm text-text-sec">The next opponent has not been established for this player.</p>}
            {nextGame && nextOpponent && <MatchupDefense league={league} opponent={nextOpponent} game={nextGame} position={profile?.position} stat={stat} accent={accent} />}
          </section>}
          {sidePanel === "availability" && <section className={card} role="tabpanel">
            <p className="ql-kicker">AVAILABILITY</p>
            <h2 className="ql-section-title mt-1">Injury Context</h2>
            {profile?.injury_status
              ? <p className="mt-3 text-sm text-text-sec">{profile.injury_status}{profile.injury_reason ? ` · ${profile.injury_reason}` : ""}</p>
              : <p className="mt-3 text-sm text-text-sec">No verified injury update is available.</p>}
          </section>}
          {sidePanel === "log" && <section className={card} role="tabpanel">
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
          </section>}
          <p className="px-1 text-xs text-text-sec">History through {summary.data?.history_through ?? "—"}{summary.data?.history_age_days != null ? ` · ${summary.data.history_age_days} days ago` : ""}</p>
        </aside>
      </div>
    </>}
  </div>;
}
