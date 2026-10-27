"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import type { PropsResponse } from "@/lib/types";
import { PropHeader, PropRow } from "@/components/props/PropRow";
import { usePlayerSheet } from "@/store/player-sheet";
import { focusFromProp } from "@/components/analysis/PlayerName";
import { PropFilters, isLineMarket, type Filters } from "@/components/props/PropFilters";
import { GameLinesBoard } from "@/components/props/GameLinesBoard";
import { ParlayBuilder } from "@/components/props/ParlayBuilder";
import { filterMinStars, sortByEdge } from "@/lib/prop-research.mjs";
import { searchProps, sortByWindow } from "@/lib/prop-rows.mjs";
import { applyBoardFilters, availableBooks, availablePlayers, selectedStats } from "@/lib/prop-filters.mjs";
import { filtersToSearch, isStaleGame, parseFilters, type BoardFilters } from "@/lib/props-url.mjs";
import { PopularBoard } from "@/components/popular/PopularBoard";
import { TrendsBoard } from "@/components/trends/TrendsBoard";
import { PropsRecord } from "@/components/props/PropsRecord";

/** How many filters differ from "show everything", for the phone's collapsed filter button. */
function activeFilterCount(filters: Filters) {
  return [!!filters.stat && filters.stat !== "All", filters.direction !== "All", filters.location !== "All", !!filters.game, filters.ev, !!filters.players, filters.minStars > 0].filter(Boolean).length;
}

const PAGE = 30;
const VIEWS = [
  { key: "props", label: "Edge" },
  { key: "hot", label: "Hot" },
  { key: "streaks", label: "Streaks" },
  { key: "alt", label: "100% Alt Lines" },
  { key: "record", label: "Record" },
] as const;
type View = (typeof VIEWS)[number]["key"];

function slateDate(value: string | null | undefined) {
  if (!value) return "Upcoming games";
  const date = new Date(`${value}T12:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
}

export default function PropsPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading props…</p>}><PropsBoard /></Suspense>;
}

function PropsBoard() {
  const { defaultStat, defaultSort, league, books, setBooks } = usePrefs();
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  // Filters live in the URL so a view can be shared, bookmarked and survive a reload.
  const defaults: BoardFilters = { stat: defaultStat, direction: "All", location: "All", sort: defaultSort, game: "", locksOnly: false, minStars: 0, window: "L10", ev: false, players: "" };
  const filters: Filters = parseFilters(search.toString(), defaults);
  const [page, setPage] = useState(1);
  const [filtersOpen, setFiltersOpen] = useState(false);
  // Popular and Trends used to be their own pages ranking these same props; they are views of this board now.
  const view = (VIEWS.find(item => item.key === search.get("view"))?.key ?? "props") as View;
  const setView = (next: View) => {
    const params = new URLSearchParams(search.toString());
    if (next === "props") params.delete("view"); else params.set("view", next);
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };
  const sheet = usePlayerSheet((state) => state.request);
  const openPlayer = usePlayerSheet((state) => state.open);
  const client = useQueryClient();
  const tapOrigin = useRef<{ x: number; y: number } | null>(null);
  const isStatic = process.env.NEXT_PUBLIC_DATA_MODE === "static";
  const budget = useQuery({ queryKey: ["budget", league], queryFn: () => api.budget(league), enabled: !isStatic });
  const altLines = useQuery({ queryKey: ["alt-lines", league], queryFn: () => api.altLines(league) });
  const record = useQuery({ queryKey: ["props-record", league], queryFn: () => api.propsRecord(league) });
  const hasRecord = !!(record.data?.total || record.data?.pending);
  const refresh = useMutation({
    mutationFn: () => api.refreshProps(league),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["props"] }),
        client.invalidateQueries({ queryKey: ["budget"] }),
      ]);
    },
  });

  // ML, spread and totals are whole-game lines, not player props, so they get their own board.
  const lineMarket = isLineMarket(filters.stat) ? filters.stat : null;
  // The API filters one stat; several are narrowed here from the loaded rows.
  const statList = selectedStats(filters.stat);
  const params = {
    league,
    direction: filters.direction.toLowerCase(),
    location: filters.location.toLowerCase(),
    sort: filters.sort === "edge" ? "ev" : filters.sort, // the API has no edge sort; it is applied below
    include_research: true,
    limit: PAGE * page,
    ...(statList.length === 1 && { stat: statList[0] }),
    ...(filters.game && { game: filters.game }),
  };
  const { data, isPending, isError, refetch } = useQuery<PropsResponse>({
    queryKey: ["props", params],
    queryFn: () => api.props(params),
    refetchInterval: 120_000,
    enabled: !lineMarket,
    // Keep the rows on screen while "Load more" or a filter fetches; swapping to the skeleton collapses the page and jumps to the top.
    placeholderData: keepPreviousData,
  });

  // The builder picks from the whole slate, not the filtered, paged rows on screen.
  const fullBoard = useQuery<PropsResponse>({
    queryKey: ["props", { league, include_research: true, limit: 500 }],
    queryFn: () => api.props({ league, include_research: true, limit: 500 }),
    refetchInterval: 120_000,
    enabled: !lineMarket,
  });

  function handleChange(update: Partial<Filters>) {
    const qs = filtersToSearch({ ...filters, ...update }, defaults);
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    setPage(1);
  }
  // A game the loaded slate does not contain can never match (other league, old link), so drop it.
  // Keyed on the data rather than on a league change, so persisted-league rehydration cannot trigger it.
  useEffect(() => {
    if (!isStaleGame(filters.game, data?.game_matchups)) return;
    const url = new URL(window.location.href);
    url.searchParams.delete("game");
    router.replace(`${url.pathname}${url.search}`, { scroll: false });
  }, [filters.game, data?.game_matchups, router]);
  const activeView = (view === "alt" && !altLines.data?.count) || (view === "record" && !hasRecord) ? "props" : view;
  const loaded = data?.props ?? [];
  const narrowed = applyBoardFilters(loaded, {
    evOnly: filters.ev, stats: statList.length > 1 ? statList : [],
    players: filters.players ? filters.players.split("|") : [], books,
  });
  const starred = searchProps(filterMinStars(narrowed, filters.minStars), filters.q);
  // "Highest hit rate" follows the chosen window (L5/L10/L20/H2H/season), not the overall sample the API sorts by.
  const shownProps = filters.sort === "edge" ? sortByEdge(starred) : filters.sort === "hit_rate" ? sortByWindow(starred, filters.window) : starred;
  const byGame = Object.groupBy(shownProps, (prop) => prop.game_matchup || "Other matchups");
  const verifiedCount = shownProps.filter((prop) => (prop.pick_type ?? (prop.recommendation_eligible ? "pick" : "research")) !== "research").length;
  // The static snapshot withholds a past slate (status "stale") instead of showing yesterday's picks.
  const pastSlate = data?.status === "stale";
  const showPlayers = activeView === "props" && !lineMarket && !pastSlate;

  return (
    <main className="ql-page mx-auto max-w-[1440px] px-4 pb-12 pt-7 sm:px-8" onPointerDownCapture={(event) => { tapOrigin.current = { x: event.clientX, y: event.clientY }; }}>
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="ql-kicker">PROPS / {league.toUpperCase()}</p>
          <h1 className="ql-heading mt-1">Props</h1>
          <p className="ql-subtitle mt-2 max-w-[80ch]">{slateDate(data?.target_date)} · every priced market, ranked by edge, form or streak.</p>
        </div>
      </header>

      <div className="mb-5 flex flex-wrap items-center justify-between gap-2 border-b border-ink/10">
        <div className="flex max-w-full items-center gap-1 overflow-x-auto" role="tablist" aria-label="Prop board views">
          {VIEWS.filter(item => (item.key !== "alt" || !!altLines.data?.count) && (item.key !== "record" || hasRecord)).map(item =>
            <button key={item.key} type="button" role="tab" onClick={() => setView(item.key)} aria-selected={activeView === item.key}
              className={`shrink-0 border-b-2 px-3 py-2.5 text-xs font-bold uppercase tracking-[.08em] ${activeView === item.key ? "border-teal-400 text-teal-400" : "border-transparent text-text-sec hover:text-ink"}`}>{item.label}</button>)}
        </div>
        {!isStatic && budget.data?.configured && (
          <button
            type="button"
            onClick={() => refresh.mutate()}
            disabled={refresh.isPending}
            className="ql-control mb-1 px-3 py-1.5 text-xs font-semibold text-teal-400 hover:bg-teal-400/10 disabled:opacity-50"
          >
            {refresh.isPending ? "Refreshing…" : "Refresh sportsbook quotes"}
          </button>
        )}
      </div>

      {league === "wnba" && !isStatic && budget.data?.configured && (
        <p className="mb-4 text-xs text-text-sec">NBA and WNBA share one free Odds API pool. Quotes refresh each morning and about two hours before tip-off, and credits go to whichever league has games, so off-season credits fund the other league. In-progress games do not produce pregame picks.</p>
      )}

      {activeView === "hot" && <PopularBoard league={league} />}
      {activeView === "streaks" && <TrendsBoard league={league} />}

      {/* On a phone the filters used to fill the whole first screen; the list comes first and filters open on demand. */}
      {activeView === "props" && <button type="button" onClick={() => setFiltersOpen(open => !open)} aria-expanded={filtersOpen} aria-controls="prop-filters"
        className="ql-control mb-3 flex w-full items-center justify-between px-4 py-3 text-sm font-semibold text-text-pri sm:hidden">
        <span>Filters{activeFilterCount(filters) ? ` · ${activeFilterCount(filters)} on` : ""}</span><span aria-hidden="true">{filtersOpen ? "−" : "+"}</span>
      </button>}
      {activeView === "props" && <div id="prop-filters" className={filtersOpen ? "" : "hidden sm:block"}>
        <PropFilters filters={filters} matchups={data?.game_matchups ?? []} statCounts={data?.stat_counts} onChange={handleChange}
          bookOptions={availableBooks(loaded)} books={books} onBooksChange={setBooks} playerOptions={availablePlayers(loaded)}
          playerImages={Object.fromEntries(loaded.map(prop => [prop.player, prop.headshot_url]))} />
      </div>}

      {!isStatic && (refresh.error || refresh.data || budget.data?.configured === false) && (
        <p role="status" className="mb-4 text-xs text-text-sec">
          {refresh.error?.message || refresh.data?.message || "No sportsbook provider is connected. Player research remains available."}
        </p>
      )}

      {data?.status === "research" && data.message && (
        <p role="status" className="ql-panel mb-4 border-lock-gold/30 px-4 py-3 text-xs text-lock-gold">{data.message} These are model projections without a sportsbook line, price, or betting pick.</p>
      )}

      <div className="min-w-0">
      {activeView === "props" && lineMarket && <GameLinesBoard league={league} market={lineMarket} />}
      {activeView === "props" && !lineMarket && pastSlate && (
        <div className="ql-panel px-6 py-12 text-center" role="status">
          <p className="text-sm text-text-pri">{"Today's picks aren't published yet."}</p>
          <p className="mt-2 text-xs text-text-sec">The board last refreshed for {slateDate(data?.target_date)}. New picks appear after the morning data refresh, around 7:15 AM ET.</p>
        </div>
      )}
      {showPlayers && fullBoard.data?.status === "ready" && fullBoard.data.target_date && (
        <ParlayBuilder props={fullBoard.data.props} league={league} date={fullBoard.data.target_date} />
      )}
      {showPlayers && isPending && (
        <div className="ql-panel overflow-hidden" aria-busy="true" aria-label="Loading props">
          {Array.from({ length: 5 }).map((_, index) => (
            <div key={index} className="h-32 animate-pulse border-b border-ink/5" />
          ))}
        </div>
      )}

      {showPlayers && isError && (
        <div className="ql-panel p-8 text-center text-sm text-danger" role="alert">
          <p>We couldn&apos;t load the prop board.</p>
          <button type="button" className="mt-2 underline" onClick={() => void refetch()}>Try again</button>
        </div>
      )}

      {showPlayers && !isPending && data?.props.length === 0 && (
        <div className="ql-panel px-6 py-12 text-center" role="status">
          <p className="text-sm text-text-sec">{data.message || "No props match the current filters."}</p>
          <p className="mt-2 text-xs text-text-sec">Try another stat, direction, game, or league.</p>
        </div>
      )}

      {showPlayers && !isPending && data && data.props.length > 0 && shownProps.length === 0 && (
        <div className="ql-panel px-6 py-12 text-center" role="status">
          <p className="text-sm text-text-sec">No loaded props match your filters.</p>
          <p className="mt-2 text-xs text-text-sec">Try another name, lower the minimum stars, or load more props below.</p>
        </div>
      )}

      {showPlayers && !isPending && data && shownProps.length > 0 && (
        <div className="min-w-0 space-y-5">
          <p className="ql-data-label text-text-sec">{data.status === "research" ? "MODEL RESEARCH · NO PRICED PICKS" : `${verifiedCount} VERIFIED PRICED ${verifiedCount === 1 ? "MARKET" : "MARKETS"} SHOWN`} · SELECT A PLAYER FOR GAME HISTORY</p>
          <PropHeader window={filters.window} />
          {Object.entries(byGame).map(([matchup, props]) => (
            <section key={matchup} aria-label={matchup}>
              <div className="mb-2 flex items-baseline justify-between border-b border-ink/10 pb-2">
                <h2 className="ql-section-title">{matchup}</h2>
                <span className="ql-data-label text-text-sec">{props?.length ?? 0} PROPS</span>
              </div>
              <div className="ql-panel overflow-hidden">
                {props?.map((prop) => <PropRow key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`} prop={prop} targetDate={data.target_date} selected={sheet?.player === prop.player && sheet.focus?.stat === prop.stat} record={record.data} window={filters.window} onAnalyze={(next) => openPlayer({ player: next.player, league, origin: tapOrigin.current, focus: focusFromProp(next) })} />)}
              </div>
            </section>
          ))}
        </div>
      )}


      {showPlayers && data && data.props.length > 0 && data.props.length < data.count && (
        <button
          type="button"
          onClick={() => setPage((current) => current + 1)}
          className="ql-control mt-4 w-full py-3 text-sm font-semibold text-teal-400 hover:bg-teal-400/10"
        >
          Load more ({data.count - data.props.length} remaining)
        </button>
      )}

      {activeView === "alt" && altLines.data && <section className="ql-panel overflow-hidden">
        <div className="flex items-center justify-between border-b border-ink/10 px-5 py-4">
          <h2 className="text-lg font-bold text-text-pri">100% Alt Lines</h2>
          <span className="text-xs text-text-sec">{altLines.data.count} historical streaks</span>
        </div>
        <p className="px-5 py-3 text-xs text-text-sec">Historical streaks only. Alternate-line sportsbook prices are not validated.</p>
        {altLines.data.alt_lines.map((line, index) => <div key={`${line.player}-${line.stat}-${index}`} className="grid grid-cols-[55px_minmax(0,1fr)_auto_auto] items-center gap-3 border-t border-ink/[0.06] px-5 py-3 text-sm">
          <span className="text-xs font-bold text-text-sec">{line.team}</span>
          <span className="min-w-0 truncate font-semibold text-text-pri">{line.player}</span>
          <span className="rounded-md border border-teal-400/25 bg-teal-400/10 px-2 py-1 text-xs font-bold text-teal-400">{line.threshold}+ {line.stat_label}</span>
          <span className="text-xs font-bold text-text-sec">{line.trend}</span>
        </div>)}
      </section>}

      {activeView === "record" && record.data && <PropsRecord record={record.data} />}

      </div>
      <p className="mt-5 text-center text-xs text-text-sec">Historical hit rates are descriptive, not guaranteed outcomes.</p>
    </main>
  );
}
