"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import type { Prop, PropsResponse } from "@/lib/types";
import { useSlip } from "@/store/slip";
import { BetSlip } from "@/components/props/BetSlip";
import { PropCard } from "@/components/props/PropCard";
import { PropAnalysisPanel } from "@/components/props/PropAnalysisPanel";
import { PropFilters, type Filters } from "@/components/props/PropFilters";

const PAGE = 30;

function slateDate(value: string | null | undefined) {
  if (!value) return "Upcoming games";
  const date = new Date(`${value}T12:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
}

export default function PropsPage() {
  const slipCount = useSlip(state => state.legs.length);
  const { defaultStat, defaultSort, league } = usePrefs();
  const [filters, setFilters] = useState<Filters>({
    stat: defaultStat,
    direction: "All",
    location: "All",
    sort: defaultSort,
    game: "",
    locksOnly: false,
  });
  const [page, setPage] = useState(1);
  const [view, setView] = useState<"props" | "alt" | "record">("props");
  const [selected, setSelected] = useState<{ league: "nba" | "wnba"; prop: Prop } | null>(null);
  const client = useQueryClient();
  const isStatic = process.env.NEXT_PUBLIC_DATA_MODE === "static";
  const budget = useQuery({ queryKey: ["budget", league], queryFn: () => api.budget(league), enabled: !isStatic });
  const altLines = useQuery({ queryKey: ["alt-lines", league], queryFn: () => api.altLines(league) });
  const record = useQuery({ queryKey: ["props-record"], queryFn: api.propsRecord });
  const refresh = useMutation({
    mutationFn: () => api.refreshProps(league),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["props"] }),
        client.invalidateQueries({ queryKey: ["budget"] }),
      ]);
    },
  });

  const params = {
    league,
    direction: filters.direction.toLowerCase(),
    location: filters.location.toLowerCase(),
    sort: filters.sort,
    include_research: true,
    limit: PAGE * page,
    ...(filters.stat && { stat: filters.stat }),
    ...(filters.game && { game: filters.game }),
  };
  const { data, isPending, isError, refetch } = useQuery<PropsResponse>({
    queryKey: ["props", params],
    queryFn: () => api.props(params),
    refetchInterval: 120_000,
  });

  function handleChange(update: Partial<Filters>) {
    setFilters((current) => ({ ...current, ...update }));
    setPage(1);
  }
  const activeView = (view === "alt" && !altLines.data?.count) || (view === "record" && !record.data?.total) ? "props" : view;
  const byGame = Object.groupBy(data?.props ?? [], (prop) => prop.game_matchup || "Other matchups");
  const verifiedCount = data?.props.filter((prop) => (prop.pick_type ?? (prop.recommendation_eligible ? "pick" : "research")) !== "research").length ?? 0;
  const selectedProp = selected?.league === league ? selected.prop : null;

  return (
    <main className="ql-page mx-auto max-w-[1440px] px-4 pb-12 pt-7 sm:px-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="ql-kicker">03 / PLAYER MARKETS</p>
          <h1 className="ql-heading mt-1">Best Props</h1>
          <p className="ql-subtitle mt-2 max-w-[80ch]">Find your edge, inspect a player, and add priced picks to your slip.</p>
        </div>
        <div className="text-left sm:text-right">
          <p className="ql-data-label text-[#a9bec0]">SLATE / {league.toUpperCase()}</p>
          <p className="mt-1 text-sm text-[#eaf8f6]">{slateDate(data?.target_date)}</p>
          <p className="mt-1 text-xs font-semibold text-[#59e0c8]">{data?.count ?? 0} {data?.status === "research" ? "players in research" : "props"}</p>
        </div>
      </header>

      <div className="mb-5 flex flex-wrap items-center justify-between gap-2 border-b border-white/10">
        <div className="flex items-center gap-1" role="group" aria-label="Prop board views">
          <button type="button" onClick={() => setView("props")} aria-pressed={activeView === "props"}
            className={`border-b-2 px-3 py-2.5 text-xs font-bold uppercase tracking-[.08em] ${activeView === "props" ? "border-[#59e0c8] text-[#59e0c8]" : "border-transparent text-[#a9bec0] hover:text-white"}`}>Props</button>
          {!!altLines.data?.count && <button type="button" onClick={() => setView("alt")} aria-pressed={activeView === "alt"}
            className={`border-b-2 px-3 py-2.5 text-xs font-bold uppercase tracking-[.08em] ${activeView === "alt" ? "border-[#59e0c8] text-[#59e0c8]" : "border-transparent text-[#a9bec0] hover:text-white"}`}>100% Alt Lines</button>}
          {!!record.data?.total && <button type="button" onClick={() => setView("record")} aria-pressed={activeView === "record"}
            className={`border-b-2 px-3 py-2.5 text-xs font-bold uppercase tracking-[.08em] ${activeView === "record" ? "border-[#59e0c8] text-[#59e0c8]" : "border-transparent text-[#a9bec0] hover:text-white"}`}>Record ✓</button>}
        </div>
        {!isStatic && budget.data?.configured && (
          <button
            type="button"
            onClick={() => refresh.mutate()}
            disabled={refresh.isPending}
            className="ql-control mb-1 px-3 py-1.5 text-xs font-semibold text-[#59e0c8] hover:bg-[#59e0c8]/10 disabled:opacity-50"
          >
            {refresh.isPending ? "Refreshing…" : "Refresh sportsbook quotes"}
          </button>
        )}
      </div>

      {league === "wnba" && !isStatic && budget.data?.configured && (
        <p className="mb-4 text-xs text-[#a9bec0]">NBA and WNBA share one free Odds API pool. Quotes refresh each morning and about two hours before tip-off, and credits go to whichever league has games, so off-season credits fund the other league. In-progress games do not produce pregame picks.</p>
      )}

      {activeView === "props" && <PropFilters filters={filters} matchups={data?.game_matchups ?? []} statCounts={data?.stat_counts} onChange={handleChange} />}

      {!isStatic && (refresh.error || refresh.data || budget.data?.configured === false) && (
        <p role="status" className="mb-4 text-xs text-[#a9bec0]">
          {refresh.error?.message || refresh.data?.message || "No sportsbook provider is connected. Player research remains available."}
        </p>
      )}

      {data?.status === "research" && data.message && (
        <p role="status" className="ql-panel mb-4 border-[#f5ba64]/30 px-4 py-3 text-xs text-[#f5ba64]">{data.message} These are model projections without a sportsbook line, price, or betting pick.</p>
      )}

      <a href="#props-slip" className="ql-control mb-4 block px-4 py-3 text-center text-sm font-semibold text-[#59e0c8] xl:hidden">View your slip · {slipCount} {slipCount === 1 ? 'leg' : 'legs'} ↓</a>
      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_350px]"><div className="min-w-0">
      {activeView === "props" && isPending && (
        <div className="ql-panel overflow-hidden" aria-busy="true" aria-label="Loading props">
          {Array.from({ length: 5 }).map((_, index) => (
            <div key={index} className="h-32 animate-pulse border-b border-white/5" />
          ))}
        </div>
      )}

      {activeView === "props" && isError && (
        <div className="ql-panel p-8 text-center text-sm text-[#fa837b]" role="alert">
          <p>We couldn&apos;t load the prop board.</p>
          <button type="button" className="mt-2 underline" onClick={() => void refetch()}>Try again</button>
        </div>
      )}

      {activeView === "props" && !isPending && data?.props.length === 0 && (
        <div className="ql-panel px-6 py-12 text-center" role="status">
          <p className="text-sm text-[#a9bec0]">{data.message || "No props match the current filters."}</p>
          <p className="mt-2 text-xs text-[#a9bec0]">Try another stat, direction, game, or league.</p>
        </div>
      )}

      {activeView === "props" && !isPending && data && data.props.length > 0 && (
        <div className={selectedProp ? "grid gap-5 2xl:grid-cols-[minmax(0,1fr)_360px] 2xl:items-start" : ""}>
        <div className="min-w-0 space-y-5">
          <p className="ql-data-label text-[#a9bec0]">{data.status === "research" ? "MODEL RESEARCH · NO PRICED PICKS" : `${verifiedCount} VERIFIED PRICED ${verifiedCount === 1 ? "MARKET" : "MARKETS"} SHOWN`} · SELECT A PLAYER FOR GAME HISTORY</p>
          {Object.entries(byGame).map(([matchup, props]) => (
            <section key={matchup} aria-label={matchup}>
              <div className="mb-2 flex items-baseline justify-between border-b border-white/10 pb-2">
                <h2 className="ql-section-title">{matchup}</h2>
                <span className="ql-data-label text-[#a9bec0]">{props?.length ?? 0} PROPS</span>
              </div>
              <div className="ql-panel overflow-hidden">
                {props?.map((prop) => <PropCard key={`${prop.player}-${prop.stat}-${prop.direction}-${prop.line}`} prop={prop} targetDate={data.target_date} selected={selectedProp === prop} onAnalyze={(next) => setSelected({ league, prop: next })} />)}
              </div>
            </section>
          ))}
        </div>
        {selectedProp && <PropAnalysisPanel key={`${league}-${selectedProp.player}-${selectedProp.stat}`} prop={selectedProp} league={league} onClose={() => setSelected(null)} />}
        </div>
      )}

      {activeView === "props" && data && data.props.length > 0 && data.props.length < data.count && (
        <button
          type="button"
          onClick={() => setPage((current) => current + 1)}
          className="ql-control mt-4 w-full py-3 text-sm font-semibold text-[#59e0c8] hover:bg-[#59e0c8]/10"
        >
          Load more ({data.count - data.props.length} remaining)
        </button>
      )}

      {activeView === "alt" && altLines.data && <section className="ql-panel overflow-hidden">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <h2 className="text-lg font-bold text-[#eaf8f6]">100% Alt Lines</h2>
          <span className="text-xs text-[#a9bec0]">{altLines.data.count} historical streaks</span>
        </div>
        <p className="px-5 py-3 text-xs text-[#a9bec0]">Historical streaks only. Alternate-line sportsbook prices are not validated.</p>
        {altLines.data.alt_lines.map((line, index) => <div key={`${line.player}-${line.stat}-${index}`} className="grid grid-cols-[55px_minmax(0,1fr)_auto_auto] items-center gap-3 border-t border-white/[0.06] px-5 py-3 text-sm">
          <span className="text-xs font-bold text-[#a9bec0]">{line.team}</span>
          <span className="min-w-0 truncate font-semibold text-[#eaf8f6]">{line.player}</span>
          <span className="rounded-md border border-[#59e0c8]/25 bg-[#59e0c8]/10 px-2 py-1 text-xs font-bold text-[#59e0c8]">{line.threshold}+ {line.stat_label}</span>
          <span className="text-xs font-bold text-[#a9bec0]">{line.trend}</span>
        </div>)}
      </section>}

      {activeView === "record" && record.data && <section className="ql-panel p-5">
        <h2 className="mb-4 text-lg font-bold text-[#eaf8f6]">Props record</h2>
        <div className="grid gap-3 sm:grid-cols-3">
          <RecordMetric label="All-time hit rate" value={`${record.data.pct}%`} detail={`${record.data.hit} of ${record.data.total} graded`} />
          <RecordMetric label="Recent seven days" value={`${record.data.recent_7d}%`} detail="graded props" />
          <RecordMetric label="Graded props" value={`${record.data.total}`} detail={`${record.data.miss} misses`} />
        </div>
        {Object.keys(record.data.by_stat).length > 0 && <div className="mt-5 flex flex-wrap gap-2">
          {Object.entries(record.data.by_stat).map(([stat, result]) => <span key={stat} className="rounded-md border border-white/10 px-3 py-2 text-xs text-[#a9bec0]">{stat} <strong className="ml-1 text-[#eaf8f6]">{result.pct}%</strong> <span>({result.hit}/{result.total})</span></span>)}
        </div>}
      </section>}

      </div><div id="props-slip" className="scroll-mt-20 xl:sticky xl:top-24"><BetSlip /></div></div>
      <p className="mt-5 text-center text-xs text-[#a9bec0]">Historical hit rates are descriptive, not guaranteed outcomes.</p>
    </main>
  );
}

function RecordMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="rounded-lg border border-white/10 bg-white/[0.03] p-4">
    <p className="text-xs text-[#a9bec0]">{label}</p>
    <p className="mt-1 text-2xl font-extrabold text-[#eaf8f6]">{value}</p>
    <p className="mt-1 text-xs text-[#a9bec0]">{detail}</p>
  </div>;
}
