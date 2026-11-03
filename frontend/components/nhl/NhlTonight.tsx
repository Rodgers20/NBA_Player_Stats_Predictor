"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { nhlApi } from "@/lib/api";
import { defaultDay } from "@/lib/cbb.mjs";
import { MARKETS, lineLabel } from "@/lib/nhl.mjs";
import { boardRows, initialLine, lineOptions } from "@/lib/nhl-props.mjs";
import { todayEt } from "@/lib/slate-freshness.mjs";
import type { NhlMarketKey } from "@/lib/nhl-types";
import { NhlGameCard } from "./NhlGameCard";
import { NhlPropRow } from "./NhlPropRow";
import { NhlPlayerSheet } from "./NhlPlayerSheet";
import { ErrorBox, FairOddsNote, Loading, NhlHeader, FRAME } from "./parts";

const TOP_PER_MARKET = 3;
const SLATE_PREVIEW = 8;
const dayLabel = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", timeZone: "UTC" });

/** NBA's Tonight page for hockey: the best model plays by market, then the slate. */
export function NhlTonight() {
  const index = useQuery({ queryKey: ["nhl-schedule-index"], queryFn: nhlApi.scheduleIndex, staleTime: 300_000 });
  const board = useQuery({ queryKey: ["nhl-props"], queryFn: nhlApi.props, staleTime: 120_000 });
  const date = index.data ? defaultDay(index.data.days, todayEt()) : undefined;
  const schedule = useQuery({ queryKey: ["nhl-schedule", date], queryFn: () => nhlApi.schedule(date!), enabled: !!date, staleTime: 120_000 });
  const [open, setOpen] = useState<number | null>(null);

  const games = useMemo(() => [...(schedule.data?.games ?? [])].sort((a, b) => a.start.localeCompare(b.start)), [schedule.data]);
  const players = useMemo(() => (board.data?.players ?? []).filter(p => !date || p.date === date), [board.data, date]);
  const groups = useMemo(() => MARKETS.map(m => {
    const line = initialLine(lineOptions(players, m.key), m.key);
    const rows = line == null ? [] : boardRows(players, { market: m.key, line, date, sort: "p" }).slice(0, TOP_PER_MARKET);
    return { market: m, line, rows };
  }).filter(g => g.rows.length > 0), [players, date]);

  const pending = index.isPending || schedule.isPending || board.isPending;
  return (
    <div className={FRAME}>
      <NhlHeader kicker="TONIGHT" title="Tonight's plays" sub={`${date ? dayLabel(date) : "Upcoming games"} · the model's strongest probabilities by market, with fair odds to compare against your book.`} />
      {(index.isError || board.isError) && <ErrorBox onRetry={() => { void index.refetch(); void board.refetch(); }}>Tonight&apos;s board could not be loaded.</ErrorBox>}
      {pending && !index.isError && !board.isError && <Loading label="Loading tonight's plays" />}
      {!pending && <>
        <p className="mb-5 flex flex-wrap items-baseline gap-x-3 gap-y-1 border-y border-ink/10 py-3 text-sm text-text-sec" role="status">
          <span><strong className="text-lg font-extrabold tabular-nums text-text-pri">{games.length}</strong> {games.length === 1 ? "game" : "games"} · <strong className="tabular-nums text-text-pri">{players.length}</strong> players projected</span>
          <Link href="/nhl/model" className="ml-auto text-xs font-semibold text-teal-400 hover:underline">Model record ↗</Link>
        </p>
        <div className="mb-5"><FairOddsNote /></div>
        {groups.map(({ market, line, rows }) => (
          <section key={market.key} className="mb-7" aria-label={market.label}>
            <div className="mb-2 flex items-baseline justify-between border-b border-ink/10 pb-2">
              <h2 className="ql-section-title">{market.label}{line != null && market.key !== "goal" ? ` ${lineLabel(market.key as NhlMarketKey, line)}` : ""}</h2>
              <Link href="/nhl/props" className="ql-data-label hover:text-text-pri">ALL IN PROPS ↗</Link>
            </div>
            <div className="ql-panel overflow-hidden">
              {rows.map(r => <NhlPropRow key={r.player.player_id} row={r} market={market.key as NhlMarketKey} onOpen={setOpen} />)}
            </div>
          </section>
        ))}
        {groups.length === 0 && <section className="ql-panel px-6 py-10 text-center" role="status">
          <h2 className="ql-section-title">No props projected yet</h2>
          <p className="mx-auto mt-2 max-w-[60ch] text-sm text-text-sec">The board fills in once lineups for the next slate are projected.</p>
        </section>}
        {games.length > 0 && <section className="mt-8" aria-label="Slate">
          <div className="mb-2 flex items-baseline justify-between border-b border-ink/10 pb-2">
            <h2 className="ql-section-title">The slate</h2>
            <Link href="/nhl/games" className="ql-data-label hover:text-text-pri">ALL GAMES</Link>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">{games.slice(0, SLATE_PREVIEW).map(g => <NhlGameCard key={g.id} game={g} />)}</div>
        </section>}
      </>}
      {open != null && <NhlPlayerSheet playerId={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
