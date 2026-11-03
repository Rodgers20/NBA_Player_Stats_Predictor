"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { nhlApi } from "@/lib/api";
import { defaultDay } from "@/lib/cbb.mjs";
import { hourLabel, pct } from "@/lib/nhl.mjs";
import { todayEt } from "@/lib/slate-freshness.mjs";
import { DateStrip } from "@/components/games/DateStrip";
import { BoardGroups } from "@/components/games/shared/Board";
import { NhlGameCard } from "./NhlGameCard";
import { ErrorBox, Loading, NhlHeader, FRAME } from "./parts";
import type { NhlGame } from "@/lib/nhl-types";

function RecordBar({ games }: { games: NhlGame[] }) {
  const graded = games.filter(g => g.pick?.correct != null);
  if (!graded.length) return null;
  const right = graded.filter(g => g.pick?.correct).length;
  return (
    <dl className="ql-panel mb-4 flex items-baseline gap-4 p-4" aria-label="Model record for the day">
      <div><dt className="ql-data-label">Winners picked</dt><dd className="ql-metric !text-2xl">{right}–{graded.length - right}<span className="ml-2 text-xs font-medium text-text-sec">{pct(right / graded.length)}</span></dd></div>
      <p className="text-xs text-text-sec">Pre-game calls on the finished games of this day.</p>
    </dl>
  );
}

export function NhlSlate() {
  const params = useSearchParams();
  const index = useQuery({ queryKey: ["nhl-schedule-index"], queryFn: nhlApi.scheduleIndex, staleTime: 300_000 });
  const days = index.data?.days;
  const today = todayEt();
  const requested = params.get("date");
  const [picked, setPicked] = useState<string | null>(requested && /^\d{4}-\d{2}-\d{2}$/.test(requested) ? requested : null);
  const date = picked ?? (days ? defaultDay(days, today) : undefined);
  const schedule = useQuery({ queryKey: ["nhl-schedule", date], queryFn: () => nhlApi.schedule(date!), enabled: !!date, staleTime: 120_000 });
  const games = useMemo(() => [...(schedule.data?.games ?? [])].sort((a, b) => a.start.localeCompare(b.start)), [schedule.data]);

  useEffect(() => {
    document.querySelector<HTMLElement>('nav[aria-label="Game dates"] [aria-pressed="true"]')?.scrollIntoView({ inline: "center", block: "nearest" });
  }, [date, days]);

  const choose = (next: string) => {
    setPicked(next);
    const url = new URL(window.location.href);
    url.searchParams.set("date", next);
    window.history.replaceState(null, "", url);
  };

  return (
    <div className={FRAME}>
      <NhlHeader kicker="GAMES" title="NHL games" sub="Every game with a model win probability, projected total and the pre-game call kept for finished games. There are no sportsbook lines in this data.">
        <span className="flex gap-2"><Link href="/nhl/standings" className="ql-chip">Standings</Link><Link href="/nhl/model" className="ql-chip">Model record</Link></span>
      </NhlHeader>
      {index.isError && <ErrorBox onRetry={() => void index.refetch()}>NHL data is unavailable.</ErrorBox>}
      {days && date && <DateStrip dates={days.map(d => d.date)} date={date} today={today} onChange={choose} />}
      <div className="mt-4">
        {schedule.isPending && !index.isError && <Loading label="Loading games" />}
        {schedule.isError && <ErrorBox onRetry={() => void schedule.refetch()}>We couldn&apos;t load this day.</ErrorBox>}
        {schedule.data && <>
          <RecordBar games={games} />
          {games.length === 0
            ? <p role="status" className="ql-panel p-5 text-sm text-text-sec">No games scheduled on this day.</p>
            : <BoardGroups items={games} label={g => hourLabel(g.start)} card={g => <NhlGameCard key={g.id} game={g} />} />}
        </>}
      </div>
    </div>
  );
}
