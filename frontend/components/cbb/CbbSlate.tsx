"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { cbbApi } from "@/lib/api";
import { conferencesOf, defaultDay, dayRecord, filterGames, pct } from "@/lib/cbb.mjs";
import { todayEt } from "@/lib/slate-freshness.mjs";
import { DateStrip } from "@/components/games/DateStrip";
import { CbbGameCard } from "./CbbGameCard";
import { CbbHeader, ErrorBox, Loading } from "./parts";

const VIEWS = [
  { id: "all", label: "All games" },
  { id: "top25", label: "Top 25" },
  { id: "lines", label: "With lines" },
  { id: "disagree", label: "Model vs book" },
] as const;
const PAGE = 48;

function RecordBar({ games }: { games: Parameters<typeof dayRecord>[0] }) {
  const rec = dayRecord(games);
  if (!rec.graded) return null;
  const cell = (label: string, r: { wins: number; losses: number }) => (
    <div><dt className="ql-data-label">{label}</dt><dd className="ql-metric !text-2xl">{r.wins}–{r.losses}{r.wins + r.losses > 0 && <span className="ml-2 text-xs font-medium text-text-sec">{pct(r.wins / (r.wins + r.losses))}</span>}</dd></div>
  );
  return <dl className="ql-panel mb-4 grid grid-cols-3 gap-3 p-4" aria-label="Model record for the day">{cell("Winners", rec.winner)}{cell("Vs spread", rec.ats)}{cell("Over / under", rec.total)}</dl>;
}

export function CbbSlate() {
  const params = useSearchParams();
  const health = useQuery({ queryKey: ["cbb-health"], queryFn: cbbApi.health, staleTime: 300_000 });
  const days = health.data?.schedule.days;
  const today = todayEt();
  const requested = params.get("date");
  const [picked, setPicked] = useState<string | null>(requested && /^\d{4}-\d{2}-\d{2}$/.test(requested) ? requested : null);
  const date = picked ?? (days ? defaultDay(days, today) : undefined);
  const [view, setView] = useState<(typeof VIEWS)[number]["id"]>("all");
  const [conference, setConference] = useState("");
  const [shown, setShown] = useState(PAGE);

  const schedule = useQuery({ queryKey: ["cbb-schedule", date], queryFn: () => cbbApi.schedule(date!), enabled: !!date, staleTime: 120_000 });
  const games = useMemo(() => schedule.data?.games ?? [], [schedule.data]);
  const visible = useMemo(() => filterGames(games, { view, conference }), [games, view, conference]);
  const conferences = useMemo(() => conferencesOf(games), [games]);
  const available = days?.map(d => d.date) ?? [];

  // The strip starts at the oldest day; bring the open day into view.
  useEffect(() => {
    document.querySelector<HTMLElement>('nav[aria-label="Game dates"] [aria-pressed="true"]')?.scrollIntoView({ inline: "center", block: "nearest" });
  }, [date, days]);

  const choose = (next: string) => {
    setPicked(next); setShown(PAGE);
    const url = new URL(window.location.href);
    url.searchParams.set("date", next);
    window.history.replaceState(null, "", url);
  };

  return (
    <div className="ql-page">
      <CbbHeader title="College hoops" sub={`Every Division I game with a model win probability, spread and total next to the sportsbook line.${health.data?.ap_week ? ` AP poll: ${health.data.ap_week}.` : ""}`} />
      {health.isError && <ErrorBox onRetry={() => void health.refetch()}>College data is unavailable.</ErrorBox>}
      {days && date && <DateStrip dates={available} date={date} today={today} onChange={choose} />}
      <div className="mt-4 flex flex-wrap items-center gap-2" role="group" aria-label="Filters">
        {VIEWS.map(v => <button key={v.id} type="button" aria-pressed={view === v.id} onClick={() => { setView(v.id); setShown(PAGE); }} className={`ql-chip ${view === v.id ? "ql-chip-active" : ""}`}>{v.label}</button>)}
        {conferences.length > 0 && (
          <select aria-label="Conference" value={conference} onChange={e => { setConference(e.target.value); setShown(PAGE); }} className="ql-control max-w-full px-3 text-xs font-bold">
            <option value="">All conferences</option>
            {conferences.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        )}
        {schedule.data && <span className="ml-auto font-mono text-[11px] text-text-sec">{visible.length} of {games.length} games</span>}
      </div>
      <div className="mt-4">
        {schedule.isPending && !health.isError && <Loading label="Loading games" />}
        {schedule.isError && <ErrorBox onRetry={() => void schedule.refetch()}>We couldn&apos;t load this day.</ErrorBox>}
        {schedule.data && <>
          <RecordBar games={games} />
          {games.length === 0 ? <p role="status" className="ql-panel p-5 text-sm text-text-sec">No games scheduled on this day.</p>
            : visible.length === 0 ? <p role="status" className="ql-panel p-5 text-sm text-text-sec">No games match these filters.</p>
            : <>
              <div className="grid items-start gap-3 md:grid-cols-2 xl:grid-cols-3">{visible.slice(0, shown).map(g => <CbbGameCard key={g.id} game={g} />)}</div>
              {visible.length > shown && <button type="button" onClick={() => setShown(s => s + PAGE)} className="ql-chip mx-auto mt-4">Show more ({visible.length - shown})</button>}
            </>}
        </>}
      </div>
    </div>
  );
}
