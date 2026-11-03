"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { nhlApi } from "@/lib/api";
import { MARKETS, lineLabel, tipText } from "@/lib/nhl.mjs";
import { SORTS, boardGames, boardRows, initialLine, lineOptions } from "@/lib/nhl-props.mjs";
import type { NhlMarketKey } from "@/lib/nhl-types";
import { ErrorBox, FairOddsNote, Loading, NhlHeader } from "./parts";
import { NhlPropRow } from "./NhlPropRow";
import { NhlPlayerSheet } from "./NhlPlayerSheet";

const PAGE = 40;
const dayLabel = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });

export function NhlProps() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["nhl-props"], queryFn: nhlApi.props, staleTime: 120_000 });
  const [market, setMarket] = useState<NhlMarketKey>("goal");
  const [lines, setLines] = useState<Partial<Record<NhlMarketKey, number>>>({});
  const [pickedDate, setDate] = useState("");
  const [gameId, setGameId] = useState("");
  const [team, setTeam] = useState("");
  const [sort, setSort] = useState("p");
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);
  const [open, setOpen] = useState<number | null>(null);

  const players = useMemo(() => data?.players ?? [], [data]);
  const date = pickedDate || data?.dates[0] || "";
  const dayPlayers = useMemo(() => players.filter(p => p.date === date), [players, date]);
  const options = useMemo(() => lineOptions(dayPlayers, market), [dayPlayers, market]);
  const line = lines[market] != null && options.includes(lines[market]!) ? lines[market]! : initialLine(options, market);
  const games = useMemo(() => boardGames(players, date), [players, date]);
  const teams = useMemo(() => [...new Set(dayPlayers.map(p => p.team))].sort(), [dayPlayers]);
  const rows = useMemo(() => boardRows(players, { market, line, date, gameId, team, query, sort }), [players, market, line, date, gameId, team, query, sort]);
  const reset = () => setShown(PAGE);

  return (
    <div className="ql-page">
      <NhlHeader title="NHL props" sub="Anytime goal, shots on goal, assists and points for every player projected to dress, as model probability and fair odds." />
      <div className="mb-4"><FairOddsNote extra={data?.lineup_note ? <> {data.lineup_note}</> : undefined} /></div>
      {isPending && <Loading label="Loading props" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>The props board is unavailable.</ErrorBox>}
      {data && players.length === 0 && <p role="status" className="ql-panel p-5 text-sm text-text-sec">No props are posted right now. The board fills in once lineups for the next slate are projected.</p>}
      {data && players.length > 0 && <>
        {data.dates.length > 1 && (
          <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Day">
            {data.dates.map(d => <button key={d} type="button" aria-pressed={d === date} onClick={() => { setDate(d); setGameId(""); setTeam(""); reset(); }} className={`ql-chip ${d === date ? "ql-chip-active" : ""}`}>{dayLabel(d)}</button>)}
          </div>
        )}
        <div className="mb-3 flex gap-2 overflow-x-auto pb-1 [scrollbar-width:none]" role="tablist" aria-label="Market">
          {MARKETS.map(m => <button key={m.key} type="button" role="tab" aria-selected={market === m.key} onClick={() => { setMarket(m.key); reset(); }} className={`ql-chip shrink-0 ${market === m.key ? "ql-chip-active" : ""}`}>{m.label}</button>)}
        </div>
        {options.length > 1 && (
          <div className="mb-3 flex flex-wrap items-center gap-2" role="group" aria-label="Line">
            <span className="ql-data-label">LINE</span>
            {options.map(l => <button key={l} type="button" aria-pressed={line === l} onClick={() => { setLines(s => ({ ...s, [market]: l })); reset(); }} className={`ql-chip ${line === l ? "ql-chip-active" : ""}`}>{l} <span className="opacity-70">({lineLabel(market, l)})</span></button>)}
          </div>
        )}
        <div className="mb-3 flex flex-wrap gap-2">
          <input type="search" value={query} onChange={e => { setQuery(e.target.value); reset(); }} placeholder="Find a player or team" aria-label="Find a player or team" className="ql-control min-w-0 basis-full px-3 text-sm sm:max-w-xs sm:basis-auto sm:flex-1" />
          <select aria-label="Game" value={gameId} onChange={e => { setGameId(e.target.value); reset(); }} className="ql-control max-w-full px-3 text-xs font-bold">
            <option value="">All games</option>
            {games.map(g => <option key={g.id} value={g.id}>{g.away} @ {g.home} · {tipText(g.start).replace(" ET", "")}</option>)}
          </select>
          <select aria-label="Team" value={team} onChange={e => { setTeam(e.target.value); reset(); }} className="ql-control px-3 text-xs font-bold">
            <option value="">All teams</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <select aria-label="Sort by" value={sort} onChange={e => { setSort(e.target.value); reset(); }} className="ql-control px-3 text-xs font-bold">
            {SORTS.map(s => <option key={s.key} value={s.key}>Sort: {s.label}</option>)}
          </select>
        </div>
        <p className="mb-2 font-mono text-[11px] text-text-sec">{rows.length} players · over {line} {MARKETS.find(m => m.key === market)?.unit}</p>
        {rows.length === 0 ? <p role="status" className="ql-panel p-5 text-sm text-text-sec">No players match these filters.</p> : (
          <ul className="ql-panel divide-y divide-ink/[0.07] overflow-hidden">
            {rows.slice(0, shown).map(r => <NhlPropRow key={r.player.player_id} row={r} market={market} onOpen={setOpen} />)}
          </ul>
        )}
        {rows.length > shown && <button type="button" onClick={() => setShown(s => s + PAGE)} className="ql-chip mx-auto mt-4">Show more ({rows.length - shown})</button>}
      </>}
      {open != null && <NhlPlayerSheet playerId={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
