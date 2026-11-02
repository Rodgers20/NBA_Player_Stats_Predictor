"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { cbbApi } from "@/lib/api";
import { STATS, POSITIONS, bestFirst, boardRows, defaultSeason, formatStat, seasonLabel, statValue } from "@/lib/cbb-players.mjs";
import type { BoardRow } from "@/lib/cbb-players.mjs";
import { TeamLogo } from "@/components/teams/TeamLogo";
import { CbbHeader, ErrorBox, Loading } from "./parts";
import { CbbPlayerSheet } from "./CbbPlayerSheet";
import { Headshot } from "./Headshot";

const PAGE = 50;
const MIN_GAMES = [1, 5, 10, 20];

export function CbbPlayers() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["cbb-players"], queryFn: cbbApi.players, staleTime: 600_000 });
  const [pickedSeason, setSeason] = useState<number | null>(null);
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "pts", dir: -1 });
  const [q, setQ] = useState("");
  const [position, setPosition] = useState("");
  const [minGames, setMinGames] = useState<number | null>(null);
  const [shown, setShown] = useState(PAGE);
  const [open, setOpen] = useState<BoardRow | null>(null);

  const season = pickedSeason ?? (data ? defaultSeason(data) : 0);
  const finished = !!data && season !== data.season;
  const floor = minGames ?? (finished ? 10 : 1);
  const rows = useMemo(() => data ? boardRows(data.players, { season, sort: sort.key, dir: sort.dir, minGames: floor, query: q, position }) : [], [data, season, sort, floor, q, position]);
  const lead = rows[0]?.value || 1;
  const reset = () => setShown(PAGE);
  const click = (key: string) => { setSort(s => s.key === key ? { key, dir: s.dir === 1 ? -1 : 1 } : { key, dir: bestFirst(key) }); reset(); };
  const sortSpec = STATS.find(s => s.key === sort.key);

  return (
    <div className="ql-page">
      <CbbHeader title="Players"
        sub={data ? `${rows.length.toLocaleString()} Division I players ranked by ${sortSpec?.title.toLowerCase()}, ${seasonLabel(season)}.${finished && pickedSeason == null ? ` The ${seasonLabel(data.season)} season has only just started, so last season's finished numbers show first. Switch seasons any time.` : ""}` : undefined} />
      {isPending && <Loading label="Loading players" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>Player stats are unavailable.</ErrorBox>}
      {data && <>
        <div className="mb-3 flex flex-wrap gap-2">
          <input type="search" value={q} onChange={e => { setQ(e.target.value); reset(); }} placeholder="Find a player or team" aria-label="Find a player or team" className="ql-control min-w-0 flex-1 px-3 text-sm sm:max-w-xs" />
          <select aria-label="Season" value={season} onChange={e => { setSeason(Number(e.target.value)); setMinGames(null); reset(); }} className="ql-control px-3 text-xs font-bold">
            {data.seasons.map(s => <option key={s} value={s}>{seasonLabel(s)}</option>)}
          </select>
          <select aria-label="Position" value={position} onChange={e => { setPosition(e.target.value); reset(); }} className="ql-control px-3 text-xs font-bold">
            {POSITIONS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
          <select aria-label="Minimum games" value={floor} onChange={e => { setMinGames(Number(e.target.value)); reset(); }} className="ql-control px-3 text-xs font-bold">
            {MIN_GAMES.map(n => <option key={n} value={n}>{n === 1 ? "Any games played" : `${n}+ games`}</option>)}
          </select>
        </div>
        <div className="ql-panel overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm">
            <thead><tr className="border-b border-border-card text-left">
              <th scope="col" className="ql-data-label sticky left-0 z-10 w-10 bg-bg-card px-3 py-3">#</th>
              <th scope="col" className="ql-data-label sticky left-10 z-10 bg-bg-card px-3 py-3">Player</th>
              {STATS.map(c => (
                <th key={c.key} scope="col" aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"} className="p-0">
                  <button type="button" title={c.title} onClick={() => click(c.key)}
                    className={`ql-data-label w-full px-3 py-3 text-right hover:text-text-pri ${sort.key === c.key ? "!text-text-pri" : ""}`}>{c.label}{sort.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}</button>
                </th>
              ))}
            </tr></thead>
            <tbody>
              {rows.slice(0, shown).map(r => (
                <tr key={r.id} className="border-b border-ink/[0.06] last:border-0 hover:bg-surface-hover/40">
                  <td className="sticky left-0 z-10 bg-bg-card px-3 py-2 font-mono text-xs tabular-nums text-text-sec">{r.rank}</td>
                  <td className="sticky left-10 z-10 bg-bg-card px-3 py-2">
                    <button type="button" onClick={() => setOpen(r)} className="flex min-w-[190px] items-center gap-2.5 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
                      <Headshot id={r.id} name={r.name} size={34} />
                      <span className="min-w-0">
                        <span className="block truncate font-bold text-text-pri">{r.name}</span>
                        <span className="flex items-center gap-1.5 font-mono text-[10px] text-text-sec"><TeamLogo team={{ abbrev: r.abbr ?? "", name: r.team ?? "", logo: r.logo }} size={14} />{r.team}{r.pos ? ` · ${r.pos}` : ""}</span>
                      </span>
                    </button>
                  </td>
                  {STATS.map(c => {
                    const active = c.key === sort.key;
                    return (
                      <td key={c.key} className={`relative px-3 py-2 text-right font-mono text-xs tabular-nums ${active ? "font-bold text-text-pri" : "text-text-soft"}`}>
                        {active && !c.low && <span aria-hidden="true" className="absolute inset-y-1.5 right-0 rounded-l bg-teal-400/15" style={{ width: `${Math.max(4, (r.value / lead) * 100)}%` }} />}
                        <span className="relative">{formatStat(statValue(r.line, c.key), c.key)}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length > shown && <button type="button" onClick={() => setShown(s => s + 100)} className="ql-chip mx-auto mt-4">Show more ({rows.length - shown})</button>}
        {rows.length === 0 && <p role="status" className="mt-3 text-sm text-text-sec">No players match. Lower the minimum games or clear the search.</p>}
      </>}
      {open && <CbbPlayerSheet onClose={() => setOpen(null)} player={{ id: open.id, name: open.name, pos: open.pos, team: open.team, abbr: open.abbr, logo: open.logo, season, line: open.line }} />}
    </div>
  );
}
