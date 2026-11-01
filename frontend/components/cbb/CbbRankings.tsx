"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { cbbApi } from "@/lib/api";
import { signed } from "@/lib/cbb.mjs";
import type { CbbRankTeam } from "@/lib/cbb-types";
import { CbbHeader, CbbLogo, ErrorBox, Loading, RankTag } from "./parts";

type Key = "power_rank" | "ap_rank" | "off" | "def" | "net100" | "name";
const COLS: { key: Key; label: string; cls?: string; title: string }[] = [
  { key: "power_rank", label: "#", title: "Model power rank" },
  { key: "name", label: "Team", title: "Team name" },
  { key: "ap_rank", label: "AP", cls: "hidden sm:table-cell", title: "AP Top 25 rank" },
  { key: "off", label: "Off", cls: "hidden md:table-cell", title: "Points per game above an average D1 offense" },
  { key: "def", label: "Def", cls: "hidden md:table-cell", title: "Points per game allowed vs average; lower is better" },
  { key: "net100", label: "Net/100", title: "Net rating per 100 possessions" },
];

const value = (t: CbbRankTeam, key: Key): number | string => key === "ap_rank" ? (t.ap_rank ?? 999) : key === "name" ? t.short : t[key];

export function CbbRankings() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["cbb-rankings"], queryFn: cbbApi.rankings, staleTime: 600_000 });
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: "power_rank", dir: 1 });
  const [q, setQ] = useState("");
  const [conf, setConf] = useState("");
  const [shown, setShown] = useState(75);
  const conferences = useMemo(() => [...new Set((data?.teams ?? []).map(t => t.conference).filter(Boolean) as string[])].sort(), [data]);
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (data?.teams ?? [])
      .filter(t => (!conf || t.conference === conf) && (!needle || t.name.toLowerCase().includes(needle)))
      .sort((a, b) => {
        const x = value(a, sort.key), y = value(b, sort.key);
        return (typeof x === "string" ? x.localeCompare(String(y)) : Number(x) - Number(y)) * sort.dir;
      });
  }, [data, sort, q, conf]);
  // Lower is better for ranks and defense; everything else starts high-to-low.
  const click = (key: Key) => setSort(s => s.key === key ? { key, dir: s.dir === 1 ? -1 : 1 } : { key, dir: ["power_rank", "ap_rank", "def", "name"].includes(key) ? 1 : -1 });

  return (
    <div className="ql-page">
      <CbbHeader title="Power rankings" sub={data ? `${data.teams.length} Division I teams by the model's ratings, season ${data.season}.${data.ap_week ? ` AP poll: ${data.ap_week}.` : ""} ${data.ratings_note ?? ""}` : undefined} />
      {isPending && <Loading label="Loading rankings" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>Rankings are unavailable.</ErrorBox>}
      {data && <>
        <div className="mb-3 flex flex-wrap gap-2">
          <input type="search" value={q} onChange={e => { setQ(e.target.value); setShown(75); }} placeholder="Find a team" aria-label="Find a team" className="ql-control min-w-0 flex-1 px-3 text-sm sm:max-w-xs" />
          <select aria-label="Conference" value={conf} onChange={e => { setConf(e.target.value); setShown(75); }} className="ql-control px-3 text-xs font-bold">
            <option value="">All conferences</option>
            {conferences.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="ql-panel overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b border-border-card text-left">
              {COLS.map(c => (
                <th key={c.key} scope="col" aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"} className={`${c.cls ?? ""} p-0`}>
                  <button type="button" title={c.title} onClick={() => click(c.key)} className="ql-data-label w-full px-3 py-3 text-left hover:text-text-pri">{c.label}{sort.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}</button>
                </th>
              ))}
              <th scope="col" className="ql-data-label hidden px-3 py-3 lg:table-cell">Record</th>
              <th scope="col" className="ql-data-label hidden px-3 py-3 lg:table-cell">Last 10</th>
            </tr></thead>
            <tbody>
              {rows.slice(0, shown).map(t => (
                <tr key={t.id} className="border-b border-ink/[0.06] last:border-0">
                  <td className="px-3 py-2 font-mono text-xs tabular-nums text-text-sec">{t.power_rank}</td>
                  <td className="px-3 py-2"><span className="flex items-center gap-2.5"><CbbLogo team={t} size={26} /><span className="min-w-0"><span className="block truncate font-bold text-text-pri"><RankTag rank={t.ap_rank} />{t.short}</span><span className="block truncate font-mono text-[10px] text-text-sec">{t.conference} · {t.record}</span></span></span></td>
                  <td className="hidden px-3 py-2 font-mono text-xs tabular-nums sm:table-cell">{t.ap_rank ?? "—"}{t.ap_rank && t.ap_previous ? <span className="ml-1 text-text-sec">({t.ap_previous})</span> : null}</td>
                  <td className="hidden px-3 py-2 font-mono text-xs tabular-nums md:table-cell">{signed(t.off)}</td>
                  <td className="hidden px-3 py-2 font-mono text-xs tabular-nums md:table-cell">{signed(t.def)}</td>
                  <td className="px-3 py-2 font-mono text-xs font-bold tabular-nums text-teal-300">{signed(t.net100)}</td>
                  <td className="hidden px-3 py-2 font-mono text-xs tabular-nums lg:table-cell">{t.record}</td>
                  <td className="hidden px-3 py-2 font-mono text-xs tabular-nums lg:table-cell">{t.last10 ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length > shown && <button type="button" onClick={() => setShown(s => s + 100)} className="ql-chip mx-auto mt-4">Show more ({rows.length - shown})</button>}
        {rows.length === 0 && <p role="status" className="mt-3 text-sm text-text-sec">No teams match.</p>}
      </>}
    </div>
  );
}
