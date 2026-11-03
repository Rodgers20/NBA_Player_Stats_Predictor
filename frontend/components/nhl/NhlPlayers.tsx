"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { nhlApi } from "@/lib/api";
import { num } from "@/lib/nhl.mjs";
import { AbbrLogo, ErrorBox, Headshot, Loading, NhlHeader, FRAME } from "./parts";
import { NhlPlayerAnalysis } from "./NhlPlayerAnalysis";

const PAGE = 50;
const fold = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

/** The analysis page for one player (?player=ID). */
function PlayerPage({ id }: { id: string }) {
  const q = useQuery({ queryKey: ["nhl-player", Number(id)], queryFn: () => nhlApi.player(id), staleTime: 300_000, retry: false, enabled: /^\d{5,9}$/.test(id) });
  const back = <Link href="/nhl/players" className="mb-4 inline-flex items-center gap-1.5 text-sm text-text-sec hover:text-text-pri"><ArrowLeft size={16} aria-hidden /> All players</Link>;
  return (
    <div className={FRAME}>
      {back}
      {!/^\d{5,9}$/.test(id) ? <ErrorBox>That player link is not valid.</ErrorBox>
        : q.isPending ? <Loading label="Loading player" />
        : q.isError ? <ErrorBox>This player isn&apos;t in the data. Goalies and players with fewer than 5 games are not covered.</ErrorBox>
        : <NhlPlayerAnalysis d={q.data} />}
    </div>
  );
}

export function NhlPlayers() {
  const picked = useSearchParams().get("player");
  if (picked) return <PlayerPage id={picked} />;
  return <PlayerList />;
}

function PlayerList() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["nhl-players"], queryFn: nhlApi.players, staleTime: 600_000 });
  const [q, setQ] = useState("");
  const [pos, setPos] = useState("");
  const [shown, setShown] = useState(PAGE);
  const rows = useMemo(() => {
    const needle = fold(q).trim();
    return (data?.players ?? []).filter(p => (!pos || (pos === "D" ? p.pos === "D" : p.pos !== "D")) && (!needle || fold(p.name).includes(needle) || fold(p.team).includes(needle)));
  }, [data, q, pos]);
  return (
    <div className={FRAME}>
      <NhlHeader kicker="PLAYERS" title="Players" sub="Search any skater for his game log, hit rates against each line, splits and the model's projection for his next game. Goalies are not covered." />
      {isPending && <Loading label="Loading players" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>Player data is unavailable.</ErrorBox>}
      {data && <>
        <div className="mb-3 flex flex-wrap gap-2">
          <input type="search" value={q} onChange={e => { setQ(e.target.value); setShown(PAGE); }} placeholder="Find a player or team" aria-label="Find a player or team" className="ql-control min-w-0 basis-full px-3 text-sm sm:max-w-xs sm:basis-auto sm:flex-1" />
          <select aria-label="Position" value={pos} onChange={e => { setPos(e.target.value); setShown(PAGE); }} className="ql-control px-3 text-xs font-bold">
            <option value="">All positions</option><option value="F">Forwards</option><option value="D">Defensemen</option>
          </select>
          <span className="self-center font-mono text-[11px] text-text-sec">{rows.length.toLocaleString()} players · sorted by points</span>
        </div>
        <div className="ql-panel overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead><tr className="border-b border-border-card text-left"><th className="ql-data-label px-3 py-3">Player</th>{["GP", "G", "A", "PTS", "SOG", "TOI"].map(h => <th key={h} className="ql-data-label px-3 py-3 text-right">{h}</th>)}</tr></thead>
            <tbody>{rows.slice(0, shown).map(p => (
              <tr key={p.id} className="border-b border-ink/[0.06] last:border-0 hover:bg-surface-hover/40">
                <td className="px-3 py-2"><Link href={`/nhl/players?player=${p.id}`} className="flex min-w-[190px] items-center gap-2.5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
                  <Headshot src={p.headshot} name={p.name} size={34} />
                  <span className="min-w-0"><span className="block truncate font-bold text-text-pri">{p.name}</span><span className="flex items-center gap-1.5 font-mono text-[10px] text-text-sec"><AbbrLogo abbr={p.team} size={14} />{p.team} · {p.pos}</span></span>
                </Link></td>
                {[p.gp, p.goals, p.assists, p.points, p.sog, num(p.toi)].map((v, i) => <td key={i} className={`px-3 py-2 text-right font-mono text-xs tabular-nums ${i === 3 ? "font-bold text-text-pri" : "text-text-soft"}`}>{v}</td>)}
              </tr>
            ))}</tbody>
          </table>
        </div>
        {rows.length > shown && <button type="button" onClick={() => setShown(s => s + 100)} className="ql-chip mx-auto mt-4">Show more ({rows.length - shown})</button>}
        {rows.length === 0 && <p role="status" className="mt-3 text-sm text-text-sec">No players match.</p>}
      </>}
    </div>
  );
}
