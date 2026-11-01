"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { cbbApi } from "@/lib/api";
import { signed } from "@/lib/cbb.mjs";
import { CbbHeader, CbbLogo, ErrorBox, Loading, RankTag } from "./parts";

export function CbbStandings() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["cbb-standings"], queryFn: cbbApi.standings, staleTime: 600_000 });
  const [conf, setConf] = useState("");
  const list = useMemo(() => (data?.conferences ?? []).filter(c => c.teams.length && (!conf || c.id === conf)), [data, conf]);
  return (
    <div className="ql-page">
      <CbbHeader title="Standings" sub={data ? `Conference standings, season ${data.season}. Sorted by conference win percentage.` : undefined}>
        {data && <select aria-label="Conference" value={conf} onChange={e => setConf(e.target.value)} className="ql-control px-3 text-xs font-bold">
          <option value="">All conferences</option>
          {data.conferences.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>}
      </CbbHeader>
      {isPending && <Loading label="Loading standings" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>Standings are unavailable.</ErrorBox>}
      <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
        {list.map(c => (
          <section key={c.id} aria-label={c.name} className="ql-panel min-w-0 overflow-hidden">
            <h2 className="ql-section-title border-b border-border-card px-4 py-3 !text-lg">{c.name}</h2>
            <table className="w-full text-sm">
              <thead><tr className="text-left">
                <th className="ql-data-label px-4 py-2">Team</th><th className="ql-data-label px-2 py-2 text-right">Conf</th>
                <th className="ql-data-label px-2 py-2 text-right">All</th><th className="ql-data-label hidden px-2 py-2 text-right sm:table-cell">Diff</th>
                <th className="ql-data-label px-4 py-2 text-right">Strk</th>
              </tr></thead>
              <tbody>
                {c.teams.map(t => (
                  <tr key={t.id} className="border-t border-ink/[0.06]">
                    <td className="px-4 py-2"><span className="flex items-center gap-2.5"><CbbLogo team={t} size={22} /><span className="truncate font-bold text-text-pri"><RankTag rank={t.ap_rank} />{t.short}</span></span></td>
                    <td className="px-2 py-2 text-right font-mono text-xs tabular-nums">{t.conf_record}</td>
                    <td className="px-2 py-2 text-right font-mono text-xs tabular-nums">{t.overall}</td>
                    <td className="hidden px-2 py-2 text-right font-mono text-xs tabular-nums sm:table-cell">{t.avg_diff == null ? "—" : signed(t.avg_diff)}</td>
                    <td className={`px-4 py-2 text-right font-mono text-xs ${t.streak?.startsWith("W") ? "text-teal-300" : t.streak ? "text-rose-300" : "text-text-sec"}`}>{t.streak ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        ))}
      </div>
    </div>
  );
}
